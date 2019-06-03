import bpy, bmesh, time, mathutils;
import numpy as np;
import scipy as sp;
import scipy.sparse as spsp;
from scipy.sparse.linalg import eigsh;

from mathutils import Vector;

#Return object bounds as minvector and maxvector
def getObjectBounds(mesh):
    minx = miny = minz = 9999999999999999;
    maxx = maxy = maxz = -9999999999999999;    
    bounds = [Vector(b) for b in mesh.bound_box];

    xvalues = [b.x for b in bounds];
    yvalues = [b.y for b in bounds];
    zvalues = [b.z for b in bounds];
    
    minx = min(xvalues);
    maxx = max(xvalues);

    miny = min(yvalues);
    maxy = max(yvalues);

    minz = min(zvalues);
    maxz = max(zvalues);
    
    return Vector((min(xvalues), min(yvalues), min(zvalues))), Vector((max(xvalues), max(yvalues),  max(zvalues)));

def getBBox(m):
    min_coords, max_coords = getObjectBounds(m);
    dimensionvector = max_coords - min_coords;
    diameter = dimensionvector.length;
    return min_coords, max_coords, diameter;

def meanCurvatureLaplaceWeights(context, mesh, symmetric = False, normalized=False):
    start = time.time();
    
    bm = getBMMesh(context, mesh, useeditmode=False);
    ensurelookuptable(bm);
    
    rows = [];
    cols = [];
    data = [];
    n = len(mesh.data.vertices);
    for f in bm.faces:
        v1, v2, v3 = [l.vert for l in f.loops];
        v1v2 = v1.co - v2.co;
        v1v3 = v1.co - v3.co;
        
        v2v1 = v2.co - v1.co;
        v2v3 = v2.co - v3.co;
        
        v3v1 = v3.co - v1.co;
        v3v2 = v3.co - v2.co;            
        
        cot1 = v2v1.dot(v3v1) / max(v2v1.cross(v3v1).length,1e-06);
        cot2 = v3v2.dot(v1v2) / max(v3v2.cross(v1v2).length,1e-06);
        cot3 = v1v3.dot(v2v3) / max(v1v3.cross(v2v3).length,1e-06);
        
        rows.append(v2.index);
        cols.append(v3.index);
        data.append(cot1);
        
        rows.append(v3.index);
        cols.append(v2.index);
        data.append(cot1);
        
        rows.append(v3.index);
        cols.append(v1.index);
        data.append(cot2);
        
        rows.append(v1.index);
        cols.append(v3.index);
        data.append(cot2);
        
        rows.append(v1.index);
        cols.append(v2.index);
        data.append(cot3);
        
        rows.append(v2.index);
        cols.append(v1.index);
        data.append(cot3);           
    
    W = spsp.csr_matrix((data, (rows, cols)), shape = (n,n));
    
    if(symmetric and not normalized):
        sum_vector = W.sum(axis=0);
        d = spsp.dia_matrix((sum_vector, [0]), shape=(n,n));
        L = d - W;
    elif(symmetric and normalized):
        sum_vector = W.sum(axis=0);
        sum_vector_powered = np.power(sum_vector, -0.5);
        d = spsp.dia_matrix((sum_vector_powered, [0]), shape=(n,n));
        eye = spsp.identity(n);
        L = eye - d * W * d;
    elif (not symmetric and normalized):
        sum_vector = W.sum(axis=0);
        sum_vector_powered = np.power(sum_vector, -1.0);
        d = spsp.dia_matrix((sum_vector_powered, [0]), shape=(n,n));
        eye = spsp.identity(n);
        L = eye - d * W;
    else:
        L = W;
    
    bm.free();
       
    if(not context.mode == "OBJECT"):
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False);
    
    end = time.time();
    print("FINISHED CONSTRUCTING WEIGHTS FOR ", mesh.name, " IN ", (end - start)); 
    return L;

#Purpose: To return a sparse matrix representing a laplacian matrix with
#cotangent weights in the upper square part and anchors as the lower rows
#Inputs: mesh (polygon mesh object), anchorsIdx (indices of the anchor points)
#Returns: L (An (N+K) x N sparse matrix, where N is the number of vertices
#and K is the number of anchors)
def getLaplacianMatrixCotangent(context, mesh, anchorsIdx=[], anchorWeights=[], *, defaultWeight=1.0):
    (I, J, V, _) = getLaplacianMeshUpperIdxs(context, mesh, True);
    #Now fill in the anchors and finish making the Laplacian matrix
    N = len(mesh.data.vertices);
    K = len(anchorsIdx);
    for i in range(K):
        I.append(N+i);
        J.append(anchorsIdx[i]);
        V.append(1.0);
    [I, J, V] = [np.array(I), np.array(J), np.array(V)];
    L = spsp.coo_matrix((V, (I, J)), shape=(N+K, N)).tocsr();
    return L;

def getLaplacianMeshNormalized(context, mesh, cotangent = False):
    (I, J, V, weights) = getLaplacianMeshUpperIdxs(context, mesh, cotangent);
    N = len(mesh.data.vertices);
    L = spsp.coo_matrix((V, (I, J)), shape=(N, N)).tocsr();
    weights = np.array(weights)[:, None];
    weights[weights == 0] = 1;
    L = L/weights;
    return L;

def getFaceCotangent(v1, v2, f, mesh):
    if not f:
        return 0.0;
    vs = f.verts;
    v3 = None
    for v in vs:
        if (not v == v1) and (not v == v2):
            v3 = v
            break;
    if not v3:
        return 0.0;
    
    [i1, i2, i3] = [v1.index, v2.index, v3.index];
    
    dV1 = v1.co - v3.co;
    dV2 = v2.co - v3.co;
    
    m1 = dV1.length;
    m2 = dV2.length;
    
    num = dV1.dot(dV2);
    denom = dV1.cross(dV2);
    denom = denom.length;
    if np.abs(denom) > 0:
        return num/denom;
    return 0.0;

#Helper function for laplacian mesh
def getLaplacianMeshUpperIdxs(context, mesh, cotangent = False, overwriteIdxs = []):
    
    bm = getBMMesh(context, mesh, useeditmode=False);
    ensurelookuptable(bm);
    
    N = len(bm.verts);
    I = [];
    J = [];
    V = [];
    weights = [];
    overwriteIdxs = sorted(overwriteIdxs);
    oidx = 0;
    for i in range(N):
        if oidx < len(overwriteIdxs):
            if overwriteIdxs[oidx] == i:
                I.append(i);
                J.append(i);
                V.append(1.0);
                oidx += 1;
                continue;
        v1 = bm.verts[i];
        neighbs = [e.other_vert(v1) for e in v1.link_edges];#v1.getVertexNeighbors()
        edges = [e for e in v1.link_edges];
        totalWeight = 0.0;
        for j, v2 in enumerate(neighbs):
            I.append(i);
            J.append(v2.index);
            weight = 1.0;
            if cotangent:
                weight = 0.0;
                e = edges[j];
                try:
                    weight += 0.5 * getFaceCotangent(v1, v2, e.link_faces[0], mesh);
                except IndexError:
                    pass;
                try:
                    weight += 0.5 * getFaceCotangent(v1, v2, e.link_faces[1], mesh);
                except IndexError:
                    pass;
            V.append(-weight);
            totalWeight += weight;
        I.append(i);
        J.append(i);
        V.append(totalWeight);
        weights.append(totalWeight);
    
    bm.free();
    return (I, J, V, weights);

##############################################################
##                  Laplacian Mesh Editing                  ##
##############################################################

#Purpose: To return a sparse matrix representing a laplacian matrix with
#the graph laplacian (D - A) in the upper square part and anchors as the
#lower rows
#Inputs: mesh (polygon mesh object), anchorsIdx (indices of the anchor points)
#Returns: L (An (N+K) x N sparse matrix, where N is the number of vertices
#and K is the number of anchors)
def getLaplacianMatrixUmbrella(context, mesh, anchorsIdx=[]):
    (I, J, V, _) = getLaplacianMeshUpperIdxs(context, mesh);
    #Now fill in the anchors and finish making the Laplacian matrix
    N = len(mesh.data.vertices);
    K = len(anchorsIdx);
    for i in range(K):
        I.append(N+i);
        J.append(anchorsIdx[i]);
        V.append(1.0);
    [I, J, V] = [np.array(I), np.array(J), np.array(V)];
    L = spsp.coo_matrix((V, (I, J)), shape=(N+K, N)).tocsr();
    return L;

#Purpose: To return a sparse matrix representing a laplacian matrix with
#cotangent weights in the upper square part and anchors as the lower rows
#Inputs: mesh (polygon mesh object), anchorsIdx (indices of the anchor points)
#Returns: L (An (N+K) x N sparse matrix, where N is the number of vertices
#and K is the number of anchors)
def getLaplacianMatrixCotangent(context, mesh, anchorsIdx=[], anchorWeights=[], *, defaultWeight=1.0):
    (I, J, V, _) = getLaplacianMeshUpperIdxs(context, mesh, True);
    #Now fill in the anchors and finish making the Laplacian matrix
    N = len(mesh.data.vertices);
    K = len(anchorsIdx);
    for i in range(K):
        I.append(N+i);
        J.append(anchorsIdx[i]);
        V.append(1.0);
    [I, J, V] = [np.array(I), np.array(J), np.array(V)];
    L = spsp.coo_matrix((V, (I, J)), shape=(N+K, N)).tocsr();
    return L;

def getLaplacianMeshNormalized(context, mesh, cotangent = False):
    (I, J, V, weights) = getLaplacianMeshUpperIdxs(context, mesh, cotangent);
    N = len(mesh.data.vertices);
    L = spsp.coo_matrix((V, (I, J)), shape=(N, N)).tocsr();
    weights = np.array(weights)[:, None];
    weights[weights == 0] = 1;
    L = L/weights;
    return L;

def angle(e1,e2):
    return np.arccos((e1*e2).sum(axis=1)/(np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)));


#Sparse matrix of weights for each vertex to its neighbours
def get_matC(context, mesh):
    ver = getMeshVPos(mesh);
    tri = getMeshFaces(mesh);
    
    v1 = ver[tri[:,0],:]
    v2 = ver[tri[:,1],:]
    v3 = ver[tri[:,2],:]        
    e1 = v2-v1
    e2 = v3-v2
    e3 = v1-v3  
    
    hcot_12 = 0.5/np.tan(angle(-e2,e3))
    hcot_13 = 0.5/np.tan(angle(-e1,e2))
    hcot_23 = 0.5/np.tan(angle(-e3,e1))
    
    data = np.array([hcot_12,hcot_23,hcot_13]).flatten()
    row_ind = np.array([tri[:,0],tri[:,1],tri[:,2]]).flatten()
    col_ind = np.array([tri[:,1],tri[:,2],tri[:,0]]).flatten()
    M = N = ver.shape[0]
    matC = spsp.csr_matrix((data,(row_ind,col_ind)),shape=(M,N))
    matC = matC + matC.T
    dia = -matC.sum(axis=1).A.flatten()
    matC = matC + spsp.dia_matrix((dia,0), shape=(M,N))
    
    return matC;

#Sparse matrix of weights for each vertex to its neighbours
def get_matC2(context, mesh):
    vertices = getMeshVPos(mesh);
    faces = getMeshFaces(mesh);
    angles = getMeshFaceAngles(mesh);
    num_vertices = vertices.shape[0];    
    cot_angles = 1.0 / np.tan(angles);
    L = spsp.csr_matrix((num_vertices, num_vertices), dtype=np.double);
    for i in range(1,4):
        i1 = (i-1) % 3;
        i2 = (i) % 3;
        i3 = (i+1) % 3;
        v = -1.0 / np.tan(angles[:, i3]);
        i = faces[:,i1];
        j = faces[:,i2];
        add_mat = spsp.csr_matrix((v, (i, j)), shape=(num_vertices, num_vertices), dtype=np.double);
        L = L + add_mat;
        
    L = 0.5 * (L + L.T);
    diagonals = -np.sum(L,1).reshape(num_vertices, );
    D = spsp.dia_matrix((diagonals, [0]), shape=(num_vertices, num_vertices), dtype=np.double);
    L = D + L;    
    return L;

#Matrix M is always a diagonal matrix that is used for AX=b (example eigen decomposition_
def get_matM(context, mesh, ANormalize=False):
    ver = getMeshVPos(mesh);
    tri = getMeshFaces(mesh);
    
    v1 = ver[tri[:,0],:]
    v2 = ver[tri[:,1],:]
    v3 = ver[tri[:,2],:]        
    e1 = v2-v1
    e3 = v1-v3    
    
    norm_l = np.cross(e1,-e3,axis=1)
    norm_mag = np.linalg.norm(norm_l,axis=1)
#    norm_1 = norm_l/norm_mag[:,None]
    area = norm_mag/2
    
    data = (area/3).repeat(3)
    row_ind = col_ind = tri.flatten()
    M = N = ver.shape[0]
    matM = spsp.csr_matrix((data,(row_ind,col_ind)),shape=(M,N))
    return matM, matM;

#Matrix M is always a diagonal matrix that is used for AX=b (example eigen decomposition_
def get_matM2(context, mesh, ANormalize=True):
    def col(d, key):
        val = d[key];
        d[key] += 1;
        return val; 
    vertices = getMeshVPos(mesh);
    faces = getMeshFaces(mesh);
    num_vertices = vertices.shape[0];
    num_faces = faces.shape[0];
    angles = getMeshFaceAngles(mesh);
    cotangles = 1.0 / np.tan(angles);
    squared_edge_length = np.zeros((num_faces, 3));
    faces_area = np.zeros((num_faces, 1));
    onebyeight = 1.0 / 8.0;
    
    for i in range(1,4):
        i1 = ((i-1) % 3);
        i2 = (i % 3);
        i3 = ((i+1) % 3);        
        squared_edge_length[:,i1] = ((vertices[faces[:,i2],np.array([0])] - vertices[faces[:,i3],np.array([0])])**2);
            
    print('STARTING WITH FACES AREA COMPUTATION');
#    faces_area = 0.25 * np.sum(np.multiply(squared_edge_length, (1.0 / np.tan(angles))), axis=1);
    faces_area = 0.25 * np.sum(squared_edge_length * cotangles, axis=1);
    
    print('FINISHED FACES_AREA COMPUTATION');
    
    print('FINIDING INDICES OF VORONOI UNSAFE REGIONS ');
    #Voronoi safe triangles and their vertex ids
    n_o_t_i = np.where(np.max(angles, axis=1) < 1.571)[0];
    #Voronoi inappropriate vertices
    o_t_i = np.where(np.max(angles, axis=1) >= 1.571)[0];
    #vertex ids causing the obtuseness
    o_t_i_rows, o_t_i_vertices = np.where(angles[o_t_i] >= 1.571);
    #vertex ids causing the obtuseness
    o_t_i_n_o_rows, o_t_i_n_o_vertices = np.where(angles[o_t_i] < 1.571);
    
    o_t_i_rows = o_t_i[o_t_i_rows];    
    o_t_i_n_o_rows = o_t_i[o_t_i_n_o_rows];           
    o_t_i_vertices = faces[o_t_i_rows, o_t_i_vertices];
    o_t_i_n_o_vertices = faces[o_t_i_n_o_rows, o_t_i_n_o_vertices];
    
    print('FINDING VORONOI SAFE AREA VALUES');
    all_values = [];
    all_rows = [];
    all_cols = [];
    d = {i:0 for i in range(num_vertices)};
    for i in range(3):
        n1 = (i+1)%3;
        n2 = (i+2)%3;
        values = onebyeight * ((cotangles[n_o_t_i, n1] * squared_edge_length[n_o_t_i, n1]) + (cotangles[n_o_t_i, n2] * squared_edge_length[n_o_t_i, n2]));    
        rows = faces[n_o_t_i, i];
        cols = np.array([col(d, j) for j in rows]);
        all_values.append(values);
        all_rows.append(rows);
        all_cols.append(cols);
            
    values = np.array(all_values);
    rows = np.array(all_rows);
    cols = np.array(all_cols);
    addA = spsp.csr_matrix((values.flatten(), (rows.flatten(), cols.flatten())), shape=(num_vertices, num_vertices));    
    A = spsp.dia_matrix((addA.sum(axis=1).reshape(num_vertices, ), [0]), shape=(num_vertices, num_vertices));
    print('FINISHED FINDING VORONOI SAFE AREAS VALUES');
    A = A.diagonal();

    print('FINDING VORONOI UNSAFE AREAS VALUES');
    for i in range(o_t_i_vertices.shape[0]):
        vid = o_t_i_vertices[i];
        fid = o_t_i_rows[i];
        A[vid] = A[vid] + faces_area[fid] * 0.5;
    
    for i in range(o_t_i_n_o_vertices.shape[0]):
        vid = o_t_i_n_o_vertices[i];
        fid = o_t_i_n_o_rows[i];
        A[vid] = A[vid] + faces_area[fid] * 0.25;    
    
    print('FINISHED FINDING VORONOI UNSAFE AREAS VALUES');
    
    print('STARTING WITH AREA COMPUTATION FINALIZATION');
    
    A = np.maximum(A, 1e-8).reshape(A.shape[0], );
    if(ANormalize):
        area = A.sum();
        A = A / area;
    Am = spsp.dia_matrix((A, [0]), shape=(num_vertices, num_vertices));    
#     np.set_printoptions(precision=4, suppress=True);
    print('FINISHING WITH AREA COMPUTATION FINALIZATION');
#     sio.savemat(bpy.path.abspath('//matlab/%s.mat'%(mesh.name)), {'vertices':vertices, 'faces':faces+1});
    return Am, A;

# Only god knows how many more versions of voronoi and cotangent weights are lying out there
# Matrix M is always a diagonal matrix that is used for AX=b (example eigen decomposition_
def get_matM3(context, mesh, ANormalize=True):
    NP_PI = np.pi * 0.5;
    onebyeight = 1.0 / 8.0;
    #Positions of each vertex as a numpy N x 3  (float)
    vertices = getMeshVPos(mesh);
    #Vertex indices in each row representing a face index as numpy N x 3 (int)
    faces = getMeshFaces(mesh);    
    num_vertices = vertices.shape[0];
    num_faces = faces.shape[0];    
    #Angle of the corner of a face in each row representing a face index as numpy N x 3 (float)
    angles = getMeshFaceAngles(mesh);
    #Cotangent applied to all the angles in a face
    cotangles = 1.0 / np.tan(angles);
    squared_edge_length = np.zeros((num_faces, 3));
        
    print('STARTING WITH FACES AREA COMPUTATION');
#     Edges AB, BC, CA, AC
    AB = vertices[faces[:,1]] - vertices[faces[:, 0]];
    AC = vertices[faces[:,2]] - vertices[faces[:, 0]];
    BC = vertices[faces[:,2]] - vertices[faces[:, 1]];
    
    squared_edge_length[:,0] = np.sum(AB**2, axis=1);
    squared_edge_length[:,1] = np.sum(BC**2, axis=1);
    squared_edge_length[:,2] = np.sum(AC**2, axis=1);
    
#     CA = vertices[faces[:,0]] - vertices[faces[:, 2]];
    faces_area =  (np.sqrt(np.sum(np.cross(AB, AC)**2, axis=1)));
#     faces_area = 0.25 * np.sum(squared_edge_length * cotangles, axis=1);
    print('FINISHED FACES_AREA COMPUTATION ', faces_area.shape);
    faces_area = faces_area.repeat(3).reshape(num_faces,3) * 1.0;
    
    print('FINIDING INDICES OF VORONOI UNSAFE REGIONS ');
    #Voronoi safe triangles and their vertex ids
    n_o_t_i = np.where(np.max(angles, axis=1) < NP_PI)[0];
    
    #Voronoi inappropriate vertices. Find the rows of all face ids from angles > 90 degrees
    o_t_i = np.where(np.max(angles, axis=1) >= NP_PI)[0];    
    
    #vertex ids causing the obtuseness
    o_t_i_rows, o_t_i_vertices_cols = np.where(angles[o_t_i] >= NP_PI);
       
    #vertex ids of the neighbours of the vertices causing the obtuseness
    o_t_i_n_o_rows, o_t_i_n_o_vertices_cols = np.where(angles[o_t_i] < NP_PI);    
    
    o_t_i_rows = o_t_i[o_t_i_rows];    
#     o_t_i_vertices = faces[o_t_i_rows, o_t_i_vertices_cols];
    
    o_t_i_n_o_rows = o_t_i[o_t_i_n_o_rows];           
#     o_t_i_n_o_vertices = faces[o_t_i_n_o_rows, o_t_i_n_o_vertices_cols];    
    
    print('FINDING VORONOI SAFE AREA VALUES');
    all_values = [];
    all_rows = [];
    all_cols = [];
    d = {i:0 for i in range(num_vertices)};
    
    for i1 in range(3):
        i2 = (i1+1)%3;
        i3 = (i1+2)%3;
        rows = faces[n_o_t_i, i1];
        cols = faces[n_o_t_i, i1];
#         idea is edgelength x cot(opposite vertex angle) for all connected edges
        values = onebyeight * ((squared_edge_length[n_o_t_i, i3] * cotangles[n_o_t_i, i2]) + (squared_edge_length[n_o_t_i, i1] * cotangles[n_o_t_i, i3]));
        all_values.append(values);
        all_rows.append(rows);
        all_cols.append(cols);   
    
    
    rows1 = np.array(all_rows);
    cols1 = np.array(all_cols);
    values1 = np.array(all_values);
    print('FINISHED FINDING VORONOI SAFE AREAS VALUES');
    
    print('FINDING VORONOI UNSAFE AREAS VALUES');
    
    rows2 = faces[o_t_i_rows , o_t_i_vertices_cols];
    cols2 = faces[o_t_i_rows, o_t_i_vertices_cols];
    values2 = faces_area[o_t_i_rows, o_t_i_vertices_cols] * 0.25;
     
    rows3 = faces[o_t_i_n_o_rows, o_t_i_n_o_vertices_cols];
    cols3 = faces[o_t_i_n_o_rows, o_t_i_n_o_vertices_cols];
    values3 = faces_area[o_t_i_n_o_rows, o_t_i_n_o_vertices_cols] * 0.125;
    
    print('FINISHED FINDING VORONOI UNSAFE AREAS VALUES');
    
    print('STARTING WITH AREA COMPUTATION FINALIZATION');
    addA = spsp.csr_matrix((values1.flatten(), (rows1.flatten(), cols1.flatten())), shape=(num_vertices, num_vertices));
    addB = spsp.csr_matrix((values2.flatten(), (rows2.flatten(), cols2.flatten())), shape=(num_vertices, num_vertices));
    addC = spsp.csr_matrix((values3.flatten(), (rows3.flatten(), cols3.flatten())), shape=(num_vertices, num_vertices));     
    
    A = addA + addB + addC;
    
    if(ANormalize):
        area = A.data.sum();
        A = A.data / area;
    
    Am = spsp.dia_matrix((np.array(A.data).reshape(num_vertices, ), [0]), shape=(num_vertices, num_vertices));        
    print('FINISHING WITH AREA COMPUTATION FINALIZATION');
    
    return Am, np.array(A.data);

def get_eigen(matM,matC,n):
    eva, eve = eigsh(matC,k=n,M=matM,sigma=-1e-8,which='LM');
    print('SHAPE OF EIGENS :: %s, %s'%(eva.shape, eve.shape));
    return eva,eve

def get_eigen2(matM, matC, n):
    num_vertices = matC.shape[0];
    #Calculate the eigen values and vectors
#     eva, eve = eigs(L,k=K,M=A,sigma=-1e-5,which='LM');
    eva, eve = eigsh(matC,k=n,M=matM,sigma=-1e-8,which='LM');
    #Sort the eigen values from smallest to highest
    # and ensure to use the real part of eigen values, because there might be complex numbers
    eva = np.abs(np.real(eva));
    idx = np.argsort(eva);
    eva = eva[idx];
    eve = eve[:, idx];    
    eve = np.real(eve);
    print('SHAPE OF EIGENS :: %s, %s'%(eva.shape, eve.shape));
    return eva, eve;

def getWKSLaplacianMatrixCotangent(context, mesh, model=2):
    if(model == 1):
        return get_matC(context, mesh);
    return get_matC2(context, mesh);

def getMeshVoronoiAreas(context, mesh, model=2):
    if(model == 1):
        return get_matM(context, mesh);
    return get_matM2(context, mesh);

def getWKSEigens(mesh, L, A, K=3, model=1):
    if(model == 1):
        return get_eigen(A, L, K);    
    return get_eigen2(A, L, K);

def getMeshVoronoiAreasSlow(context, mesh):
    vertices = getMeshVPos(mesh);
    faces = getMeshFaces(mesh);
    angles = getMeshFaceAngles(mesh);
    squared_edge_length = 0*faces;
    
    num_vertices = vertices.shape[0];
    num_faces = faces.shape[0];
    print('STARTING WITH SQUARED LENGTH COMPUTATION');
    for i in range(1,4):
        i1 = ((i-1) % 3);
        i2 = (i % 3);
        i3 = ((i+1) % 3);
        diff = vertices[faces[:,i2]] - vertices[faces[:,i3]];
        squared_edge_length[:,i1] = np.sum(diff**2, axis=1);
    
    faces_area = np.zeros((num_faces,1));
    A = np.zeros((num_vertices, 1));
    
    onebyeight = 1.0 / 8.0;
    print('STARTING WITH FACES AREA COMPUTATION');
    for i in range(3):
        faces_area = faces_area + (0.25 * (squared_edge_length[:,i].dot((1.0 / np.tan(angles[:,i])))));    
    
    print('STARTING WITH AREA SORROUNDING POINTS COMPUTATION');
    angles = 1.0 / np.tan(angles);
    for i in range(num_vertices):
        for j in range(3):
            j1 = (j-1) % 3;
            j2 = j % 3;
            j3 = (j+1) % 3;
            ind_i = np.where(faces[:,j1] == i)[0];
            for l in ind_i:
                if(np.max(angles[l,:]) < 1.57):
#                     A[i] = A[i] + onebyeight * (1.0 / np.tan(angles[l, j2])) * squared_edge_length[l, j2] + (1.0 / np.tan(angles[l, j3])) * squared_edge_length[l, j3];
                    A[i] = A[i] + onebyeight * angles[l, j2] * squared_edge_length[l, j2] + angles[l, j3] * squared_edge_length[l, j3];
                elif (angles[l, j1] > 1.57):
                    A[i] = A[i] + faces_area[l] * 0.5;
                else:
                    A[i] = A[i] + faces_area[l] * 0.25;
    
    
    print('STARTING WITH AREA COMPUTATION FINALIZATION');
    A = np.maximum(A, 1e-8).reshape(A.shape[0], );
#     A.shape = (A.shape[0],);
    area = np.sum(A);
    A = A/area;
#     Am = spsp.dia_matrix(np.diag(A));
    Am = spsp.dia_matrix((A, [0]), shape=(num_vertices, num_vertices));
    print('FINISHING WITH AREA COMPUTATION FINALIZATION');
    return Am, A;

def setMeshVPOS(mesh, vpos):
    for index, vect in enumerate(vpos):
        mesh.data.vertices[index].co = vect;

def getMeshVPos(mesh, extra_points=[]):
    vpos = [];
    for v in mesh.data.vertices:
        vpos.append([v.co.x, v.co.y, v.co.z]);
    
    for p in extra_points:
        vpos.append([p.x, p.y, p.z]);
    
    return np.array(vpos);

def getMeshNormals(mesh):
    normals = np.zeros((N,3));
    for v in mesh.data.vertices:
        normals[v.index] = v.normal.normalized().to_tuple(); 

def getMeshFaceAngles(mesh):
    bm = getBMMesh(bpy.context, mesh, useeditmode=False);
    ensurelookuptable(bm);
    f_angles = [];
    for f in bm.faces:
        v_angles = [l.calc_angle() for l in f.loops];
        f_angles.append(v_angles);
    bm.free();
    return np.array(f_angles, dtype=float);

def getMeshFaces(mesh):
    loops = mesh.data.loops;
    faces = mesh.data.polygons;
    f_vids = [];
    for f in faces:
        vids = [];
        for lid in f.loop_indices:
            vids.append(loops[lid].vertex_index);
        f_vids.append(vids);
    
    return np.array(f_vids, dtype=int);

def getMeshFaceNormals(mesh):
    faces = mesh.data.polygons;
    np_face_normals = np.zeros((len(faces), 3), dtype=float);
    for f in faces:
        np_face_normals[f.index] = f.normal.to_tuple();
    
    return np_face_normals;

def getEdgeFaces(mesh):
    bm = getBMMesh(bpy.context, mesh, useeditmode=False);
    ensurelookuptable(bm);
    np_edge_faces = np.zeros((len(bm.edges), 2), dtype=int);
    for e in bm.edges:
        np_edge_faces[e.index] = [f.index for f in e.link_faces];
    
    bm.free();
    return np_edge_faces;

def getEdgeVertices(mesh):
    np_edge_vertices = np.zeros((len(mesh.data.edges), 2), dtype=int);
    for e in mesh.data.edges:
        np_edge_vertices[e.index] = [vid for vid in e.vertices];    
    return np_edge_vertices;

def getMeshVertexWeights(mesh, group_name):
    assert(group_name != '');
    N = len(mesh.data.vertices);
    weights = np.zeros((N), dtype=float);
    try:
        vgroup = mesh.vertex_groups[group_name];
    except KeyError:
        return weights;    
    return np.array([vgroup.weight(i) for i in range(N)], dtype=float); 
    

def getDuplicatedObject(context, meshobject, meshname="Duplicated", wire = False):
        if(not context.mode == "OBJECT"):
            bpy.ops.object.mode_set(mode = 'OBJECT', toggle = False);

        bpy.ops.object.select_all(action='DESELECT') #deselect all object
        
        hide_selection = meshobject.hide_select;
        hide_view = meshobject.hide;
        
        meshobject.hide_select = False;
        meshobject.hide = False;
        
        #The next step is to duplicate these objects and then apply fairing on them
        meshobject.select = True;
        context.scene.objects.active = meshobject;
        bpy.ops.object.duplicate_move();

        meshobject.select = False;

        duplicated = context.active_object;
        duplicated.location.x = 0;
        duplicated.location.y = 0;
        duplicated.location.z = 0;
        duplicated.name = meshname;
        duplicated.show_wire = wire;
        duplicated.show_all_edges = wire;
        duplicated.data.name = meshname;

        meshobject.hide_select = hide_selection;
        meshobject.hide = hide_view;

        return duplicated;

def getVoronoiAreas(context, mesh):
    vpos = getMeshVPos(mesh);
    faces = getMeshFaces(mesh);
    num_vertices = len(mesh.data.vertices);
    num_faces = len(mesh.data.polygons);
    
    return faces;

def averageFaceArea(c, mesh):
    bm = getBMMesh(c, mesh, useeditmode=False);
    ensurelookuptable(bm);
    area = [];
    for f in bm.faces:
        v1, v2, v3 = [l.vert for l in f.loops];
        area.append(f.calc_area());
    bm.free();    
    return 1.0 / (10.0 * np.sqrt(np.mean(area)));

def getFaceAreas(c, mesh):
    faces = mesh.data.polygons;
    loops = mesh.data.loops;
    verts = mesh.data.vertices;
    vpos = getMeshVPos(mesh);
    fids = np.array([[loops[lid].vertex_index for lid in f.loop_indices] for f in faces]);
    ab = vpos[fids[:, 1]] - vpos[fids[:, 0]];
    ac = vpos[fids[:, 2]] - vpos[fids[:, 0]];
    cross_vectors = np.cross(ab, ac);
    areas_mags = np.sqrt(np.sum(cross_vectors**2, axis=1));
    
    return areas_mags * 0.5, fids;

def getOneRingAreas(c, mesh):
    onebyeight = 1.0 / 8.0;
    oneringareas = [];
    bm = getBMMesh(c, mesh, useeditmode=False);
    ensurelookuptable(bm);
    
    for v in bm.verts:
        v_one_ring_area = [];
        for f in v.link_faces:
            v_one_ring_area.append(f.calc_area());
        
        oneringareas.append(np.min(np.sqrt(np.sum(np.square(v_one_ring_area)))));
        
    bm.free();    
    return np.array(oneringareas);


def getBMMesh(context, obj, useeditmode = True):
    if(not useeditmode):
        if(context.mode == "OBJECT"):
            bm = bmesh.new();
            bm.from_mesh(obj.data);
        else:
            bm = bmesh.from_edit_mesh(obj.data);
            
            if context.mode != 'EDIT_MESH':
                bpy.ops.object.mode_set(mode = 'EDIT', toggle = False);

        return bm;

    else:
        if(context.mode != "OBJECT"):
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False);

        bpy.ops.object.select_all(action='DESELECT') #deselect all object
        context.scene.objects.active = obj;
        obj.select = True;
        bpy.ops.object.mode_set(mode = 'EDIT', toggle = False);
        bm = bmesh.from_edit_mesh(obj.data);
        return bm;

def ensurelookuptable(bm):
    try:
        bm.verts.ensure_lookup_table();
        bm.edges.ensure_lookup_table();
        bm.faces.ensure_lookup_table();
    except:
        print('THIS IS AN OLD BLENDER VERSION, SO THIS CHECK NOT NEEDED');


#Can be of kd_type, 1-VERT, 2-EDGE, 3-FACE, 4-FACEVERT (contains vertices co with face Index)
#5-EDGEVERT (contains vertices co with EDGE INDEX)
def buildKDTree(context, meshobject, kd_type="VERT", points=[], use_bm =None,*, return_points = False):
#     print('BUILDING KDTREE FOR ', object.name);
    if(meshobject):
        mesh = meshobject.data;
        data = [];    
    if(kd_type == "VERT"):
        size = len(mesh.vertices);
        kd = mathutils.kdtree.KDTree(size);
        for i, v in enumerate(mesh.vertices):
            kd.insert(v.co, v.index);
    elif(kd_type =="EDGE"):
        subdivisions = 3;
        subdivisionsF = 3.0;
        size = len(mesh.edges) * subdivisions;
        edge_points = [];
        kd = mathutils.kdtree.KDTree(size);
        
        for i, e in enumerate(mesh.edges):
            v0 = mesh.vertices[e.vertices[0]].co;
            v1 = mesh.vertices[e.vertices[1]].co;
            vect = (v1 - v0);
            
            for i in range(subdivisions):
                ratio = float(i) / (subdivisionsF - 1.0);
                point = v0 + (vect * ratio);
                kd.insert(point, e.index);
                edge_points.append({'index':e.index, 'co':point});
                
        if(return_points):
            kd.balance();
            return kd, edge_points;
        
        
    elif(kd_type =="FACE"):
        size = len(mesh.polygons);
        kd = mathutils.kdtree.KDTree(size);
        oneby3 = 1.0 / 3.0;
        loops = meshobject.data.loops;
        vertices = meshobject.data.vertices;
        
        for i, f in enumerate(mesh.polygons):
                        
#             v0 = vertices[loops[f.loop_indices[0]].vertex_index].co;
#             v1 = vertices[loops[f.loop_indices[1]].vertex_index].co;
#             v2 = vertices[loops[f.loop_indices[2]].vertex_index].co;
            
#             point = (v0 * oneby3) + (v1 * oneby3) + (v2 * oneby3);
            kd.insert(f.center.copy(), f.index);
#             point, mat = getCentroid([v0, v1, v2]);
#             kd.insert(mat * point, f.index);
    
    elif(kd_type == "FACEVERT"):
        size = len(mesh.polygons) * 3;
        kd = mathutils.kdtree.KDTree(size);

        for i, f in enumerate(mesh.polygons):
            if(len(f.loop_indices) > 3):
                print('GOTCHA :: THE BLACK SHEEP ::: ', object.name, f.index);
            for ind in f.loop_indices:
                l = mesh.loops[ind];
                v = mesh.vertices[l.vertex_index];
                kd.insert(v.co, f.index);            
    
    elif(kd_type == "CUSTOM"):
        size = len(points);
        kd = mathutils.kdtree.KDTree(size);
        useDict = True;
        try:
            index = points[0]['index'];
        except KeyError:            
            useDict = False;
        except TypeError:
            useDict = False;
        except IndexError:
            useDict = False;
        
        for i, point in enumerate(points):
            if(useDict):
                index = point['index'];
                co = point['co'];
                kd.insert(co, index);
            else:
                kd.insert(point, i);          
                
    kd.balance();
#     print('BUILD KD TREE OF SIZE : ', size, ' FOR :: ', object.name, ' USING TYPE : ', kd_type);
    return kd;

'''
Copyright: Carlo Nicolini, 2013
Code adapted from the Mark Paskin Matlab version
from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
'''
"""
 RALIGN - Rigid alignment of two sets of points in k-dimensional
          Euclidean space.  Given two sets of points in
          correspondence, this function computes the scaling,
          rotation, and translation that define the transform TR
          that minimizes the sum of squared errors between TR(X)
          and its corresponding points in Y.  This routine takes
          O(n k^3)-time.
 
 Inputs:
   X - a k x n matrix whose columns are points 
   Y - a k x n matrix whose columns are points that correspond to
       the points in X
 Outputs: 
   c, R, t - the scaling, rotation matrix, and translation vector
             defining the linear map TR as 
 
                       TR(x) = c * R * x + t
 
             such that the average norm of TR(X(:, i) - Y(:, i))
             is minimized.
"""

def ralign(X,Y):
    m, n = X.shape
 
    mx = X.mean(1);
    my = Y.mean(1);
    
    Xc =  X - np.tile(mx, (n, 1)).T;
    Yc =  Y - np.tile(my, (n, 1)).T;
 
    sx = np.mean(np.sum(Xc*Xc, 0));
    sy = np.mean(np.sum(Yc*Yc, 0));
 
    Sxy = np.dot(Yc, Xc.T) / n;
 
    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
#     r = np.rank(Sxy);
    r = Sxy.ndim;
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1;
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1  
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t
 
    R = np.dot( np.dot(U, S ), V.T)
 
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)
 
    return R,c,t