# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import bpy;
from mathutils import Vector;
import numpy as np;
from GenericMarkerCreator.misc.mathandmatrices import getBMMesh, ensurelookuptable, getMeshFaces, getMeshVPos;
#Implementation of Principal curvatures and directions as per the paper
#Rusinkiewicz, Szymon. "Estimating curvatures and their derivatives on 
#triangle meshes." 3D Data Processing, Visualization and Transmission, 
#2004. 3DPVT 2004. Proceedings. 2nd International Symposium on. IEEE, 2004.
#The original implementation was done in C++ in the TriMesh library
#http://gfx.cs.princeton.edu/proj/trimesh2/
#So this is essentially a reimplementation for Blender using Python

def need_normals(mesh):       
    
    #PURE NUMPY STEPS BELOW
    vertices = getMeshVPos(mesh);
    faces = getMeshFaces(mesh);
    n = vertices.shape[0];
    f_n = faces.shape[0];
    #Create an initial empty list of vectors for size equal to no. of vertices in mesh
    normals = np.zeros((vertices.shape[0], 3));
    a = vertices[faces[:,0]]-vertices[faces[:,1]];
    b = vertices[faces[:,1]]-vertices[faces[:,2]];
    c = vertices[faces[:,2]]-vertices[faces[:,0]];
    
    edges = np.hstack(( np.array(a),np.array(b) , np.array(c)));
    facenormals = np.cross(a, b);
    
    l2a, l2b, l2c = np.sum(a*a, axis=1), np.sum(b*b, axis=1), np.sum(c*c, axis=1); 
    
    l2al2c = (l2a * l2c).reshape(faces.shape[0],1);
    l2bl2a = (l2b * l2a).reshape(faces.shape[0],1);
    l2cl2b = (l2c * l2b).reshape(faces.shape[0],1);
    
    l2al2c_normals = facenormals / l2al2c;
    l2bl2a_normals = facenormals / l2bl2a;
    l2cl2b_normals = facenormals / l2cl2b;
    
    
    for i in range(faces.shape[0]):
        vid1, vid2, vid3 = faces[i];
        normals[vid1] = normals[vid1] + l2al2c_normals[i];
        normals[vid2] = normals[vid2] + l2bl2a_normals[i];
        normals[vid3] = normals[vid3] + l2cl2b_normals[i];
    
    
    normals = normals / np.sqrt(np.sum(normals**2, axis=1)).reshape(n,1);
    
    normals_blend_vectors = [];
    for nx, ny, nz in normals:
        normals_blend_vectors.append(Vector((nx, ny, nz)));
    
    return normals_blend_vectors, normals;


def need_pointareas(mesh):
    
    loops = mesh.data.loops;
    faces = mesh.data.polygons;
    vertices = mesh.data.vertices;
    n = len(mesh.data.vertices);
    f_n = len(mesh.data.polygons);
    #Create empty list of areas(scalar) for all vertices using size of no. of vertices
    pointareas = [0.0] * n;
    #Create empty list of vectors (list of scalar with size = 3) using the 
    #size of no. of faces
    cornerareas = [None] * f_n;
    
    #Iterate through all the faces
    for f in faces:
        #Creating reference to all the vertices in this face 'f'
        p0 = vertices[loops[f.loop_indices[0]].vertex_index];
        p1 = vertices[loops[f.loop_indices[1]].vertex_index];
        p2 = vertices[loops[f.loop_indices[2]].vertex_index];
        verts = [p0, p1, p2];
        cornerareas[f.index] = {};
        #Creating the edges list, possibly improve this algorithm by 
        #using blenders in built structures but the refernce to calculated values
        #might miss. So for a safe implementation create it ourselves
        e = [p2.co - p1.co, p0.co - p2.co, p1.co - p0.co];        
        #Find the area of the face, cross any two vectors we get the area covered
        # by those two vectors using their <outerproduct>'s length
        area = 0.5 * e[0].cross(e[1]).length;   
        #Size of the edges by using their <innerproduct>
        l2 = [e[0].dot(e[0]),e[1].dot(e[1]),e[2].dot(e[2])];
        
        #The below values are calculated when needed to find the corner weights.
        #This will help to find the weight of a face or its influence over a 
        #Vertex
        ew = [
            l2[0] * (l2[1] + l2[2] - l2[0]), 
            l2[1] * (l2[2] + l2[0] - l2[1]), 
            l2[2] * (l2[0] + l2[1] - l2[2])
            ];
        
        #If the weight is negative at vertex 0
        if (ew[0] <= 0.0):
            cornerareas[f.index][p1.index] = -0.25 * l2[2] * area / (e[0].dot(e[2]));
            cornerareas[f.index][p2.index] = -0.25 * l2[1] * area / (e[0].dot(e[1]));
            cornerareas[f.index][p0.index] = area - cornerareas[f.index][p1.index] - cornerareas[f.index][p2.index];
        #If the weight is negative at vertex 1    
        elif (ew[1] <= 0.0):
            cornerareas[f.index][p2.index] = -0.25 * l2[0] * area / (e[1].dot(e[0]));
            cornerareas[f.index][p0.index] = -0.25 * l2[2] * area / (e[1].dot(e[2]));
            cornerareas[f.index][p1.index] = area - cornerareas[f.index][p2.index] - cornerareas[f.index][p0.index];
        #if the weight is negative at vertex 3
        elif (ew[2] <= 0.0):
            cornerareas[f.index][p0.index] = -0.25 * l2[1] * area / (e[2].dot(e[1]));
            cornerareas[f.index][p1.index] = -0.25 * l2[0] * area / (e[2].dot(e[0]));
            cornerareas[f.index][p2.index] = area - cornerareas[f.index][p0.index] - cornerareas[f.index][p1.index];
        else:
            ewscale = 0.5 * area / (ew[0] + ew[1] + ew[2]);
            for j, v in enumerate(verts):
                cornerareas[f.index][v.index] = ewscale * (ew[(j + 1) % 3] + ew[(j + 2) % 3]);
        
        pointareas[p0.index] = pointareas[p0.index] + cornerareas[f.index][p0.index];
        pointareas[p1.index] = pointareas[p1.index] + cornerareas[f.index][p1.index];
        pointareas[p2.index] = pointareas[p2.index] + cornerareas[f.index][p2.index];
    
    return pointareas, cornerareas;

def need_curvatures(mesh):
    loops = mesh.data.loops;
    faces = mesh.data.polygons;
    vertices = mesh.data.vertices;
    n = len(mesh.data.vertices);
    f_n = len(mesh.data.polygons);
    
    #get the normals and pointareas associated with each vertices
    normals, normals_np = need_normals(mesh);
    pointareas, cornerareas = need_pointareas(mesh);
    
    curv1, curv2, curv12 = [0.0] * n, [0.0] * n, [0.0] * n;
    pdir1, pdir2 = [Vector((0,0,0))] * n, [Vector((0,0,0))] * n;
    principalvalues = [];    
    
    for f in faces:
        p0 = vertices[loops[f.loop_indices[0]].vertex_index];
        p1 = vertices[loops[f.loop_indices[1]].vertex_index];
        p2 = vertices[loops[f.loop_indices[2]].vertex_index];
        pdir1[p0.index] = p1.co - p0.co;
        pdir1[p1.index] = p2.co - p1.co;
        pdir1[p2.index] = p0.co - p2.co;
        
    
    for v in vertices:
        pdir1[v.index] = pdir1[v.index].cross(normals[v.index]);
        pdir1[v.index].normalize();
        pdir2[v.index] = normals[v.index].cross(pdir1[v.index]);
        
    
    for f in faces:
        #Creating reference to all the vertices in this face 'f'
        p0 = vertices[loops[f.loop_indices[0]].vertex_index];
        p1 = vertices[loops[f.loop_indices[1]].vertex_index];
        p2 = vertices[loops[f.loop_indices[2]].vertex_index];
        
        verts = [p0, p1, p2];
        e = [p2.co - p1.co, p0.co - p2.co, p1.co - p0.co];
        #Normal-Tangent-Basis coordinate system per face
        t = e[0].copy();        
        t.normalize();
        
        n = e[0].cross(e[1]);
        b = n.cross(t);
        b.normalize();
        
        #Estimating curvatures based on the variation of normals along the edges
        
        m = [0.0, 0.0, 0.0];
        w = [[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]];
        
        for j in range(3):
            prev_ii = (j + 3 - 1) % 3;
            next_ii = (j + 1) % 3;
            
            u = e[j].dot(t);
            v = e[j].dot(b);
            
            w[0][0] = w[0][0] + (u * u);
            w[0][1] = w[0][1] + (u * v);
            w[2][2] = w[2][2] + (v * v);
            
            #Find the normal between the "normals of vertices" connected to this 
            #vertex on this face 'f'
            dn = normals[verts[prev_ii].index] - normals[verts[next_ii].index];
            #Find their angular deviation (cosine theta) w.r.to tangent t and basis b
            dnu = dn.dot(t);
            dnv = dn.dot(b);
            
            m[0] = m[0] + (dnu * u);
            m[1] = m[1] + ((dnu * v) + (dnv * u));
            m[2] = m[2] + (dnv * v);
            
        w[1][1] = (w[0][0] + w[2][2]);
        w[1][2] = (w[0][1]);
        
        diag = [0.0, 0.0, 0.0];
        
        solvable, A, rdiag = ldltdc(w, diag);
        
        if(not solvable):
            continue;
            
        x, A, B, rdiag = ldltsl(A, rdiag, m, m);
        
        for index, v in enumerate(verts):
            new_ku, newkuv, newkv = proj_curv(t, b, m[0], m[1], m[2], pdir1[v.index], pdir2[v.index]);
            wt = (cornerareas[f.index][v.index] / pointareas[v.index]);
            curv1[v.index] = curv1[v.index] + (wt * new_ku);
            curv12[v.index] = curv12[v.index]+ (wt * newkuv);
            curv2[v.index] = curv2[v.index] + (wt * newkv);
            
    k1_list, k2_list, p1_list, p2_list, mean_list, gaussian_list  = [], [], [], [], [], [];
    
    bm = getBMMesh(bpy.context, mesh, useeditmode=False);
    ensurelookuptable(bm);
    
    for v in bm.verts:
        averagelength = sum([e.calc_length() for e in v.link_edges]) * (1.0 / len(v.link_edges)) * 0.4;
        k1, k2, p1, p2 = diagonalize_curv(pdir1[v.index], pdir2[v.index], curv1[v.index], curv12[v.index], curv2[v.index], normals[v.index]);        
        principalvalues.append(((k1+k2)*0.5,k1*k2 ,k1, k2, p1, p2));
        p1 = p1 * averagelength;
        p2 = p2 * averagelength;       
        
        o_values = np.array([k1, k2]);
        p_values = [p1, p2];
        abs_values = np.abs(o_values);
        indices_max = np.argmax(abs_values);
        indices_min = np.argmin(abs_values);
        
        k1_list.append(o_values[indices_max]);
        k2_list.append(o_values[indices_min]);
        p1_list.append(p_values[indices_max]);
        p2_list.append(p_values[indices_min]);
        mean_list.append((k1+k2)*0.5);
        gaussian_list.append(k1*k2);
    sx = np.abs(k1_list) - np.abs(k2_list);
    
    bm.free();
    
    return np.array(k1_list), np.array(k2_list), sx, np.array(p1_list), np.array(p2_list), mean_list, gaussian_list, normals_np;


def diagonalize_curv(old_u, old_v, ku, kuv, kv, new_norm):
    
    r_old_u, r_old_v = rot_coord_sys(old_u, old_v, new_norm);
    c,s,tt = 1.0,0.0,0.0;
    
    if(kuv != 0.0):
        #Jacobi rotation to diagonalize
        h = (0.5 * (kv - ku)) / kuv;
        if(h < 0.0):
            tt = 1.0 / (h - (1.0 + (h * h))**0.5);
        else:
            tt = 1.0 / (h + (1.0 + (h * h))**0.5);
            
        c = 1.0 / (1.0 + (tt * tt))**0.5;        
        s = tt * c;

    k1 = ku - (tt * kuv);
    k2 = kv + (tt * kuv);

    if (abs(k1) >= abs(k2)):
        pdir1 = (c * r_old_u) - (s*r_old_v);
    else:
        k1, k2 = k2, k1;
        pdir1 = (s * r_old_u) + (c*r_old_v);
        
    pdir2 = new_norm.cross(pdir1);
    
    return k1, k2, np.array(pdir1), np.array(pdir2);
    

def rot_coord_sys(old_u, old_v, new_norm):   
    
    new_u = old_u.copy();
    new_v = old_v.copy();    
    old_norm = old_u.cross(old_v);
    ndot = old_norm.dot(new_norm);
    
    if (ndot <= -1.0):
        return -new_u, -new_v;
    
    perp_old = new_norm - (ndot * old_norm);
    dperp = (1.0 / (1.0 + ndot)) * (old_norm + new_norm);
    new_u = new_u - (dperp * (new_u.dot(perp_old)));
    new_v = new_v - (dperp * (new_v.dot(perp_old)));
    
    return new_u, new_v;

def proj_curv(old_u, old_v,old_ku, old_kuv, old_kv, new_u, new_v):
    
    r_new_u, r_new_v = rot_coord_sys(new_u, new_v, old_u.cross(old_v));

    u1 = r_new_u.dot(old_u);
    v1 = r_new_u.dot(old_v);
    u2 = r_new_v.dot(old_u);
    v2 = r_new_v.dot(old_v);

    new_ku = (old_ku * u1 * u1) + (old_kuv * (2.0 * u1 * v1)) + (old_kv * v1 * v1);
    new_kuv = (old_ku * u1 * u2) + (old_kuv * (u1 * v2 + u2 * v1)) + (old_kv * v1 * v2);
    new_kv = (old_ku * u2 * u2) + (old_kuv * (2.0 * u2 * v2)) + (old_kv * v2 * v2);

    return new_ku, new_kuv, new_kv;
        

#This is a LDL^T decomposition of a positive definite matrix
def ldltdc(A, rdiag):
    v = [0] * 2;    
    for i in range(3):
        
        for k in range(i):
            v[k] = (A[i][k] * rdiag[k]);
            
        for j in range(i, 3):
            
            sum = A[i][j];
            
            for k in range(i):
                sum = sum - (v[k] * A[j][k]);
            
            if(i == j):       
                if(sum <= 0.0):
                    return False, A, rdiag;
                
                rdiag[i] = (1.0 / sum);
            
            else:
                A[j][i] = sum;
                
    return True, A, rdiag;

def ldltsl(A, rdiag, B, x):
    
    for i in range(3):
        sum = B[i];
        
        for k in range(i):
            sum = sum - (A[i][k] * x[k]);            
        x[i] = sum * rdiag[i];
    
    for i in range(3,0,-1):
        sum = 0;
        for k in range(i, 3, 1):
            sum = sum + (A[k][i-1] * x[k]);
        
        x[i-1] = x[i-1] - (sum * rdiag[i-1]);
        
    return x, A, B, rdiag;
            
        
    