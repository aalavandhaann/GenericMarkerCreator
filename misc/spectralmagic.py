import bpy;
import numpy as np;
import scipy as sp;
import scipy.sparse as spsp;
from scipy.sparse.linalg import eigsh, eigs;
import numpy.matlib as np_mlib;

from GenericMarkerCreator.misc.mathandmatrices import getBMMesh, ensurelookuptable, getMeshFaces, getMeshVPos, setMeshVPOS, getMeshFaceAngles, getMeshVoronoiAreas;
from GenericMarkerCreator.misc.mathandmatrices import meanCurvatureLaplaceWeights, getLaplacianMatrixCotangent, getLaplacianMeshNormalized, getWKSLaplacianMatrixCotangent;

CACHE = {
    'WKS_D':{}, 
    'HKS_D':{},
    'GISIF_D': {}
}

##############################################################
##        Spectral Representations / Heat Flow              ##
##############################################################

#Purpose: Given a mesh, to compute first K eigenvectors of its Laplacian
#and the corresponding eigenvalues
#Inputs: mesh (polygon mesh object), K (number of eigenvalues/eigenvectors)
#Returns: (eigvalues, eigvectors): a tuple of the eigenvalues and eigenvectors
def getLaplacianSpectrum(context, mesh, K):
    L = getLaplacianMatrixCotangent(context, mesh, []);
    (eigvalues, eigvectors) = eigsh(L, K, which='LM', sigma = 0);
    return (eigvalues, eigvectors);

#Purpose: Given a mesh, to use the first K eigenvectors of its Laplacian
#to perform a lowpass filtering
#Inputs: mesh (polygon mesh object), K (number of eigenvalues/eigenvectors)
#Returns: Nothing (should update mesh.VPos)
def doLowpassFiltering(context, mesh, K):
    (_, U) = getLaplacianSpectrum(context, mesh, K);
    return U.dot(U.T.dot(getMeshVPos(mesh)));    
    
#Purpose: Given a mesh, to simulate heat flow by projecting initial conditions
#onto the eigenvectors of the Laplacian matrix, and then to sum up the heat
#flow of each eigenvector after it's decayed after an amount of time t
#Inputs: mesh (polygon mesh object), eigvalues (K eigenvalues), 
#eigvectors (an NxK matrix of eigenvectors computed by your laplacian spectrum
#code), t (the time to simulate), initialVertices (indices of the verticies
#that have an initial amount of heat), heatValue (the value to put at each of
#the initial vertices at the beginning of time
#Returns: heat (a length N array of heat values on the mesh)
def getHeat(context, mesh, eigvalues, eigvectors, t, initialVertices, heatValue = 100.0):
    N = len(mesh.data.vertices);
    I = np.zeros(N);
    I[initialVertices] = heatValue;
    coeffs = I[None, :].dot(eigvectors);
    coeffs = coeffs.flatten();
    coeffs = coeffs*np.exp(-eigvalues*t);
    heat = eigvectors.dot(coeffs[:, None]);
    return heat;

#Purpose: Given a mesh, to approximate its curvature at some measurement scale
#by recording the amount of heat that stays at each vertex after a unit impulse
#of heat is applied.  This is called the "Heat Kernel Signature" (HKS)
#Inputs: mesh (polygon mesh object), K (number of eigenvalues/eigenvectors to use)
#t (the time scale at which to compute the HKS)
#Returns: hks (a length N array of the HKS values)
def getHKS(mesh, L, K=5, t=20.0):
    eva, eve = eigsh(L, K, which='LM', sigma=0);
#     eva, eve = eigs(L, K, which='LM', sigma=0);
#     print(eva);
    eve = (eve**2)*np.exp(-eva*t)[None, :];
    heat = np.sum(eve, 1);
#     print(np.min(heat), np.max(heat));
    return heat;


def getHKSColors(context, mesh, K=5, HKS_T=20.0):
    try:
        L = CACHE['HKS_D'][mesh.name]['L'];
    except KeyError:
        L = getLaplacianMatrixCotangent(context, mesh, []);
        CACHE['HKS_D'][mesh.name] = {};
        CACHE['HKS_D'][mesh.name]['L'] = L;
        
#     L = getLaplacianMeshNormalized(context, mesh, cotangent = True);
    K = min(len(mesh.data.vertices)-1, K);
    heat = getHKS(mesh, L, K, HKS_T);    
    heat = heat/np.max(heat);
    return heat, K;
#     applyColoringForMeshErrors(context, mesh, heat, v_group_name='hks', use_weights=False);

def getWKS(mesh, L, A, K=3, WKS_E=10, wks_variance=6):
    num_vertices = L.shape[0];
    #Calculate the eigen values and vectors
#     eva, eve = eigs(L,k=K,M=A,sigma=-1e-5,which='LM');
    eva, eve = eigsh(L,k=K,M=A,sigma=-1e-5,which='LM');
    #Sort the eigen values from smallest to highest
    # and ensure to use the real part of eigen values, because there might be complex numbers
    eva = np.abs(np.real(eva));
    idx = np.argsort(eva);
    eva = eva[idx];
    eve = eve[:, idx];    
    eve = np.real(eve);
    
#     print('E = ', eva);
#     print('PHI = ', eve);
    #Calculation of WKS Signature
    WKS = np.zeros((num_vertices,WKS_E));
    log_E = np.log(np.maximum(np.abs(eva), 1e-6)).T;
    e = np.linspace(log_E[1], np.max(log_E) / 1.02, WKS_E);
    sigma = (e[1]-e[0])*wks_variance;
    sigma_inv = (2*sigma**2);
    C = np.zeros((1, WKS_E));
        
    for i in range(WKS_E):
        np_tiled = np.exp((-(e[i] - log_E)**2) / sigma_inv);
        np_sum = np.sum( eve**2 * np_mlib.repmat(np_tiled, num_vertices, 1), 1);
        np_sum.shape = (np_sum.shape[0], );
        WKS[:,i] = np_sum;
        expo = np.exp((-(e[i]-log_E)**2) / sigma_inv);
        np_sum_expo = np.sum(expo);
        C[:,i] = np_sum_expo;
    
    #Normalized the WKS Matrix
    WKS[:,:] = WKS[:,:] / np_mlib.repmat(C, num_vertices, 1);    
    return WKS;

def getWKSColors(context, mesh, K=3, WKS_E=6, wks_variance=6):
    try:
        L = CACHE['WKS_D'][mesh.name]['L'];
        A = CACHE['WKS_D'][mesh.name]['Voronoi'];
    except KeyError:
        L = getWKSLaplacianMatrixCotangent(context, mesh);
        A, A_np = getMeshVoronoiAreas(context, mesh);
        CACHE['WKS_D'][mesh.name] = {};
        CACHE['WKS_D'][mesh.name]['L'] = L;
        CACHE['WKS_D'][mesh.name]['Voronoi'] = A;
    
#     L = meanCurvatureLaplaceWeights(context, mesh, symmetric=False, normalized=False);
    K = min(len(mesh.data.vertices)-1, K);
    wks_matrix = getWKS(mesh, L, A, K, WKS_E, wks_variance );
    wks_sum = np.sum(wks_matrix, 1);
    wks = wks_sum/np.max(wks_sum);
    return wks, K;
    


def getGISIFEigens(mesh, L, A, K=3, threshold_ratio=0.3, show_group_index = 0):    
    num_vertices = L.shape[0];
    #Calculate the eigen values and vectors
#     eva, eve = eigs(L,k=K,M=A,sigma=-1e-5,which='LM');
    eva, eve = eigsh(L,k=K,M=A,sigma=-1e-5,which='LM');
    #Sort the eigen values from smallest to highest
    # and ensure to use the real part of eigen values, because there might be complex numbers
    eva = np.abs(np.real(eva));
    idx = np.argsort(eva);
    eva = eva[idx];
    eve = eve[:, idx];    
    eve = np.real(eve);
    
    return eva, eve;
        
def getGISIFGroups(mesh, eva, eve, threshold_ratio=0.1):
    eva_diff = np.diff(eva);
    eva_min = np.min(eva);
    eva_max = np.max(eva);
    eva_range = eva_max - eva_min;
    eva_select_value = eva_range * threshold_ratio;
    
    eva_selection = np.where(eva_diff >= eva_select_value)[0];
    eva_groups = np.array([0] + (eva_selection+1).tolist());
    
    all_groups = [];
    
    for i in range(eva_groups.shape[0]):
        if(i == eva_groups.shape[0]-1):
            selection_e_vectors = eve[:,eva_groups[i]:];
            label = '%d_%d'%(eva_groups[i], selection_e_vectors.shape[1]);
        else:
            selection_e_vectors = eve[:,eva_groups[i]:eva_groups[i+1]];
            label = '%d_%d'%(eva_groups[i], selection_e_vectors.shape[1]);
        all_groups.append((label, selection_e_vectors));
            
    return all_groups;

def getGISIFColors(context, mesh, K=20, threshold_ratio=0.1, show_group_index = 0):
    K = min(len(mesh.data.vertices)-1, K);
    try:
        L = CACHE['GISIF_D'][mesh.name]['L'];
        A = CACHE['GISIF_D'][mesh.name]['Voronoi'];
    except KeyError:
        L = getWKSLaplacianMatrixCotangent(context, mesh);
        A, A_np = getMeshVoronoiAreas(context, mesh);
        eva, eve = getGISIFEigens(mesh, L, A, K);
        CACHE['GISIF_D'][mesh.name] = {};
        CACHE['GISIF_D'][mesh.name]['L'] = L;
        CACHE['GISIF_D'][mesh.name]['Voronoi'] = A;
        try:
            k_dict = CACHE['GISIF_D'][mesh.name]['K'];
            if(k_dict['k'] != K):
                CACHE['GISIF_D'][mesh.name]['K'] = {'k':K, 'threshold':threshold_ratio};
        except KeyError:
            CACHE['GISIF_D'][mesh.name]['K'] = {'k':K, 'threshold':threshold_ratio};
        
    try:
        evalues = CACHE['GISIF_D'][mesh.name]['K']['eigenvalues'];
        evectors = CACHE['GISIF_D'][mesh.name]['K']['evectors'];
    except KeyError:
        evalues, evectors = getGISIFEigens(mesh, L, A, K);        
        all_groups = getGISIFGroups(mesh, evalues, evectors, threshold_ratio);
        CACHE['GISIF_D'][mesh.name]['K']['eigenvalues']= evalues;
        CACHE['GISIF_D'][mesh.name]['K']['eigenvalues']= evectors;
        CACHE['GISIF_D'][mesh.name]['K']['groups'] = all_groups;
    
    if(CACHE['GISIF_D'][mesh.name]['K']['threshold'] != threshold_ratio):
        all_groups = getGISIFGroups(mesh, evalues, evectors, threshold_ratio);
        CACHE['GISIF_D'][mesh.name]['K']['threshold'] = threshold_ratio;
        CACHE['GISIF_D'][mesh.name]['K']['groups'] = all_groups;
    else:
        all_groups = CACHE['GISIF_D'][mesh.name]['K']['groups'];
    
    show_group_index = min(show_group_index, len(all_groups)-1);
    label, evectors_group  = all_groups[show_group_index];    
    eve = (evectors_group**2);
    gisifs = np.sum(eve, 1);
    gisifs = gisifs/np.max(gisifs);
    return gisifs, K, label;
    