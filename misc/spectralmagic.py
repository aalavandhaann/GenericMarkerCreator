import bpy;
import numpy as np;
import scipy as sp;
import scipy.sparse as spsp;
from scipy.sparse.linalg import eigsh, eigs;
import numpy.matlib as np_mlib;

from GenericMarkerCreator.misc.mathandmatrices import getBMMesh, ensurelookuptable, getMeshFaces, getMeshVPos, setMeshVPOS, getMeshFaceAngles;
from GenericMarkerCreator.misc.mathandmatrices import meanCurvatureLaplaceWeights, getLaplacianMatrixCotangent, getLaplacianMeshNormalized;
from GenericMarkerCreator.misc.mathandmatrices import getWKSEigens, getWKSLaplacianMatrixCotangent, getMeshVoronoiAreas;

CACHE = {};

def setMatrixCache(context, mesh, property, value):
    updateMatrixCache(context, mesh);
    CACHE[mesh.name][property] = value;

#For some known properties the matrix cache can be set if it doesn't exist. For other properties 
#just return false and null value if they don't exist in the memory
def getMatrixCache(context, mesh, property):
    updateMatrixCache(context, mesh);    
    try:
        reading = CACHE[mesh.name][property];
    except KeyError:
        if(property == 'WKS_L'):
            WKS_L = getWKSLaplacianMatrixCotangent(context, mesh);
            setMatrixCache(context, mesh, 'WKS_L', WKS_L);
            return True, WKS_L;
        elif (property == 'WKS_VORONOI'):
            WKS_VORONOI, __ = getMeshVoronoiAreas(context, mesh);
            setMatrixCache(context, mesh, 'WKS_VORONOI', WKS_VORONOI);
            return True, WKS_VORONOI;                
        return False, [];    
    return True, reading;
        
def updateMatrixCache(context, mesh):
    try:
        meshdata = CACHE[mesh.name];
    except KeyError:
        CACHE[mesh.name] = {};

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
def getHKSEigens(mesh, L, K=5, t=20.0, *,eva=None, eve=None, A = None):
    eva, eve = eigsh(L, K, M=A, which='LM', sigma=-1e-8);
    return eva, eve;

#eigenvalues, eigenvectors, the diagonal matrix M, and times for which HKS is propogated;
def get_hks(eival, eivec,mat_M, num_times=100):
    times = np.logspace(np.log(0.1),np.log(10),num=num_times);
    vertex_areas = spsp.csr_matrix(mat_M).data;
    k = np.zeros((eivec.shape[0],len(times)))
    for idx,t in enumerate(times):
        k[:,idx] = (np.exp(-t*eival)[None,:]*eivec*eivec).sum(axis=1)
    average_temperature = (vertex_areas[:,None]*k).sum(axis=0) / vertex_areas.sum();
    hks = k/average_temperature;
    print('HKS SHAPE : ', hks.shape);
    return hks;

def get_wks(eigen_values, eigen_vectors, energy_steps=None, absolute_sigma=None, num_steps=None, relative_sigma=None):
    eigen_values = np.abs(eigen_values);
    idx = np.argsort(eigen_values);
    eigen_values = eigen_values[idx[1:]];
    eigen_vectors = eigen_vectors[:, idx[1:]];

    if energy_steps is None:
        assert num_steps is not None
        energy_steps = np.log(np.linspace(eigen_values[1], eigen_values[-1], num_steps))

    if absolute_sigma is None:
        if relative_sigma is not None:
            absolute_sigma = (energy_steps.max() - energy_steps.min()) * relative_sigma
        else:
            # from paper
            delta = (energy_steps.max() - energy_steps.min()) / energy_steps.size
            absolute_sigma = 7 * delta

    nv = eigen_vectors.shape[0]
#    nev = eigen_vectors.shape[1]
    num_steps = energy_steps.size

    desc = np.zeros((nv, num_steps))
    for idx, e in enumerate(energy_steps):
        coeff = np.exp(-(e-np.log(eigen_values))**2/(2*absolute_sigma))
        desc[:, idx] = 1/coeff.sum() * (eigen_vectors**2).dot(coeff)
    
    print('WKS SHAPE ', desc.shape);
    return desc;

def getWKS(mesh, eva, eve, WKS_E=10, wks_variance=6):
    num_vertices = len(mesh.data.vertices);
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
    print('WKS SHAPE ', WKS.shape);
    return WKS;


def getHKSColors(context, mesh, K=5, HKS_T=20.0, HKS_CURRENT_T = 20):
    K = min(len(mesh.data.vertices)-1, K);
    k_exists, cache_k = getMatrixCache(context, mesh, 'WKS_k');
    
    WKS_L_Exists, WKS_L = getMatrixCache(context, mesh, 'WKS_L');
    WKS_VORONOI_Exists, WKS_VORONOI = getMatrixCache(context, mesh, 'WKS_VORONOI');
    
    WKS_EVA_Exists, WKS_EVA = getMatrixCache(context, mesh, 'WKS_eva');
    WKS_EVE_Exists, WKS_EVE = getMatrixCache(context, mesh, 'WKS_eve');
        
    hks_t_exists, cache_hks_t = getMatrixCache(context, mesh, 'HKS_T');    
    hks_matrix_exists, HKS_MATRIX = getMatrixCache(context, mesh, 'HKS_MATRIX');
           
    if(not k_exists):
        setMatrixCache(context, mesh, 'WKS_k', K);
    
    if(not WKS_L_Exists):
        print('GETTING HKS LAPLACIANS :');
#         HKS_L = getLaplacianMatrixCotangent(context, mesh, []);
        WKS_L = getWKSLaplacianMatrixCotangent(context, mesh);
        setMatrixCache(context, mesh, 'WKS_L', WKS_L);
    
    if(not WKS_VORONOI_Exists):    
        print('GETTING WKS VORONOI :');    
        WKS_VORONOI, __ = getMeshVoronoiAreas(context, mesh);
        setMatrixCache(context, mesh, 'WKS_VORONOI', WKS_VORONOI);
    
    if(cache_k != K or not WKS_EVA_Exists or not WKS_EVE_Exists):
        print('GETTING HKS EIGENS : ', '%s = %s'%(cache_k, K), WKS_EVA_Exists, WKS_EVE_Exists);
#         HKS_EVA, HKS_EVE = getHKSEigens(mesh, HKS_L, K);
        WKS_EVA, WKS_EVE = getWKSEigens(mesh, WKS_L, WKS_VORONOI, K); 
        setMatrixCache(context, mesh, 'WKS_k', K);
        setMatrixCache(context, mesh, 'WKS_eva', WKS_EVA);
        setMatrixCache(context, mesh, 'WKS_eve', WKS_EVE);
    
    if(cache_hks_t != HKS_T or not hks_t_exists or not hks_matrix_exists):
        HKS_MATRIX = get_hks(WKS_EVA, WKS_EVE, WKS_VORONOI, num_times=HKS_T);
        setMatrixCache(context, mesh, 'HKS_T', HKS_T);
        setMatrixCache(context, mesh, 'HKS_MATRIX', HKS_MATRIX);
        
    
    print('GETTING HKS COLOR VALUES');    
    heat = HKS_MATRIX[:,min(int(HKS_T-1), HKS_CURRENT_T)];
    print('FINISHED AND RETURNING THE COMPUTED HKS VALUES :');
    return heat, K;

def getWKSColors(context, mesh, K=3, WKS_E=6, WKS_CURRENT_E=0, wks_variance=0.0):
    K = min(len(mesh.data.vertices)-1, K);
    
    k_exists, cache_k = getMatrixCache(context, mesh, 'WKS_k');
    WKS_L_Exists, WKS_L = getMatrixCache(context, mesh, 'WKS_L');
    WKS_VORONOI_Exists, WKS_VORONOI = getMatrixCache(context, mesh, 'WKS_VORONOI');
    WKS_EVA_Exists, WKS_EVA = getMatrixCache(context, mesh, 'WKS_eva');
    WKS_EVE_Exists, WKS_EVE = getMatrixCache(context, mesh, 'WKS_eve');
    
    wks_e_exists, cache_wks_e = getMatrixCache(context, mesh, 'WKS_E');    
    wks_matrix_exists, WKS_MATRIX = getMatrixCache(context, mesh, 'WKS_MATRIX');
    wks_relative_sigma_exists, WKS_RELATIVE_SIGMA = getMatrixCache(context, mesh, 'WKS_RELATIVE_SIGMA');
    
    if(not k_exists):
        setMatrixCache(context, mesh, 'WKS_k', K);
    
    if(not WKS_L_Exists):        
        WKS_L = getWKSLaplacianMatrixCotangent(context, mesh);
        print('GETTING WKS LAPLACIANS :');
        setMatrixCache(context, mesh, 'WKS_L', WKS_L);
        
    if(not WKS_VORONOI_Exists):    
        print('GETTING WKS VORONOI :');    
        WKS_VORONOI, __ = getMeshVoronoiAreas(context, mesh);
        setMatrixCache(context, mesh, 'WKS_VORONOI', WKS_VORONOI);
    
    if(cache_k != K or not WKS_EVA_Exists or not WKS_EVE_Exists):
        print('GETTING WKS EIGENS : ', '%s = %s'%(cache_k, K), WKS_EVA_Exists, WKS_EVE_Exists);
        WKS_EVA, WKS_EVE = getWKSEigens(mesh, WKS_L, WKS_VORONOI, K);        
        setMatrixCache(context, mesh, 'WKS_k', K);
        setMatrixCache(context, mesh, 'WKS_eva', WKS_EVA);
        setMatrixCache(context, mesh, 'WKS_eve', WKS_EVE);
    
    if(cache_wks_e != WKS_E or not wks_e_exists or not wks_matrix_exists or wks_variance !=  WKS_RELATIVE_SIGMA):
        if(wks_variance > 0.0):
            WKS_MATRIX = get_wks(WKS_EVA, WKS_EVE, num_steps=WKS_E, relative_sigma=wks_variance);
        else:
            WKS_MATRIX = get_wks(WKS_EVA, WKS_EVE, num_steps=WKS_E);
            
        setMatrixCache(context, mesh, 'WKS_E', WKS_E);
        setMatrixCache(context, mesh, 'WKS_MATRIX', WKS_MATRIX);
        setMatrixCache(context, mesh, 'WKS_RELATIVE_SIGMA', wks_variance);
    
    print('GETTING WKS COLORS VALUES');
#     wks_matrix = getWKS(mesh, WKS_EVA, WKS_EVE, WKS_E, wks_variance );
#     wks_sum = np.sum(wks_matrix, 1);
#     wks = wks_sum/np.max(wks_sum);
    wks = WKS_MATRIX[:,min(int(WKS_E-1), WKS_CURRENT_E)];
    print('FINISHED AND RETURNING THE COMPUTED WKS VALUES :');
    return wks, K;
        
def getGISIFGroups(mesh, eva, eve, threshold_ratio=0.1):
    eva_diff = np.diff(eva);
    eva_min = np.min(eva);
    eva_max = np.max(eva);
#     eva_range = eva_max - eva_min;
#     eva_select_value = eva_range * threshold_ratio;
    eva_select_value = threshold_ratio;
    
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

def getGISIFColors(context, mesh, K=20, threshold_ratio=0.1, show_group_index = 0, linear_gisif_iterations=0):
    K = min(len(mesh.data.vertices)-1, K);
    
    k_exists, cache_k = getMatrixCache(context, mesh, 'WKS_k');
    WKS_L_Exists, WKS_L = getMatrixCache(context, mesh, 'WKS_L');
    WKS_VORONOI_Exists, WKS_VORONOI = getMatrixCache(context, mesh, 'WKS_VORONOI');
    WKS_EVA_Exists, WKS_EVA = getMatrixCache(context, mesh, 'WKS_eva');
    WKS_EVE_Exists, WKS_EVE = getMatrixCache(context, mesh, 'WKS_eve');
    
    GISIF_THRESHOLD_Exists, GISIF_Threshold = getMatrixCache(context, mesh, 'GISIF_Threshold');
    GISIF_GROUPS_Exists, GISIF_Groups = getMatrixCache(context, mesh, 'GISIF_Groups');
    
    if(not k_exists):
        setMatrixCache(context, mesh, 'WKS_k', K);
    
    if(not WKS_L_Exists):        
        WKS_L = getWKSLaplacianMatrixCotangent(context, mesh);
        print('GETTING GISIF LAPLACIANS :');
        setMatrixCache(context, mesh, 'WKS_L', WKS_L);
        
    if(not WKS_VORONOI_Exists):    
        print('GETTING GISIF VORONOI :');    
        WKS_VORONOI, __ = getMeshVoronoiAreas(context, mesh);
        setMatrixCache(context, mesh, 'WKS_VORONOI', WKS_VORONOI);
    
    if(cache_k != K or not WKS_EVA_Exists or not WKS_EVE_Exists):
        print('GETTING GISIF EIGENS : ', '%s = %s'%(cache_k, K), WKS_EVA_Exists, WKS_EVE_Exists);
        WKS_EVA, WKS_EVE = getWKSEigens(mesh, WKS_L, WKS_VORONOI, K);
        
        mesh.spectral_soft_update = True;
        #Find the standard deviation between the eigenvalues and apply
        mesh.gisif_threshold = np.std(WKS_EVA);
        threshold_ratio = mesh.gisif_threshold;
        mesh.spectral_soft_update = False;
        
        setMatrixCache(context, mesh, 'WKS_k', K);
        setMatrixCache(context, mesh, 'WKS_eva', WKS_EVA);
        setMatrixCache(context, mesh, 'WKS_eve', WKS_EVE);
    
    if(GISIF_Threshold != threshold_ratio or not GISIF_THRESHOLD_Exists):
        print('GETTING GISIF GROUPS :');
        GISIF_Groups = getGISIFGroups(mesh, WKS_EVA, WKS_EVE, threshold_ratio);
        setMatrixCache(context, mesh, 'GISIF_Threshold', threshold_ratio);
        setMatrixCache(context, mesh, 'GISIF_Groups', GISIF_Groups);
    
    print('SELECTING GISIF GROUPS :',WKS_EVA);
    show_group_index = min(show_group_index, len(GISIF_Groups)-1);
    linear_gisifs = [];
    if(linear_gisif_iterations > 0):
        linear_gisif_iterations = linear_gisif_iterations + 1;
        max_linear_gisifs = min(show_group_index+linear_gisif_iterations, len(GISIF_Groups));
        iterations_count = (max_linear_gisifs - show_group_index);
        linear_gisifs = np.zeros((GISIF_Groups[0][1].shape[0]));
        uselabel = None;
        print('ITERATIONS COUNT ', show_group_index, iterations_count, len(GISIF_Groups));
        for i in range(iterations_count):
            label, evectors_group  = GISIF_Groups[show_group_index+i];
            if(i == 0):
                uselabel = label;                
            eve = (evectors_group**2);
            gisifs = np.sum(eve, axis=1);
            linear_gisifs = gisifs - linear_gisifs;
            
        return linear_gisifs, K, uselabel;
        
    else:
        label, evectors_group  = GISIF_Groups[show_group_index];
        eve = (evectors_group**2);
        gisifs = np.sum(eve, axis=1);
        
    print('DOING GISIF COMPUTATION :');    
    
    print('FINISHED AND RETURNING THE COMPUTED GISIF VALUES :');
#     gisifs = gisifs/np.max(gisifs);
    return gisifs, K, label;
    