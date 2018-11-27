import bpy;
import numpy as np;
import scipy as sp;
import scipy.sparse as spsp;
from scipy.sparse.linalg import eigsh, eigs;

from GenericMarkerCreator.misc.mathandmatrices import getBMMesh, ensurelookuptable, getMeshFaces, getMeshVPos, setMeshVPOS, getMeshFaceAngles, getLaplacianMatrixCotangent, getLaplacianMeshNormalized, getMeshVoronoiAreas;
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
    eve = (eve**2)*np.exp(-eva*t)[None, :];
    return np.sum(eve, 1);


def getHKSColors(context, mesh, K=5, HKS_T=20.0):
    L = getLaplacianMatrixCotangent(context, mesh, []);
    heat = getHKS(mesh, L, K, HKS_T);
    heat = heat/np.max(heat);
    return heat;
#     applyColoringForMeshErrors(context, mesh, heat, v_group_name='hks', use_weights=False);

def getWKS(mesh, L, A, K=3, WKS_E=100, wks_variance=6):
    num_vertices = L.shape[0];
    eva, eve = eigs(L,k=K,M=A,sigma=-1e-5,which='LM');
    eva = np.abs(np.real(eva));
    idx = np.argsort(eva);
    eva = eva[idx];
    eve = eve[:, idx];
    eve = np.real(eve);
    WKS = np.zeros((num_vertices,WKS_E));
    log_E = np.log(np.maximum(np.abs(eva), 1e-6)).T;
    e = np.linspace(log_E[1], np.max(log_E) / 1.02, WKS_E);
    sigma = (e[1]-e[0])*wks_variance;
    sigma_inv = 1.0 / sigma;
    sigma_inv_2 = (2*sigma**2);
    C = np.zeros((1, WKS_E));
    
    for i in range(WKS_E):
        np_tiled = np.exp((-(e[i] - log_E)**2) * sigma_inv_2);
        WKS[:,i] = np.sum( eve**2 * np.matlib.repmat(np_tiled, num_vertices, 1), 1);        
        C[i] = np.sum(np.exp((-(e[i]-log_E)**2) * sigma_inv_2), axis=np.newaxis);
    return 0;

def getWKSColors(context, mesh, L, K=100, WKS_E=6):
    L = getLaplacianMatrixCotangent(context, mesh);
    A, A_np = getMeshVoronoiAreas(context, mesh);
    wks = getWKS(mesh, L, A);
    return wks;
    




