import os;
import sys;
import time;

from scipy.io import loadmat
from scipy.sparse import csr_matrix, dia_matrix
from scipy.sparse.linalg import eigsh, inv
import numpy as np
from numpy import cross, arccos, array
from numpy.linalg import norm
from sklearn.neighbors import BallTree
from GenericMarkerCreator.misc.lib_rigid_ICP import compute_best_rigid_deformation;
from GenericMarkerCreator.misc.mathandmatrices import getWKSEigens, getWKSLaplacianMatrixCotangent, getMeshVoronoiAreas;
from GenericMarkerCreator.misc.mathandmatrices import getMeshFaces, getMeshVPos;
from GenericMarkerCreator.misc.spectralmagic import get_hks, get_wks, getMatrixCache, setMatrixCache;

class FunctionalMapResult():
    #The parameters are 
    # evecS = original eigenvectors of Source;
    # evecT = original eigenvectors of Target;
    # evecSMapped = transformed eigenvectors of Source after finding FunctionalMap C
    # C - The functional map itself
    # mode - The mode (wks or hks) used for finding the coeffs used in finding the functional maps
    # coeffS - The coefficeints of Source calculated based on the mode selected
    # coeffT - The coefficeints of Target calculated based on the mode selected
    def __init__(self, evecS, evecT, evecSMapped, C, mode, coeffS, coeffT):
        self.eivecS = evecS;
        self.eivecT = evecT;
        self.eivecSMapped = evecSMapped;
        self.C = C;
        self.mode = mode;
        self.coeffsS = coeffS;
        self.coeffsT = coeffT;
    
    def getDictionary(self):

def get_coef(eivec, matM, coefficients):
    coef = eivec.T.dot(matM.dot(coefficients));
    return coef;

def get_funcMap(coefS, coefT, eivalS, eivalT):
    k = coefS.shape[0]
    t = coefS.shape[1]
    w_func = 1
    w_com = 1
    w_reg = 1

    data_1 = (coefS.T.repeat(k,axis=0).flatten())*w_func
    row_ind_1 = np.arange(k*t).repeat(k)
    col_ind_1 = np.tile(np.arange(k*k),t)    
    
    lS = np.tile(eivalS,(eivalS.shape[0],1))    
    lT = np.tile(eivalT,(eivalS.shape[0],1)).T
    matEival = (lT-lS) * (lT-lS)
    data_2 = (matEival.T.repeat(k,axis=0).flatten())*w_com
    row_ind_2 = (np.arange(k*k)+(k*t)).repeat(k)
    col_ind_2 = np.tile(np.arange(k*k),k)
        
    data_3 = (np.ones(k*k))*w_reg
    row_ind_3 = np.arange(k*k)+((k*t)+(k*k))
    col_ind_3 = np.arange(k*k)
    
    M = (k*t)+(k*k)+(k*k)
    N = k*k
    data = np.concatenate((data_1,data_2,data_3),axis=0)
    row_ind = np.concatenate((row_ind_1,row_ind_2,row_ind_3),axis=0)
    col_ind = np.concatenate((col_ind_1,col_ind_2,col_ind_3),axis=0)
    a = csr_matrix((data,(row_ind,col_ind)),shape=(M,N))
    b1 = (coefT.T.flatten())*w_func
    b2 = (np.zeros(k*k))*w_com
    b3 = (np.zeros(k*k))*w_reg
    b = (np.concatenate((b1,b2,b3),axis=0))[:,None]
    aTa = a.T.dot(a)
    aTb = a.T.dot(b)
    funcMap = (inv(aTa).dot(aTb)).reshape(k,k)
    return funcMap

def extract_mapping_original(F, evecs_from, evecs_to):
    bt_ = BallTree(F.dot(evecs_from.T).T)
    dists, others = bt_.query(evecs_to)
    return others.flatten()

def funcmap_correspondences(context, source, target, n_eigen=30, spectral_steps=100, coeffsmode='wks'):    
#     verS, verT, triS, triT = getMeshVPos(source), getMeshVPos(target), getMeshFaces(source), getMeshFaces(target);
    
    n_eigen = min(len(source.data.vertices)-1, len(target.data.vertices)-1, n_eigen);
        
     #%% 1. LapBel Operator & Basis (eigenfunc)    
    __, matMS = getMatrixCache(context, source, 'WKS_VORONOI');
    __, matCS = getMatrixCache(context, source, 'WKS_L');
    __, matMT = getMatrixCache(context, target, 'WKS_VORONOI');
    __, matCT = getMatrixCache(context, target, 'WKS_L');
    
    s_k_exists, s_cache_k = getMatrixCache(context, source, 'WKS_k');
    WKS_EVA_Exists, WKS_EVA = getMatrixCache(context, source, 'WKS_eva');
    WKS_EVE_Exists, WKS_EVE = getMatrixCache(context, source, 'WKS_eve');
        
    if(not s_k_exists):
        setMatrixCache(context, source, 'WKS_k', n_eigen);
    
    if(s_cache_k != n_eigen or not WKS_EVA_Exists or not WKS_EVE_Exists):
        eivalS, eivecS = getWKSEigens(source, matMS,matCS,n_eigen);
        eivalT, eivecT = getWKSEigens(target, matMT,matCT,n_eigen);
        
        setMatrixCache(context, source, 'WKS_k', n_eigen);
        setMatrixCache(context, target, 'WKS_k', n_eigen);
        
        setMatrixCache(context, source, 'WKS_eva', eivalS);
        setMatrixCache(context, target, 'WKS_eva', eivalT);
        
        setMatrixCache(context, source, 'WKS_eve', eivecS);
        setMatrixCache(context, target, 'WKS_eve', eivecT);
    else:
        __, eivalS = getMatrixCache(context, source, 'WKS_eva')
        __, eivalT = getMatrixCache(context, target, 'WKS_eva');
        __, eivecS = getMatrixCache(context, source, 'WKS_eve')
        __, eivecT = getMatrixCache(context, target, 'WKS_eve');
        
#     eivalS = eivalS[:-1]
#     eivecS = eivecS[:,:-1]
#     eivalT = eivalT[:-1]
#     eivecT = eivecT[:,:-1]
    
    #%% 2. Function Representation (HKS & WKS) & its Coefficient
    if(coeffsmode == 'hks'):
        hksS = get_hks(eivalS, eivecS, matMS, spectral_steps);
        hksT = get_hks(eivalT, eivecT, matMT, spectral_steps);
        coefS = get_coef(eivecS, matMS, hksS);
        coefT = get_coef(eivecT, matMT, hksT);
    elif (coeffsmode=='wks'):
        wksS = get_wks(eivalS, eivecS, num_steps=spectral_steps);
        wksT = get_wks(eivalT, eivecT, num_steps=spectral_steps);
        coefS = get_coef(eivecS, matMS, wksS);
        coefT = get_coef(eivecT, matMT, wksT);
    
    else:
        raise NotImplementedError('The mode %s not known and has not been implemented');
    
    funcMap = get_funcMap(coefS, coefT, eivalS, eivalT);
    eivecS_mapped = (funcMap.dot(eivecS.T)).T;
    
    return FunctionalMapResult(eivecS, eivecT, eivecS_mapped, funcMap, coeffsmode, coefS, coefT);
    
    
