__author__="ashok"
__date__ ="$Mar 23, 2015 8:16:11 PM$"


import gc
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import bpy, time;
import numpy as np;
import scipy.io as sio;
from sklearn.decomposition import PCA as sklearnPCA;
from sklearn.preprocessing import StandardScaler;
from sklearn.mixture  import GaussianMixture;

from bpy.props import StringProperty;
from GenericMarkerCreator.misc.spectralmagic import getHKSColors, getWKSColors, getGISIFColors, doLowpassFiltering;
from GenericMarkerCreator.misc.staticutilities import applyColoringForMeshErrors;
from GenericMarkerCreator.misc.mathandmatrices import getDuplicatedObject;
from GenericMarkerCreator.misc.mathandmatrices import setMeshVPOS;
from GenericMarkerCreator.misc.TrimeshCurvatures import need_curvatures;

def getGISIFColorsInner(context, mesh):
    if(mesh.linear_gisif_combinations):
        gisif_colors, k, gisif_name = getGISIFColors(context, mesh, mesh.eigen_k, mesh.gisif_threshold, mesh.gisif_group_index, linear_gisif_iterations=mesh.linear_gisif_n);
    else:
        gisif_colors, k, gisif_name = getGISIFColors(context, mesh, mesh.eigen_k, mesh.gisif_threshold, mesh.gisif_group_index);
    
    return gisif_colors, k , gisif_name;


#The operators for creating landmarks
class SpectralHKS(bpy.types.Operator):
    bl_idname = "genericlandmarks.spectralhks";
    bl_label = "HKS Colors";
    bl_description = "HKS Properties of a mesh as colors over time";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");
     
        
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            heat_colors, k = getHKSColors(context, mesh, mesh.eigen_k, mesh.hks_t);
            applyColoringForMeshErrors(context, mesh, heat_colors, v_group_name='hks', use_weights=False);
                
        return{'FINISHED'};

#The operators for creating landmarks
class SpectralWKS(bpy.types.Operator):
    bl_idname = "genericlandmarks.spectralwks";
    bl_label = "WKS Colors";
    bl_description = "WKS Properties of a mesh as colors over variance and evaluations";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");
     
        
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            wks_colors, k = getWKSColors(context, mesh, mesh.eigen_k, mesh.wks_e, mesh.wks_variance);
            applyColoringForMeshErrors(context, mesh, wks_colors, v_group_name='wks', use_weights=False);
                
        return{'FINISHED'};


#The operators for creating landmarks
class SpectralGISIF(bpy.types.Operator):
    bl_idname = "genericlandmarks.spectralgisif";
    bl_label = "GISIF Colors";
    bl_description = "GISIF Properties of a mesh as colors over eigen threshold";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");     
        
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            gisif_colors, k, gisif_name = getGISIFColorsInner(context, mesh);
            mesh.gisif_group_name = gisif_name;
            normalized_gisifs = np.interp(gisif_colors, (gisif_colors.min(), gisif_colors.max()), (0.0,1.0));
            mesh.gisif_signatures.clear();
            applyColoringForMeshErrors(context, mesh, normalized_gisifs, v_group_name='gisif', use_weights=False, A=np.min(normalized_gisifs), B=np.max(normalized_gisifs));
                
        return{'FINISHED'};

#The operators for creating landmarks
class AddSpectralSignatures(bpy.types.Operator):
    bl_idname = "genericlandmarks.addspectralsignatures";
    bl_label = "Add Spectral Signatures";
    bl_description = "Compute GISIF and add other signatures";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");     
    
    def pcaTransform(self, context, mesh, features, K=5):
#         X_std = features;#StandardScaler().fit_transform(X);
        X_std = StandardScaler().fit_transform(features);
        sklearn_pca = sklearnPCA(n_components=K);
        Y_sklearn = sklearn_pca.fit_transform(X_std);
        
        mu = sklearn_pca.mean_;
        mu.shape = (mu.shape[0], 1);
        D = sklearn_pca.explained_variance_;
        D_ratio = sklearn_pca.explained_variance_ratio_;
        V = sklearn_pca.components_;
        print('*'*40);
        print('DATA ENTRIES SHAPE ::: ', features.shape);
        print('MEAN MATRIX SHAPE ::: ', mu.shape);
        print('EIGEN VALUES SHAPE ::: ', D.shape);
        print('EIGEN VECTORS SHAPE ::: ', V.shape);
        print('TRANSFORMED SHAPE ::: ', Y_sklearn.shape);        
        sio.savemat(bpy.path.abspath('%s/%s.mat'%(mesh.signatures_dir, mesh.name)), {'eigenvectors':V.T, 'eigenvalues':D, 'mu':mu, 'X':X_std,'XMinusMu':(X_std.T - mu), 'transformed':Y_sklearn});        
        print('FINISHED SAVING ::: %s/%s.mat'%(mesh.signatures_dir, mesh.name));
        return mu;
    
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            gisif_colors, k, gisif_name = getGISIFColorsInner(context, mesh);
            k1_list, k2_list, sx, p1_list, p2_list, mean_list, gaussian_list, normals = need_curvatures(mesh);
            normalized_gisif_signatures = gisif_colors / np.sqrt(np.sum(gisif_colors**2));
            
            print(normals.shape, k1_list.shape, k2_list.shape, p1_list.shape, p2_list.shape, normalized_gisif_signatures.shape);
            
            features = np.hstack((normals, k1_list.reshape(k1_list.shape[0],1), k2_list.reshape(k2_list.shape[0],1), p1_list, p2_list, normalized_gisif_signatures.reshape(normalized_gisif_signatures.shape[0],1)));
            print(features.shape);
            mu = self.pcaTransform(context, mesh, features, K=12);
                
                
        return{'FINISHED'};

class SpectralShape(bpy.types.Operator):
    bl_idname = "genericlandmarks.spectralshape";
    bl_label = "Spectral Shape";
    bl_description = "Low pass geometric filtering of a mesh";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");
     
        
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            try:
                dup_mesh = context.scene.objects["%s-spectral-shape"%(mesh.name)];
            except KeyError:
                dup_mesh = getDuplicatedObject(context, mesh, meshname="%s-spectral-shape"%(mesh.name));
                bpy.ops.object.select_all(action="DESELECT");
                context.scene.objects.active = mesh;
                mesh.select = True;
            vpos = doLowpassFiltering(context, mesh, mesh.eigen_k);
            setMeshVPOS(dup_mesh, vpos);
                
        return{'FINISHED'};

#The operators for creating landmarks
class AddSpectralSignatureLandmarks(bpy.types.Operator):
    bl_idname = "genericlandmarks.addspectralsignaturelandmarks";
    bl_label = "Add Spectral Signature Landmarks";
    bl_description = "Compute GISIF and add other signatures and use them to create landmarks";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");     
    
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            gisif_colors, k, gisif_name = getGISIFColorsInner(context, mesh);
            gisif_colors = gisif_colors.reshape(gisif_colors.shape[0], 1);
            gisif_colors = StandardScaler().fit_transform(gisif_colors);            
            count_n = mesh.gisif_markers_n;      
                  
            gmm = GaussianMixture(n_components=count_n, covariance_type='full').fit(gisif_colors);
            labels_gmm = gmm.predict(gisif_colors);
            labels_gmm.shape = (labels_gmm.shape[0], 1);
            
            proba_gmm = gmm.predict_proba(gisif_colors);
            score_gmm = gmm.score(gisif_colors);
            
            gmm_sorted_indices = np.argsort(gmm.means_.T).flatten();
            gmm_sorted_values = np.sort(gmm.means_.T).flatten();
            
            print(score_gmm);
            print('='*40);
            for i in range(count_n):
                gmm_label_index = gmm_sorted_indices[i];
                gmm_value = gmm_sorted_values[gmm_label_index];
                gmm_subset, __ = np.where(labels_gmm == gmm_label_index);
                print('-----------------');                
                print(proba_gmm[gmm_subset[:,None],[i]]);
                print('GMM VALUE ::: ', gmm_value);
                
                
            print(gmm.means_.T, gisif_colors.shape);
            print(gisif_colors.min(), gisif_colors.max());
            print(labels_gmm.shape);
                
                
        return{'FINISHED'};
    
    