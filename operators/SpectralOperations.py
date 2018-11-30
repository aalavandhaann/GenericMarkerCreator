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

from bpy.props import StringProperty;
from GenericMarkerCreator.misc.spectralmagic import getHKSColors, getWKSColors, getGISIFColors, doLowpassFiltering;
from GenericMarkerCreator.misc.staticutilities import applyColoringForMeshErrors;
from GenericMarkerCreator.misc.mathandmatrices import getDuplicatedObject;
from GenericMarkerCreator.misc.mathandmatrices import setMeshVPOS;
from GenericMarkerCreator.misc.TrimeshCurvatures import need_curvatures;

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
            gisif_colors, k, gisif_name = getGISIFColors(context, mesh, mesh.eigen_k, mesh.gisif_threshold, mesh.gisif_group_index);
            mesh.gisif_group_name = gisif_name;
            normalized_gisifs = np.interp(gisif_colors, (gisif_colors.min(), gisif_colors.max()), (0.0,1.0));
            mesh.gisif_signatures.clear();
            normalized_gisif_signatures = gisif_colors / np.sqrt(np.sum(gisif_colors**2));
            for i in range(normalized_gisif_signatures.shape[0]):
                gisif_function_value = normalized_gisif_signatures[i];
                gisif_signature = mesh.gisif_signatures.add();
                gisif_signature.index = i;
                gisif_signature.gisif = gisif_function_value;
#             print(gisif_colors);
#             print(normalized_gisifs);
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
        X_std = features;#StandardScaler().fit_transform(X);
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
        return mu;
        
    
    
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            gisif_colors, k, gisif_name = getGISIFColors(context, mesh, mesh.eigen_k, mesh.gisif_threshold, mesh.gisif_group_index);
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