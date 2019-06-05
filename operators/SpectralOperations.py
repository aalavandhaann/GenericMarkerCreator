__author__="ashok"
__date__ ="$Mar 23, 2015 8:16:11 PM$"


import gc
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import bpy, time;
import numpy as np;
import scipy.io as sio;
import scipy.sparse as spsp;

from sklearn.decomposition import PCA as sklearnPCA;
from sklearn.preprocessing import StandardScaler;
from sklearn.mixture  import GaussianMixture;
from sklearn.metrics import pairwise_distances_argmin_min;

from bpy.props import StringProperty;
from GenericMarkerCreator.misc.spectralmagic import getHKSColors, getWKSColors, getGISIFColors, doLowpassFiltering;
from GenericMarkerCreator.misc.staticutilities import applyColoringForMeshErrors;
from GenericMarkerCreator.misc.mathandmatrices import getDuplicatedObject;
from GenericMarkerCreator.misc.mathandmatrices import setMeshVPOS;
from GenericMarkerCreator.misc.TrimeshCurvatures import need_curvatures;
from GenericMarkerCreator.misc.mathandmatrices import getMeshFaces;
from GenericMarkerCreator.misc.staticutilities import addConstraint, getBlenderMarker, detectMorN;

from GenericMarkerCreator.misc.mathandmatrices import getMeshVPos, get_matC2;
from GenericMarkerCreator.misc.mathandmatrices import get_matM3;

def getGISIFColorsInner(context, mesh, applyMesh=None):    
    use_mesh = mesh;
    if(applyMesh):
        use_mesh = applyMesh;
    if(mesh.linear_gisif_combinations):
        gisif_colors, k, gisif_name = getGISIFColors(context, use_mesh, mesh.eigen_k, mesh.gisif_threshold, mesh.gisif_group_index, linear_gisif_iterations=mesh.linear_gisif_n);
    else:
        gisif_colors, k, gisif_name = getGISIFColors(context, use_mesh, mesh.eigen_k, mesh.gisif_threshold, mesh.gisif_group_index);
    
    return gisif_colors, k , gisif_name;

def pcaTransform(context, mesh, features, K=5):
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
        return mu, Y_sklearn;

#The operators for creating landmarks
class MeanCurvatures(bpy.types.Operator):
    bl_idname = "genericlandmarks.meancurvatures";
    bl_label = "Mean Curvatures";
    bl_description = "Plot the mean curvature values as a color. This method uses Mark Meyer with post processing the curvatures using histogram";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    bl_options = {'REGISTER', 'UNDO'}
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");
    postprocess = bpy.props.BoolProperty(name="Post Process", default=False);
    percent_min = bpy.props.FloatProperty(name="Min Histogram", default=0.1);
    percent_max = bpy.props.FloatProperty(name="Max Histogram", default=0.9);
    
    def mesh_mean_curvatures(self, context, mesh):
        vpos = getMeshVPos(mesh);
        N = vpos.shape[0];
        normals = np.zeros((N,3));
        for v in mesh.data.vertices:
            normals[v.index] = v.normal.to_tuple();    
        vdotnormal = np.sum(vpos * normals, axis=1);
        vdotnormal[vdotnormal < 0.0] = -1.0;
        vdotnormal[vdotnormal > 0.0] = 1.0;        
        normals_weight = spsp.dia_matrix((vdotnormal, 0), shape=(N,N));
        
        C = get_matC2(context, mesh);
        A, Aa = get_matM3(context, mesh, ANormalize=False);
        AMixed = spsp.dia_matrix(((1.0 / (2.0*A.data)), 0), shape=(N,N));
        delta = AMixed.dot(C.dot(vpos));
        K = np.sqrt(np.sum(delta**2, axis=1));
        K = normals_weight.dot(K);
        return K;
    
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
        
        K = self.mesh_mean_curvatures(context, mesh);
        
        pp = mesh.post_process_colors;
        ppmin = mesh.post_process_min;
        ppmax = mesh.post_process_max;
        if(pp):
            applyColoringForMeshErrors(context, mesh, K, v_group_name='mean curvatures', use_weights=True, normalize_weights=True, use_histogram_preprocess=pp, percent_min=ppmin, percent_max=ppmax);
        else:
            applyColoringForMeshErrors(context, mesh, K, v_group_name='mean curvatures', use_weights=True, normalize_weights=True, use_histogram_preprocess=self.postprocess, percent_min=self.percent_min, percent_max=self.percent_max);
        return {'FINISHED'};
        

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
            heat_colors, k = getHKSColors(context, mesh, mesh.eigen_k, mesh.hks_t, mesh.hks_current_t);
            
            pp = mesh.post_process_colors;
            ppmin = mesh.post_process_min;
            ppmax = mesh.post_process_max;
            applyColoringForMeshErrors(context, mesh, heat_colors, v_group_name='hks', use_weights=False, use_histogram_preprocess=pp, percent_min=ppmin, percent_max=ppmax);
            
            if(mesh.spectral_sync):
                paired_mesh = detectMorN(mesh);
                pp = paired_mesh.post_process_colors;
                ppmin = paired_mesh.post_process_min;
                ppmax = paired_mesh.post_process_max;
                if(paired_mesh):
                    heat_colors, k = getHKSColors(context, paired_mesh, mesh.eigen_k, mesh.hks_t, mesh.hks_current_t);
                    applyColoringForMeshErrors(context, paired_mesh, heat_colors, v_group_name='hks', use_weights=False, use_histogram_preprocess=pp, percent_min=ppmin, percent_max=ppmax);
                
                
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
            wks_colors, k = getWKSColors(context, mesh, mesh.eigen_k, mesh.wks_e, mesh.wks_current_e, mesh.wks_variance);
            pp = mesh.post_process_colors;
            ppmin = mesh.post_process_min;
            ppmax = mesh.post_process_max;
            
            applyColoringForMeshErrors(context, mesh, wks_colors, v_group_name='wks', use_weights=False, use_histogram_preprocess=pp, percent_min=ppmin, percent_max=ppmax);
            
            if(mesh.spectral_sync):
                paired_mesh = detectMorN(mesh);
                pp = paired_mesh.post_process_colors;
                ppmin = paired_mesh.post_process_min;
                ppmax = paired_mesh.post_process_max;
                if(paired_mesh):
                    wks_colors, k = getWKSColors(context, paired_mesh, mesh.eigen_k, mesh.wks_e, mesh.wks_current_e, mesh.wks_variance);
                    applyColoringForMeshErrors(context, paired_mesh, wks_colors, v_group_name='wks', use_weights=False, use_histogram_preprocess=pp, percent_min=ppmin, percent_max=ppmax);
                
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
            mesh.gisif_signatures.clear();
            
            if(not mesh.gisif_symmetries or not len(mesh.generic_landmarks)):
                normalized_gisifs = np.interp(gisif_colors, (gisif_colors.min(), gisif_colors.max()), (0.0,1.0));                
            else:
                if(len(mesh.generic_landmarks)):
                    mindex = min(len(mesh.generic_landmarks), mesh.gisif_symmetry_index) - 1;
                    marker = mesh.generic_landmarks[mindex];
                    bmarker = getBlenderMarker(mesh, marker);
                    bpy.ops.object.select_all(action="DESELECT");
                    bmarker.select = True;
                    
                    indices = np.array(marker.v_indices);
                    uvw = np.array(marker.v_ratios);
                    
                    o_vid = indices[np.argmax(uvw)];
                    normalized_gisifs = gisif_colors / np.sqrt(np.sum(gisif_colors**2));                    
                    delta_gisif_colors = np.sqrt((normalized_gisifs[o_vid] - normalized_gisifs)**2);                    
                    normalized_gisifs = np.interp(delta_gisif_colors, (delta_gisif_colors.min(), delta_gisif_colors.max()), (0.0,1.0));            
            
            
            pp = mesh.post_process_colors;
            ppmin = mesh.post_process_min;
            ppmax = mesh.post_process_max;
            applyColoringForMeshErrors(context, mesh, normalized_gisifs, v_group_name='gisif', use_weights=False, use_histogram_preprocess=pp, percent_min=ppmin, percent_max=ppmax);
            
            if(mesh.spectral_sync):
                paired_mesh = detectMorN(mesh);
                pp = paired_mesh.post_process_colors;
                ppmin = paired_mesh.post_process_min;
                ppmax = paired_mesh.post_process_max;
                if(paired_mesh):
                    gisif_colors, k, gisif_name = getGISIFColorsInner(context, mesh, applyMesh=paired_mesh);
                    paired_mesh.gisif_group_name = gisif_name;
                    normalized_gisifs = np.interp(gisif_colors, (gisif_colors.min(), gisif_colors.max()), (0.0,1.0)); 
                    applyColoringForMeshErrors(context, paired_mesh, normalized_gisifs, v_group_name='gisif', use_weights=False, use_histogram_preprocess=pp, percent_min=ppmin, percent_max=ppmax);
            
        return{'FINISHED'};

#The operators for creating landmarks
class SpectralFeatures(bpy.types.Operator):
    bl_idname = "genericlandmarks.spectralfeatures";
    bl_label = "Features Colors";
    bl_description = "Feature Properties of a mesh as colors";
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
            m_path = bpy.path.abspath('%s/%s.mat'%(mesh.signatures_dir, mesh.name));
            data = sio.loadmat(m_path)['X'];
            normalized_gisifs = np.sum(data, axis=1);   
            normalized_gisifs = np.interp(normalized_gisifs, (normalized_gisifs.min(), normalized_gisifs.max()), (0.0,1.0));    
            applyColoringForMeshErrors(context, mesh, normalized_gisifs, v_group_name='features', use_weights=False, A=np.min(normalized_gisifs), B=np.max(normalized_gisifs));
            
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
    
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            gisif_colors, k, gisif_name = getGISIFColorsInner(context, mesh);
            k1_list, k2_list, sx, p1_list, p2_list, mean_list, gaussian_list, normals = need_curvatures(mesh);
            normalized_gisif_signatures = gisif_colors / np.sqrt(np.sum(gisif_colors**2));            
            features = np.hstack((normals, k1_list.reshape(k1_list.shape[0],1), k2_list.reshape(k2_list.shape[0],1), p1_list, p2_list, normalized_gisif_signatures.reshape(normalized_gisif_signatures.shape[0],1)));
            mu, transformedFeatures = pcaTransform(context, mesh, features, K=12);                
        return{'FINISHED'};

class SpectralShape(bpy.types.Operator):
    bl_idname = "genericlandmarks.spectralshape";
    bl_label = "Spectral Shape";
    bl_description = "Low pass geometric filtering of a mesh";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");
    use_eigen_k = bpy.props.IntProperty(name="Initialize for Object", default = -1);
    
    def applySpectralShape(self, context, mesh, eigen_k=5):
        try:
            dup_mesh = context.scene.objects["%s-spectral-shape"%(mesh.name)];
        except KeyError:
            dup_mesh = getDuplicatedObject(context, mesh, meshname="%s-spectral-shape"%(mesh.name));
            bpy.ops.object.select_all(action="DESELECT");
            context.scene.objects.active = mesh;
            mesh.select = True;
            
        vpos = doLowpassFiltering(context, mesh, eigen_k);
        setMeshVPOS(dup_mesh, vpos);
         
    
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
        
        if(mesh is not None):
            if(self.use_eigen_k != -1):
                self.applySpectralShape(context, mesh, self.use_eigen_k);
            else:
                self.applySpectralShape(context, mesh, mesh.eigen_k);
                if(mesh.spectral_sync):
                    paired_mesh = detectMorN(mesh);
                    if(paired_mesh):
                        self.applySpectralShape(context, paired_mesh, mesh.eigen_k);
                
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
            only_gisif_colors, k, gisif_name = getGISIFColorsInner(context, mesh);
            only_gisif_colors = only_gisif_colors.reshape(only_gisif_colors.shape[0], 1);
            #Normalize the gisif colors
            only_gisif_colors = only_gisif_colors / np.sqrt(np.sum(only_gisif_colors**2));
            
#             k1_list, k2_list, sx, p1_list, p2_list, mean_list, gaussian_list, normals = need_curvatures(mesh);             
#             features = np.hstack((normals, k1_list.reshape(k1_list.shape[0],1), k2_list.reshape(k2_list.shape[0],1), p1_list, p2_list, only_gisif_colors.reshape(only_gisif_colors.shape[0],1)));
#             mu, transformedFeatures = pcaTransform(context, mesh, features, K=12);
            
            gisif_colors = only_gisif_colors;
            
            gisif_colors = StandardScaler().fit_transform(gisif_colors);            
            count_n = mesh.gisif_markers_n;      
                  
            gmm = GaussianMixture(n_components=count_n, covariance_type='full').fit(gisif_colors);
            labels_gmm = gmm.predict(gisif_colors);
            labels_gmm.shape = (labels_gmm.shape[0], 1);
            
#             gmm_sorted_indices = np.argsort(gmm.means_.T).flatten();
#             gmm_sorted_values = np.sort(gmm.means_.T).flatten();
            
            gmm_sorted_indices = np.array([i for i in range(count_n)]);
            gmm_sorted_values = gmm.means_;
            
            print(gmm.means_, gmm_sorted_indices);
            
            keyindices = [];
            print('='*40);
            for i in range(count_n):
                gmm_label_index = gmm_sorted_indices[i];
                gmm_value = gmm_sorted_values[gmm_label_index];
                gmm_subset, __ = np.where(labels_gmm == gmm_label_index);
                cluster_values = gisif_colors[gmm_subset];
                print(gmm_value, gmm_value.shape, cluster_values.shape);
                closest, __ = pairwise_distances_argmin_min(gmm_value.reshape(1, -1), cluster_values);
                closest_index = gmm_subset[closest[0]];
                closest_value = gisif_colors[closest_index];
                keyindices.append(closest_index);
                print('-----------------');
#                 print('GMM VALUES (Mean: %f, Closest: %f, Closest Index: %d, In Subset Value: %f, In Subset Index: %d) ::: '%(gmm_value, closest_value, closest_index, cluster_values[closest[0]], closest[0]));
            
            faces = getMeshFaces(mesh);
            for vid in keyindices:
                uvw = [0.0, 0.0, 0.0];
                faces_rows, faces_column = np.where(faces == vid);
                face_row_index, face_column_index = faces_rows[0], faces_column[0];
                face_row = faces[face_row_index];
                uvw[face_column_index] = 1.0;
                vid1, vid2, vid3 = face_row.tolist();
                print(vid1, vid2, vid3);
                co = mesh.data.vertices[face_row[face_column_index]].co;
                addConstraint(context, mesh, uvw, [vid1, vid2, vid3], co, faceindex=face_row_index, create_visual_landmarks = False);
            
            if(mesh.gisif_symmetries):
                print('~'*40);
                for o_vid in keyindices:
                    #EQuation 10 in the paper for finding the symmetry points where the euclidean distance will be zero for symmetry
                    delta_gisif_colors = np.sqrt((only_gisif_colors[o_vid] - only_gisif_colors)**2);
#                     delta_gisif_colors[o_vid] = np.finfo(float).max;                    
                    vidrows, __ = np.where(delta_gisif_colors == 0.0);
                    
                    print(delta_gisif_colors[vidrows]);
                    print(vidrows);
                    
                    filtered_vid_values = delta_gisif_colors[vidrows];                 
                    vid = vidrows[filtered_vid_values.argmin()];
                    print(o_vid, vid);
                    
                    uvw = [0.0, 0.0, 0.0];
                    faces_rows, faces_column = np.where(faces == vid);
                    face_row_index, face_column_index = faces_rows[0], faces_column[0];
                    face_row = faces[face_row_index];
                    uvw[face_column_index] = 1.0;
                    vid1, vid2, vid3 = face_row.tolist();
                    print(vid1, vid2, vid3);
                    co = mesh.data.vertices[face_row[face_column_index]].co;
                    addConstraint(context, mesh, uvw, [vid1, vid2, vid3], co, faceindex=face_row_index, create_visual_landmarks = False);
                    
            
            
#             bpy.ops.genericlandmarks.createlandmarks('EXEC_DEFAULT', currentobject=mesh.name, updatepositions = True);
            bpy.ops.genericlandmarks.changelandmarks('EXEC_DEFAULT', currentobject=mesh.name);                
                
        return{'FINISHED'};
    
    