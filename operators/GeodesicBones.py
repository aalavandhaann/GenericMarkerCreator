__author__="ashok"
__date__ ="$Mar 23, 2015 8:16:11 PM$"


import gc
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import bpy, bmesh, time;
from bpy.props import StringProperty;
from bpy.props import FloatVectorProperty;
from mathutils import Vector;
from mathutils.bvhtree import BVHTree;

import numpy as np;
from scipy.sparse import csr_matrix;
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components;


from GenericMarkerCreator import constants;
from GenericMarkerCreator.propertiesregister import changeMarkerColor, changeUnlinkedMarkerColor;
from GenericMarkerCreator.misc.interactiveutilities import ScreenPoint3D;
from GenericMarkerCreator.misc.staticutilities import detectMN, applyMarkerColor, addConstraint, getConstraintsKD, deleteObjectWithMarkers, reorderConstraints;
from GenericMarkerCreator.misc.staticutilities import getMarkersForMirrorX, getGenericLandmark, getMeshForBlenderMarker, getBlenderMarker;
from GenericMarkerCreator.misc.meshmathutils import getBarycentricCoordinateFromPolygonFace, getBarycentricCoordinate, getCartesianFromBarycentre, getGeneralCartesianFromBarycentre, getTriangleArea;
from GenericMarkerCreator.misc.mathandmatrices import getBMMesh, ensurelookuptable, getMeshVPos; 


from chenhan_pp.MeshData import RichModel
from chenhan_pp.GraphPaths import ChenhanGeodesics, isFastAlgorithmLoaded;
from chenhan_pp.helpers import createGeodesicPathMesh;


class GeodesicBones(bpy.types.Operator):
    """Store the object names in the order they are selected, """ \
    """use RETURN key to confirm selection, ESCAPE key to cancel"""
    bl_idname = "genericlandmarks.geodesicbones";
    bl_label = "Geodesic Bones";
    bl_options = {'UNDO'};
    bl_description="Create bones using landmarks and their in-between geodesic distances";
    paired_operations = bpy.props.BoolProperty(name='paired operation', description='Create bones for the mesh that is assigned as a pair', default=False);
    
    def getGeodesicAlgorithm(self, context, mesh):
        richmodel = None;
        bm = getBMMesh(context, mesh, False);
        alg = None;
        if(not isFastAlgorithmLoaded):     
            try:
                richmodel = RichModel(bm, mesh);
                richmodel.Preprocess();
            except: 
                print('CANNOT CREATE RICH MODEL');
                richmodel = None;
       
        #If py_chenhann the c port of geodesics is available then richmodel is not necessary
        #if py_chenhann is not available and the richmodel is available then proceed to geodesics calculation
        if(richmodel or isFastAlgorithmLoaded):
            #Intitialize the chenhan algorithm and fill the K x K matrix of graph paths for minimal spanning tree
            alg = ChenhanGeodesics(context, mesh, bm, richmodel);
                        
        bm.free();        
        return alg;
    
    def getGeodesicGraphMatrix(self, context, mesh, seed_points):
        K = len(seed_points);
        np_matrix = np.zeros((K, K));
        alg = self.getGeodesicAlgorithm(context, mesh);       
        #Intitialize the chenhan algorithm and fill the K x K matrix of graph paths for minimal spanning tree
        wm = context.window_manager;
        wm.progress_begin(0, len(seed_points));
        for i, seed_index in enumerate(seed_points):
            alg.addSeedIndex(seed_index);
            v_distances = alg.getVertexDistances(seed_index);                
            np_matrix[i] = np.array(v_distances)[seed_points];                
            wm.progress_update(i+1);
        wm.progress_end();
        
        return np_matrix;
    
    def getSpanningTreePath(self, context, csgraph):
        graph_paths = [];
        use_seed_vertices = [];
        use_vertex_sequence = [];
        
        np_matrix = csr_matrix(csgraph);
        spanningtree = minimum_spanning_tree(np_matrix, overwrite=False);        
#===============================================================================
# #         plotting Minimum spanning tree
        coo = spanningtree.tocoo();
        edges = zip(coo.row, coo.col);
        edges = sorted(tuple(sorted(pair)) for pair in edges);        
        return edges;
    
    def getSeedIndicesFromLandmarks(self, context, mesh, landmarks=None):
        if(not landmarks):
            landmarks = mesh.generic_landmarks;
        return [lm.v_indices[np.argmax(lm.v_ratios)] for lm in landmarks];
    
    def bonesSingle(self, context, M):
        seedindices_m = self.getSeedIndicesFromLandmarks(context, M);
        seedindices_m_cos = getMeshVPos(M)[seedindices_m];
        csgraph_m = self.getGeodesicGraphMatrix(context, M, seedindices_m);
        bones_path = np.array(self.getSpanningTreePath(context, csgraph_m));
        
        return np.array(seedindices_m), np.array(seedindices_m)[bones_path], np.array(seedindices_m_cos)[bones_path], np.array(bones_path);
    
    def bonesPair(self, context, M, N):        
        landmarks_m = M.generic_landmarks;
        landmarks_m_paired_ids = [lm.linked_id for lm in landmarks_m];
        
        landmarkds_n_ids = [lm.id for lm in N.generic_landmarks];
        landmarks_n = [N.generic_landmarks[landmarkds_n_ids.index(lm_linked_id)] for lm_linked_id in landmarks_m_paired_ids]; 
        
        seedindices_m = self.getSeedIndicesFromLandmarks(context, M);
        seedindices_n = self.getSeedIndicesFromLandmarks(context, N, landmarks_n);
        
        seedindices_m_cos = getMeshVPos(M)[seedindices_m];
        seedindices_n_cos = getMeshVPos(N)[seedindices_n];
        
        csgraph_m = self.getGeodesicGraphMatrix(context, M, seedindices_m);
        csgraph_n = self.getGeodesicGraphMatrix(context, N, seedindices_n);
        bones_path = np.array(self.getSpanningTreePath(context, csgraph_m+csgraph_n));
        
        vids_m, vcos_m = np.array(seedindices_m)[bones_path], np.array(seedindices_m_cos)[bones_path];
        vids_n, vcos_n = np.array(seedindices_n)[bones_path], np.array(seedindices_n_cos)[bones_path];        
                
        return np.array(seedindices_m), vids_m, vcos_m, np.array(seedindices_n), vids_n, vcos_n, np.array(bones_path);
    
    def createBonesMesh(self, context, mesh, bone_vertices, edge_indices):
        iso_name = mesh.name+"_geodesics_bones";    
        try:
            existing = context.scene.objects[iso_name];
            existing.name = "READY_FOR_DELETE";
            bpy.ops.object.select_all(action="DESELECT");
            existing.select = True;
            context.scene.objects.active = existing;
            bpy.ops.object.delete();
        except KeyError:
            pass;
        
        iso_mesh = bpy.data.meshes.new(iso_name);
        vertex_cos = getMeshVPos(mesh);
        # Make a new BMesh
        bm = bmesh.new();
        
        for vid in bone_vertices:
            x, y, z = vertex_cos[vid];
            v = bm.verts.new();
            v.co = (x,y,z);
        
        ensurelookuptable(bm);
        
        for (id_1, id_2) in edge_indices:
            v1, v2 = bm.verts[id_1], bm.verts[id_2];
            e = bm.edges.new([v1, v2]);
            print(id_1, id_2);
        
        bm.to_mesh(iso_mesh);        
        
        iso_mesh_obj = bpy.data.objects.new(iso_name, iso_mesh);
        iso_mesh_obj.data = iso_mesh;
        context.scene.objects.link(iso_mesh_obj);
        iso_mesh_obj.location = mesh.location.copy();
        
        return iso_mesh_obj;
        
    def execute(self, context):
        if(context.active_object):
            activeobject = context.active_object;
            if(activeobject and self.paired_operations):                
                M, N = detectMN(activeobject);
                if(not M and not N):
                    message = "Select a mesh pair to do bones operation";
                    bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
                    return {'FINISHED'};
                else:
                    vids_m, vids_m_path, vcos_m_path, vids_n, vids_n_path, vcos_n_path, edge_indices = self.bonesPair(context, M, N);
                    bones_m_mesh = self.createBonesMesh(context, M, vids_m, edge_indices);
                    bones_n_mesh = self.createBonesMesh(context, N, vids_n, edge_indices);
                    
            elif (activeobject and not self.paired_operations):
                vids_m, vids_m_path, vcos_m_path, edge_indices = self.bonesSingle(context, activeobject);
                bones_m_mesh = self.createBonesMesh(context, activeobject, vids_m, edge_indices);
            else:
                message = "No valid meshes available";
                bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
                return {'FINISHED'};
        return {'FINISHED'};