__author__="ashok"
__date__ ="$Mar 23, 2015 8:16:11 PM$"


import gc
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import bpy, time;

from bpy.props import StringProperty;
from GenericMarkerCreator.misc.spectralmagic import getHKSColors, getWKSColors, getGISIFColors, doLowpassFiltering;
from GenericMarkerCreator.misc.staticutilities import applyColoringForMeshErrors;
from GenericMarkerCreator.misc.mathandmatrices import getDuplicatedObject;
from GenericMarkerCreator.misc.mathandmatrices import setMeshVPOS;

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
            wks_colors, k, gisif_name = getGISIFColors(context, mesh, mesh.eigen_k, mesh.gisif_threshold, mesh.gisif_group_index);
            mesh.gisif_group_name = gisif_name;
            applyColoringForMeshErrors(context, mesh, wks_colors, v_group_name='gisif', use_weights=False);
                
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