import bpy;

from GenericMarkerCreator.misc.staticutilities import getMeshForBlenderMarker;
from GenericMarkerCreator.operators.LandmarksPair import AssignMeshPair;
from GenericMarkerCreator.operators.LiveOperators import LiveLandmarksCreator, SignaturesMatching;
from GenericMarkerCreator.operators.LandmarksCreator import CreateLandmarks, ReorderLandmarks, \
ChangeLandmarks, UnLinkLandmarks, LinkLandmarks, RemoveLandmarks, LandmarkStatus, LandmarksPairFinder, TransferLandmarkNames, AutoLinkLandmarksByID, SnapLandmarksToVertex, \
LoadBIMLandmarks;
from GenericMarkerCreator.operators.SpectralOperations import SpectralHKS, SpectralWKS, SpectralGISIF, SpectralShape, AddSpectralSignatures, AddSpectralSignatureLandmarks,SpectralFeatures, MeanCurvatures;

class SpectralGeneralPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_spectralpanel"
    bl_label = "Spectral Controls";
    bl_space_type = "VIEW_3D";
    bl_region_type = "TOOLS";
    bl_category = "Generic Landmarks"
    bl_description = "Panel to operate for adding landmarks to a mesh";
        
    def draw(self, context):
        if(context.active_object):
            layout = self.layout;
            mainbox = layout.box();
            mainbox.label('Spectral Properties');
            
            row = mainbox.row();
            row.prop(context.active_object, 'spectral_sync');
            
            row = mainbox.row();
            row.prop(context.active_object, 'post_process_colors', 'Post Process Colors?');
            
            col = row.column();
            col.prop(context.active_object, 'post_process_min', 'Min');
            
            col = row.column();
            col.prop(context.active_object, 'post_process_max', 'Max');
            
            row = mainbox.row();
            row.prop(context.active_object, 'eigen_k');
            
            box = mainbox.box();
            box.label('Low Pass Filtering (Eigen Shapes)')
            row = box.row();
            row.prop(context.active_object, 'live_spectral_shape');
            row.operator(SpectralShape.bl_idname);           
            
            row = box.row();
            row.operator(SpectralFeatures.bl_idname);
            
            box = layout.box();
            box.label('Mean Curvatures')
            
            row = box.row();
            col = row.column();
            col.prop(context.active_object, 'mean_curvatures_use_normal');
            
            col = row.column();
            op = row.operator(MeanCurvatures.bl_idname);
            
#             row = box.row();
#             row.prop(op, "percent_min");
#             
#             row = box.row();
#             row.prop(op, "percent_max");
    

class HKSPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_hksspanel"
    bl_label = "HKS Controls";
    bl_space_type = "VIEW_3D";
    bl_region_type = "TOOLS";
    bl_category = "Generic Landmarks"
    bl_description = "Panel to operate for adding landmarks to a mesh";   
     
    def draw(self, context):
        if(context.active_object):
            layout = self.layout;
            box = layout.box();
            box.label('HKS');
            row = box.row();       
            
            row.prop(context.active_object, 'hks_t');
            row.prop(context.active_object, 'hks_current_t');
            
            row = box.row();
            row.prop(context.active_object, 'hks_log_start');
            row.prop(context.active_object, 'hks_log_end');
            
            row.prop(context.active_object, 'live_hks');
            row = box.row();
            row.operator(SpectralHKS.bl_idname);

class WKSPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_wksspanel"
    bl_label = "WKS Controls";
    bl_space_type = "VIEW_3D";
    bl_region_type = "TOOLS";
    bl_category = "Generic Landmarks"
    bl_description = "Panel to operate for adding landmarks to a mesh";  
    
    def draw(self, context):
        if(context.active_object):
            layout = self.layout;
            box = layout.box();
            box.label('WKS');
            row = box.row();       
            row.prop(context.active_object, 'wks_e');
            row.prop(context.active_object, 'wks_variance');
            row.prop(context.active_object, 'wks_current_e');
            row.prop(context.active_object, 'live_wks');
            row = box.row();
            row.operator(SpectralWKS.bl_idname);

class GISIFPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_gisifpanel"
    bl_label = "GISIF Controls";
    bl_space_type = "VIEW_3D";
    bl_region_type = "TOOLS";
    bl_category = "Generic Landmarks"
    bl_description = "Panel to operate for adding landmarks to a mesh";  
    
    def draw(self, context):
        if(context.active_object):
            layout = self.layout;
            box = layout.box();
            box.label('GISIF');
            row = box.row();
            row.prop(context.active_object, 'gisif_threshold');
            row.prop(context.active_object, 'gisif_group_index');
            row.prop(context.active_object, 'live_gisif');
            
            box_linear = box.box();
            box_linear.label('Linear Combinations');
            row = box_linear.row();
            row.prop(context.active_object, 'linear_gisif_combinations');
            row = box_linear.row();
            row.prop(context.active_object, 'linear_gisif_n');
            
            row = box.row();
            row.label('GISIF:%s'%(context.active_object.gisif_group_name));
            row.prop(context.active_object, 'gisif_symmetries');
            row.prop(context.active_object, 'gisif_symmetry_index');
            row = box.row();
            row.operator(SpectralGISIF.bl_idname);
            
            kp_box = box.box();
            kp_box.label('Generate Keypoints');
            row = kp_box.row();
            row.prop(context.active_object, 'gisif_markers_n');                        
            row.operator(AddSpectralSignatureLandmarks.bl_idname);
            
            save_box = box.box();
            save_box.label('Save Signatures')
            row = save_box.row();
            row.prop(context.active_object, 'signatures_dir');
            row.operator(AddSpectralSignatures.bl_idname);        

class LandmarksPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_landmarksspanel"
    bl_label = "Generic Landmarks";
    bl_space_type = "VIEW_3D";
    bl_region_type = "TOOLS";
#     bl_space_type = 'PROPERTIES'
#     bl_region_type = "WINDOW";
    bl_category = "Generic Landmarks"
    bl_description = "Panel to operate for adding landmarks to a mesh"
    
    def draw(self, context):        
        if(context.active_object):
            layout = self.layout;
            box = layout.box();
            box.label('Global properties');
            row = box.row();
            row.prop(context.scene, 'use_mirrormode_x');
            row.prop(context.scene, 'landmarks_use_selection');
            
            row = box.row();
            row.prop(context.active_object, 'snap_landmarks');
            
            box = layout.box();
            box.label('Load Landmarks from a file');
            
            row = box.row(align=True);
            col = row.column(align=True);
            col.prop(context.active_object, 'landmarks_file');
            
            col = row.column(align=True);
            col.operator(LoadBIMLandmarks.bl_idname);
            
            if(len(context.active_object.generic_landmarks)):
                box = layout.box();
                box.label('Unpaired mesh operations');    
                
                row = box.row();
                row.prop(context.active_object, 'hide_landmarks');
                
                row = box.row();
                op = row.operator(CreateLandmarks.bl_idname, text="Update Positions");
                op.updatepositions = True;
                
                row = box.row();
                row.operator(ChangeLandmarks.bl_idname);
                
                row = box.row();
                row.operator(ReorderLandmarks.bl_idname);
                
                row = box.row();
                row.operator(SnapLandmarksToVertex.bl_idname);
                
                row = box.row();
                row.operator(AutoLinkLandmarksByID.bl_idname);
                
                row = box.row();
                row.operator(LiveLandmarksCreator.bl_idname);
                
            if(context.active_object.is_landmarked_mesh):            
                box = layout.box();
                box.label('Mesh with a pair operations');              
                
                row = box.row();
                row.prop(context.active_object, 'linked_landmark_color');
                row = box.row();
                row.prop(context.active_object, 'unlinked_landmark_color');
                
                row = box.row();
                row.operator(LandmarkStatus.bl_idname);
                
                row = box.row();
                row.operator(TransferLandmarkNames.bl_idname);
                
                row = box.row();
                row.operator(LandmarksPairFinder.bl_idname);
                
                box = layout.box();
                box.label('Experimnetal: Signatures Matching');
                
                row = box.row();
                row.operator(SignaturesMatching.bl_idname);
                            
            if(context.active_object.is_visual_landmark):
                box = layout.box();
                box.label('Global Landmark settings');
                
                row = box.row();
                row.operator(RemoveLandmarks.bl_idname);
                
                belongs_to = getMeshForBlenderMarker(context.active_object);       
                if(belongs_to.is_landmarked_mesh):
                    box = layout.box();
                    box.label('Mesh Paired Landmark settings');
                    
                    row = box.row();
                    row.prop(context.active_object, 'edit_landmark_name');
                    
                    row = box.row();
                    row.operator(LinkLandmarks.bl_idname);
                    
                    row = box.row();
                    row.operator(UnLinkLandmarks.bl_idname);                
                

class PanelHelpAddMappingFromFile(bpy.types.Operator):
    bl_idname = "ashok.panel_help_add_mapping_from_file";
    bl_label = "Add Mapping";
    bl_description = "Operator to help add a mapping entry that can be filled with details and constructed. Select two meshes in scene. Active Object gets the mapping from non-active object in selection";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    bl_options = {'REGISTER', 'UNDO'};
    m1 = bpy.props.StringProperty(name='m1', default="---");
    m2 = bpy.props.StringProperty(name='m2', default="---");
    
    def execute(self, context):
        meshes = [m for m in context.selected_objects if m.type == 'MESH'];
        argument_meshes = [];
        try:
            argument_meshes = [context.scene.objects[self.m1], context.scene.objects[self.m2] ];
        except KeyError:
            pass;
                
        if(len(meshes) < 2 and len(argument_meshes) <2 ):
            self.report({'WARNING'}, "Select two meshes from the scene");
            return {'FINISHED'};
        if(len(argument_meshes) > 1):
            meshes = argument_meshes;
            
        m1, m2 = meshes[0], meshes[1];
        
        if(m1 == context.active_object):
            s = m1;
            t = m2;
        else:
            s = m2;
            t = m1;
        
        map = s.surfacemappings.add();
        map.map_to_mesh = t.name;
        map.apply_on_duplicate = True;
        map.mapping_name = 'Mapping';        
        return {'FINISHED'};   

class PanelHelpRemoveMappingFromFile(bpy.types.Operator):
    bl_idname = "ashok.panel_help_remove_mapping_from_file";
    bl_label = "Remove Last Mapping";
    bl_description = "Operator to help remove a mapping entry for the selected mesh";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    bl_options = {'REGISTER', 'UNDO'};
    currentobject = bpy.props.StringProperty(name='currentobject', default="---");
    
    def execute(self, context):
        mesh = None;
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(not mesh):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;
        
        if(not mesh.type in {"MESH"}):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;        
        
        mesh.surfacemappings.remove( len(mesh.surfacemappings) - 1 );        
        return {'FINISHED'};
    
class PanelHelpClearMappingFromFile(bpy.types.Operator):
    bl_idname = "ashok.panel_help_clear_mapping_from_file";
    bl_label = "Clear All Mappings";
    bl_description = "Operator to help clear all mapping entries for the selected mesh";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";    
    bl_context = "objectmode";
    bl_options = {'REGISTER', 'UNDO'};
    currentobject = bpy.props.StringProperty(name='currentobject', default="---");
    
    def execute(self, context):
        mesh = None;
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(not mesh):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;
        
        if(not mesh.type in {"MESH"}):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;        
        
        mesh.surfacemappings.clear();
        return {'FINISHED'};

class PanelHelpConstructMappingFromFile(bpy.types.Operator):
    bl_idname = "ashok.panel_help_construct_mapping_from_file";
    bl_label = "Construct Mapping";
    bl_description = "Operator to construct a mapping given the index and the mesh name";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    bl_options = {'REGISTER', 'UNDO'};
    current_object = bpy.props.StringProperty(name='currentobject', default="---");
    mapping_index = bpy.props.IntProperty(name='Mapping Index', default=0);
    
    def execute(self, context):
        mesh = None;
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(not mesh):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;
        
        if(not mesh.type in {"MESH"}):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;        
        
        print(self.mapping_index, self.current_object, mesh.surfacemappings[self.mapping_index].mapping_name);
        
#         try:
        mapping = mesh.surfacemappings[self.mapping_index];
        mapping.constructMapping();
#         except IndexError:
#             self.report({'WARNING'}, "No valid Mapping");
                
        return {'FINISHED'};
class PanelHelpExportMappingFromFile(bpy.types.Operator):
    bl_idname = "ashok.panel_help_export_mapping_from_file";
    bl_label = "Export  Mapping";
    bl_description = "Operator to export a mapping (format: v1, v2, v3, u, v, w) given the index and the mesh name";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    bl_options = {'REGISTER', 'UNDO'};
    current_object = bpy.props.StringProperty(name='currentobject', default="---");
    mapping_index = bpy.props.IntProperty(name='Mapping Index', default=0);
    
    def execute(self, context):
        mesh = None;
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(not mesh):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;
        
        if(not mesh.type in {"MESH"}):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;        
        
        print(self.mapping_index, self.current_object);
        
        try:
            mapping = mesh.surfacemappings[self.mapping_index];
            mapping.exportMapping();
        except IndexError:
            self.report({'WARNING'}, "No valid Mapping");
                
        return {'FINISHED'};
    
class PanelHelpDeformationMappingFromFile(bpy.types.Operator):
    bl_idname = "ashok.panel_help_deformation_mapping_from_file";
    bl_label = "Deform  Mapping";
    bl_description = "Operator to deform to the shape using a mapping given the index and the mesh name";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    bl_options = {'REGISTER', 'UNDO'};
    current_object = bpy.props.StringProperty(name='currentobject', default="---");
    mapping_index = bpy.props.IntProperty(name='Mapping Index', default=0);
    
    def execute(self, context):
        mesh = None;
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(not mesh):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;
        
        if(not mesh.type in {"MESH"}):
            self.report({'WARNING'}, "Select a mesh from the scene");
            return;        
        
        print(self.mapping_index, self.current_object);
        
        try:
            mapping = mesh.surfacemappings[self.mapping_index];      
            mapping.deformWithMapping();
        except IndexError:
            self.report({'WARNING'}, "No valid Mapping");
                
        return {'FINISHED'};       

class MultiMappings_ui_entries(bpy.types.UIList):    
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        box = layout.box();
        row = box.row(align=True);
        
        col = row.box();        
        rowS = col.row();
        rowS.prop(item, 'mapping_name', text='Name');                                
        rowS = col.row();
        rowS.label('Mapped To: %s'%(item.map_to_mesh));
                
        col = row.box();
        rowS = col.row();
        rowS.prop(item, 'file_path', text='');
        rowS = col.row();
        op = rowS.operator(PanelHelpConstructMappingFromFile.bl_idname, text='Import');
        op.mapping_index = index;
        op.current_object = data.name;
        
        col = row.box();
        rowS = col.row();
        rowS.prop(item, 'export_file_path', text='');
        if(item.mapping_is_valid):
            rowS = col.row();
            op = rowS.operator(PanelHelpExportMappingFromFile.bl_idname, text='Export');
            op.mapping_index = index;
            op.current_object = data.name;        
        
        
        col = row.box();
        if(item.mapping_is_valid):
            rowS = col.row();
            op = rowS.operator(PanelHelpDeformationMappingFromFile.bl_idname, text='Deform');
            op.mapping_index = index;
            op.current_object = data.name;        
        rowS = col.row();
        rowS.prop(item, 'apply_on_duplicate');
            
#         print(data, item, active_data, active_propname, index);

class MappingFromFilePanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_mappingfromfilepanel"
    bl_label = "Surface mapping";
    bl_space_type = "VIEW_3D";
    bl_region_type = "TOOLS";
#     bl_space_type = 'PROPERTIES'
#     bl_region_type = "WINDOW";
    bl_category = "Generic Landmarks"
    bl_description = "Panel to create loading of mappings between two surfaces";
    
    def draw(self, context):
        if(context.active_object):
            layout = self.layout;
            box = layout.box();
            box.label('Mapping Entries From File');
            
            row = box.row();            
            row.template_list("MultiMappings_ui_entries", "", context.active_object, "surfacemappings", context.active_object,"multimappings_entries_count");
            
            col = row.column(align=True);
            col.operator(PanelHelpAddMappingFromFile.bl_idname, text='', icon='ZOOMIN');
            col.operator(PanelHelpRemoveMappingFromFile.bl_idname, text='', icon='ZOOMOUT');
            col.operator(PanelHelpClearMappingFromFile.bl_idname, text='', icon='PANEL_CLOSE');


         
def register():
    bpy.utils.register_class(LandmarksPanel);
    bpy.utils.register_class(SpectralGeneralPanel);
    bpy.utils.register_class(HKSPanel);
    bpy.utils.register_class(WKSPanel);
    bpy.utils.register_class(GISIFPanel);
    
    bpy.utils.register_class(PanelHelpAddMappingFromFile);
    bpy.utils.register_class(PanelHelpRemoveMappingFromFile);
    bpy.utils.register_class(PanelHelpClearMappingFromFile);
    bpy.utils.register_class(PanelHelpConstructMappingFromFile);
    bpy.utils.register_class(PanelHelpDeformationMappingFromFile);
    bpy.utils.register_class(PanelHelpExportMappingFromFile);
    
    
    bpy.utils.register_class(MultiMappings_ui_entries);
    bpy.utils.register_class(MappingFromFilePanel);
    
    

def unregister():
    bpy.utils.unregister_class(LandmarksPanel);
    bpy.utils.unregister_class(SpectralGeneralPanel);
    bpy.utils.unregister_class(HKSPanel);
    bpy.utils.unregister_class(WKSPanel);
    bpy.utils.unregister_class(GISIFPanel);
    
    bpy.utils.unregister_class(PanelHelpAddMappingFromFile);
    bpy.utils.unregister_class(PanelHelpRemoveMappingFromFile);
    bpy.utils.unregister_class(PanelHelpClearMappingFromFile);
    bpy.utils.unregister_class(PanelHelpConstructMappingFromFile);
    bpy.utils.unregister_class(PanelHelpDeformationMappingFromFile);
    bpy.utils.unregister_class(PanelHelpExportMappingFromFile);
    
    bpy.utils.unregister_class(MultiMappings_ui_entries);
    bpy.utils.unregister_class(MappingFromFilePanel);
    
    
    
    