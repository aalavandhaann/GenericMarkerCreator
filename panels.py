import bpy;

from GenericMarkerCreator.misc.staticutilities import getMeshForBlenderMarker;
from GenericMarkerCreator.operators.LandmarksPair import AssignMeshPair;
from GenericMarkerCreator.operators.LiveOperators import LiveLandmarksCreator;
from GenericMarkerCreator.operators.LandmarksCreator import CreateLandmarks, ReorderLandmarks, \
ChangeLandmarks, UnLinkLandmarks, LinkLandmarks, RemoveLandmarks, LandmarkStatus, LandmarksPairFinder, TransferLandmarkNames, AutoLinkLandmarksByID;



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
def register():
    bpy.utils.register_class(LandmarksPanel);
    pass;

def unregister():
    bpy.utils.unregister_class(LandmarksPanel);
    pass;