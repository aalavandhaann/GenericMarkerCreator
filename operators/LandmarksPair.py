import bpy;
from mathutils import Vector, Matrix;

class AssignMeshPair(bpy.types.Operator):
    bl_idname = "genericlandmarks.assignmeshpair";
    bl_label = "Assign Pair";
    bl_description="Map two meshes are a pair to create linked landmarks between them";
    bl_options = {'REGISTER', 'UNDO'};
        
    def assignPairsMAndN(self, context, M, N):      
        M.is_landmarked_mesh = N.is_landmarked_mesh = True;  
        M.mapped_mesh = N.name;
        N.mapped_mesh = M.name;

        bpy.ops.object.select_all(action="DESELECT");
        
        M.select = True;
        context.scene.objects.active = M;
        bpy.ops.object.transform_apply(rotation=True, scale=True);
        
        bpy.ops.object.select_all(action="DESELECT");
        N.select = True;
        context.scene.objects.active = N;
        bpy.ops.object.transform_apply(rotation=True, scale=True);
                
        bpy.ops.object.select_all(action="DESELECT");      
        
        message = M.name+" is the source and "+N.name+" is the target. \nYou can now start creating the landmarks";        
        bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='INFO',message=message,messagelinesize=60);
        
        return M, N;
        
    def useSelectionsForMN(self, context, selections):
        M = None;
        N = None;        
        
        if(selections[0].location.x < selections[1].location.x):
            M, N = self.assignPairsMAndN(context, selections[0], selections[1]);
        else:
            M, N = self.assignPairsMAndN(context, selections[1], selections[0]);
    
        return M, N;
    
    def execute(self, context):        
        selections = [o for o in context.selected_objects];
        
        if(len(selections) == 2):            
            self.useSelectionsForMN(context, selections);
        else:
            message = "Please select a combination for surfaces M and N";
            bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
        
        return {'FINISHED'};