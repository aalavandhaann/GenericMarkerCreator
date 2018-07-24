__author__="ashok"
__date__ ="$Mar 23, 2015 8:16:11 PM$"


import gc
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import bpy, time;

from bpy.props import StringProperty;
from bpy.props import FloatVectorProperty;
from mathutils import Vector;
from mathutils.bvhtree import BVHTree;

from GenericMarkerCreator import constants;
from GenericMarkerCreator.propertiesregister import changeMarkerColor, changeUnlinkedMarkerColor;
from GenericMarkerCreator.misc.interactiveutilities import ScreenPoint3D;
from GenericMarkerCreator.misc.staticutilities import detectMN, applyMarkerColor, addConstraint, getConstraintsKD, deleteObjectWithMarkers, reorderConstraints;
from GenericMarkerCreator.misc.staticutilities import getMarkersForMirrorX, getGenericLandmark, getMeshForBlenderMarker, getBlenderMarker;
from GenericMarkerCreator.misc.meshmathutils import getBarycentricCoordinateFromPolygonFace, getBarycentricCoordinate, getCartesianFromBarycentre, getGeneralCartesianFromBarycentre, getTriangleArea;

#The operators for creating landmarks
class CreateLandmarks(bpy.types.Operator):
    bl_idname = "genericlandmarks.createlandmarks";
    bl_label = "Creating landmarks on a mesh";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");
    updatepositions = bpy.props.BoolProperty(name="Update Landmarks Flag", default = False);
    hidemarkers = bpy.props.BoolProperty(name="Hide Landmarks Flag", default = False);
    markersource = bpy.props.StringProperty(name="Mesh to use as Landmark", default = "");
    colorizemarkers = bpy.props.BoolProperty(name="Recolorize Landmarks", default = False);
        
    def hideConstraintsVisually(self, context, mesh):        
        for marker in mesh.generic_landmarks:
            try:
                bmarker = bpy.data.objects[mesh.name + "_marker_"+str(marker.id)];
                bmarker.hide = mesh.hide_landmarks;
                bmarker.hide_render = mesh.hide_landmarks;
            except KeyError:
                pass;
            
    def updateConstraints(self, context, mesh):        
        if(context.mode != "OBJECT"):
            bpy.ops.object.mode_set(mode = 'OBJECT', toggle = False);
        
        for marker in mesh.generic_landmarks:
            vertex1 = mesh.data.vertices[marker.v_indices[0]].co;
            vertex2 = mesh.data.vertices[marker.v_indices[1]].co;
            vertex3 = mesh.data.vertices[marker.v_indices[2]].co;
            
            location = getGeneralCartesianFromBarycentre(marker.v_ratios, [vertex1, vertex2, vertex3]);
            marker.location = location;
            bmarker = None;
            
            try:
                bmarker = bpy.data.objects[mesh.name + "_marker_"+str(marker.id)];            
                bmarker.parent = None;
                bmarker.location = location;        
            except KeyError:
                pass;
            
            if(context.mode != "OBJECT"):
                bpy.ops.object.mode_set(mode = 'OBJECT', toggle = False);
            
            bpy.ops.object.select_all(action='DESELECT') #deselect all object
            if(bmarker):
                bmarker.parent = mesh;
        
    def createConstraintsVisual(self, context, mesh):
        useprimitive = False;
        referencemesh = None;
        if(self.markersource in "~PRIMITIVE~"):
            useprimitive = True;
        else:
            referencemesh = bpy.data.objects[self.markersource];
        
        material = changeMarkerColor(mesh);
        unlinkedmaterial = changeUnlinkedMarkerColor(mesh);
        
        if(not context.mode == "OBJECT"):
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False);
        
        deleteObjectWithMarkers(context, mesh);
        
        temp = -1;
        
        for index, marker in enumerate(mesh.generic_landmarks):            
            markername = mesh.name + "_marker_"+str(marker.id);
            try:
                markerobj = context.data.objects[markername];
                createmarker = False;
            except:
                createmarker = True;
            
            vertex1 = mesh.data.vertices[marker.v_indices[0]].co;
            vertex2 = mesh.data.vertices[marker.v_indices[1]].co;
            vertex3 = mesh.data.vertices[marker.v_indices[2]].co;
            
            location = getGeneralCartesianFromBarycentre(marker.v_ratios, [vertex1, vertex2, vertex3]);
            marker.location = location;
            
            if(useprimitive):
                bpy.ops.mesh.primitive_cube_add(location=location, radius = 0.15);
            else:
                mk_mesh = bpy.data.meshes.new(mesh.name + "_marker_"+str(marker.id));
                # Create new object associated with the mesh
                ob_new = bpy.data.objects.new(mesh.name + "_marker_"+str(marker.id), mk_mesh);
                ob_new.data = referencemesh.data.copy();
                ob_new.scale = referencemesh.scale;
                # Link new object to the given scene and select it
                context.scene.objects.link(ob_new);
                bpy.ops.object.select_all(action='DESELECT') #deselect all object
                ob_new.select = True;
                ob_new.location = location;
                bpy.context.scene.objects.active = ob_new;
                
#             markerobj = context.object;
            markerobj = context.active_object;
            markerobj.is_visual_landmark = True;
            markerobj.landmark_id = marker.id;
            markerobj.name = mesh.name + "_marker_"+str(marker.id);
            markerobj.belongs_to = mesh.name;
            
            markerobj.data.materials.clear();
            
            if(marker.is_linked):
                markerobj.data.materials.append(material);
            else:
                markerobj.data.materials.append(unlinkedmaterial);
                
            bpy.ops.object.select_all(action='DESELECT') #deselect all object            
            markerobj.parent = mesh;
            
            if(marker.id > temp):
                temp = marker.id;
            
        for area in context.screen.areas: # iterate through areas in current screen
            if area.type == 'VIEW_3D':
                for space in area.spaces: # iterate through spaces in current VIEW_3D area
                    if space.type == 'VIEW_3D': # check if space is a 3D view
                        space.viewport_shade = 'SOLID' # set the viewport shading to rendered
        
        context.scene.objects.active = mesh;
        
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            # unselect all  
            for item in bpy.context.selectable_objects:  
                item.select = False;  
            mesh.select = True;
            bpy.context.scene.objects.active = mesh;
            
            
            if(self.updatepositions):
                self.updateConstraints(context, mesh);
            elif(self.hidemarkers):
                self.hideConstraintsVisually(context, mesh);
            else:
                self.createConstraintsVisual(context, mesh);
                
        return{'FINISHED'};
    
# Button
class ReorderLandmarks(bpy.types.Operator):
    bl_idname = "genericlandmarks.reorderlandmarks";
    bl_label = "Reorder Landmarks";
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
            M, N = detectMN(mesh);
            if(not M and not N):
                M = mesh;
                reorderConstraints(context, mesh, None);
            else:    
                reorderConstraints(context, M, N);
                
            tempmarkersource = context.scene.landmarks_use_selection;
            if(tempmarkersource.strip() is ""):
                tempmarkersource = "~PRIMITIVE~";
            
            bpy.ops.genericlandmarks.createlandmarks('EXEC_DEFAULT',currentobject=M.name, markersource=tempmarkersource);
            if(N):
                bpy.ops.genericlandmarks.createlandmarks('EXEC_DEFAULT',currentobject=N.name, markersource=tempmarkersource);
        
        return {'FINISHED'};
    

class ChangeLandmarks(bpy.types.Operator):
    bl_idname = "genericlandmarks.changelandmarks";
    bl_label = "Change Landmarks";
    bl_description = "Use a different mesh to show landmarks";
    bl_options = {'REGISTER', 'UNDO'};
    currentobject = bpy.props.StringProperty(name="A mesh with markers", default = "--");
    
    def execute(self, context):        
        try:            
            meshobject = bpy.data.objects[self.currentobject];
        except:
            meshobject = context.active_object;
            
        if(meshobject is not None):            
            M, N = detectMN(meshobject);
            if(not M and not N):
                M = meshobject;
                
            tempmarkersource = context.scene.landmarks_use_selection;
            if(tempmarkersource.strip() is ""):
                tempmarkersource = "~PRIMITIVE~";

            if(M):
                bpy.ops.genericlandmarks.createlandmarks('EXEC_DEFAULT',currentobject=M.name, markersource=tempmarkersource);
            if(N):
                bpy.ops.genericlandmarks.createlandmarks('EXEC_DEFAULT',currentobject=N.name, markersource=tempmarkersource);
            
        return {'FINISHED'};   

class UnLinkLandmarks(bpy.types.Operator):
    bl_idname = "genericlandmarks.unlinklandmarks";
    bl_label = "Unlink Landmarks";
    bl_description = "Operator to unlink landmarks between meshes";
    bl_options = {'REGISTER', 'UNDO'};
    
    def unLinkMarkers(self, context, bmarkerobjects):        
        if(len(bmarkerobjects) > 0):
            source, target = detectMN(bmarkerobjects[0]);
            if(not source and not target):
                message = "Linking or Unlinking of landmarks is a concept applied only to mesh pairs with a bijective landmarks correspondence";
                bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
                return;
            for m in bmarkerobjects:
                info = m.name.split("_marker_");
                belongsto = bpy.data.objects[info[0]];
                hmarker = getGenericLandmark(belongsto, m);
                
                if(hmarker.is_linked):
                    hmarker.is_linked = False;
                    hmarker.linked_id = -1;
                else:
                    if(len(bmarkerobjects) < 2):
                        message = "You cannot unlink a marker that is not linked to any markers";
                        bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
            
            
            for m in source.generic_landmarks:
                if(m.is_linked):
                    tm = [tm for tm in target.generic_landmarks if tm.id == m.linked_id][0];
                    if(not tm.is_linked):
                        m.is_linked = False;
                        m.linked_id = -1;
                        changeUnlinkedMarkerColor(source, getBlenderMarker(source, m));
                        changeUnlinkedMarkerColor(target, getBlenderMarker(target, tm));
                else:
                    changeUnlinkedMarkerColor(source, getBlenderMarker(source, m));
                    
                    
            for m in target.generic_landmarks:
                if(m.is_linked):
                    sm = [sm for sm in source.generic_landmarks if sm.id == m.linked_id][0];
                    if(not sm.is_linked):
                        m.is_linked = False;
                        m.linked_id = -1;
                        changeUnlinkedMarkerColor(target, getBlenderMarker(target, m));
                        changeUnlinkedMarkerColor(source, getBlenderMarker(source, sm));                
                else:
                    changeUnlinkedMarkerColor(target, getBlenderMarker(target, m));
                
                
    def execute(self, context):
        bmarkerobjects = [o for o in context.selected_objects if o.is_visual_landmark];
        self.unLinkMarkers(context, bmarkerobjects);
        if(context.scene.use_mirrormode_x):        
            reflections = getMarkersForMirrorX(context, bmarkerobjects);
            bpy.ops.object.select_all(action="DESELECT");
            for reflected in reflections:
                reflected.select = True;        
            self.unLinkMarkers(context, reflections);
        
        return {'FINISHED'};

class LinkLandmarks(bpy.types.Operator):
    bl_idname = "genericlandmarks.linklandmarks";
    bl_label = "Link Landmarks";
    bl_description = "Operator to link landmarks";
    bl_options = {'REGISTER', 'UNDO'};
    silence = bpy.props.BoolProperty(name='silence',description="Just add markers without message box", default=False);
    
    def linkMarkers(self, context, bmarkerobjects):
        if(len(bmarkerobjects) > 1):
            adam = getMeshForBlenderMarker(bmarkerobjects[0]);
            eve = getMeshForBlenderMarker(bmarkerobjects[1]);
            
            if(adam.name != eve.name):                
                ahmarker = getGenericLandmark(adam, bmarkerobjects[0]);
                ehmarker = getGenericLandmark(eve, bmarkerobjects[1]);
                
                if(ahmarker.is_linked or ehmarker.is_linked):
                    message = "You cannot link markers that are already linked";
                    bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
                    return;
                
                ahmarker.is_linked = True;
                ehmarker.is_linked = True;
                ahmarker.linked_id = ehmarker.id;
                ehmarker.linked_id = ahmarker.id;
                    
                changeMarkerColor(adam, bmarkerobjects[0]);
                changeMarkerColor(eve, bmarkerobjects[1]);
                
                
                slinkedmarkers = [m for m in adam.generic_landmarks if m.is_linked];
                tlinkedmarkers = [m for m in eve.generic_landmarks if m.is_linked];

                smarkersnum = len(adam.generic_landmarks);
                tmarkersnum = len(eve.generic_landmarks);

                slinkednum = len(slinkedmarkers);
                tlinkednum = len(tlinkedmarkers);

                sunlinkednum = smarkersnum - slinkednum;
                tunlinkednum = tmarkersnum - tlinkednum;
                
                print(sunlinkednum, tunlinkednum, slinkednum, tlinkednum, smarkersnum, tmarkersnum);
                bpy.ops.object.select_all(action="DESELECT");
                if(not self.silence):
                    if((sunlinkednum + tunlinkednum) == 0):
                        message = "All the landmarks in Source and Target are Linked \n";
                    else:
                        message = "Landmark "+str(ahmarker.id)+" in "+adam.name;
                        message += "\n is now linked to Landmark"+str(ehmarker.id)+" in "+eve.name;
                
                    bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='INFO',message=message,messagelinesize=60);
            
            else:
                if(not self.silence):
                    message = "You cannot link markers in the same mesh";
                    bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
    
    def execute(self, context):
        bmarkerobjects = [o for o in context.selected_objects if o.is_visual_landmark];
        self.linkMarkers(context, bmarkerobjects);
        
        if(context.scene.use_mirrormode_x):        
            reflections = getMarkersForMirrorX(context, bmarkerobjects);
            bpy.ops.object.select_all(action="DESELECT");

            for reflected in reflections:
                reflected.select = True;                
            self.linkMarkers(context, reflections);
            
        return {'FINISHED'};    


class RemoveLandmarks(bpy.types.Operator):
    bl_idname = "genericlandmarks.removelandmarks";
    bl_label = "Remove Landmarks";
    bl_description = "Operator to remove landmarks in mesh";
    bl_options = {'REGISTER', 'UNDO'};
    
    def removeMarkers(self, context, bmarkerobjects):
        bmarkerobjects = [o for o in context.selected_objects if o.is_visual_landmark];
        meshes = [];
        
        bpy.ops.object.select_all(action="DESELECT");
        
        #Check if there are any markers in selection to be removed
        if(len(bmarkerobjects) > 0):
            #iterate through the selection to do the removal process
            for bmarker in bmarkerobjects:
                #For the current marker in iteration, find the mesh it belongs to
                info = bmarker.name.split("_marker_");
                #Access the blender mesh of the marker
                belongstoobject = getMeshForBlenderMarker(bmarker);
                #Access the original id of the marker
                originalid = int(info[1]);
                #Index of the landmark object for the current blender marker in iteration
                hmindex = [index for index, hm in enumerate(belongstoobject.generic_landmarks) if hm.id == originalid];
                #The below check will work if the blender marker indeed has a landmark object pointer
                if(len(hmindex)):
                    hmindex = hmindex[0];
                    #Select the visual blender marker
                    bmarker.select = True;           
                    #Remove the pointer landmark based on the found index
                    belongstoobject.generic_landmarks.remove(hmindex);                   
                    #since there are many markers in selection, keep a track 
                    #of the meshes (source or target) for which the markers were removed
                    try:
                        #The index method fails if the mesh was not in the list
                        meshes.index(belongstoobject.name);
                    except:
                        #The idea is to unlink markers for the markers removed
                        #in this iteration on the pair mesh. For example,
                        #removing a marker in source should ensure that the 
                        #linked marker in the target should be unlinked and vice-versa
                        meshes.append(belongstoobject.name);
        
            #Blender object based delete operator to remove the marker objects
            #that would have been selected in the above iteration
            bpy.ops.object.delete();

            #Access the source and target meshes
            source, target = detectMN(bpy.data.objects[meshes[0]]);
            
            if(not source and not target):
                return;
            
            #iterate through the source landmarks
            for m in source.generic_landmarks:
                #Check if it the marker in source is a linked one
                if(m.is_linked):
                    #If linked, try accessing the target's landmark with the
                    #current iterated marker in source. If this operation fails
                    #It means the paired marker was removed in the previous iteration
                    #So, its time to unlink this source marker
                    try:
                        tm = [tm for tm in target.generic_landmarks if tm.id == m.linked_id][0];
                    except IndexError:
                        m.linked_id = -1;
                        m.is_linked = False;
                        changeUnlinkedMarkerColor(source, getBlenderMarker(source, m));
            
            #iterate through the target landmarks
            for m in target.generic_landmarks:
                #Check if it the marker in target is a linked one
                if(m.is_linked):
                    #If linked, try accessing the source's landmark with the
                    #current iterated marker in target. If this operation fails
                    #It means the paired marker was removed in the previous iteration
                    #So, its time to unlink this target marker
                    try:
                        sm = [sm for sm in source.generic_landmarks if sm.id == m.linked_id][0];
                    except IndexError:
                        m.linked_id = -1;
                        m.is_linked = False;
                        changeUnlinkedMarkerColor(target, getBlenderMarker(target, m));
            
            if(len(source.generic_landmarks) == 0):
                source.total_landmarks = len(source.generic_landmarks);
            
            if(len(target.generic_landmarks) == 0):
                target.total_landmarks = len(target.generic_landmarks);
    
    def execute(self, context):
        bmarkerobjects = [o for o in context.selected_objects if o.is_visual_landmark];
        if(context.scene.use_mirrormode_x):        
            reflections = getMarkersForMirrorX(context, bmarkerobjects);
            
            for reflected in reflections:
                reflected.select = True;
                
            bmarkerobjects += reflections;
            
        self.removeMarkers(context, bmarkerobjects);        
        return {'FINISHED'};

class LandmarkStatus(bpy.types.Operator):
    bl_idname = "genericlandmarks.landmarkstatus";
    bl_label = "Landmarks Status";
    bl_description = "An operators to know about the status of landmarks in M and N. This info will provide details of how many constraints in M and N are linked and unlinked.";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    
    def showStatus(self, context, source, target):
        slinkedmarkers = [m for m in source.generic_landmarks if m.is_linked];
        tlinkedmarkers = [m for m in target.generic_landmarks if m.is_linked];
        
        smarkersnum = len(source.generic_landmarks);
        tmarkersnum = len(target.generic_landmarks);
        
        slinkednum = len(slinkedmarkers);
        tlinkednum = len(tlinkedmarkers);
        
        sunlinkednum = smarkersnum - slinkednum;
        tunlinkednum = tmarkersnum - tlinkednum;
        
        message = "There are :"+str(smarkersnum)+" Landmarks in M\n";
        message += "There are :"+str(tmarkersnum)+" Landmarks in N\n";
        message += str(slinkednum)+" Linked Landmarks in M \n";
        message += str(tlinkednum)+" Linked Landmarks in N \n";
        message += str(sunlinkednum)+" UnLinked Landmarks in M \n";
        message += str(tunlinkednum)+" UnLinked Landmarks in N \n";
        
        if(sunlinkednum == tunlinkednum == 0):
            message += "All the Landmarks in M and N are linked \n";
        else:
            message += "There are Landmarks in M and N yet to be linked \n";
            
        bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='INFO',message=message,messagelinesize=60);
        
    
    def execute(self, context):
        
        if(context.active_object):
            activeobject = context.active_object;
            M, N = detectMN(activeobject);
            if(M and N):
                self.showStatus(context, M, N);
        return {'FINISHED'}



class LandmarksPairFinder(bpy.types.Operator):
    """Store the object names in the order they are selected, """ \
    """use RETURN key to confirm selection, ESCAPE key to cancel"""
    bl_idname = "genericlandmarks.landmarkspairfinder";
    bl_label = "Find Linked landmarks";
    bl_options = {'UNDO'};
    bl_description="Find the linked landmark to a landmark on the mesh. Press this button and keep selecting landmarks to find its linked landmarks on the other mesh. Press ESCAPE key to finish the modal operation.";
    num_selected = 0;
    
    @classmethod
    def poll(self, context):
        return bpy.context.mode == 'OBJECT';    
    
    def getPairMarker(self, context, sourcemarker, source, target):
        markeridslist = [marker.id for marker in target.generic_landmarks];
        if(sourcemarker.is_linked):
            targetmarker = target.generic_landmarks[markeridslist.index(sourcemarker.linked_id)];        
            return getBlenderMarker(target, targetmarker);
        
        return None;
        
        
    def update(self, context):
        # Get the currently selected objects
        sel = context.selected_objects;
        
        if(len(sel) < 1):
            return;
        
        marker = sel[0];
        
        if(marker.is_visual_landmark):
            M, N = detectMN(marker);
        else:
            return;
        
        belongsto = getMeshForBlenderMarker(marker);
        
        bpy.ops.object.select_all(action='DESELECT') #deselect all object
        
        hmarker = getGenericLandmark(belongsto, marker);
        
        if(belongsto == M):
            sourcemarker = marker;
            targetmarker = self.getPairMarker(context, hmarker, M, N);
            
        elif(belongsto == N):
            targetmarker = marker;
            sourcemarker = self.getPairMarker(context, hmarker, N, M);
                    
        if(sourcemarker is not None and targetmarker is not None):
            sourcemarker.select = True;
            targetmarker.select = True;      
        

    def modal(self, context, event):
        if event.type == 'RET':
            # If return is pressed, finish the operators
            return {'FINISHED'}
        elif event.type == 'ESC':
            # If escape is pressed, cancel the operators
            return {'CANCELLED'}

        # Update selection if we need to
        self.update(context)
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        self.update(context)

        context.window_manager.modal_handler_add(self);
        return {'RUNNING_MODAL'}


class TransferLandmarkNames(bpy.types.Operator):
    """Store the object names in the order they are selected, """ \
    """use RETURN key to confirm selection, ESCAPE key to cancel"""
    bl_idname = "genericlandmarks.transferlandmarknames";
    bl_label = "Transfer landmark names";
    bl_options = {'UNDO'};
    bl_description="Transfer names of landmarks to the linked ones";
    num_selected = 0;
    
    def execute(self, context):
        if(context.active_object):
            activeobject = context.active_object;
            if(activeobject.is_landmarked_mesh):
                M, N = detectMN(activeobject);
                if(not M and not N):
                    message = "Transfer of landmark names is a concept applied only to mesh pairs with a bijective landmarks correspondence";
                    bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
                    return {'FINISHED'};
            
                from_mesh, to_mesh = None, None;
                
                if(activeobject == M):
                    from_mesh, to_mesh = M, N;
                if(activeobject == N):
                    from_mesh, to_mesh = N, M;
                
                to_landmarks_ids = [m.linked_id for m in to_mesh.generic_landmarks];
                for from_m in from_mesh.generic_landmarks:
                    if(from_m.is_linked):
                        try:
                            linked_m = to_mesh.generic_landmarks[to_landmarks_ids.index(from_m.id)];
                            linked_m.landmark_name = from_m.landmark_name;
                        except IndexError:
                            pass;
        return {'FINISHED'};







