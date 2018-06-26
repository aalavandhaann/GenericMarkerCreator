import bpy, mathutils;
from mathutils import Vector;
from GenericMarkerCreator.misc.meshmathutils import getKDTree;

def getConstraintsKD(context, mesh):
    coords = [];
    for m in mesh.generic_landmarks:
        co = Vector((m.location[0], m.location[1], m.location[2]));
        coords.append(co);
    kd = getKDTree(context, mesh, "CUSTOM", coords);
    return kd, coords;

def applyMarkerColor(marker):            
    try:
        material = bpy.data.materials[marker.name+'_MouseMarkerMaterial'];
    except:
        material = bpy.data.materials.new(marker.name+'_MouseMarkerMaterial');
    
    material.diffuse_color = (0.0, 0.0, 1.0);
    material.alpha = 1;
    material.specular_color = (0.0, 0.0, 1.0);
    
    marker.data.materials.clear();
    marker.data.materials.append(material);

def detectMN(mesh):
    M = None;
    N = None;
    
    if(mesh.is_landmarked_mesh):
        M = mesh;
        N = bpy.data.objects[mesh.mapped_mesh];
        
    elif(mesh.is_visual_landmark):
        belongsto = bpy.data.objects[mesh.name.split("_marker_")[0]];
        return detectMN(belongsto);
    
    return M, N;

def addConstraint(context, mesh, bary_ratios, bary_indices, co, should_reorder=False):
    m = mesh.generic_landmarks.add();
    m.id = mesh.total_landmarks;
    m.linked_id = -1;
    m.is_linked = False;
    m.v_indices = bary_indices;
    m.v_ratios = bary_ratios;
    m.location = [co.x, co.y, co.z];
    m.landmark_name = 'Original Id: %s'%(m.id);
    
    mesh.total_landmarks = len(mesh.generic_landmarks);
    
    if(should_reorder):
        bpy.ops.genericlandmarks.reorderlandmarks('EXEC_DEFAULT',currentobject=mesh.name);
    else:
        tempmarkersource = context.scene.landmarks_use_selection;
        if(tempmarkersource.strip() is ""):
            tempmarkersource = "~PRIMITIVE~";
        bpy.ops.genericlandmarks.createlandmarks('EXEC_DEFAULT',currentobject=mesh.name, markersource=tempmarkersource);
        
    context.scene.objects.active = mesh;
    return m;

def reorderConstraints(context, M, N):
    if(M and N):
        sourcemarkers = [m for m in M.generic_landmarks];
        targetmarkers = [m for m in N.generic_landmarks];
        
        targetmarkerids = [m.id for m in N.generic_landmarks];
        markercouples = [];
        
        nonlinkedsourcemarkers = [m for m in M.generic_landmarks if not m.is_linked];
        nonlinkedtargetmarkers = [m for m in N.generic_landmarks if not m.is_linked];
        
        index = 0;
        
        for sm in sourcemarkers:
            if(sm.is_linked):
                tm = targetmarkers[targetmarkerids.index(sm.linked_id)]; 
                markercouples.append((sm, tm));
        
        for index, (sm, tm) in enumerate(markercouples):
            sm.id = index;
            tm.id = index;
            sm.linked_id = tm.id;
            tm.linked_id = sm.id;
            
        markerindex = index + 1;
        
        for m in nonlinkedsourcemarkers:
            m.id = markerindex;
            markerindex += 1;
            
        M.total_landmarks = markerindex;
        
        markerindex = index + 1;
        
        for m in nonlinkedtargetmarkers:    
            m.id = markerindex;
            markerindex += 1;
        
        N.total_landmarks = markerindex;
    
    else:
        sourcemarkers = [m for m in M.generic_landmarks];
        nonlinkedsourcemarkers = [m for m in M.generic_landmarks if not m.is_linked];
        markerindex = 0;
        for m in nonlinkedsourcemarkers:
            m.id = markerindex;
            markerindex += 1;
            
        M.total_landmarks = markerindex;
    
    
def deleteObjectWithMarkers(context, mesh, onlymarkers=True):
    
    if(context.mode != "OBJECT"):
        bpy.ops.object.mode_set(mode = 'OBJECT', toggle = False);
                        
    if(mesh.is_landmarked_mesh):
        if(mesh.hide_landmarks):
            mesh.hide_landmarks = False;
    
    bpy.ops.object.select_all(action="DESELECT");
    
    context.scene.objects.active =  mesh;
    mesh.select = True;
    bpy.ops.object.select_grouped(type="CHILDREN_RECURSIVE");
    if(onlymarkers):
        mesh.select = False;
    else:
        mesh.select = True;
    bpy.ops.object.delete();


def getMarkersForMirrorXByHM(context, mesh, hmarkerslist):
    reflectionmarkers = [];
    belongsto = mesh;
    for hmarker in hmarkerslist:
        reflectionmarker = getMarkerXPlane(belongsto, hmarker);
        if(reflectionmarker):
            reflectionmarkers.append(reflectionmarker);
    
    return reflectionmarkers;
     

def getMarkersForMirrorX(context, bmarkerslist):
    reflectionmarkers = [];
    
    for bmarker in bmarkerslist:
        belongsto = getMeshForBlenderMarker(bmarker);
        hmarker = getGenericLandmark(belongsto, bmarker);
        reflectionmarker = getMarkerXPlane(belongsto, hmarker);
        if(reflectionmarker):
            mirrorbmarker = getBlenderMarker(belongsto, reflectionmarker);
            reflectionmarkers.append(mirrorbmarker);
    
    return reflectionmarkers;

def getMarkerXPlane(meshobject, landmark):
    bl = landmark.location;
    baselocation = Vector((bl[0], bl[1], bl[2]));
    
    hmarkers = [m for m in meshobject.generic_landmarks if m.id != landmark.id];
    
    for m in hmarkers:
        mlocation = Vector((m.location[0], m.location[1], m.location[2]));
        fliplocation = Vector((mlocation.x * -1, mlocation.y, mlocation.z));
        diffDist = (fliplocation - baselocation).length;
        if(diffDist < 0.0001):
            return m;
    return None;

def getGenericLandmark(meshobject, bmarker):
    if(bmarker.is_visual_landmark):
        bnamelist = bmarker.name.split('_marker_');
        originalid = int(bnamelist[1]);
        
        return [m for m in meshobject.generic_landmarks if m.id == originalid][0];
    
    return None;
    
def getBlenderMarker(meshobject, landmark):
    mname = meshobject.name + "_marker_"+str(landmark.id);    
    return bpy.data.objects[mname];

def getMeshForBlenderMarker(blendermarker):
    if(blendermarker.is_visual_landmark):
        if(blendermarker.belongs_to):
            return bpy.data.objects[blendermarker.belongs_to];
        else:
            bnamelist = blendermarker.name.split('_marker_');
            return bpy.data.objects[bnamelist[0]];
    
def getMarkerOwner(markerobj):
    if(markerobj.is_visual_landmark):
        belongsto = bpy.data.objects[markerobj.name.split("_marker_")[0]];
        return belongsto, False, False;    
    return None, False, False;
