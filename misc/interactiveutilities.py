import bpy, bmesh, bgl, math, os, mathutils, sys;
from mathutils import Vector;
import blf, time, datetime;
from bpy_extras.view3d_utils import location_3d_to_region_2d, region_2d_to_vector_3d, region_2d_to_location_3d, region_2d_to_origin_3d
from bpy_extras import view3d_utils;


def getViewports(context):    
    view_ports_count = 0;
    view_ports = [];
    
    for window in context.window_manager.windows:
        screen = window.screen;     
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                view_ports_count += 1;
                view_ports.append(area);
    
    return view_ports, view_ports_count;

def getActiveView(context, event):
    view_ports, view_ports_count = getViewports(context);
    region = None;
    rv3d = None;
    coord = event.mouse_x, event.mouse_y;
    for area in view_ports:
        for r in area.regions:
            if(r.type == "WINDOW" and r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height):
                rv3d = [space.region_3d for space in area.spaces if space.type == 'VIEW_3D'][0];
                region = r;
                coord = r.width - ((r.x + r.width) - event.mouse_x), r.height - ((r.y + r.height) - event.mouse_y);
                break;
    return region, rv3d, coord;

def ScreenPoint3D(context, event, *, ray_max=1000.0, position_mouse = True, use_mesh = None):
    # get the context arguments
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    #coord = event.mouse_region_x, event.mouse_region_y;
    
    region, rv3d, coord = getActiveView(context, event);
    
    if(not region and not rv3d):
        return Vector((0,0,0)), False, -1, None;
    
#     print('COORD : ', coord, " ID ::: ", region.id);
    
    # get the ray from the viewport and mouse
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord);
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord);


    if bpy.app.version < (2, 77, 0):
        if rv3d.view_perspective == 'ORTHO':
            # move ortho origin back
#             ray_origin = ray_origin - (view_vector * (ray_max / 2.0));
            pass;
    
    else:
        ray_max = 1.0;
    
    ray_target = ray_origin + (view_vector * ray_max);


    def obj_ray_cast(obj, matrix):
        """Wrapper for ray casting that moves the ray into object space"""
        # get the ray relative to the object
        matrix_inv = matrix.inverted();
        ray_origin_obj = matrix_inv * ray_origin;
        ray_target_obj = matrix_inv * ray_target;
        ray_direction_obj = ray_target_obj - ray_origin_obj;


        # cast the ray
        try:
            hit, normal, face_index = obj.ray_cast(ray_origin_obj, ray_target_obj);
        except ValueError:
            result, hit, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj);
#        hit, normal, face_index, distance = bvhtree.ray_cast(ray_origin_obj, ray_target_obj);
        
        if face_index != -1:
            return hit, normal, face_index;
        else:
            return None, None, None;


    # no need to loop through other objects since we are interested in the active object only
    if(use_mesh):
        obj = use_mesh;
    else:
        obj = context.scene.objects.active;
    
    if(obj):
        matrix = obj.matrix_world.copy();
        if(position_mouse):
            mousemarker = bpy.data.objects["Marker"];
        
        if obj.type == 'MESH':
            hit, normal, face_index = obj_ray_cast(obj, matrix);
            if hit is not None:
                hit_world = matrix * hit;
                if(position_mouse):
                    mousemarker.location = hit_world;
                return hit_world, True, face_index, hit;
    return view_vector, False, None, None;