import bpy, time;
from bpy.props import StringProperty;
from bpy.props import FloatVectorProperty;
from mathutils import Vector;
from mathutils.bvhtree import BVHTree;

import bgl;

from chenhan_pp.helpers2 import drawHollowCircleBillBoard, drawPoint, drawText;

from GenericMarkerCreator import constants;
from GenericMarkerCreator.propertiesregister import changeMarkerColor, changeUnlinkedMarkerColor;
from GenericMarkerCreator.misc.interactiveutilities import ScreenPoint3D;
from GenericMarkerCreator.misc.staticutilities import detectMN, applyMarkerColor, addConstraint, getConstraintsKD, deleteObjectWithMarkers, reorderConstraints;
from GenericMarkerCreator.misc.staticutilities import getMarkersForMirrorX, getGenericLandmark, getMeshForBlenderMarker, getBlenderMarker;
from GenericMarkerCreator.misc.meshmathutils import getBarycentricCoordinateFromPolygonFace, getBarycentricCoordinate, getCartesianFromBarycentre, getGeneralCartesianFromBarycentre, getTriangleArea;

def DrawGL(self, context):
    
    bgl.glDisable(bgl.GL_DEPTH_TEST);
    bgl.glColor4f(*(1.0, 1.0, 0.0,1.0));
    
    drawPoint(Vector((0,0,0)), (1,1,1,1));
    drawHollowCircleBillBoard(context, Vector((0, 0, 0)), 5.0);
    drawText(context, "Add Markers", Vector((0, 0, 0)), text_scale_value = 0.001, constant_scale = False);
    # restore opengl defaults
    bgl.glLineWidth(1);
    bgl.glDisable(bgl.GL_BLEND);
    bgl.glEnable(bgl.GL_DEPTH_TEST);
    bgl.glColor4f(0.0, 0.0, 0.0, 1.0);


class LiveLandmarksCreator(bpy.types.Operator):
    """Draw a line with the mouse"""
    bl_idname = "genericlandmarks.livelandmarkscreator";
    bl_label = "Landmarks Creator";
    bl_description = "Create Landmarks for surface(s)"
    hit = FloatVectorProperty(name="hit", size=3);    
    
    def modal(self, context, event):
        if event.type in {'ESC'}:
            if(self._handle):
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW'); 
            context.area.header_text_set();
            bpy.ops.object.select_all(action="DESELECT");
            self.mousepointer.select = True;
            context.scene.objects.active = self.mousepointer;
            bpy.ops.object.delete();                
            return {'CANCELLED'}
        
        elif (event.type in {"A", "a"} and event.value in {"PRESS"}):
            self.hit, onM, m_face_index, m_hitpoint = ScreenPoint3D(context, event, position_mouse = False, use_mesh=self.M);
            self.hit, onN, n_face_index, n_hitpoint = ScreenPoint3D(context, event, position_mouse = False, use_mesh=self.N);
            
            the_mesh = None;
            face_index = -1;
            hitpoint = None;
            use_bvh_tree = None;
            onMesh = False;
            
            if(onM):
                the_mesh = self.M;
                face_index = m_face_index;
                hitpoint = m_hitpoint;
                use_bvh_tree = self.bvhtree_m;
                onMesh = onM;
            if(onN):
                the_mesh = self.N;
                face_index = n_face_index;
                hitpoint = n_hitpoint;
                use_bvh_tree = self.bvhtree_n;
                onMesh = onN;

            if(face_index and onMesh):
                proceedToAddMarker = False;                
                diffmouse = time.time() - self.lastmousepress;
                
                if(diffmouse > constants.TIME_INTERVAL_MOUSE_PRESS and onMesh):
                    self.lastmousepress = time.time();
                    markerskd, markercos = getConstraintsKD(context, the_mesh);
                    co, index, dist = markerskd.find(hitpoint);
                    if(not len(the_mesh.generic_landmarks)):
                        dist = 9999999.0;
                        
                    if(dist is not None):
                        if(dist > constants.MARKER_MIN_DISTANCE):
                            proceedToAddMarker = True;
                        else:
                            proceedToAddMarker = False;
                if(proceedToAddMarker):                
                    face = the_mesh.data.polygons[face_index];
                    loops = the_mesh.data.loops;
                    vertices = the_mesh.data.vertices;
                    a = vertices[loops[face.loop_indices[0]].vertex_index];
                    b = vertices[loops[face.loop_indices[1]].vertex_index];
                    c = vertices[loops[face.loop_indices[2]].vertex_index];
                    
                    u,v,w,ratio,isinside = getBarycentricCoordinate(hitpoint, a.co, b.co, c.co);
                    finalPoint = getCartesianFromBarycentre(Vector((u,v,w)), a.co, b.co, c.co);
                    
                    if(isinside):
                        print('ADDING MARKER WITH BARYCENTRIC VALUES ::: ',u,v,w, face.index);
                        addConstraint(context, the_mesh, [u,v,w], [a.index, b.index, c.index], hitpoint);
                        
                        if(context.scene.use_mirrormode_x):
                            center = finalPoint.copy();
                            center.x = -center.x;
                            delta = (finalPoint - center).length;
                            print('DELTA BETWEEN SYMMETRY POINTS ', delta);
                            if(delta > constants.EPSILON):
                                try:
                                    co, n, index, distance = use_bvh_tree.find(center);
                                except AttributeError:
                                    co, n, index, distance = use_bvh_tree.find_nearest(center);
                                    
                                face = the_mesh.data.polygons[index];
                                a = vertices[loops[face.loop_indices[0]].vertex_index];
                                b = vertices[loops[face.loop_indices[1]].vertex_index];
                                c = vertices[loops[face.loop_indices[2]].vertex_index];
                                u,v,w,ratio,isinside = getBarycentricCoordinate(co, a.co, b.co, c.co);
                                addConstraint(context, the_mesh, [u,v,w], [a.index, b.index, c.index], center);
                                print('ADD MIRROR MARKER AT ', center, u, v, w);
                                print('IT HAS A NEAREST FACE : ', index, ' AT A DISTANCE ::: ', distance);
            
            return {'RUNNING_MODAL'};

            
        elif event.type == 'MOUSEMOVE':    
            self.hit, onM, m_face_index, m_hitpoint = ScreenPoint3D(context, event, position_mouse = False, use_mesh=self.M);
            self.hit, onN, n_face_index, n_hitpoint = ScreenPoint3D(context, event, position_mouse = False, use_mesh=self.N);
            
            the_mesh = None;
            face_index = -1;
            hitpoint = None;
            
            if(onM):
                the_mesh = self.M;
                face_index = m_face_index;
                hitpoint = m_hitpoint;
            if(onN):
                the_mesh = self.N;
                face_index = n_face_index;
                hitpoint = n_hitpoint;
#             context.area.header_text_set("hit: %.4f %.4f %.4f" % tuple(self.hit));
            if(onM or onN):
                if(face_index):
                    face = the_mesh.data.polygons[face_index];
                    loops = the_mesh.data.loops;
                    vertices = the_mesh.data.vertices;
                    a = vertices[loops[face.loop_indices[0]].vertex_index];
                    b = vertices[loops[face.loop_indices[1]].vertex_index];
                    c = vertices[loops[face.loop_indices[2]].vertex_index];
                    area, area2 = getTriangleArea(a.co, b.co, c.co);
                    u,v,w,ratio,isinside = getBarycentricCoordinate(hitpoint, a.co, b.co, c.co, epsilon=area * 0.1);
                        
                    newco = getCartesianFromBarycentre(Vector((u,v,w)), a.co, b.co, c.co);
                    context.area.header_text_set("Barycentric Values: %.8f %.8f %.8f %.8f" % tuple((u,v,w,(u+v+w))));
                        
                    self.mousepointer.location = the_mesh.matrix_world * newco;
        
        return {'PASS_THROUGH'};
        
    def invoke(self, context, event):
        if(context.active_object):
            M, N = detectMN(context.active_object);
            if(not M and not N):
                message = "Landmark creator needs both M and N";
                self.M = context.active_object;
                self.N = context.active_object;
#                 bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='INFO',message=message,messagelinesize=60);
#                 return {'FINISHED'};
            elif(M and not N):
                self.M = M;
                self.N = M;
            elif (not M and N):
                self.M = N;
                self.N = N;
            else:
                self.M = M;
                self.N = N;
            
        self.lastkeypress = time.time();
        self.lastmousepress = time.time();
        
        self.mesh = context.active_object;
        self.bvhtree_m = BVHTree.FromObject(self.M, context.scene);
        self.bvhtree_n = BVHTree.FromObject(self.N, context.scene);
        
        maxsize = max(self.mesh.dimensions.x, self.mesh.dimensions.y, self.mesh.dimensions.z);
        markersize = maxsize * 0.01;            
        tempmarkersource = "Marker";
        
        try:
            tempmarker = bpy.data.objects[tempmarkersource];
        except KeyError:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=36, ring_count = 36);
            tempmarker = context.object;
            tempmarker.name = "Marker";

        tempmarker.dimensions = (markersize,markersize,markersize);
        self.mousepointer = tempmarker;
        
        self.key = [];
        self.time = [];
        
        applyMarkerColor(self.mousepointer);
        context.scene.objects.active = self.mesh;
        
        context.window_manager.modal_handler_add(self);
        
        args = (self, context); 
        self._handle = bpy.types.SpaceView3D.draw_handler_add(DrawGL, args, 'WINDOW', 'POST_VIEW');
        
        
        return {'RUNNING_MODAL'};