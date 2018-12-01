import bpy, time;
from bpy.props import StringProperty;
from bpy.props import FloatVectorProperty;
from mathutils import Vector;
from mathutils.bvhtree import BVHTree;
from scipy.io import loadmat;
from sklearn.neighbors import NearestNeighbors

import bgl;

from chenhan_pp.helpers2 import drawHollowCircleBillBoard, drawPoint, drawText;

from GenericMarkerCreator import constants;
from GenericMarkerCreator.propertiesregister import changeMarkerColor, changeUnlinkedMarkerColor;
from GenericMarkerCreator.misc.interactiveutilities import ScreenPoint3D;
from GenericMarkerCreator.misc.staticutilities import detectMN, applyMarkerColor, addConstraint, getConstraintsKD, deleteObjectWithMarkers, reorderConstraints;
from GenericMarkerCreator.misc.staticutilities import getMarkersForMirrorX, getGenericLandmark, getMeshForBlenderMarker, getBlenderMarker;
from GenericMarkerCreator.misc.meshmathutils import getBarycentricCoordinateFromPolygonFace, getBarycentricCoordinate, getCartesianFromBarycentre, getGeneralCartesianFromBarycentre, getTriangleArea;
from GenericMarkerCreator.misc.mathandmatrices import getObjectBounds;

def DrawGL(self, context):
    
    bgl.glDisable(bgl.GL_DEPTH_TEST);
    bgl.glColor4f(*(1.0, 1.0, 0.0,1.0));
    
    for (co, id) in self.M_markers:
        drawPoint(Vector((0,0,0)), (1,1,1,1), size=15.0);
        drawHollowCircleBillBoard(context, co, self.marker_ring_size);
        drawText(context, "id: %d"%(id), co, text_scale_value = 0.001, constant_scale = False);
        
    for (co, id) in self.N_markers:
        drawPoint(Vector((0,0,0)), (1,1,1,1));
        drawHollowCircleBillBoard(context, co, self.marker_ring_size);
        drawText(context, "id: %d"%(id), co, text_scale_value = 0.001, constant_scale = False);
    
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
                    
                    u,v,w,ratio,isinside = getBarycentricCoordinate(hitpoint, a.co, b.co, c.co, snapping=the_mesh.snap_landmarks);
                    finalPoint = getCartesianFromBarycentre(Vector((u,v,w)), a.co, b.co, c.co);
                    print('PROCEED TO ADD MARKER : %s, IS INSIDE : %s'%(proceedToAddMarker, isinside));
                    if(isinside):
                        print('ADDING MARKER WITH BARYCENTRIC VALUES ::: ',u,v,w, face.index);
                        addConstraint(context, the_mesh, [u,v,w], [a.index, b.index, c.index], hitpoint, faceindex=face_index);
                        
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
                                u,v,w,ratio,isinside = getBarycentricCoordinate(co, a.co, b.co, c.co, snapping=the_mesh.snap_landmarks);
                                addConstraint(context, the_mesh, [u,v,w], [a.index, b.index, c.index], co, faceindex=face_index);
                                print('ADD MIRROR MARKER AT ', center, u, v, w);
                                print('IT HAS A NEAREST FACE : ', index, ' AT A DISTANCE ::: ', distance);
                
                        del self.M_markers[:];
                        del self.N_markers[:];
                        
                        for gm in self.M.generic_landmarks:
                            loc = Vector([dim for dim in gm.location]);
                            loc = self.M.matrix_world * loc;
                            dictio = (loc, gm.id);
                            self.M_markers.append(dictio);
                        
                        if(self.M != self.N and self.N is not None):
                            for gm in self.N.generic_landmarks:
                                loc = Vector([dim for dim in gm.location]);
                                loc = self.N.matrix_world * loc;
                                dictio = (loc, gm.id);
                                self.N_markers.append(dictio);
                            
                            
                
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
                    u,v,w,ratio,isinside = getBarycentricCoordinate(hitpoint, a.co, b.co, c.co, epsilon=area * 0.1, snapping=the_mesh.snap_landmarks);
                        
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
        
        self.M_markers = [];
        self.N_markers = [];
        
        for gm in self.M.generic_landmarks:
            loc = Vector([dim for dim in gm.location]);
            loc = self.M.matrix_world * loc;
            dictio = (loc, gm.id);
            self.M_markers.append(dictio);
        
        if(self.M != self.N and self.N is not None):
            for gm in self.N.generic_landmarks:
                loc = Vector([dim for dim in gm.location]);
                loc = self.N.matrix_world * loc;
                dictio = (loc, gm.id);
                self.N_markers.append(dictio);
        
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
        
        real_marker = None;
        try:
            real_marker = context.scene.objects[context.scene.landmarks_use_selection];
        except KeyError:
            real_marker = tempmarker;
            
        min_size, max_size = getObjectBounds(real_marker);
        size_vector = max_size - min_size;
        self.marker_ring_size = size_vector.length * 0.003;
        
        self.key = [];
        self.time = [];
        
        applyMarkerColor(self.mousepointer);
        context.scene.objects.active = self.mesh;
        
        context.window_manager.modal_handler_add(self);
        
        args = (self, context); 
        self._handle = bpy.types.SpaceView3D.draw_handler_add(DrawGL, args, 'WINDOW', 'POST_VIEW');
        
        
        return {'RUNNING_MODAL'};



class SignaturesMatching(bpy.types.Operator):
    """Draw a line with the mouse"""
    bl_idname = "genericlandmarks.signaturesmatching";
    bl_label = "Signature Matching";
    bl_description = "Match Landmarks based on signatures of normals, gisif, and curvature (k1, k2)"
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
                            
        elif event.type == 'MOUSEMOVE':    
            self.hit, onM, m_face_index, m_hitpoint = ScreenPoint3D(context, event, position_mouse = False, use_mesh=self.M);
            self.hit, onN, n_face_index, n_hitpoint = ScreenPoint3D(context, event, position_mouse = False, use_mesh=self.N);
            
            the_mesh = None;
            other_mesh = None;
            face_index = -1;
            hitpoint = None;
            the_data = None;
            find_on_tree = None;
            
            c_source, c_target = None, None;
            
            
            if(onM):
                the_mesh = self.M;
                face_index = m_face_index;
                hitpoint = m_hitpoint;
                the_data = self.M_data;
                find_on_tree = self.N_kd_tree;
                c_source = self.M_markers;
                c_target = self.N_markers;
                other_mesh = self.N;
            elif(onN):
                the_mesh = self.N;
                face_index = n_face_index;
                hitpoint = n_hitpoint;
                the_data = self.N_data;
                find_on_tree = self.M_kd_tree;
                c_source = self.N_markers;
                c_target = self.M_markers;
                other_mesh = self.M;
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
                    u,v,w,ratio,isinside = getBarycentricCoordinate(hitpoint, a.co, b.co, c.co, epsilon=area * 0.1, snapping=the_mesh.snap_landmarks);
                    
                    aco = the_data[a.index];
                    bco = the_data[b.index];
                    cco = the_data[c.index];                    
                    findco = getCartesianFromBarycentre(Vector((u,v,w)), aco, bco, cco).reshape(1,-1);
                    newco = getCartesianFromBarycentre(Vector((u,v,w)), a.co, b.co, c.co);
                    distances, indices = find_on_tree.kneighbors(findco);
#                     weights = 1.0 - np.interp(distances[0], (distances.min(), distances.max()), (0.0,1.0));
                    
                    del self.M_markers[:];
                    del self.N_markers[:];
                    
                    c_source.append((the_mesh.matrix_world * newco, 0));
                    
                    for id in indices[0]:
                        vert = other_mesh.data.vertices[id];
                        loc = other_mesh.matrix_world * vert.co;
                        dictio = (loc, vert.index);
                        c_target.append(dictio);
                          
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
                bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='INFO',message=message,messagelinesize=60);
                return {'FINISHED'};
            
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
        
        self.M_markers = [];
        self.N_markers = [];
        
        for gm in self.M.generic_landmarks:
            loc = Vector([dim for dim in gm.location]);
            loc = self.M.matrix_world * loc;
            dictio = (loc, gm.id);
            self.M_markers.append(dictio);
        
        if(self.M != self.N and self.N is not None):
            for gm in self.N.generic_landmarks:
                loc = Vector([dim for dim in gm.location]);
                loc = self.N.matrix_world * loc;
                dictio = (loc, gm.id);
                self.N_markers.append(dictio);
        
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
        
        real_marker = None;
        try:
            real_marker = context.scene.objects[context.scene.landmarks_use_selection];
        except KeyError:
            real_marker = tempmarker;
            
        min_size, max_size = getObjectBounds(real_marker);
        size_vector = max_size - min_size;
        self.marker_ring_size = size_vector.length * 0.003;
        
        self.key = [];
        self.time = [];
        
        applyMarkerColor(self.mousepointer);
        context.scene.objects.active = self.mesh;
        
        try:
            m_path = bpy.path.abspath('%s/%s.mat'%(self.M.signatures_dir, self.M.name));
            n_path = bpy.path.abspath('%s/%s.mat'%(self.N.signatures_dir, self.N.name));            
            self.M_data = loadmat(m_path)['X'];
            self.N_data = loadmat(n_path)['X'];
        except FileNotFoundError:
            message = "Please provide the spectral signatures for both M(%s) and N(%s)"%(self.M.name, self.N.name);
            bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
            return {'FINISHED'};
        
        self.M_kd_tree = NearestNeighbors(n_neighbors=1).fit(self.M_data);
        self.N_kd_tree = NearestNeighbors(n_neighbors=1).fit(self.N_data);
        
        context.window_manager.modal_handler_add(self);
        
        args = (self, context); 
        self._handle = bpy.types.SpaceView3D.draw_handler_add(DrawGL, args, 'WINDOW', 'POST_VIEW');
        
        
        return {'RUNNING_MODAL'};









