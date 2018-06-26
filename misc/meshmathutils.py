import bpy, mathutils;
from mathutils import Vector;

def getTriangleArea(p0, p1, p2):        
    a = (p1 - p0).length;
    b = (p2 - p1).length;
    c = (p0 - p2).length;
    #Using Herons formula
    A = 0.25 * ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))**0.5;
    
    if(isinstance(A, complex)):
        A = cmath.phase(A);
    
    if (A < 0.0001):
        A = 0.0001;
    
    return A, A*2;

#Can be of type, 1-VERT, 2-EDGE, 3-FACE, 4-FACEVERT (contains vertices co with face Index)
#5-EDGEVERT (contains vertices co with EDGE INDEX)
def getKDTree(context, meshobject, type="VERT", points=[], use_bm =None,*, return_points = False):
    if(meshobject):
        mesh = meshobject.data;
        data = [];    
    if(type == "VERT"):
        size = len(mesh.vertices);
        kd = mathutils.kdtree.KDTree(size);
        for i, v in enumerate(mesh.vertices):
            kd.insert(v.co, v.index);
    elif(type =="EDGE"):
        subdivisions = 3;
        subdivisionsF = 3.0;
        size = len(mesh.edges) * subdivisions;
        edge_points = [];
        kd = mathutils.kdtree.KDTree(size);
        
        for i, e in enumerate(mesh.edges):
            v0 = mesh.vertices[e.vertices[0]].co;
            v1 = mesh.vertices[e.vertices[1]].co;
            vect = (v1 - v0);
            
            for i in range(subdivisions):
                ratio = float(i) / (subdivisionsF - 1.0);
                point = v0 + (vect * ratio);
                kd.insert(point, e.index);
                edge_points.append({'index':e.index, 'co':point});
                
        if(return_points):
            kd.balance();
            return kd, edge_points;
        
        
    elif(type =="FACE"):
        size = len(mesh.polygons);
        kd = mathutils.kdtree.KDTree(size);        
        for i, f in enumerate(mesh.polygons):
            kd.insert(f.center.copy(), f.index);
    
    elif(type == "FACEVERT"):
        size = len(mesh.polygons) * 3;
        kd = mathutils.kdtree.KDTree(size);

        for i, f in enumerate(mesh.polygons):
            if(len(f.loop_indices) > 3):
                print('GOTCHA :: THE BLACK SHEEP ::: ', object.name, f.index);
            for ind in f.loop_indices:
                l = mesh.loops[ind];
                v = mesh.vertices[l.vertex_index];
                kd.insert(v.co, f.index);            
    
    elif(type == "CUSTOM"):
        size = len(points);
        kd = mathutils.kdtree.KDTree(size);
        useDict = True;
        try:
            index = points[0]['index'];
        except KeyError:            
            useDict = False;
        except TypeError:
            useDict = False;
        except IndexError:
            useDict = False;
        
        for i, point in enumerate(points):
            if(useDict):
                index = point['index'];
                co = point['co'];
                kd.insert(co, index);
            else:
                kd.insert(point, i);          
                
    kd.balance();
    return kd;

#Given the point and three coordinates of form mathutils.Vector
#Will return the barycentric ratios
def getBarycentricCoordinate(p, a, b, c, *, epsilon=0.0000001,snapping=True):
    
    v0 = b - a;
    v1 = c - a;
    v2 = p - a;
    
    d00 = v0.dot(v0);
    d01 = v0.dot(v1);
    d11 = v1.dot(v1);
    d20 = v2.dot(v0);
    d21 = v2.dot(v1);
    denom = (d00 * d11) - (d01 * d01);
    
    try:
        v = (d11 * d20 - d01 * d21) / denom;        
    except ZeroDivisionError:
        problems = True;
        v = 0.0;
    try:
        w = (d00 * d21 - d01 * d20) / denom;
    except ZeroDivisionError:
        problems = True;
        w = 0.0;
    
    if(v > 0.95 and snapping):
        return 0.0, 1.0, 0.0,1.0, True;
    if(w > 0.95 and snapping):
            return 0.0, 0.0, 1.0,1.0, True;
    
    u = 1.0 - v - w;
    
    if(u > 0.95 and snapping):
        return 1.0, 0.0, 0.0,1.0, True;
    
    if(u < 0.0 or v < 0.0 or w < 0.0):
        return u, v, w,(u+v+w), False;

    if(snapping):    
        residue = 0.0;
        divisor = 3.0;
         
        if(u < 1.0e-01 and u > 0.0):
            residue += u;
            divisor -= 1.0;
            u = 0.0;
                 
        if(v < 1.0e-01 and v > 0.0):
            residue += v;
            divisor -= 1.0;
            v = 0.0;
                 
        if(w < 1.0e-01 and w > 0.0):
            residue += w;
            divisor -= 1.0;
            w = 0.0;
         
        if(divisor > 0.0):
            real_residue = residue / divisor;
        else:
            real_residue = 0.0;
             
        u += min(u, real_residue);
        v += min(v, real_residue);
        w += min(w, real_residue);
    
    ratio = u + v + w;
    
    
    return u,v,w,ratio, (u >= 0.0 and v >=0.0 and u >=0.0 and ratio <=1.0);


def getBarycentricCoordinateFromPolygonFace(p, pface, mesh, *, epsilon=0.0000001,snapping=True, extra_info = False):
    points = [];
    loops = mesh.data.loops;
    vertices = mesh.data.vertices;
    v_indices = [];
    
    for lid in pface.loop_indices:
        l = loops[lid];
        v = vertices[l.vertex_index];
        v_indices.append(l.vertex_index);
        
        points.append(v.co.copy());
    
    u,v,w,ratio, isinside = getBarycentricCoordinate(p, points[0],points[1],points[2], epsilon=epsilon, snapping=snapping);
    
    if(not extra_info):
        return u,v,w,ratio, isinside;
    else:
        return u,v,w,ratio, isinside, v_indices[0], v_indices[1], v_indices[2];


#Given the barycentric value as a mathutils.Vector (u=>x, v=>y, w=>z)
#And the three new points of the triangle again of the form mathutils.Vector
#Will give the new cartesian coordinate

def getCartesianFromBarycentre(ba, a, b, c):
    aa = a.copy();
    bb = b.copy();
    cc = c.copy();    
    
    p = ( aa * ba.x) + (bb * ba.y) + (cc * ba.z);
    
    return p;

def getGeneralCartesianFromBarycentre(weights, points):
    p = Vector((0.0,0.0,0.0));
    for index, r in enumerate(weights):
        p = p + (points[index] * r);    
    return p;
