import bpy, bmesh, mathutils, os;
from mathutils.bvhtree import BVHTree;
import numpy as np;

import platform;

from mathutils import Vector, Color;
from scipy.interpolate.interpolate_wrapper import logarithmic
from functools import reduce;

from GenericMarkerCreator.misc.meshmathutils import getKDTree, getBarycentricCoordinate, getBarycentricCoordinateFromPolygonFace;
from GenericMarkerCreator.misc.mathandmatrices import getBMMesh, ensurelookuptable;

print('HELPERS LOOKING INTO THE PLATFORM ::: ', platform.system());

if(platform.system() != 'Windows'):
    import matplotlib;
    from matplotlib import rcParams;
    try:
        from matplotlib import pyplot as plt;
    except ImportError:
        pass;
    from matplotlib.figure import Figure;
    from mpl_toolkits.mplot3d import Axes3D;
    from mpl_toolkits.mplot3d import proj3d;
    from matplotlib.patches import FancyArrowPatch;
    from matplotlib.font_manager import FontProperties;
    
    import matplotlib.cm as cm;
    import matplotlib.mlab as mlab;
    from matplotlib.cm import get_cmap;
    import matplotlib.colors as clrs;
else:
    print('WINDOWS SYSTEM CANNOT BE SUPPORTED. SORRY!');




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


def getMarkerType(context, mesh, landmark):
    indices = np.array([vid for vid in landmark.v_indices], dtype=int);
    ratios = np.array([r for r in landmark.v_ratios]);
    c_nz = np.count_nonzero(ratios);
    arg_sorted = np.argsort(ratios)[::-1];
    
    v_indices = None;
    bm = getBMMesh(context, mesh, useeditmode=False);    
    ensurelookuptable(bm);
    if(c_nz == 1):
        v_indices = indices[arg_sorted[[0]]];
        location = bm.verts[v_indices[0]].co.to_tuple();
        bm.free();
        return 'VERTEX', v_indices[0], location, [location];
    elif (c_nz == 2):
        v_indices = indices[arg_sorted[[0, 1]]];
        edges_1 = np.array([e.index for e in bm.verts[v_indices[0]].link_edges]);
        edges_2 = np.array([e.index for e in bm.verts[v_indices[1]].link_edges]);
        edge_common = np.intersect1d(edges_1, edges_2);
        location = tuple([val for val in landmark.location]);
        edge_locations = [bm.verts[v_indices[0]].co.to_tuple(), bm.verts[v_indices[1]].co.to_tuple()];
        bm.free();
        return 'EDGE', edge_common.tolist()[0], location, edge_locations;
    elif(c_nz == 3):
        v_indices = indices[arg_sorted[[0, 1, 2]]];
        faces_1 = np.array([f.index for f in bm.verts[v_indices[0]].link_faces]);
        faces_2 = np.array([f.index for f in bm.verts[v_indices[1]].link_faces]);
        faces_3 = np.array([f.index for f in bm.verts[v_indices[2]].link_faces]);
        face_common = reduce(np.intersect1d, (faces_1, faces_2, faces_3));
        location = tuple([val for val in landmark.location]);
        face_locations = [bm.verts[v_indices[0]].co.to_tuple(), bm.verts[v_indices[1]].co.to_tuple(), bm.verts[v_indices[2]].co.to_tuple()];
        bm.free();
        return 'FACE', face_common.tolist()[0], location, face_locations;
    
    return None, None, None;

def subdivideEdge(bm, edge, point):
        dictio = bmesh.ops.bisect_edges(bm, edges=[edge], cuts=1);
        dictio['geom_split'][0].co = point;
        return dictio['geom_split'][0];

def subdivideFace(bm, face, point):
    retu = bmesh.ops.poke(bm, faces=[face]);
    thevertex = retu['verts'][0];
    thevertex.co = point;
    return thevertex;

def remeshMarkersAsVertices(context, mesh):
    edge_indices = [];
    edge_locations = [];
    
    face_indices = [];
    face_locations = [];
    
    for gm in mesh.generic_landmarks:
        gm_on_type, gm_on_type_index, gm_location, gm_locations = getMarkerType(context, mesh, gm);
        if(gm_on_type == 'EDGE'):
            edge_locations.append(gm_location);
            edge_indices.append(gm_on_type_index);
        elif(gm_on_type == 'FACE'):
            face_locations.append(gm_location);
            face_indices.append(gm_on_type_index);
    
    verts_and_locations = [];
    bm = getBMMesh(context, mesh, useeditmode=False);
    
    ensurelookuptable(bm);
    faces = [bm.faces[ind] for ind in face_indices];
    returned_geometry_faces = bmesh.ops.poke(bm, faces=faces);
    
    for i, vert in enumerate(returned_geometry_faces['verts']):
        verts_and_locations.append((vert.index, face_locations[i]));
            
    ensurelookuptable(bm);
    edges = [bm.edges[ind] for ind in edge_indices];
    returned_geometry_edges = bmesh.ops.bisect_edges(bm, edges=edges, cuts=1);
    returned_vertices = [vert for vert in returned_geometry_edges['geom_split'] if isinstance(vert, bmesh.types.BMVert)];
        
    for i, vert in enumerate(returned_vertices):
        verts_and_locations.append((vert.index, edge_locations[i]));
            
    ensurelookuptable(bm);
    
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0);
    bm.to_mesh(mesh.data);
    
    bm.free();
   
    for vid, co in verts_and_locations:
        mesh.data.vertices[vid].co = co;
        
    autoCorrectLandmarksData(context, mesh);
    
def autoCorrectLandmarksData(context, mesh):
    use_tree = BVHTree.FromObject(mesh, context.scene);
    kd = getKDTree(context, mesh);
    bm = getBMMesh(context, mesh, False);
    ensurelookuptable(bm);
    
    for gm in mesh.generic_landmarks:
        loc = [dim for dim in gm.location];
        mco = Vector((loc[0], loc[1], loc[2]));        
        co, index, dist = kd.find(mco);
        v = bm.verts[index];        
        f = v.link_faces[0];
            
        a = f.loops[0].vert;
        b = f.loops[1].vert;
        c = f.loops[2].vert;        
        u,v,w,ratio,isinside = getBarycentricCoordinate(co, a.co, b.co, c.co);
        gm.v_ratios = [u, v, w];
        gm.v_indices = [a.index, b.index, c.index];
    bm.free();
    
def plotErrorGraph(context, reference, meshobject, algo_names, distances, *, sorting = True, xlabel="Vertex", ylabel="Error", graph_title="Laplacian errors", graph_name="LaplacianErrors_", logarithmic=False, plot_sum=False, plot_average=False, show_title = True, elaborate_title = False):
        colors = [(1,0,0), (0,1,0), (0,0,1), (1,0,1), (1,1,0), (0,1,1), (1,0.5,0.5), (0.5,1,0.5), (0.5,0.5,1.0)];
        minimum = 99999999.0;
        maximum = -99999999.0;
        max_length = -99999999;
        max_distance = [];
        my_dpi = 200;
        all_np_distances = [];
        
        distance_sums = {};
        
        title_suffixes = "";
        title_prefixes = reference.name + " and "+meshobject.name;
                
        for index, distance in enumerate(distances):
            minimum = min(minimum, min(distance));
            maximum = max(maximum, max(distance));
            distance_sums[algo_names[index]] = sum(distance);
            max_distance.append(max(distance));
            max_length = max(max_length, len(distance));            
            all_np_distances.append(np.array(distance));
        
        minimum = minimum - (maximum * 0.1);
        
        x_labels = np.array([i for i in range(max_length)]);
        
        if(platform.system() != 'Windows'):        
            try:
                fig = plt.figure(figsize=(1920/my_dpi, 1200/my_dpi), dpi=my_dpi);
            except:            
                fig = Figure(figsize=(1920/my_dpi, 1200/my_dpi), dpi=my_dpi);
        
        f_size = 20;
        
        shown_graph_title = graph_title;
        if(elaborate_title):
            shown_graph_title = graph_title+ ' '+title_prefixes;
        
        if(not show_title):
            shown_graph_title = "";
        
        if(platform.system() != 'Windows'):        
#             ax = getAxisObject(fig, minimum, maximum, usegrid=True, xlabel= xlabel, ylabel = ylabel, logarithmic = logarithmic, fontsize = f_size);#fig.add_subplot(111);
#             ax.set_title(shown_graph_title, fontsize=f_size);
            ax = getAxisObject(fig, minimum, maximum, usegrid=True, xlabel= xlabel, ylabel = ylabel, logarithmic = logarithmic);
            ax.set_title(shown_graph_title);
            ax.set_title('');
        
        fulltitle = title_prefixes+"_" +title_suffixes;
        
        path = bpy.path.abspath(context.scene.metricsexportlocation)+reference.name+"_"+meshobject.name;        
        path = bpy.path.abspath(path);
        
        if not os.path.isdir(path):
            os.makedirs(path);        
        
        colors = 'b,g,r,cyan,yellow,orange,magenta'.split(",");
        linestyles = '--,:,-,-.'.split(",");
        matlab_contents = [];
        matlab_contents.append("h = figure('Position', [0 0 1100 600]);");
        matlab_variables = []; 
        
        for index, algoname in enumerate(algo_names):            
            color = colors[index];
            distances = all_np_distances[index];
            
            if(sorting):
                distances.sort();

            if(index != 0):
                title_suffixes += " vs "+algoname;
            else:
                title_suffixes = algoname;
            
            if(platform.system() != 'Windows'):            
                ax.plot(x_labels,distances, '-', label=algoname, markersize=0.5, color=color, alpha=0.9);
                
            np.savetxt(path+"/"+graph_name+".csv", distances, delimiter=',');
            
            matlab_variable = "x"+str(index)+" = ["+",".join([str(v) for v in all_np_distances[index]])+"];";
            matlab_variables.append(matlab_variable)
            matlab_contents.append(matlab_variable);
        
        for index, mvariable in enumerate(matlab_variables):
            variable_name = 'x'+str(index);
            linestyle = linestyles[index % len(linestyles)];
            linecolor = colors[index];
            if(index == len(matlab_variables)-1):
                linestyle = '-';
                linecolor = 'r';
                
            plot_statement = "plot(sort("+variable_name+"),'-"+linecolor+"', 'LineWidth', 1, 'LineStyle','"+linestyle+"');";
            matlab_contents.append(plot_statement);
            matlab_contents.append("hold all;");
        
        
        matlab_contents.append("xlabel('"+xlabel+"');");
        matlab_contents.append("ylabel('"+ylabel+"');");
        matlab_contents.append("title('Corr-"+reference.name +" - "+meshobject.name +" - Correspondences');");
        matlab_contents.append("legend("+",".join(["'"+a+"'" for a in algo_names])+");");
        matlab_contents.append("legend('Location','northwest');");
        
        if(platform.system() != 'Windows'):        
            handles, labels = ax.get_legend_handles_labels();
            ax.legend(handles, labels,fancybox=True,shadow=False,prop={'size':f_size-7}, loc='upper left');
        
        distance_errors_txt = "";
        for key in distance_sums:
            distance_errors_txt += str(key) + ","+str(distance_sums[key])+"\n";
        
        clean_file_name = "-".join(fulltitle.split(" "));
        
        fp = open(path+"/"+graph_name+clean_file_name+".csv", 'w', encoding='utf-8');
        fp.write(distance_errors_txt);
        fp.close();
        
        
        matlab_contents.append("grid on");
        matlab_contents.append("p = mfilename('fullpath');");
        matlab_contents.append("try");
        matlab_contents.append("addpath('matlab2tikz/src/');");
        matlab_contents.append("matlab2tikz(strcat(p,'.tex'),'width', '\\fwidth');");
        matlab_contents.append("catch exception");
        matlab_contents.append("fprintf('If you are planning to use this graph on latex then visit https://tomlankhorst.nl/matlab-to-latex-with-matlab2tikz/. It is really useful, Trust me!');");        
        matlab_contents.append("end");
        
        matlab_contents.append("try");
        matlab_contents.append("saveas(h,strcat(p,'.pdf'))");
        matlab_contents.append("saveas(h,strcat(p,'.jpg'))");
        matlab_contents.append("saveas(h,strcat(p,'.png'))");
        matlab_contents.append("catch exception");
        matlab_contents.append("fprintf('[WARNING] Could not save Results');");
        matlab_contents.append("end");
        
        
        if(clean_file_name.endswith("_")):
            clean_file_name = clean_file_name[:-1];
        
#         matlab_file = open(path+"/matgraph"+reference.name.replace("-","_")+"_"+meshobject.name.replace("-","_")+".m", "w");
        matlab_file = open(path+"/"+graph_name+"_".join(clean_file_name.split("-"))+".m", "w");
        
        matlab_file.write("\n".join(matlab_contents));
        matlab_file.close();
        
        print('PROCEED TO SAVE THE FILE ::: AT PATH :: ', path);
        if(platform.system() != 'Windows'):
            try:
                plt.tight_layout();
                plt.savefig(path+"/"+graph_name+clean_file_name+".pdf", transparent=False, bbox_inches='tight');
                plt.savefig(path+"/"+graph_name+clean_file_name+".png", transparent=False, bbox_inches='tight');
                plt.savefig(path+"/"+graph_name+clean_file_name+".svg", transparent=False, bbox_inches='tight');
                print('SAVING FILES USING DEFAULT PLOT MECHANISM');
            except NameError:        
                print('SAVED FILES USING CANVAS PRINT FIGURE MECHANISM-2');
                canvas = FigureCanvas(fig);
                canvas.print_figure(path+"/"+graph_name+clean_file_name+".pdf", transparent=False, bbox_inches='tight');
                canvas.print_figure(path+"/"+graph_name+clean_file_name+".png", transparent=False, bbox_inches='tight');
                canvas.print_figure(path+"/"+graph_name+clean_file_name+".svg", transparent=False, bbox_inches='tight');
        
        print('FINISHED SAVING THE FILE');
        
def applyColoringForMeshErrors(context, error_mesh, error_values, *, A = None, B = None, v_group_name = "lap_errors"):        
        c = error_values.T;       
        norm = clrs.Normalize(vmin=A, vmax=B);
        
        cmap = get_cmap('jet');
        c = c / np.amax(c);
        final_colors = cmap(c)[:, 0:3]
        
        final_weights = norm(c);
        
        print('GIVEN ARRAY:%s AND FINAL COLORS:%s '%(c.shape, final_colors.shape));
        
        colors = {};
        L_error_color_values = {};
        
        for v in error_mesh.data.vertices:            
            try:
                (r,g,b,a) = final_colors[v.index];
            except ValueError:
                (r,g,b) = final_colors[v.index];            
            color = Color((r,g,b));
            L_error_color_values[v.index] = color;
            colors[v.index] = final_weights[v.index];
        
        if(None == error_mesh.vertex_groups.get(v_group_name)):
            error_mesh.vertex_groups.new(name=v_group_name);
        
        if(None == error_mesh.data.vertex_colors.get(v_group_name)):
            error_mesh.data.vertex_colors.new(v_group_name);
            
        group_ind = error_mesh.vertex_groups[v_group_name].index;
        lap_error_colors = error_mesh.data.vertex_colors[v_group_name];
        
        bm = getBMMesh(context, error_mesh, False);
        ensurelookuptable(bm);
        
        for v in error_mesh.data.vertices:
            n = v.index;
            error_mesh.vertex_groups[group_ind].add([n], colors[v.index], 'REPLACE');
            
            b_vert = bm.verts[v.index];
            
            for l in b_vert.link_loops:
                lap_error_colors.data[l.index].color = L_error_color_values[v.index];
            
        bm.free();
        
        try:
            material = bpy.data.materials[error_mesh.name+'_'+v_group_name];
        except KeyError:
            material = bpy.data.materials.new(error_mesh.name+'_'+v_group_name);
        
        try:
            error_mesh.data.materials[error_mesh.name+'_'+v_group_name];
        except KeyError:
            error_mesh.data.materials.append(material);
#         print('MATERIALS ::: ', error_mesh.data.materials);
        material.use_vertex_color_paint = True;   

def exportMeshColors(context, mesh, vertex_colors_name, base_location, exportname,*, retain_location=False):
    filepath = bpy.path.abspath(base_location + "/"+exportname+".ply");                 
    bpy.ops.object.select_all(action="DESELECT");
    
    mesh.data.vertex_colors.active = mesh.data.vertex_colors[vertex_colors_name];
    context.scene.objects.active = mesh;
    mesh.select = True;    
    o_location = mesh.location.copy();
    if(not retain_location):
        mesh.location = (0,0,0);    
    bpy.ops.export_mesh.ply(filepath=filepath, check_existing=False, axis_forward='-Z', axis_up='Y', filter_glob="*.ply", use_mesh_modifiers=True, use_normals=True, use_uv_coords=True, use_colors=True, global_scale=1.0);
    mesh.location = o_location;
    bpy.ops.object.select_all(action="DESELECT");




     