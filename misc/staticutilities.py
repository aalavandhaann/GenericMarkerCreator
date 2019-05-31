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

from matplotlib.colors import LinearSegmentedColormap, ListedColormap;


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

def detectMorN(mesh):
    M, N = detectMN(mesh);
    
    if(M and N):
#         print(M.name, N.name, mesh.name)
        if(mesh.name == M.name):
#             print('RETURN : ', N.name);
            return N;
#         print('RETURN : ', M.name);
        return M;
    
    return None;

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

def addConstraint(context, mesh, bary_ratios, bary_indices, co, *, should_reorder=False, faceindex = -1, useid=-1, create_visual_landmarks = True):
    current_ids = [gm.id for gm in mesh.generic_landmarks];
    
    try:
        use_id = int(max(current_ids) + 1);
    except ValueError:
        use_id = mesh.total_landmarks;
        
    if(useid != -1):
        if(useid in current_ids):
            conflicting_id = current_ids[current_ids.index(useid)];
            error_message = 'The given id: %d in argument conflicts with an existing landmark with id: %d.'%(useid, conflicting_id);
            raise ValueError(error_message);
        use_id = useid;
    
    m = mesh.generic_landmarks.add();        
    m.id = use_id;
    m.linked_id = -1;
    m.faceindex = faceindex;
    m.is_linked = False;
    m.v_indices = bary_indices;
    m.v_ratios = bary_ratios;
    m.location = [co.x, co.y, co.z];
    m.landmark_name = 'Original Id: %s'%(m.id);
    
    mesh.total_landmarks = len(mesh.generic_landmarks);
    
    if(should_reorder):
        bpy.ops.genericlandmarks.reorderlandmarks('EXEC_DEFAULT',currentobject=mesh.name);
    else:
        if(create_visual_landmarks):
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
    print('GET ALL LANDMARKS ON EDGE OR FACE');
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
    print('POKE THE FACES FIRST');
    ensurelookuptable(bm);
    faces = [bm.faces[ind] for ind in face_indices];
    returned_geometry_faces = bmesh.ops.poke(bm, faces=faces);
    
    for i, vert in enumerate(returned_geometry_faces['verts']):
        verts_and_locations.append((vert.index, face_locations[i]));
    
    print('CUT THE EDGES NOW');
    ensurelookuptable(bm);
    edges = [bm.edges[ind] for ind in edge_indices];
    returned_geometry_edges = bmesh.ops.bisect_edges(bm, edges=edges, cuts=1);
    returned_vertices = [vert for vert in returned_geometry_edges['geom_split'] if isinstance(vert, bmesh.types.BMVert)];
        
    for i, vert in enumerate(returned_vertices):
        verts_and_locations.append((vert.index, edge_locations[i]));
            
    ensurelookuptable(bm);
    print('TRIANGULATING THE MESH AS THE LAST STEP FOR MESH');
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0);
    print('NOW MAKE THIS NEW TOPOLOGY TO BE THE MESH');
    bm.to_mesh(mesh.data);
    
    bm.free();
   
    for vid, co in verts_and_locations:
        mesh.data.vertices[vid].co = co;
    print('AUTOCORRECT THE MESH LANDMARKS WITH NEW TOPOLOGY');    
    autoCorrectLandmarksData(context, mesh);
    
def autoCorrectLandmarksData(context, mesh):
    print('AUTOCORRECT: CONSTRUCT KDTREE');
    kd = getKDTree(context, mesh);
    print('AUTOCORRECT: CONSTRUCT BMESH DATA');
    bm = getBMMesh(context, mesh, False);
    print('AUTOCORRECT: CONSTRUCT BMESH DATA ENSURETABLE');
    ensurelookuptable(bm);
    
    print('AUTOCORRECT: ITERATE LANDMARKS AND FIX POSITION');
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
    
    print('AUTOCORRECT: FREE THE BMESH DATA');
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


def getInterpolatedColorValues(error_values, A = None, B = None, *, normalize=True):
    step_colors = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]];
    norm = clrs.Normalize(vmin=A, vmax=B);    
#     cmap = get_cmap('jet');
#     cmap = ListedColormap(step_colors);
#     cmap = get_cmap('Spectral');
    c = error_values;    
    final_weights = norm(c);
    final_colors = cmap(final_weights)[:, 0:3];
    if(normalize):
        return final_colors, final_weights;
    return final_colors, error_values;

def applyVertexWeights(context, mesh, weights,*, v_group_name = "lap_errors"):
    if(None == mesh.vertex_groups.get(v_group_name)):
        mesh.vertex_groups.new(name=v_group_name);
    
    group_ind = mesh.vertex_groups[v_group_name].index;
    vertex_group = mesh.vertex_groups[group_ind];
    
    bm = getBMMesh(context, mesh, False);
    ensurelookuptable(bm);
    
    for v in mesh.data.vertices:
        n = v.index;
        vertex_group.add([n], weights[v.index], 'REPLACE');
        b_vert = bm.verts[v.index];               
    bm.free();
    
    return vertex_group;


def applyVertexColors(context, mesh, colors,*, v_group_name = "lap_errors", for_vertices = True):
    if(None == mesh.data.vertex_colors.get(v_group_name)):
        mesh.data.vertex_colors.new(v_group_name);
    
    vertex_colors = mesh.data.vertex_colors[v_group_name];
    vertex_colors.active = True;
    try:
        material = bpy.data.materials[mesh.name+'_'+v_group_name];
    except KeyError:
        material = bpy.data.materials.new(mesh.name+'_'+v_group_name);
    
    try:
        mesh.data.materials[mesh.name+'_'+v_group_name];
    except KeyError:
        mesh.data.materials.append(material);
    
    if(for_vertices):
        bm = getBMMesh(context, mesh, False);
        ensurelookuptable(bm);        
        for v in mesh.data.vertices:            
            b_vert = bm.verts[v.index];            
            for l in b_vert.link_loops:
                vertex_colors.data[l.index].color = colors[v.index];                
        bm.free();
    
    else:        
        for f in mesh.data.polygons:
            for lid in f.loop_indices:
                vertex_colors.data[lid].color = colors[f.index];
    
    
    material.use_vertex_color_paint = True;
    
    return vertex_colors, material;

def getBinEdgeValue(N, hist, edges, fraction):
    sum, partsum = int(N*fraction), 0;
    i = 0;
    for i, h_count in enumerate(hist):
        partsum += h_count;
        if(partsum >= sum):
            break;
    return edges[i+1];

def getMinMax(data, histsize=10000):
    N = data.shape[0];
    hist, edges = np.histogram(data, bins=histsize);
    min_, max_ = np.min(data), np.max(data);
    threshold = histsize / 5;
    maxcount = np.max(hist);
    if(maxcount > threshold):
        #sorted_data = np.sort(data);
        Nby100 = N / 100;
        left_index = int(Nby100);
        right_index = N - left_index;
        hist, edges = np.histogram(data, bins=histsize*50);
        min_, max_ = getBinEdgeValue(N, hist, edges, 0.1), getBinEdgeValue(N, hist, edges, 0.9);
    return min_, max_;

def applyColoringForMeshErrors(context, error_mesh, error_values, *, A = None, B = None, v_group_name = "lap_errors", use_weights=False, normalize_weights=True, use_histogram_preprocess=False): 
    if(use_histogram_preprocess):
        min_, max_ = getMinMax(error_values);
        error_values[error_values <= min_] = min_;
        error_values[error_values >= max_] = max_;
    
    final_colors, final_weights = getInterpolatedColorValues(error_values, A, B, normalize=normalize_weights);
    
    colors = {};
    weights = {};
    
    iterator_model = [];    
    for_vertices = not (len(error_values) == len(error_mesh.data.polygons));
    
    if(for_vertices):
        iterator_model = error_mesh.data.vertices;
    else:
        iterator_model = error_mesh.data.polygons;
    
    for it_elem in iterator_model:            
        try:
            (r,g,b,a) = final_colors[it_elem.index];
        except ValueError:
            (r,g,b) = final_colors[it_elem.index];            
        color = Color((r,g,b));
        colors[it_elem.index] = color;
        weights[it_elem.index] = final_weights[it_elem.index];
    
    if(for_vertices and use_weights):
        applyVertexWeights(context, error_mesh, weights, v_group_name = v_group_name);
    
    applyVertexColors(context, error_mesh, colors, v_group_name=v_group_name, for_vertices=for_vertices);
    

def exportMeshColors(context, mesh, vertex_colors_name, base_location, exportname,*, retain_location=False):
#     filepath = bpy.path.abspath(base_location + "/"+exportname+".ply");
    filepath = os.path.join(base_location, exportname+".ply");
    bpy.ops.object.select_all(action="DESELECT");
    
    mesh.data.vertex_colors.active = mesh.data.vertex_colors[vertex_colors_name];
    context.scene.objects.active = mesh;
    mesh.select = True;    
    o_location = mesh.location.copy();
    if(not retain_location):
        mesh.location = (0,0,0);    
    bpy.ops.export_mesh.ply(filepath=filepath, check_existing=False, axis_forward='-Z', axis_up='Y', filter_glob="*.ply", use_mesh_modifiers=False, use_normals=False, use_uv_coords=False, use_colors=True, global_scale=1.0);
    mesh.location = o_location;
    bpy.ops.object.select_all(action="DESELECT");




     