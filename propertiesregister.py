import bpy;
from mathutils import Vector;
from mathutils.bvhtree import BVHTree;

from GenericMarkerCreator.misc.staticutilities import getMarkerOwner, getGenericLandmark;
from GenericMarkerCreator.misc.meshmathutils import getBarycentricCoordinateFromPolygonFace, getGeneralCartesianFromBarycentre;
from GenericMarkerCreator.misc.mathandmatrices import getDuplicatedObject;

#To enable edit/removal button in the custom properties panel of Blender
#its necessary to remove the property assignment to entities such as scene, object etc,
#From the register routine.

def updateSpectralProperty(self, context):
#     print('WHO AM I : ', self);
    if(self.spectral_soft_update):
        return;
    if(self.live_hks):
        bpy.ops.genericlandmarks.spectralhks('EXEC_DEFAULT', currentobject=self.name);
    elif(self.live_wks):
        bpy.ops.genericlandmarks.spectralwks('EXEC_DEFAULT', currentobject=self.name);
    elif(self.live_spectral_shape):
        bpy.ops.genericlandmarks.spectralshape('EXEC_DEFAULT', currentobject=self.name);
    elif(self.live_gisif):
        bpy.ops.genericlandmarks.spectralgisif('EXEC_DEFAULT', currentobject=self.name);
        
def updateNormalMarkerColor(self, context):
    changeMarkerColor(context.active_object);

def updateUnlinkedMarkerColor(self, context):
    changeUnlinkedMarkerColor(context.active_object);

def getLandmarkName(self):
#     print('WHO AM I : ', self);
    mesh, isM, isN = getMarkerOwner(self);
    if(mesh):
        data_landmark = getGenericLandmark(mesh, self);
        if(data_landmark):
            return data_landmark.landmark_name;
    
    return "No Name";

def setLandmarkName(self, value):
    mesh, isM, isN = getMarkerOwner(self);
    if(mesh):
        data_landmark = getGenericLandmark(mesh, self);
        if(data_landmark):
            data_landmark.landmark_name = value;
    
    self['edit_landmark_name'] = value;
    
def updateLandmarkName(self, context):
#     print('WHO AM I : ', self);
    mesh, isM, isN = getMarkerOwner(self);
    if(mesh):
        data_landmark = getGenericLandmark(mesh, self);
        if(data_landmark):
            data_landmark.landmark_name = self.edit_landmark_name;
        
def changeMarkerColor(mesh, bmarker = None):    
    try:
        material = bpy.data.materials[mesh.name+'_LinkedMaterial'];
    except:
        material = bpy.data.materials.new(mesh.name+'_LinkedMaterial');
    
    material.diffuse_color = (mesh.linked_landmark_color[0], mesh.linked_landmark_color[1], mesh.linked_landmark_color[2]);
    material.alpha = 1;
    material.specular_color = (mesh.linked_landmark_color[0], mesh.linked_landmark_color[1], mesh.linked_landmark_color[2]);
    
    if(bmarker):
        bmarker.data.materials.clear();
        bmarker.data.materials.append(material);
    
    return material;
    
def changeUnlinkedMarkerColor(mesh = None, bmarker = None):    
    try:
        material = bpy.data.materials[mesh.name+'_UnlinkedMaterial'];
    except:
        material = bpy.data.materials.new(mesh.name+'_UnlinkedMaterial');
    
    material.diffuse_color = (mesh.unlinked_landmark_color[0], mesh.unlinked_landmark_color[1], mesh.unlinked_landmark_color[2]);
    material.alpha = 1;
    material.specular_color = (mesh.unlinked_landmark_color[0], mesh.unlinked_landmark_color[1], mesh.unlinked_landmark_color[2]);
    
    if(bmarker):
        bmarker.data.materials.clear();
        bmarker.data.materials.append(material);
    
    return material;

def get_scene_meshes(self, context):
    templatenames = ["_marker","_joint","_bone","_lines","_cloud", "Template", "Marker", "Landmark"];
    return [(item.name, item.name, item.name) for item in bpy.data.objects if item.type == "MESH" and not any(word in item.name for word in templatenames)];

def get_marker_meshes(self, context):
    templatenames = ["_marker","_joint","_bone","_lines","_cloud"];
    return [(item.name, item.name, item.name) for item in bpy.data.objects if item.type == "MESH" and not any(word in item.name for word in templatenames) and item.hide_select and item.hide_render];

def showHideConstraints(self, context):
    bpy.ops.genericlandmarks.createlandmarks('EXEC_DEFAULT',hidemarkers=True, currentobject = self.name);

class GenericLandmark(bpy.types.PropertyGroup):
    is_linked = bpy.props.BoolProperty(name="Is Linked", description="Flag to check if a landmark is linked", default=False);
    id = bpy.props.IntProperty(name="Landmark Id", description="Index or indentifier that is unique for this landmark", default=-1);
    linked_id = bpy.props.IntProperty(name="Linked Constraint Id", description="Index or indentifier of the unique indexed constraint to which this landmark is mapped.", default=-1);    
    faceindex = bpy.props.IntProperty(name="Triangle Index", description="Index or indentifier of the triangle on which this landmark is placed.", default=-1);
    v_indices = bpy.props.IntVectorProperty(name="Vertex indices", description="Vertex indices on which the barycentric ratios have to be applied",default=(-1, -1, -1));
    v_ratios = bpy.props.FloatVectorProperty(name="Barycentric ratios", description="Given the vertex indices (==3) apply the barycentric ratios for the location of the marker",default=(0.0,0.0,0.0));
    location = bpy.props.FloatVectorProperty(name="Location of Landmark",default=(0.0,0.0,0.0));
    landmark_name = bpy.props.StringProperty(name="Landmark Name", default="No Name");

class GenericPointSignature(bpy.types.PropertyGroup):
    index = bpy.props.IntProperty(name="Index", description="Index or indentifier that is unique for this landmark", default=-1);
    name = bpy.props.StringProperty(name="Signature Name", default="");
    
    gisif = bpy.props.FloatProperty(name = 'GISIF', description="The signature for this point", default=0.00);    
    k1 = bpy.props.FloatProperty(name = 'k1', description="The k1 signature for this point", default=0.00);
    k2 = bpy.props.FloatProperty(name = 'k2', description="The k2 signature for this point", default=0.00);
    
    normal = bpy.props.FloatVectorProperty(name = 'Curvature Normal', description="The normal signature for this point", default=(0.0, 0.0, 0.0));
    p1 = bpy.props.FloatVectorProperty(name = 'Principal Direction 1', description="The p1 signature for this point", default=(0.0, 0.0, 0.0));
    p2 = bpy.props.FloatVectorProperty(name = 'Principal Direction 2', description="The p2 signature for this point", default=(0.0, 0.0, 0.0));

class VertexMapping(bpy.types.PropertyGroup):
    bary_indices = bpy.props.IntVectorProperty(name="Vertex indices", description="Vertex indices on which the barycentric ratios have to be applied",default=(-1, -1, -1));
    bary_ratios = bpy.props.FloatVectorProperty(name="Barycentric ratios", description="Given the vertex indices (==3) apply the barycentric ratios for the location of mapped point",default=(0.0,0.0,0.0));
    is_valid = bpy.props.BoolProperty(name="Is Valid?", description="Is this a valid mapping point", default=True);
    
class VertexToSurfaceMappingBVH(bpy.types.PropertyGroup):
    map_to_mesh = bpy.props.StringProperty(name='Map To', description="Map to Mesh surface name", default='--');
    mapped_points = bpy.props.CollectionProperty(type=VertexMapping);
    apply_on_duplicate = bpy.props.BoolProperty(name="Apply Duplicate", description="Apply the mapping to deform on duplicate", default=True);
    
    def constructMapping(self):
        c = bpy.context;
        owner_mesh = self.id_data;
        try:
            map_to = c.scene.objects[self.map_to_mesh];
            btree = BVHTree.FromObject(map_to, c.scene);
            n_count = len(owner_mesh.data.vertices);
            for vid, v in enumerate(owner_mesh.data.vertices):
                mapped_point = self.mapped_points.add();
                x, y, z = v.co.to_tuple();
                co, n, i, d = btree.find_nearest(Vector((x,y,z)));
                if(co and n and i and d):
                    face = map_to.data.polygons[i];
                    u,v,w,ratio, isinside, vid1, vid2, vid3 = getBarycentricCoordinateFromPolygonFace(co, face, map_to, snapping=False, extra_info = True);
                    mapped_point.bary_ratios = [u, v, w];
                    mapped_point.bary_indices = [vid1, vid2, vid3];
                else:
                    mapped_point.is_valid = False
        except KeyError:
            print('MAP TO MESH NEEDS TO BE SET BEFORE CONSTRUCTING A MAPPING');
            raise;
        
    def deformWithMapping(self):
        c = bpy.context;
        owner_mesh = self.id_data;
        try:
            map_to = c.scene.objects[self.map_to_mesh];
            apply_on_mesh = owner_mesh;
            
            if(self.apply_on_duplicate):
                try:
                    apply_on_mesh = c.scene.objects["%s-mapped-shape-%s"%(owner_mesh.name, self.name)];
                except KeyError:
                    apply_on_mesh = getDuplicatedObject(c, owner_mesh, meshname="%s-mapped-shape-%s"%(owner_mesh.name, self.name));
                    
            for vid, mapped_point in enumerate(self.mapped_points):
                if(mapped_point.is_valid):
                    co = getGeneralCartesianFromBarycentre(mapped_point.bary_ratios, [map_to.data.vertices[m_vid].co for m_vid in mapped_point.bary_indices]);
                    apply_on_mesh.data.vertices[vid].co = co;
            return apply_on_mesh;
        
        except KeyError:
            print('MAP TO MESH NEEDS TO BE SET BEFORE CONSTRUCTING A MAPPING');
            raise;
        
        
        
    def copyMappingToMesh(self, to_mesh):        
        from_mesh = self.id_data;
        try:
            assert (len(from_mesh.data.vertices) == len(to_mesh.data.vertices));
        except AssertionError:
            print(('Meshes %s and %s should have same connectivity'%(from_mesh.name, to_mesh.name)).upper());
            raise;
        
        mapping = to_mesh.surfacemappingsbvh.add();
        mapping.name = self.name;
        mapping.map_to_mesh = self.map_to_mesh;
        
        for mapped_point in self.mapped_points:
            mpoint = mapping.mapped_points.add();
            mpoint.is_valid = mapped_point.is_valid;
            if(mpoint.is_valid):
                mpoint.bary_indices = [id for id in mapped_point.bary_indices];
                mpoint.bary_ratios = [r for r in mapped_point.bary_ratios]; 
        
        return mapping;
    
def register():
    bpy.utils.register_class(GenericLandmark);    
    bpy.utils.register_class(GenericPointSignature);        
    bpy.utils.register_class(VertexMapping);  
    bpy.utils.register_class(VertexToSurfaceMappingBVH);  
    
    bpy.types.Object.snap_landmarks = bpy.props.BoolProperty(name="Snap Landmarks", description="Flag to enable/disable snapping", default=False);
        
    bpy.types.Object.is_landmarked_mesh = bpy.props.BoolProperty(name="Is Landmarked Mesh", description="Flag to identify meshes with landmarks", default=False);
    bpy.types.Object.hide_landmarks = bpy.props.BoolProperty(name="Hide Landmarks", description="Flag to show or hide landmarks in a mesh", default=False, update=showHideConstraints);
    
    bpy.types.Object.is_visual_landmark = bpy.props.BoolProperty(name="Is Visual Landmark", description="Flag to identify if the mesh is an object used for showing a landmark visually", default=False);
    bpy.types.Object.edit_landmark_name = bpy.props.StringProperty(name="Edit Landmark Name", description="Change Landmark name", default="No Name", set=setLandmarkName, get=getLandmarkName);#, update=updateLandmarkName);
    bpy.types.Object.belongs_to = bpy.props.StringProperty(name="Belongs To", description="The name of the mesh to which the landmark has been added", default="---");#, update=updateLandmarkName);
    
    bpy.types.Object.linked_landmark_color = bpy.props.FloatVectorProperty(name = "Landmark Color",subtype='COLOR',default=[0.0,1.0,0.0], description = "Color of a linked landmark",update=updateNormalMarkerColor);    
    bpy.types.Object.unlinked_landmark_color = bpy.props.FloatVectorProperty(name = "Unlinked Landmark Color",subtype='COLOR',default=[1.0,0.0,0.0],description = "The color of an unlinked landmark.", update=updateUnlinkedMarkerColor);
    
    bpy.types.Object.mapped_mesh = bpy.props.StringProperty(name="Mapped mesh",description="The Blender name of the mapped mesh on whom landmarks are linked",default="");
    
    bpy.types.Object.landmark_id = bpy.props.IntProperty(name = "Landmark Id",description = "The original id of a landmark",default = -1);
    bpy.types.Object.landmark_array_index = bpy.props.IntProperty(name = "Landmark Array Index",description = "The positional index of the marker in the array",default = -1);    
    bpy.types.Object.total_landmarks = bpy.props.IntProperty(name="Total Landmarks", description="Total number of landmarks for this mesh", default=0);
    
    bpy.types.Object.spectral_soft_update = bpy.props.BoolProperty(name='Spectral Soft', description="flag to avoid update methods for spectral to run", default=False);
    bpy.types.Object.eigen_k = bpy.props.IntProperty(name="Eigen K", description="Number of Eigen Ranks to solve", default=5, min=1, step=1, update=updateSpectralProperty);
    bpy.types.Object.spectral_sync = bpy.props.BoolProperty(name='Spectral Sync', description="flag to sync spectral properties with paired mesh", default=False);
    
    
    bpy.types.Object.hks_t = bpy.props.FloatProperty(name="HKS Time", description="The time for which the heat dissipation for every point is calculated", default=20.0, min=0.1, update=updateSpectralProperty);
    bpy.types.Object.hks_current_t = bpy.props.IntProperty(name="HKS Current Time", description="The current time of heat dissipation", default=20, min=0, update=updateSpectralProperty);
    
    bpy.types.Object.wks_e = bpy.props.IntProperty(name="WKS Evalautions", description="The Total evaluations for which WKS is calculated", default=100, min=2, step=1,update=updateSpectralProperty);
    bpy.types.Object.wks_current_e = bpy.props.IntProperty(name="WKS Current Evalaution", description="The current evaluation for which WKS is shown", default=0, min=0, step=1,update=updateSpectralProperty);
    bpy.types.Object.wks_variance = bpy.props.FloatProperty(name="WKS variance", description="The WKS variance to consider", default=6.0, min=0.00, update=updateSpectralProperty);
    
    bpy.types.Object.gisif_group_name = bpy.props.StringProperty(name='GISIF Group', description="The current GISIF Group and the clusters", default="");
    bpy.types.Object.gisif_group_index = bpy.props.IntProperty(name="GISIF Group", description="For a Threshold applied choose the GISIF Group to show", default=0, min=0, step=1,update=updateSpectralProperty);
    bpy.types.Object.gisif_threshold = bpy.props.FloatProperty(name="GISIF Threshold", description="The threshold for eigen values to group them as repeated", default=0.1, max=10000.0, min=0.0, update=updateSpectralProperty);
    
    bpy.types.Object.linear_gisif_combinations = bpy.props.BoolProperty(name='Linear GISIFS?', description="Use Linear GISIF combinations of gisifs from i to i+n", default=False, update=updateSpectralProperty);
    bpy.types.Object.linear_gisif_n = bpy.props.IntProperty(name='Linear GISIF (i+1)', description="Linear GISIF means combinations of gisifs from i to i+n", default=0, min=0, update=updateSpectralProperty);
    
    bpy.types.Object.gisif_symmetry_index = bpy.props.IntProperty(name='GISIF Symmetry Index', description="For the marker selected find the symmetrical gisifs", default=0, update=updateSpectralProperty);
    bpy.types.Object.gisif_markers_n = bpy.props.IntProperty(name='Spectral Clusters', description="With the computed Spectral signatures generate the lnadmarks for the count specified using GMM clustering", default=1, min=1);
    bpy.types.Object.gisif_symmetries = bpy.props.BoolProperty(name='GISIF Symmetries', description="Once the GISIFS are clustered add their symmetries also ", default=True);
    
    
    bpy.types.Object.live_hks = bpy.props.BoolProperty(name='Live HKS', description="Live HKS means reflect the changes in the scene immediately after values are changed (Eigen K or HKS Time)", default=False);
    bpy.types.Object.live_wks = bpy.props.BoolProperty(name='Live WKS', description="Live WKS means reflect the changes in the scene immediately after values are changed (Eigen K or HKS Time)", default=False);
    bpy.types.Object.live_spectral_shape = bpy.props.BoolProperty(name='Live Spectral Shape', description="Perform Live spectral shape", default=False);
    bpy.types.Object.live_gisif = bpy.props.BoolProperty(name='Live GISIF', description="Live GISIF means reflect the changes in the scene immediately after values are changed (Treshold or Group Index)", default=False);
       
    
    bpy.types.Object.generic_landmarks = bpy.props.CollectionProperty(type=GenericLandmark);
    
    bpy.types.Object.hks_signatures = bpy.props.CollectionProperty(type=GenericPointSignature);
    bpy.types.Object.wks_signatures = bpy.props.CollectionProperty(type=GenericPointSignature);
    bpy.types.Object.gisif_signatures = bpy.props.CollectionProperty(type=GenericPointSignature);
    bpy.types.Object.surfacemappingsbvh = bpy.props.CollectionProperty(type=VertexToSurfaceMappingBVH);
    
    bpy.types.Object.signatures_dir = bpy.props.StringProperty(name="Directory Signatures", description="Directory where the signatures and the pca space is saved", subtype="DIR_PATH", default="//");
    
    bpy.types.Scene.use_mirrormode_x = bpy.props.BoolProperty(name="Mirror Mode X", description="Use mirror mode on X-Axis", default=True);
    bpy.types.Scene.landmarks_use_selection = bpy.props.EnumProperty(name = "Landmarks List", items = get_marker_meshes, description = "Meshes available in the Blender scene to be used for as landmark mesh");
    
def unregister():    
    bpy.utils.unregister_class(GenericLandmark);
    bpy.utils.unregister_class(GenericPointSignature);
    bpy.utils.unregister_class(VertexMapping);  
    bpy.utils.unregister_class(VertexToSurfaceMappingBVH);  
    