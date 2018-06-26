import bpy;

from GenericMarkerCreator.misc.staticutilities import getMarkerOwner, getGenericLandmark

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
    v_indices = bpy.props.IntVectorProperty(name="Vertex indices", description="Vertex indices on which the barycentric ratios have to be applied",default=(-1, -1, -1));
    v_ratios = bpy.props.FloatVectorProperty(name="Barycentric ratios", description="Given the vertex indices (==3) apply the barycentric ratios for the location of the marker",default=(0.0,0.0,0.0));
    location = bpy.props.FloatVectorProperty(name="Location of Landmark",default=(0.0,0.0,0.0));
    landmark_name = bpy.props.StringProperty(name="Landmark Name", default="No Name");
    
        
def register():
    bpy.utils.register_class(GenericLandmark);    
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
    
    bpy.types.Object.generic_landmarks = bpy.props.CollectionProperty(type=GenericLandmark);
    
    bpy.types.Scene.use_mirrormode_x = bpy.props.BoolProperty(name="Mirror Mode X", description="Use mirror mode on X-Axis", default=True);
    bpy.types.Scene.landmarks_use_selection = bpy.props.EnumProperty(name = "Landmarks List", items = get_marker_meshes, description = "Meshes available in the Blender scene to be used for as landmark mesh");
    
def unregister():
    bpy.utils.unregister_class(GenericLandmark);
    
    
    