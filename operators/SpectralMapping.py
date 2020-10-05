import bpy;

from scipy.sparse.linalg import lsqr;
import scipy.sparse as spsp;
from mathutils import Vector, Matrix;
from mathutils.bvhtree import BVHTree;
from scipy.spatial import cKDTree;

from GenericMarkerCreator.misc.staticutilities import detectMN;
from GenericMarkerCreator.misc.mathandmatrices import getBMMesh, ensurelookuptable;
from GenericMarkerCreator.misc.mathandmatrices import getDuplicatedObject;
from GenericMarkerCreator.misc.mathandmatrices import getMeshVPos, setMeshVPOS, getMeshFaces,  getEdgeVertices, getWKSLaplacianMatrixCotangent, getColumnFilledMatrix;
from GenericMarkerCreator.misc.spectralmagic import doLowpassFiltering;


'''
@inproceedings{srinivasanspectralearscans,
  author =       {Ramachandran, Srinivasan and Paquette, Eric and Popa, Tiberiu},
  title =        {Constraint-Based Spectral Space Template Deformation for Ear Scans},
  year =  {2020},
  booktitle =  "Proceedings of Graphics Interface 2020",
}
'''

#The operators for creating landmarks
class SpectralSpaceDeformer(bpy.types.Operator):
    bl_idname = "genericlandmarks.spectralspacedeformer";
    bl_label = "Spectral Space Deformer";
    bl_description = "Find mapping between two meshes using Constraint-Based Spectral Space Template Deformation for Ear Scans";
    bl_space_type = "VIEW_3D";
    bl_region_type = "UI";
    bl_context = "objectmode";
    currentobject = bpy.props.StringProperty(name="Initialize for Object", default = "--");    
    
    def getSpectralShape(self, context, mesh, eigen_k=5, need_A=False, eigenvectors=False):
        try:
            dup_mesh = context.scene.objects["%s-spectral-shape"%(mesh.name)];
        except KeyError:
            dup_mesh = getDuplicatedObject(context, mesh, meshname="%s-spectral-shape"%(mesh.name));
            bpy.ops.object.select_all(action="DESELECT");
            context.scene.objects.active = mesh;
            mesh.select = True;
        
        if(need_A):
            vpos, A = doLowpassFiltering(context, mesh, eigen_k, a_matrix=need_A);
            setMeshVPOS(dup_mesh, vpos);
            return dup_mesh, vpos, A;
        if(eigenvectors):
            vpos, UT = doLowpassFiltering(context, mesh, eigen_k, eigenvectors=True);
            setMeshVPOS(dup_mesh, vpos);
            return dup_mesh, vpos, UT;
        else:
            vpos = doLowpassFiltering(context, mesh, eigen_k, a_matrix=need_A);
            setMeshVPOS(dup_mesh, vpos);
            return dup_mesh, vpos;
    
    
    def doSpectralSpaceMapping(self, context, M, N):
        linked_landmarks_M = [lm.linked_id for lm in M.generic_landmarks if lm.linked_id != -1];
        if(not len(linked_landmarks_M)):
            return -1;
        
        M_original_vpos = getMeshVPos(M);
        M_V_N = M_original_vpos.shape[0];
        L_M = getWKSLaplacianMatrixCotangent(context, M);
        delta_M = L_M.dot(M_original_vpos);
        
        s, original_eigen_source, UT = self.getSpectralShape(context, M, M.eigen_k, eigenvectors=True);
        t , original_eigen_target = self.getSpectralShape(context, N, M.eigen_k,);
        
        
        U = spsp.csr_matrix(UT.T);
        
        
    
    def execute(self, context):
        try:            
            mesh = bpy.data.objects[self.currentobject];
        except:
            mesh = context.active_object;
            
        if(mesh is not None):
            M, N = detectMN(mesh);
            if(not M or not N):
                message = "You need two meshes for finding a mapping between them.";
                bpy.ops.genericlandmarks.messagebox('INVOKE_DEFAULT',messagetype='ERROR',message=message,messagelinesize=60);
                return {'FINISHED'};
            
            else:
                self.doSpectralSpaceMapping(context, M, N);
        
        return {'FINISHED'};