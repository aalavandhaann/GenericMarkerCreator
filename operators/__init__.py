# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
# from sfmsuite.geodesics.GeodesicOperator import SimpleGeodesic;

bl_info = {
    "name": "Generic Landmarks: Operators",
    "description": "Suite of operators that implement the required functionalities for creating landmarks",
    "author": "#0K Srinivasan",
    "version": (0, 0, 1), 
    "blender": (2, 7, 4), 
    "location": "View3D > Tools > Generic Landmarks", 
    "warning": "Addon not complete, still under development", 
    "wiki_url": "",
    "category": "Object" }


import bpy;
from GenericMarkerCreator.operators.MessageBox import GenericLandmarksMessageBox;
from GenericMarkerCreator.operators.LandmarksPair import AssignMeshPair;
from GenericMarkerCreator.operators.LiveOperators import LiveLandmarksCreator, SignaturesMatching;
from GenericMarkerCreator.operators.LandmarksCreator import CreateLandmarks, ReorderLandmarks, \
ChangeLandmarks, UnLinkLandmarks, LinkLandmarks, RemoveLandmarks, LandmarkStatus, LandmarksPairFinder, TransferLandmarkNames, AutoLinkLandmarksByID
from GenericMarkerCreator.operators.SpectralOperations import SpectralHKS, SpectralWKS, SpectralGISIF, SpectralShape, AddSpectralSignatures, AddSpectralSignatureLandmarks, SpectralFeatures;



def register():
    print('OPERATORS BEING REGISTERED -> START');
    bpy.utils.register_class(GenericLandmarksMessageBox);
    bpy.utils.register_class(AssignMeshPair);
    bpy.utils.register_class(CreateLandmarks);
    bpy.utils.register_class(ReorderLandmarks);
    bpy.utils.register_class(ChangeLandmarks);
    bpy.utils.register_class(UnLinkLandmarks);
    bpy.utils.register_class(LinkLandmarks);
    bpy.utils.register_class(RemoveLandmarks);
    bpy.utils.register_class(LandmarkStatus);
    bpy.utils.register_class(LandmarksPairFinder);
    bpy.utils.register_class(TransferLandmarkNames);
    bpy.utils.register_class(LiveLandmarksCreator);
    bpy.utils.register_class(SignaturesMatching);
    bpy.utils.register_class(AutoLinkLandmarksByID);
    bpy.utils.register_class(SpectralHKS);
    bpy.utils.register_class(SpectralWKS);
    bpy.utils.register_class(SpectralGISIF);    
    bpy.utils.register_class(SpectralShape);
    bpy.utils.register_class(AddSpectralSignatures);
    bpy.utils.register_class(AddSpectralSignatureLandmarks);
    bpy.utils.register_class(SpectralFeatures);    
    print('OPERATORS BEING REGISTERED -> END');

def unregister():
    print('OPERATORS BEING UNREGISTERED -> START');
    bpy.utils.unregister_class(GenericLandmarksMessageBox);
    bpy.utils.unregister_class(AssignMeshPair);
    bpy.utils.unregister_class(CreateLandmarks);
    bpy.utils.unregister_class(ReorderLandmarks);
    bpy.utils.unregister_class(ChangeLandmarks);
    bpy.utils.unregister_class(UnLinkLandmarks);
    bpy.utils.unregister_class(LinkLandmarks);
    bpy.utils.unregister_class(RemoveLandmarks);
    bpy.utils.unregister_class(LandmarkStatus);
    bpy.utils.unregister_class(LandmarksPairFinder);
    bpy.utils.unregister_class(TransferLandmarkNames);
    bpy.utils.unregister_class(LiveLandmarksCreator);
    bpy.utils.unregister_class(SignaturesMatching);
    bpy.utils.unregister_class(AutoLinkLandmarksByID);
    bpy.utils.unregister_class(SpectralHKS);
    bpy.utils.unregister_class(SpectralWKS);
    bpy.utils.unregister_class(SpectralGISIF);
    bpy.utils.unregister_class(SpectralShape);
    bpy.utils.unregister_class(AddSpectralSignatures);
    bpy.utils.unregister_class(AddSpectralSignatureLandmarks);
    bpy.utils.unregister_class(SpectralFeatures);    
    print('OPERATORS BEING UNREGISTERED -> END');