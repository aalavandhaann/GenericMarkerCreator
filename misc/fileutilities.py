import os;
try:
    import bpy, bmesh;
    from mathutils import Vector, Color;
except ModuleNotFoundError:
    print('WORKING OUTSIDE BLENDER ENVIRONMENT');
    
import numpy as np;
from scipy.interpolate.interpolate_wrapper import logarithmic
import matplotlib.cm as cm;
from matplotlib.cm import get_cmap;
import matplotlib.colors as clrs;

def getFilesFromDirectory(directorypath, extension='.obj', exclusions=[]):
    found_files = [];
    for root, directories, files in os.walk(os.path.abspath(directorypath)):
        for f in files:
            if(f.endswith(extension)):
#                 Purename is the name without the file extension
                purename = f[:f.index(extension)];
                if(not (f in exclusions or purename in exclusions)):
                    found_files.append({'filename':f, 'purename':purename, 'filepath':os.path.join(root, f)});
        return found_files;
    
def getMeshesFromDirectory(directorypath, mesh_exclusions=[]):
    return getFilesFromDirectory(directorypath, extension='.obj', exclusions=mesh_exclusions);

def createDir(pathdir):
    if(not os.path.exists(pathdir)):
        os.makedirs(pathdir);    
    return os.path.abspath(pathdir);