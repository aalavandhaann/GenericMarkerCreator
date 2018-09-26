'''
Copyright (C) 2017 SRINIVASAN RAMACHANDRAN
ashok.srinivasan2002@gmail.com

Created by #0K Srinivasan Ramachandran

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

bl_info={
    "name": "Generic Landmarks",
    "author": "#0K Srinivasan",
    "version": (0, 0, 1), 
    "blender": (2, 7, 4), 
    "location": "View3D > Tools > Generic Landmarks", 
    "description": "Create linked landmarks between mesh pairs. Useful for mapping and other parametrization based approaches", 
    "warning": "Addon not complete, still under development", 
    "wiki_url": "", 
    "tracker_url": "", 
    "category": "Object"
    }

import bpy;
from GenericMarkerCreator import operators;
from GenericMarkerCreator import propertiesregister, panels, keymaps;

def register():
    print('BEGIN REGISTRATION OF MODULES');
    propertiesregister.register();
    operators.register();
    keymaps.register();
    panels.register();
    print('END REGISTRATION OF MODULES');
#     bpy.utils.register_module(__name__);    
 
def unregister():
    propertiesregister.unregister();
    panels.unregister();
    keymaps.unregister();
    operators.unregister();
#     bpy.utils.register_module(__name__);

if __name__ == "__main__":
    print('CALLING REGISTRATYION');
    register();