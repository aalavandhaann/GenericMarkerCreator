import bpy;

from GenericMarkerCreator.operators.LandmarksPair import AssignMeshPair;
from GenericMarkerCreator.operators.LiveOperators import LiveLandmarksCreator;
from GenericMarkerCreator.operators.LandmarksCreator import UnLinkLandmarks, LinkLandmarks, RemoveLandmarks;


# store keymaps here to access after registration
addon_keymaps_basics = [];
def register():
    print('KEYMAPS BEING REGISTERED -> START');
    # handle the keymap
    wm = bpy.context.window_manager;
    
    try:
        km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY');
        kmi = km.keymap_items.new(AssignMeshPair.bl_idname, 'A', 'PRESS', ctrl=True, shift=True);
        addon_keymaps_basics.append(km);
        
        km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY');
        kmi = km.keymap_items.new(LiveLandmarksCreator.bl_idname, 'M', 'PRESS', ctrl=True, shift=True);
        addon_keymaps_basics.append(km);
        
        km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY');
        kmi = km.keymap_items.new(LinkLandmarks.bl_idname, 'SLASH', 'PRESS', ctrl=False, shift=False);
        addon_keymaps_basics.append(km);       
        
        km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY');
        kmi = km.keymap_items.new(UnLinkLandmarks.bl_idname, 'PERIOD', 'PRESS', ctrl=False, shift=False);
        addon_keymaps_basics.append(km);
        
        km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY');
        kmi = km.keymap_items.new(RemoveLandmarks.bl_idname, 'MINUS', 'PRESS', ctrl=False, shift=False);
        addon_keymaps_basics.append(km);
        
    except AttributeError:
        pass;
    
    print('KEYMAPS BEING REGISTERED -> END');

def unregister():
    print('KEYMAPS BEING UNREGISTERED -> START');
    # handle the keymap
    wm = bpy.context.window_manager;
    try:
        for km in addon_keymaps_basics:
            wm.keyconfigs.addon.keymaps.remove(km);
        # clear the list
        addon_keymaps_basics.clear();
    except (AttributeError, RuntimeError):
        pass;
    print('KEYMAPS BEING UNREGISTERED -> END');