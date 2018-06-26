import bpy;

__author__="ashok"
__date__ ="$Mar 23, 2015 8:16:11 PM$"

class GenericLandmarksMessageBox(bpy.types.Operator):
    bl_idname = "genericlandmarks.messagebox"# unique identifier for buttons and menu items to reference.
    bl_label = "Message Box"         # display name in the interface.
    bl_space_type = "VIEW_3D"       # show up in: 3d-window
    bl_region_type = "UI"           # show up in: properties panel
    bl_context = "objectmode"; 
    messagelinesize = bpy.props.IntProperty(name="Message",description="What is the message you want to show", default = 60);
    message = bpy.props.StringProperty(name="Message",description="What is the message you want to show", default = "Hello World. (dot). A message revolutionized by Thalaivar Superstar");
    messagetype = bpy.props.StringProperty(name="Message Type",description="What is the message type? (INFO, ERROR, WARNING)", default = "info");
        
    def execute(self, context):
        return {'FINISHED'};
    
    def invoke(self, context, event):
        wm = context.window_manager;
        return wm.invoke_props_dialog(self);
    
    def draw(self, context):
        layout = self.layout;
        n = self.messagelinesize;       
        
        if(self.messagetype == 'INFO'):
            icon = 'INFO';
            
        if(self.messagetype == 'WARNING'):
            icon = 'ERROR';
        
        if(self.messagetype == 'ERROR'):
            icon = 'CANCEL';
        
        row = layout.row();
        row.label(text=self.messagetype,  icon=icon);
        
        allmessageslist = self.message.splitlines();
        
        for message in allmessageslist:
            for i in range(0, len(message), n):
                row = layout.row();
                row.label(text=message[i:i+n]);
                