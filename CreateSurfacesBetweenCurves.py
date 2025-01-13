import bpy

from .commons import get_greg_collection
from .DependantsOfResolution import dependants_of_resolution_dict, DependantsOfResolution_np
from .CreateSurfacesGlobalList import CreateSurfacesGlobalList

class OBJECT_OT_create_surfaces_between_curves(bpy.types.Operator):
    """Gregory: create surface"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_create_surfs"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: create surfaces"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.
    
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        active = context.active_object
        if active is None:
            selected = context.selected_objects
            if len(selected) != 1:
                return False
            active = selected[0]
        collection = get_greg_collection(active)
        return collection is not None
    
    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        active = context.active_object
        if active is None:
            active = context.selected_objects[0]
        collection = get_greg_collection(active)
        prev_nedges = collection.greg_settings.nedges
        mesh_obj = collection.greg_settings.mesh_obj
        if mesh_obj is not None:
            collection.greg_settings.nedges = mesh_obj.greg_resolution
        d = dependants_of_resolution_dict.get(collection.greg_settings.name)
        if d is None:
            d = DependantsOfResolution_np()
            dependants_of_resolution_dict[collection.greg_settings.name] = d
            d.conditional_update(collection.greg_settings.nedges)
        else:
            if prev_nedges != collection.greg_settings.nedges:
                d.conditional_update(collection.greg_settings.nedges)
        glist = CreateSurfacesGlobalList(collection)
        glist.prepare_for_greg()
        glist.add_curves_and_bpoints()
        glist.add_quads()
        glist.calculate_kk()
        glist.render_mesh(d, collection.name)

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def add_surface_menu_func(self, context: bpy.types.Context):
    self.layout.operator(OBJECT_OT_create_surfaces_between_curves.bl_idname)