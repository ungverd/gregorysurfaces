import bpy

from .commons import get_greg_collection, get_parent_collection
from .GlobalList import GlobalList, Spline

class CreateCurvesCollection(bpy.types.Operator):
    """Gregory: create curves"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_create_curves"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: create curves"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        if len(selected) != 1:
            return False
        obj = selected[0]
        if not obj.type == "CURVE":
            return False
        for spline in obj.data.splines:
            if len(spline.bezier_points) == 0:
                return False
        return get_greg_collection(obj) is None
    
    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.

        glist = GlobalList()

        active = context.selected_objects[0]
        mb = active.matrix_basis
        active.data.transform(mb)
        active.matrix_basis.identity()

        cur = active.data
        splines = cur.splines

        for s in splines:
            spline = Spline(glist)
            for p in s.bezier_points:
                spline.add_point(p.co, p.handle_left, p.handle_right)
            if s.use_cyclic_u:
                spline.round_spline()
        glist.add_many_curves(active.name, active, get_parent_collection(active), context)
        bpy.data.objects.remove(active, do_unlink=True)

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def add_collection_menu_func(self, context: bpy.types.Context):
    self.layout.operator(CreateCurvesCollection.bl_idname)