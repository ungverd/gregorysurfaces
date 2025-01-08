from typing import Optional, Tuple, List, Set

import bpy

from .commons import get_greg_collection, apply_hook, add_hook
from .GlobalForSubdivide import GlobalForSubdivide

class GregSubdivide(bpy.types.Operator):
    """Gregory: subdivide loop of curves"""
    bl_idname = "object.greg_sibdivide"
    bl_label = "Greg: subdivide loop of curves"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        if not context.selected_objects:
            return False
        collection = get_greg_collection(context.selected_objects[0])
        curves = []
        empties = []
        for obj in context.selected_objects:
            if not obj.greg_empty_settings.used_for_greg and not obj.greg_curve_settings.used_for_greg:
                return False
            if get_greg_collection(obj) != collection:
                return False
            if obj.greg_empty_settings.used_for_greg:
                empties.append(obj)
                if len(empties) > 4:
                    return False
            elif obj.greg_curve_settings.used_for_greg:
                curves.append(obj)
        if len(curves) == 0 or len(empties) == 0:
            return False
        if verify_complete_for_subdivide(context) is None:
            return False
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        collection = get_greg_collection(context.selected_objects[0])
        res = verify_complete_for_subdivide(context)
        if res is None:
            raise ValueError("invalid corners or borders selected!")
        else:
            g_list, optimal_quad, empties = res
        g_list.add_optimal_quad(optimal_quad)
        for empty_obj in empties:
            for end in empty_obj.greg_empty_settings.curve_ends:
                apply_hook(end)
                add_hook(end)
        g_list.subdivide()
        g_list.adjust_free_outers()
        g_list.add_real_curves(collection, context)
        return {'FINISHED'}    

def verify_complete_for_subdivide(context: bpy.types.Context) -> Optional[Tuple[GlobalForSubdivide, List[int], Set[bpy.types.Object]]]:
    empties = []
    curves = []
    corners = []
    for obj in context.selected_objects:
        if obj.greg_curve_settings.used_for_greg:
            curves.append(obj)
            empties.extend((obj.greg_curve_settings.end1_empty, obj.greg_curve_settings.end1_empty))
        elif obj.greg_empty_settings.used_for_greg:
            corners.append(obj)
    empties = set(empties) # remove duplicates
    g_list = GlobalForSubdivide()
    for curve in curves:
        g_list.add_curve_and_mirrors(curve)
    res = g_list.find_borders(corners)
    if res == False:
        return None
    else:
        optimal_quad = res
        return g_list, optimal_quad, empties

def add_greg_subdivide_func(self, context: bpy.types.Context):
    self.layout.operator(GregSubdivide.bl_idname)