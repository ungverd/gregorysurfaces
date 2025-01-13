import bpy

from .commons import check_curve_crosses_mirror, make_curve_mirror_bridge, add_hook, apply_hook, mirror_vec


class OBJECT_OT_make_curve_mirror_bridge(bpy.types.Operator):
    """Gregory: make curve a bridge through mirror"""
    bl_idname = "object.make_curve_mirror_bridge"
    bl_label = "Greg: make curve bridge through mirror"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        if len(selected) != 2:
            return False
        curves = [obj for obj in selected if obj.greg_curve_settings.used_for_greg]
        if len(curves) != 1:
            return False
        curve = curves[0]
        if curve.greg_curve_settings.is_mirror_bridge:
            return False
        others = [sel for sel in selected if sel != curve]
        if len(others) != 1:
            return False
        other = others[0]
        if not other.greg_empty_settings.used_for_greg:
            return False
        curve_empties = (curve.greg_curve_settings.end1_empty, curve.greg_curve_settings.end2_empty)
        if not (other in curve_empties):
            return False
        not_target = [empty for empty in curve_empties if empty != other][0]
        if len(not_target.greg_empty_settings.curve_ends) != 1:
            return False
        if check_curve_crosses_mirror(curve):
            return True
        return False

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        # target - empty connected to main structure
        selected = context.selected_objects
        curve = [obj for obj in selected if obj.greg_curve_settings.used_for_greg][0]
        i, axis = check_curve_crosses_mirror(curve)
        mirror_obj = curve.modifiers[i].mirror_object
        target = [sel for sel in selected if sel != curve][0]
        two_empties = (curve.greg_curve_settings.end1_empty, curve.greg_curve_settings.end2_empty)
        other = [empty for empty in two_empties if empty != target][0]
        target_end_name, other_end_name, target_point, other_point = make_curve_mirror_bridge(curve, target, other, mirror_obj, axis)
        end1 = target.greg_empty_settings.curve_ends[target_end_name]
        end2 = other.greg_empty_settings.curve_ends[other_end_name]
        apply_hook(end1)
        apply_hook(end2)
        other_point.handle_left = mirror_vec(target_point.handle_right, mirror_obj, axis)
        other_point.handle_right = mirror_vec(target_point.handle_left, mirror_obj, axis)
        add_hook(end1)
        add_hook(end2)
        return {'FINISHED'}    

def add_bridge_mirror_func(self, context: bpy.types.Context):
    self.layout.operator(OBJECT_OT_make_curve_mirror_bridge.bl_idname)
