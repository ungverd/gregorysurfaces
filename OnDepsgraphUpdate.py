from typing import Optional
from enum import Enum
import bpy
import mathutils
from inspect import getouterframes, currentframe
from commons import apply_hook, add_hook, mirror_vec, get_greg_collection, are_collinear
from commons import TH2, rotate_end_to_vec, remove_coplanar

mode = [None]

@bpy.app.handlers.persistent
def on_depsgraph_update(scene):
    level = len(getouterframes(currentframe()))
    if level < 2:
        now_mode = bpy.context.mode
        if now_mode == "EDIT_CURVE":
            if bpy.context.active_object.greg_curve_settings.used_for_greg:
                if now_mode != mode[0]:
                    on_curve_enter_edit_mode(bpy.context.active_object)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                for update in depsgraph.updates:
                    if update.is_updated_geometry:
                        on_curve_edit_mode(bpy.context.active_object)
        elif now_mode == "OBJECT":
            verify_curve_deleted_or_returned()
        if now_mode != mode[0]:
            mode[0] = now_mode

def on_curve_enter_edit_mode(curve_obj: bpy.types.Object):
    bpy.ops.object.mode_set(mode='OBJECT')
    for end_empty in (curve_obj.greg_curve_settings.end1_empty, curve_obj.greg_curve_settings.end2_empty):
        for end in end_empty.greg_empty_settings.curve_ends:
            apply_hook(end)
            add_hook(end)
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.mode_set(mode='EDIT')

def on_curve_edit_mode(curve_obj: bpy.types.Object):
    if len(curve_obj.data.splines) != 1 or len(curve_obj.data.splines[0].bezier_points) != 2:
        bpy.ops.object.mode_set(mode='OBJECT')
        remove_curve_from_greg_structure(curve_obj)
        bpy.ops.object.mode_set(mode='EDIT')
        return
    preserve_points(curve_obj)
    preserve_coplanar(curve_obj)
    preserve_collinear(curve_obj)
    if curve_obj.greg_curve_settings.is_mirror_bridge:
        preserve_mirror_bridge(curve_obj)

def on_curve_edit_mode(curve_obj: bpy.types.Object):
    if len(curve_obj.data.splines) != 1 or len(curve_obj.data.splines[0].bezier_points) != 2:
        bpy.ops.object.mode_set(mode='OBJECT')
        remove_curve_from_greg_structure(curve_obj)
        bpy.ops.object.mode_set(mode='EDIT')
        return
    preserve_points(curve_obj)
    preserve_coplanar(curve_obj)
    preserve_collinear(curve_obj)
    if curve_obj.greg_curve_settings.is_mirror_bridge:
        preserve_mirror_bridge(curve_obj)

def preserve_mirror_bridge(curve_obj: bpy.types.Object):
    if curve_obj.greg_curve_settings.bridge_mirror_other_i == 0:
        p_other = curve_obj.data.splines[0].bezier_points[0]
        p_target = curve_obj.data.splines[0].bezier_points[1]
    elif curve_obj.greg_curve_settings.bridge_mirror_other_i == 1:
        p_other = curve_obj.data.splines[0].bezier_points[1]
        p_target = curve_obj.data.splines[0].bezier_points[0]
    p_other.handle_left = mirror_vec(p_target.handle_right,
                                     curve_obj.greg_curve_settings.bridge_mirror_object,
                                     curve_obj.greg_curve_settings.bridge_mirror_axis)
    p_other.handle_right = mirror_vec(p_target.handle_left,
                                      curve_obj.greg_curve_settings.bridge_mirror_object,
                                      curve_obj.greg_curve_settings.bridge_mirror_axis)

def traverse_tree(t):
    yield t
    for child in t.children:
        yield from traverse_tree(child)

def verify_curve_deleted_or_returned():
    coll = bpy.context.scene.collection
    for collection in traverse_tree(coll):
        if collection.greg_settings.used_for_greg:
            verify_arrow_returned(collection)
            for setting in collection.greg_settings.curves:
                curve_obj = setting.curve
                if curve_obj:
                    if not bpy.context.scene.objects.get(curve_obj.name):
                        remove_curve_from_greg_structure(curve_obj, collection)
                        bpy.data.objects.remove(curve_obj, do_unlink=True)
                    else:
                        splines = curve_obj.data.splines
                        if len(splines) != 1 or len(splines[0].bezier_points) != 2:
                            remove_curve_from_greg_structure(curve_obj, collection)
                        else:
                            settings = curve_obj.greg_curve_settings
                            for i, (end_name, end_empty) in enumerate(((settings.end1_name, settings.end1_empty),
                                                                    (settings.end2_name, settings.end2_empty))):
                                if end_empty.greg_empty_settings.curve_ends.find(end_name) == -1:
                                    repare_end(curve_obj, end_empty, end_name, i)
                                    repare_hooks(end_empty)

def remove_curve_from_greg_structure(curve_obj: bpy.types.Object, collection: Optional[bpy.types.Collection] = None):
    #print("before remove")
    #print_structure(collection)
    #print("****************************")
    #requires Object mode
    curve_obj.greg_curve_settings.used_for_greg = False
    curve_name = curve_obj.greg_curve_settings.name
    if collection is None:
        collection = get_greg_collection(curve_obj)
    end1_empty = curve_obj.greg_curve_settings.end1_empty
    end2_empty = curve_obj.greg_curve_settings.end2_empty
    end1_name = curve_obj.greg_curve_settings.end1_name
    end2_name = curve_obj.greg_curve_settings.end2_name
    for empty, end_name in ((end1_empty, end1_name), (end2_empty, end2_name)):
        #print("some ends", [end.name for end in empty.greg_empty_settings.curve_ends])
        end = empty.greg_empty_settings.curve_ends[end_name]
        apply_hook(end)
        empty_name = empty.greg_empty_settings.name
        if end.is_collinear:
            for other_basic_end in end.collinear_to:
                other_end = empty.greg_empty_settings.curve_ends[other_basic_end.name]
                other_end.collinear_to.remove(other_end.collinear_to.find(end_name))
        if end.is_coplanar:
            remove_coplanar(end, end_name, empty, collection)
        empty_setting = empty.greg_empty_settings.curve_ends.find(end_name)
        empty.greg_empty_settings.curve_ends.remove(empty_setting)
        if len(empty.greg_empty_settings.curve_ends) == 0:
            collection_setting = collection.greg_settings.empties.find(empty_name)
            collection.greg_settings.empties.remove(collection_setting)
            bpy.data.objects.remove(empty, do_unlink=True)
    collection_setting = collection.greg_settings.curves.find(curve_name)
    collection.greg_settings.curves.remove(collection_setting)
    curve_obj.greg_curve_settings.name = ""
    curve_obj.greg_curve_settings.end1_empty = None
    curve_obj.greg_curve_settings.end2_empty = None
    curve_obj.greg_curve_settings.end1_name = ""
    curve_obj.greg_curve_settings.end2_name = ""
    if collection in curve_obj.users_collection:
        parent_collection = bpy.context.scene.collection
        parent_collection.objects.link(curve_obj)
        collection.objects.unlink(curve_obj)
    #print("after remove")
    #print_structure(collection)
    #print("****************************")

def verify_arrow_returned(collection):
    for setting in collection.greg_settings.arrows:
        arrow_obj = setting.arrow
        if arrow_obj:
            if bpy.context.scene.objects.get(arrow_obj.name):
                arrow_settings = arrow_obj.greg_arrow_settings
                arrow_name = arrow_settings.name
                empty_obj = arrow_obj.parent
                empty_settings = empty_obj.greg_empty_settings
                if empty_settings.coplanars.find(arrow_name) == -1:
                    new_arrow_item = empty_settings.coplanars.add()
                    new_arrow_item.name = arrow_name
                    new_arrow_item.arrow = arrow_obj

def repare_end(curve_obj, empty_obj, end_name, i):
    verify_coplanar_and_add(empty_obj, curve_obj, i)
    collinear_end = verify_collinear(empty_obj, curve_obj, i)
    setting = empty_obj.greg_empty_settings.curve_ends.add()
    setting.basic_end.end = i
    setting.basic_end.name = end_name
    setting.basic_end.curve = curve_obj
    setting.name = end_name
    setting.empty = empty_obj
    if collinear_end is not None:
        c1 = setting.collinear_to.add()
        c1.name = collinear_end.name
        c1.curve = collinear_end.basic_end.curve
        c1.end = collinear_end.basic_end.end
        for other_basic_end in collinear_end.collinear_to:
            c1 = setting.collinear_to.add()
            c1.name = other_basic_end.name
            c1.curve = other_basic_end.curve
            c1.end = other_basic_end.end
            other_end = empty_obj.greg_empty_settings.curve_ends[other_basic_end.name]
            c2 = other_end.collinear_to.add()
            c2.end = i
            c2.name = end_name
            c2.curve = curve_obj
        c3 = collinear_end.collinear_to.add()
        c3.end = i
        c3.name = end_name
        c3.curve = curve_obj
    add_arrow_to_end_in_empty(empty_obj, setting)

def repare_hooks(empty_obj):
    empty_settings = empty_obj.greg_empty_settings
    for end in empty_settings.curve_ends:
        curve = end.basic_end.curve
        curve_name = curve.greg_curve_settings.name
        verify_coplanar_and_add(empty_obj, curve, end.basic_end.end)
        add_arrow_to_end_in_empty(empty_obj, end)
        if end.is_coplanar and len(end.coplanar_vectors) == 1:
            arrow = end.coplanar_vectors[0]
            new_hook_name = f"hook_{curve_name}_{arrow.name}"
            old_hook_name = f"hook_{curve_name}_{empty_settings.name}" #possible old name
            res = verify_if_change_hook(curve.modifiers, old_hook_name, new_hook_name)
            if res == HookRes.ADD:
                add_hook(end)
            elif res == HookRes.APPLY_AND_ADD:
                apply_hook(end)
                add_hook(end)
            if end.hook != new_hook_name:
                end.hook = new_hook_name
        else:
            hook_name = f"hook_{curve_name}_{empty_settings.name}"
            res = verify_if_change_hook(curve.modifiers, None, hook_name)
            if res == HookRes.ADD:
                add_hook(end)
            if end.hook != hook_name:
                end.hook = hook_name

class HookRes(Enum):
    FINE = 1
    APPLY_AND_ADD = 2
    ADD = 3

def verify_if_change_hook(modifiers, old_hook_name, new_hook_name):
    for modifier in modifiers:
        if modifier.name == new_hook_name:
            return HookRes.FINE
        if modifier.name == old_hook_name:
            return HookRes.APPLY_AND_ADD
        return HookRes.ADD

def verify_collinear(empty, curve, i):
    p = curve.data.splines[0].bezier_points[i]
    co = p.co
    h = p.handle_right if i == 0 else p.handle_left
    hh = h - co
    for end in empty.greg_empty_settings.curve_ends:
        curve2 = end.basic_end.curve
        i2 = end.basic_end.end
        p2 = curve2.data.splines[0].bezier_points[i2]
        co2 = p2.co
        h2 = p2.handle_right if i2 == 0 else p2.handle_left
        hh2 = h2 - co2
        if are_collinear(hh, hh2):
            return end
    return None

def verify_coplanar_and_add(empty, curve, i):
    if i == 0:
        end_name = curve.greg_curve_settings.end1_name
    else:
        end_name = curve.greg_curve_settings.end2_name
    for arrow_item in empty.greg_empty_settings.coplanars:
        expected_modifier_name = f"hook_{curve.greg_curve_settings.name}_{arrow_item.name}"
        for modifier in curve.modifiers:
            if modifier.name == expected_modifier_name:
                add_end_to_arrow(arrow_item.arrow, end_name, i, curve)
                return

def add_end_to_arrow(arrow, end_name, i, curve):
    basic_end = arrow.greg_arrow_settings.coplanars.add()
    basic_end.name = end_name
    basic_end.end = i
    basic_end.curve = curve

def add_arrow_to_end_in_empty(empty_obj, end):
    empty = empty_obj.greg_empty_settings
    for arrow_item in empty.coplanars:
        arrow_settings = arrow_item.arrow.greg_arrow_settings
        if arrow_settings.coplanars.find(end.name) != -1:
            if end.coplanar_vectors.find(arrow_item.name) == -1:
                new_arrow_item = end.coplanar_vectors.add()
                new_arrow_item.name = arrow_item.name
                new_arrow_item.arrow = arrow_item.arrow

def preserve_points(curve_obj):
    # edit mode
    spline = curve_obj.data.splines[0]
    settings = curve_obj.greg_curve_settings
    for i, empty in enumerate((settings.end1_empty, settings.end2_empty)):
        if (spline.bezier_points[i].co - empty.matrix_world.translation).length_squared > TH2:
            spline.bezier_points[i].co = empty.matrix_world.translation

def preserve_collinear(curve_obj):
    # edit mode
    settings = curve_obj.greg_curve_settings
    for i, (end_name, empty) in enumerate(((settings.end1_name, settings.end1_empty),
                                           (settings.end2_name, settings.end2_empty))):
        end = empty.greg_empty_settings.curve_ends[end_name]
        if end.is_collinear:
            spline = end.basic_end.curve.data.splines[0]
            i = end.basic_end.end
            p = spline.bezier_points[i]
            if i == 0:
                h = p.handle_right - p.co
            else:
                h = p.handle_left - p.co
            for basic_end in end.collinear_to:
                rotate_end_to_vec(h, basic_end)

def preserve_coplanar(curve_obj):
    # edit mode
    settings = curve_obj.greg_curve_settings
    for i, (end_name, empty) in enumerate(((settings.end1_name, settings.end1_empty),
                                           (settings.end2_name, settings.end2_empty))):
        end = empty.greg_empty_settings.curve_ends[end_name]
        if end.is_coplanar:
            if len(end.coplanar_vectors) == 1:
                arrow = end.coplanar_vectors[0].arrow
                mat = arrow.matrix_world.copy()
                mat.invert()
                vec = mathutils.Vector((0,0,1)) @ mat
                p = curve_obj.data.splines[0].bezier_points[i]
                for ii, handle in enumerate((p.handle_right, p.handle_left)):
                    h = handle - p.co
                    project = h.project(vec)
                    if project.length_squared > TH2:
                        h2 = h - project
                        quat = h.rotation_difference(h2)
                        h.rotate(quat)
                        if ii == 0:
                            p.handle_right = h + p.co
                        else:
                            p.handle_left = h + p.co

def preserve_coplanar_to_two_planes(end):
    # edit mode
    if len(end.coplanar_vectors) == 2:
        arrow1 = end.coplanar_vectors[0].arrow
        arrow2 = end.coplanar_vectors[1].arrow
        mat1 = arrow1.matrix_world.copy()
        mat2 = arrow2.matrix_world.copy()
        mat1.invert()
        vec1 = mathutils.Vector((0,0,1)) @ mat1
        mat2.invert()
        vec2 = mathutils.Vector((0,0,1)) @ mat2
        cross = vec1.cross(vec2)
        rotate_end_to_vec(cross, end.basic_end)

def preserve_coplanar_to_two_planes_object_mode(end):
    if len(end.coplanar_vectors) == 2:
        arrow1 = end.coplanar_vectors[0].arrow
        arrow2 = end.coplanar_vectors[1].arrow
        mat1 = arrow1.matrix_world.copy()
        mat2 = arrow2.matrix_world.copy()
        mat1.invert()
        vec1 = mathutils.Vector((0,0,1)) @ mat1
        mat2.invert()
        vec2 = mathutils.Vector((0,0,1)) @ mat2
        cross = vec1.cross(vec2)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        basic_end = end.basic_end
        spline = basic_end.curve.evaluated_get(depsgraph).data.splines[0]
        i = basic_end.end
        p = spline.bezier_points[i]
        if i == 0:
            h = p.handle_right - p.co
        else:
            h = p.handle_left - p.co
        if not are_collinear(h, cross):
            apply_hook(end)
            rotate_end_to_vec(cross, end.basic_end)
            add_hook(end)