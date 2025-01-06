from typing import Optional, List
import math

import bpy
import mathutils

from .propertyGroups import GregBasicEnd, GregCurveEndItem

TH = 0.0001
TH2 = TH**2

def apply_hook(end, context: Optional[bpy.types.Context]=None):
    #requires Object mode
    curve_obj = end.basic_end.curve
    if context is None:
        context = bpy.context
    if context.scene.objects.get(curve_obj.name):
        context.view_layer.objects.active = curve_obj
        hook_name = end.hook
        for modifier in curve_obj.modifiers:
            if modifier.name == hook_name:
                bpy.ops.object.modifier_apply(modifier=hook_name)
                return

def add_hook(end, context: Optional[bpy.types.Context]=None):
    #requires Object mode
    if context is None:
        context = bpy.context
    curve_obj = end.basic_end.curve
    if bpy.context.scene.objects.get(curve_obj.name):
        i = end.basic_end.end
        if end.is_coplanar and len(end.coplanar_vectors) == 1:
            empty_to_link = end.coplanar_vectors[0].arrow
            empty_name = empty_to_link.greg_arrow_settings.name
        else:
            empty_to_link = end.empty
            empty_name = empty_to_link.greg_empty_settings.name
        curve_name = curve_obj.greg_curve_settings.name
        hook_name = f"hook_{curve_name}_{empty_name}"
        possible_hooks = get_all_possible_hooks(end)
        for modifier in curve_obj.modifiers:
            if modifier.name == hook_name: #modifier exists
                return
            if modifier.name in possible_hooks:
                context.view_layer.objects.active = curve_obj
                bpy.ops.object.modifier_apply(modifier=modifier.name)
        hook = curve_obj.modifiers.new(name=hook_name, type='HOOK')
        hook.vertex_indices_set([i*3, i*3+1, i*3+2])
        context.evaluated_depsgraph_get()
        hook.object = empty_to_link
        end.hook = hook.name
        curve_obj.modifiers.move(len(curve_obj.modifiers) - 1, 0) # move new modifier to the first position
                                                                  # to be before mirror modifiers if there are any

def get_all_possible_hooks(end):
    curve_obj = end.basic_end.curve
    curve_name = curve_obj.greg_curve_settings.name
    empty_obj = end.empty
    empty_name = empty_obj.greg_empty_settings.name
    res = [f"hook_{curve_name}_{empty_name}"]
    for coplanar in empty_obj.greg_empty_settings.coplanars:
        res.append(f"hook_{curve_name}_{coplanar.name}")
    return res

def mirror_vec_with_vec(vec, normal, pos):
    reflected = vec.reflect(normal)
    delta_vec = pos.project(normal)
    return reflected + 2*delta_vec

def mirror_vec(vec, mirror_object, axis): #axis: 0 - x, 1 - y, 2 - z
    basic_vec = mathutils.Vector((0, 0, 0))
    basic_vec[axis] = 1
    if mirror_object is not None:
        mat = mirror_object.matrix_world.copy()
        mat.invert()
        mirror = basic_vec @ mat
        return mirror_vec_with_vec(vec, mirror, mirror_object.matrix_world.translation)
    else:
        return vec.reflect(basic_vec)


def get_greg_collection(obj: bpy.types.Object) -> Optional[bpy.types.Collection]:
    for collection in obj.users_collection:
        if collection.greg_settings.used_for_greg:
            return collection
    return None

def are_collinear(v1: mathutils.Vector, v2: mathutils.Vector) -> bool:
    return (v1.normalized().cross(v2.normalized())).length < TH

def rotate_end_to_vec(vec: mathutils.Vector, basic_end: "GregBasicEnd"):
    spline = basic_end.curve.data.splines[0]
    i = basic_end.end
    p = spline.bezier_points[i]
    if i == 0:
        h = p.handle_right - p.co
    else:
        h = p.handle_left - p.co
    if not are_collinear(h, vec):
        quat = h.rotation_difference(vec)
        if quat.angle > math.pi / 2:
            axis, angle = quat.to_axis_angle()
            new_angle = math.pi - angle
            new_axis = -1 * axis
            quat = mathutils.Quaternion(new_axis, new_angle)
        h.rotate(quat)
        if i == 0:
            p.handle_right = h + p.co
            if p.handle_right_type == "ALIGNED":
                oh = p.handle_left - p.co
                oh.rotate(quat)
                p.handle_left = oh + p.co
        else:
            p.handle_left = h + p.co
            if p.handle_left_type == "ALIGNED":
                oh = p.handle_right - p.co
                oh.rotate(quat)
                p.handle_right = oh + p.co

def remove_coplanar(end: "GregCurveEndItem", end_name: str, empty: bpy.types.Object, collection: bpy.types.Collection):
    for arrow_item in end.coplanar_vectors:
        arrow = arrow_item.arrow
        arrow_name = arrow_item.name
        end_setting = arrow.greg_arrow_settings.coplanars.find(end_name)
        arrow.greg_arrow_settings.coplanars.remove(end_setting)
        groups_num = get_coplanar_groups_num(arrow)
        if groups_num < 3:
            empty_setting = empty.greg_empty_settings.coplanars.find(arrow_name)
            empty.greg_empty_settings.coplanars.remove(empty_setting)
            collection_setting = collection.greg_settings.arrows.find(arrow_name)
            collection.greg_settings.arrows.remove(collection_setting)
            for basic_end in arrow.greg_arrow_settings.coplanars:
                other_end = empty.greg_empty_settings.curve_ends[basic_end.name]
                other_end_setting = other_end.coplanar_vectors.find(arrow_name)
                other_end.coplanar_vectors.remove(other_end_setting)
                if len(other_end.coplanar_vectors) == 0:
                    if other_end.name != end_name:
                        apply_hook(other_end)
                        add_hook(other_end)
            bpy.data.objects.remove(arrow, do_unlink=True)

def get_coplanar_groups_num(arrow: bpy.types.Object):
    empty = arrow.parent
    added = {basic_end.name: False for basic_end in arrow.greg_arrow_settings.coplanars}
    counter = 0
    for basic_end in arrow.greg_arrow_settings.coplanars:
        bn = basic_end.name
        if not added[bn]:
            added[bn] = True
            end = empty.greg_empty_settings.curve_ends[bn]
            counter += 1
            if end.is_collinear:
                for other_basic_end in end.collinear_to:
                    obn = other_basic_end.name
                    added[obn] = True
    return counter

def same_coords(c1: mathutils.Vector, c2: mathutils.Vector) -> bool:
    return (c1-c2).length_squared < TH2

def get_next_id(collection: bpy.types.Collection):
    name = str(collection.greg_settings.max_id)
    collection.greg_settings.max_id += 1
    return name

def add_curve_obj(collection, co_s, handles_left, handles_right):
    curve = bpy.data.curves.new(name="greg_curve", type='CURVE')
    curve.dimensions = '3D'
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(1)
    for i, (co, handle_left, handle_right) in enumerate(zip(co_s, handles_left, handles_right)):
        spline.bezier_points[i].co = co
        spline.bezier_points[i].handle_left = co + handle_left
        spline.bezier_points[i].handle_right = co + handle_right
        if are_collinear(handle_left, handle_right) and handle_left.dot(handle_right) < 0:
            spline.bezier_points[i].handle_left_type = 'ALIGNED'
            spline.bezier_points[i].handle_right_type = 'ALIGNED'
        else:
            spline.bezier_points[i].handle_left_type = 'FREE'
            spline.bezier_points[i].handle_right_type = 'FREE'
    curve_obj = bpy.data.objects.new(name="greg_curve_obj", object_data=curve)
    curve_obj.greg_curve_settings.used_for_greg = True
    curve_obj.greg_curve_settings.name = get_next_id(collection)
    constraint = curve_obj.constraints.new(type="LIMIT_LOCATION")
    constraint.owner_space = "WORLD"
    constraint.max_x = 0
    constraint.max_y = 0
    constraint.max_z = 0
    constraint.min_x = 0
    constraint.min_y = 0
    constraint.min_z = 0
    constraint.use_max_x = True
    constraint.use_max_y = True
    constraint.use_max_z = True
    constraint.use_min_x = True
    constraint.use_min_y = True
    constraint.use_min_z = True
    constraint.enabled = True
    curve_prop = collection.greg_settings.curves.add()
    curve_prop.name = curve_obj.greg_curve_settings.name
    curve_prop.curve = curve_obj
    collection.objects.link(curve_obj)
    return curve_obj, curve_prop # DANGER! Do not create new curve_prop while you are using this one

def add_empty_obj(collection, co):
    empty_obj = bpy.data.objects.new("greg_empty", None)
    empty_obj.location = co
    empty_obj.greg_empty_settings.used_for_greg = True
    empty_obj.greg_empty_settings.name = get_next_id(collection)
    empty_prop = collection.greg_settings.empties.add()
    empty_prop.name = empty_obj.greg_empty_settings.name
    empty_prop.empty = empty_obj
    collection.objects.link(empty_obj)
    return empty_obj

def add_curve_end(collection, empty_obj, curve_obj, i):
    curve_end = empty_obj.greg_empty_settings.curve_ends.add()
    curve_end.basic_end.curve = curve_obj
    curve_end.basic_end.end = i
    curve_end.empty = empty_obj
    curve_end.name = get_next_id(collection)
    curve_end.basic_end.name = curve_end.name
    if i == 0:
        curve_obj.greg_curve_settings.end1_name = curve_end.name
        curve_obj.greg_curve_settings.end1_empty = empty_obj
    else:
        curve_obj.greg_curve_settings.end2_name = curve_end.name
        curve_obj.greg_curve_settings.end2_empty = empty_obj
    return curve_end.name

def coplanar_collinear(empty: bpy.types.Object, collection: bpy.types.Collection):
    for i, end1 in enumerate(empty.greg_empty_settings.curve_ends):
        for end2 in empty.greg_empty_settings.curve_ends[:i]:
            check_ends_collinear(end1, end2)
    collinear_groups = []
    used_dict = {end.name: False for end in empty.greg_empty_settings.curve_ends}
    for end in empty.greg_empty_settings.curve_ends:
        if not end.is_collinear:
            collinear_groups.append([end])
        else:
            if used_dict[end.name] == False:
                used_dict[end.name] = True
                collinear_groups.append([end])
                for basic_end in end.collinear_to:
                    collinear_end_name = basic_end.name
                    used_dict[collinear_end_name] = True
                    collinear_groups[-1].append(empty.greg_empty_settings.curve_ends[collinear_end_name])
    sets_and_vectors = []
    for i, ends1 in enumerate(collinear_groups):
        for j, ends2 in enumerate(collinear_groups[:i]):
            for ends3 in collinear_groups[:j]:
                res = check_ends_coplanar(ends1, ends2, ends3, empty, collection)
                if res is not None:
                    sets_and_vectors.append((set(ends1).union(set(ends2)).union(set(ends3)), res))
    final_sets_and_vectors = []
    for se, vec in sets_and_vectors:
        i = 0
        added = False
        while i < len(final_sets_and_vectors) and not added:
            if are_collinear(vec, final_sets_and_vectors[i][1]):
                final_sets_and_vectors[i][0] = final_sets_and_vectors[i][0].union(se)
                added = True
            i += 1
        if not added:
            final_sets_and_vectors.append([se.copy(), vec])
    for se, vec in final_sets_and_vectors:
        add_coplanar_arrow(se, vec, empty, collection)

def add_coplanar_arrow(set_ends, coplanar_vector: mathutils.Vector, empty: bpy.types.Object, collection: bpy.types.Collection):
    feasible = True
    for end in set_ends:
        if len(end.coplanar_vectors) >= 2:
            feasible = False
    if feasible:
        name = get_next_id(collection)
        arrow = bpy.data.objects.new("greg_arrow", None)
        arrow.greg_arrow_settings.name = name
        arrow.greg_arrow_settings.used_for_greg = True
        empty_arrow_settings = empty.greg_empty_settings.coplanars.add()
        empty_arrow_settings.name = name
        empty_arrow_settings.arrow = arrow
        collection_arrow_settings = collection.greg_settings.arrows.add()
        collection_arrow_settings.name = name
        collection_arrow_settings.arrow = arrow
        arrow.parent = empty
        arrow.empty_display_type = "SINGLE_ARROW"
        collection.objects.link(arrow)
        vec = mathutils.Vector((0,0,1)) @ arrow.matrix_parent_inverse
        vec.normalize()
        quat = vec.rotation_difference(coplanar_vector)
        arrow.rotation_mode = "QUATERNION"
        arrow.rotation_quaternion = quat
        constraint = arrow.constraints.new(type="LIMIT_LOCATION")
        constraint.owner_space = "LOCAL"
        constraint.max_x = 0
        constraint.max_y = 0
        constraint.max_z = 0
        constraint.min_x = 0
        constraint.min_y = 0
        constraint.min_z = 0
        constraint.use_max_x = True
        constraint.use_max_y = True
        constraint.use_max_z = True
        constraint.use_min_x = True
        constraint.use_min_y = True
        constraint.use_min_z = True
        constraint.enabled = True
        for end in set_ends:
            end_setting = end.coplanar_vectors.add()
            end_setting.arrow = arrow
            end_setting.name = name
            arrow_setting = arrow.greg_arrow_settings.coplanars.add()
            arrow_setting.name = end.name
            arrow_setting.curve = end.basic_end.curve
            arrow_setting.end = end.basic_end.end

def check_ends_collinear(end1: "GregCurveEndItem", end2: "GregCurveEndItem"):
    ends = (end1, end2)
    handles = extract_vectors_from_ends(ends)
    if are_collinear(*handles):
        common_empty = end1.empty
        set_ends_collinear_to_one_another(end1, end2, common_empty)

def set_ends_collinear_to_one_another(end1: GregCurveEndItem, end2: GregCurveEndItem, common_empty: bpy.types.Object):
    ends = (end1, end2)
    end_groups = [[end.basic_end] + list(end.collinear_to) for end in ends]
    for first_basic_ends, second_basic_ends in zip(end_groups, reversed(end_groups)):
        for first_basic_end in first_basic_ends:
            not_basic_end = extract_end_from_basic_end_and_empty(first_basic_end, common_empty)
            for second_basic_end in second_basic_ends:
                if second_basic_end.name != not_basic_end.name:
                    if not second_basic_end.name in not_basic_end.collinear_to:
                        added_basic_end = not_basic_end.collinear_to.add()
                        added_basic_end.curve = second_basic_end.curve
                        added_basic_end.end = second_basic_end.end
                        added_basic_end.name = second_basic_end.name

def extract_end_from_name_and_empty(name: str, empty: bpy.types.Object):
    return empty.greg_empty_settings.curve_ends[name]

def extract_end_from_basic_end_and_empty(basic_end: GregBasicEnd, empty: bpy.types.Object) -> str:
    return extract_end_from_name_and_empty(basic_end.name, empty)

def check_ends_coplanar(ends1, ends2, ends3, empty, collection: bpy.types.Collection):
    handles: List[mathutils.Vector] = []
    ends = (ends1[0], ends2[0], ends3[0])
    handles = extract_vectors_from_ends(ends)
    if are_coplanar(*handles):
        coplanar_vector = (handles[0].cross(handles[1])).normalized()
        return coplanar_vector
    return None

def are_coplanar(v1: mathutils.Vector, v2: mathutils.Vector, v3: mathutils.Vector) -> bool:
    return abs(mathutils.Matrix((v1.normalized(), v2.normalized(), v3.normalized())).determinant()) < TH

def extract_vectors_from_ends(ends: List["GregCurveEndItem"]):
    handles: List[mathutils.Vector] = []
    for end in ends:
        handles.append(extract_vector_from_basic_end(end.basic_end))
    return handles

def extract_vector_from_basic_end(basic_end: "GregBasicEnd"):
    spline = basic_end.curve.data.splines[0]
    i = basic_end.end
    point = spline.bezier_points[i]
    co = point.co
    if i == 0:
        ha = point.handle_right
    else:
        ha = point.handle_left
    return ha - co

def copy_mirrors_from_one_obj_to_another(obj_with_mirrors: bpy.types.Object,
                                         obj_without_mirrors: bpy.types.Object):
    i = 1
    for modifier in obj_with_mirrors.modifiers:
        if modifier.type == 'MIRROR':
            new_mirror = obj_without_mirrors.modifiers.new(f"mirror_{i}", 'MIRROR')
            new_mirror.mirror_object = modifier.mirror_object
            new_mirror.use_axis = modifier.use_axis
            i += 1