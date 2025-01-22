from typing import Optional, List, Tuple
import math

import numpy as np

import bpy
import mathutils

from .propertyGroups import GregBasicEnd, GregCurveEndItem

TH = 0.0001
TH2 = TH**2

def apply_hook(end: GregCurveEndItem,
               context: Optional[bpy.types.Context]=None):
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

def add_hook(end: GregCurveEndItem,
             context: Optional[bpy.types.Context]=None):
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

def set_properties_for_collinears(ends: Tuple[GregCurveEndItem, GregCurveEndItem],
                                  common_empty: bpy.types.Object): # must be 2 ends
    end_groups = [[end.basic_end] + list(end.collinear_to) for end in ends]
    for first_basic_ends, second_basic_ends in zip(end_groups, reversed(end_groups)):
        for first_basic_end in first_basic_ends:
            not_basic_end = extract_end_from_basic_end_and_empty(first_basic_end, common_empty)
            for second_basic_end in second_basic_ends:
                added_basic_end = not_basic_end.collinear_to.add()
                added_basic_end.curve = second_basic_end.curve
                added_basic_end.end = second_basic_end.end
                added_basic_end.name = second_basic_end.name
    end1, end2 = ends
    if end1.is_coplanar and end2.is_coplanar:
        collection = get_greg_collection(common_empty)
        #removes coplanar arrows if there's np more 3 collinear groups
        remove_coplanar(end1, end1.name, common_empty, collection)

def turn_all_collinears(first_end: GregCurveEndItem):
    first_vec = extract_vectors_from_ends([first_end])[0]
    for basic_end in first_end.collinear_to:
        vec = extract_vector_from_basic_end(basic_end)
        new_vec = vec.project(first_vec)
        rotate_end_to_vec(new_vec, basic_end)

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

def extract_end_from_name_and_empty(name: str, empty: bpy.types.Object) -> GregCurveEndItem:
    return empty.greg_empty_settings.curve_ends[name]

def extract_end_from_basic_end_and_empty(basic_end: GregBasicEnd, empty: bpy.types.Object):
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

def get_approx_bezier_length(co_s: Tuple[mathutils.Vector, mathutils.Vector],
                             handles: Tuple[mathutils.Vector, mathutils.Vector]) -> float:
    points_handles = [co + handle for co, handle in zip(co_s, handles)]
    length_extremities = (co_s[1] - co_s[0]).length
    lengths_handles = [handle.length for handle in handles]
    between_handles_length = (points_handles[1] - points_handles[0]).length
    approx_length = (length_extremities + between_handles_length + sum(lengths_handles)) / 2
    return approx_length

def make_collinear(v1: mathutils.Vector, v2: mathutils.Vector):
    dif = (v1 - v2).normalized()
    new_v1 = dif * v1.length
    new_v2 = -1 * dif * v2.length
    return new_v1, new_v2

def check_curve_crosses_mirror(obj, use_matrix_world = True):
    p1, p2 = None, None
    for i, modifier in enumerate(obj.modifiers):
        if modifier.type == 'MIRROR':
            if p1 is None:
                if use_matrix_world:
                    p1 = obj.greg_curve_settings.end1_empty.matrix_world.translation
                    p2 = obj.greg_curve_settings.end2_empty.matrix_world.translation
                else:
                    p1 = obj.greg_curve_settings.end1_empty.location
                    p2 = obj.greg_curve_settings.end2_empty.location
            mirror_object = modifier.mirror_object
            for axis in range(3):
                if modifier.use_axis[axis]:
                    normal = mathutils.Vector((0, 0, 0))
                    normal[axis] = 1
                    if mirror_object is not None:
                        mat = mirror_object.matrix_world.copy()
                        mat.invert()
                        normal = normal @ mat
                        pos = mirror_object.matrix_world.translation
                    else:
                        pos = mathutils.Vector((0,0,0))
                    q1 = normal.dot(pos - p1)
                    q2 = normal.dot(pos - p2)
                    if (q1 < -TH and q2 > TH) or (q1 > TH and q2 < -TH):
                        return i, axis
    return False

def make_curve_mirror_bridge(curve, target, other, mirror_obj, axis):
    curve_name = curve.greg_curve_settings.name
    add_mirror_empty_constraints(other, target, mirror_obj, axis, curve_name)
    if target == curve.greg_curve_settings.end1_empty:
        target_end_name = curve.greg_curve_settings.end1_name
        other_end_name = curve.greg_curve_settings.end2_name
        target_point = curve.data.splines[0].bezier_points[0]
        other_point = curve.data.splines[0].bezier_points[1]
        other_i = 1
    elif target == curve.greg_curve_settings.end2_empty:
        target_end_name = curve.greg_curve_settings.end2_name
        other_end_name = curve.greg_curve_settings.end1_name
        target_point = curve.data.splines[0].bezier_points[1]
        other_point = curve.data.splines[0].bezier_points[0]
        other_i = 0
    new_other_name = target.greg_empty_settings.mirror_bridge_other_names.add()
    new_other_name.name = other.greg_empty_settings.name
    new_target_name = other.greg_empty_settings.mirror_bridge_other_names.add()
    new_target_name.name = target.greg_empty_settings.name
    curve.greg_curve_settings.is_mirror_bridge = True
    curve.greg_curve_settings.bridge_mirror_object = mirror_obj
    curve.greg_curve_settings.bridge_mirror_axis = axis
    curve.greg_curve_settings.bridge_mirror_other_i = other_i
    return target_end_name, other_end_name, target_point, other_point

def add_mirror_empty_constraints(empty, target, mirror_object, axis, curve_name):
    constraint = empty.constraints.new('COPY_LOCATION')
    if mirror_object:
        constraint.owner_space = 'CUSTOM'
        constraint.space_object = mirror_object
    else:
        constraint.owner_space = 'WORLD'
    constraint.use_x = True
    constraint.use_y = True
    constraint.use_z = True
    if axis == 0:
        constraint.invert_x = True
    elif axis == 1:
        constraint.invert_y = True
    elif axis == 2:
        constraint.invert_z = True
    constraint.target = target
    constraint.name = f"copy_location_{curve_name}"

    constraint2 = empty.constraints.new('COPY_ROTATION')
    if mirror_object:
        constraint2.owner_space = 'CUSTOM'
        constraint2.space_object = mirror_object
    else:
        constraint2.owner_space = 'WORLD'
    constraint2.use_x = True
    constraint2.use_y = True
    constraint2.use_z = True
    if axis == 0:
        constraint2.invert_y = True
        constraint2.invert_z = True
    elif axis == 1:
        constraint2.invert_x = True
        constraint2.invert_z = True
    elif axis == 2:
        constraint2.invert_x = True
        constraint2.invert_y = True
    constraint2.target = target
    constraint2.name = f"copy_rotation_{curve_name}"

def coplanar_collinear_add_one_end(empty_obj, collection, new_end):
    for end in empty_obj.greg_empty_settings.curve_ends:
        apply_hook(end)
    
    for end1 in empty_obj.greg_empty_settings.curve_ends:
        if end1 != new_end:
            check_ends_collinear(end1, new_end)
    collinear_groups = []
    used_dict = {end.name: False for end in empty_obj.greg_empty_settings.curve_ends}
    for end in empty_obj.greg_empty_settings.curve_ends:
        if not end.is_collinear:
            collinear_groups.append([end])
        else:
            if used_dict[end.name] == False:
                used_dict[end.name] = True
                collinear_groups.append([end])
                for basic_end in end.collinear_to:
                    collinear_end_name = basic_end.name
                    used_dict[collinear_end_name] = True
                    collinear_groups[-1].append(empty_obj.greg_empty_settings.curve_ends[collinear_end_name])
    sets_and_vectors = []
    for i, ends1 in enumerate(collinear_groups):
        for j, ends2 in enumerate(collinear_groups[:i]):
            for ends3 in collinear_groups[:j]:
                if new_end in ends1 or new_end in ends2 or new_end in ends3:
                    res = check_ends_coplanar(ends1, ends2, ends3, empty_obj, collection)
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
        common_arrows = None
        for end in se:
            if end != new_end:
                if common_arrows is None:
                    common_arrows = set(vec.arrow for vec in end.coplanar_vectors)
                else:
                    common_arrows = common_arrows.intersection(set(vec.arrow for vec in end.coplanar_vectors))
        if len(common_arrows) > 0:
            target_arrow = common_arrows[0]
            add_one_end_to_arrow(new_end, target_arrow)
        else:
            add_coplanar_arrow(se, vec, empty_obj, collection)
    
    for end in empty_obj.greg_empty_settings.curve_ends:
        add_hook(end)


def add_one_end_to_arrow(end: GregCurveEndItem, arrow_obj: bpy.types.Object):
    end_setting = end.coplanar_vectors.add()
    end_setting.arrow = arrow_obj
    end_setting.name = arrow_obj.greg_arrow_settings.name
    arrow_setting = arrow_obj.greg_arrow_settings.coplanars.add()
    arrow_setting.name = end.name
    arrow_setting.curve = end.basic_end.curve
    arrow_setting.end = end.basic_end.end

def extract_end_name_from_curve_and_i(curve: bpy.types.Object, i: int):
    if i == 0:
        end_name = curve.greg_curve_settings.end1_name
    elif i == 1:
        end_name = curve.greg_curve_settings.end2_name
    else:
        raise Exception("I must be 0 or 1!")
    return end_name

def get_parent_collection(obj):
    for coll in obj.users_collection:
        if bpy.context.scene.user_of_id(coll):
            return coll

def get_angles(ve: mathutils.Vector) -> Tuple[float, float]:
    r = ve.length
    th = np.arccos(ve.z / r)
    xy = np.sqrt(ve.x**2 + ve.y**2)
    if xy == 0:
        ph = 0
    else:
        ph = np.sign(ve.y) * np.arccos(ve.x / xy)
    return th, ph
