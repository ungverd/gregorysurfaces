# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

from typing import List, Optional, Tuple, Dict, Set
import numpy as np


import bpy
import mathutils

from .propertyGroups import *
from .commons import apply_hook, add_hook, get_greg_collection, are_collinear
from .commons import rotate_end_to_vec, remove_coplanar, add_curve_obj, add_empty_obj, add_curve_end
from .commons import extract_vectors_from_ends, are_coplanar
from .commons import add_coplanar_arrow, extract_end_from_basic_end_and_empty
from .commons import extract_end_from_name_and_empty, extract_vector_from_basic_end
from .commons import make_collinear, extract_end_name_from_curve_and_i
from .commons import coplanar_collinear_add_one_end, add_one_end_to_arrow
from .OnDepsgraphUpdate import on_depsgraph_update
from .CbUpdatePatches import cb_update
from .CreateSurfacesBetweenCurves import CreateSurfacesBetweenCurves, add_surface_menu_func
from .MakeCurveMirrorBridge import MakeCurveMirrorBridge, add_bridge_mirror_func
from .GregSubdivide import GregSubdivide, add_greg_subdivide_func
from .CreateCurvesCollection import CreateCurvesCollection, add_collection_menu_func
from .PartialGlobalList import PartialGlobalList, PartialCurve



def print_structure(collection):
    print("collection greg", collection.greg_settings.used_for_greg)
    for item in collection.greg_settings.empties:
        print("empty", item.name)
        print("used",  item.empty.greg_empty_settings.used_for_greg)
        print("name again",  item.empty.greg_empty_settings.name)
        print("ends")
        for end in item.empty.greg_empty_settings.curve_ends:
            print("    name", end.name)
            print("    basic end name", end.basic_end.name)
            print("    basic end curve name", end.basic_end.curve.greg_curve_settings.name)
            print("    basic end i", end.basic_end.end)
            print("    empty name", end.empty.greg_empty_settings.name)
            print("    is coplanar", end.is_coplanar)
            print("    len coplanar vectors", len(end.coplanar_vectors))
            for coplanar_vector in end.coplanar_vectors:
                print("        arrow name", coplanar_vector.name)
                print("        internal arrow name", coplanar_vector.arrow.greg_arrow_settings.name)
            print("    is collinear", end.is_collinear)
            print("    len collinear_to", len(end.collinear_to))
            for basic_end in end.collinear_to:
                print("        collinear name", basic_end.name)
                print("        collinear curve name", basic_end.curve.greg_curve_settings.name)
                print("        collinear i", basic_end.end)
            print("    hook", end.hook)
        print("arrows")
        for arrow in item.empty.greg_empty_settings.coplanars:
            print("    arrow name", arrow.name)
            print("    internal arrow name", arrow.arrow.greg_arrow_settings.name)
    for item in collection.greg_settings.curves:
        print("curve", item.name)
        print("used",  item.curve.greg_curve_settings.used_for_greg)
        print("end1", item.curve.greg_curve_settings.end1_name)
        print("end1_empty", item.curve.greg_curve_settings.end1_empty.greg_empty_settings.name)
        print("end2", item.curve.greg_curve_settings.end2_name)
        print("end2_empty", item.curve.greg_curve_settings.end2_empty.greg_empty_settings.name)
    for item in collection.greg_settings.arrows:
        print("arrow", item.name)
        print("used",  item.arrow.greg_arrow_settings.used_for_greg)
        print("name again",  item.arrow.greg_arrow_settings.name)
        print("len ends", len(item.arrow.greg_arrow_settings.coplanars))
        print("ends")
        for basic_end in item.arrow.greg_arrow_settings.coplanars:
            print("    basic end name", basic_end.name)
            print("    basic end curve name", basic_end.curve.greg_curve_settings.name)


def get_possible_curves_dict(curve, end):
    return {curve_end.basic_end.curve.greg_curve_settings.name: curve_end.basic_end.end
            for curve_end in end.empty.greg_empty_settings.curve_ends
            if curve_end.basic_end.curve != curve}

def add_one_curve(curves, is_added, now_curve, end_to_add):
    possible_curves = get_possible_curves_dict(now_curve, end_to_add)
    if len(possible_curves) == 0:
        return False
    for curve in curves:
        curve_name = curve.greg_curve_settings.name
        if not is_added[curve_name]:
            if curve_name in possible_curves:
                is_added[curve_name] = True
                i = possible_curves[curve_name]
                if i == 0:
                    new_empty = curve.greg_curve_settings.end2_empty
                    new_name = curve.greg_curve_settings.end2_name
                else:
                    new_empty = curve.greg_curve_settings.end1_empty
                    new_name = curve.greg_curve_settings.end1_name
                new_end = new_empty.greg_empty_settings.curve_ends[new_name]
                return is_added, curve, new_end
    return False

def verify_is_closed_path_and_of_length_4(curves: List[bpy.types.Object]):
    is_added = {curve.greg_curve_settings.name: False for curve in curves}
    first_curve = curves[0]
    first_empty = first_curve.greg_curve_settings.end1_empty
    first_name = first_curve.greg_curve_settings.end1_name
    first_end = first_empty.greg_empty_settings.curve_ends[first_name]
    is_added[curves[0].greg_curve_settings.name] = True
    curve = first_curve
    end = first_end
    for i in range(len(curves) - 1):
        res = add_one_curve(curves, is_added, curve, end)
        if not res:
            return False
        is_added, curve, end = res
    # verify that path is closed
    possible_curves = get_possible_curves_dict(curve, end)
    if len(possible_curves) == 0:
        return False
    first_curve_name = first_curve.greg_curve_settings.name
    if not first_curve_name in possible_curves:
        return False
    if possible_curves[first_curve_name] == 0:
        return False
    return True # TODO

def print_end(end):
    print("end:")
    print("\tbasic end name", end.basic_end.name)
    print("\tbasic end curve name", end.basic_end.curve.greg_curve_settings.name)
    print("\tbasic end i", end.basic_end.end)
    print("\tempty name", end.empty.greg_empty_settings.name)
    print("\tis coplanar", end.is_coplanar)
    print("\tcoplanar_vectors:")
    for i, arrow in enumerate(end.coplanar_vectors):
        print("\t  arrow", i+1)
        print("\t\tarrow name", arrow.name)
        print("\t\tarrow object name", arrow.arrow.name)
    print("\tis collinear", end.is_collinear)
    print("\tcollinear_to:")
    for i, basic_end in enumerate(end.collinear_to):
        print("\t  collinear end", i+1)
        print("\t\tcollinear end name", basic_end.name)
        print("\t\tcollinear end curve name", basic_end.curve.greg_curve_settings.name)
        print("\t\tcollinear end i", basic_end.end)
    print("\thook", end.hook)
    print("\tname", end.name)

class OBJECT_PT_greg_curve_properties(bpy.types.Panel):
    bl_idname = "OBJECT_PT_greg_curve_propertirs"
    bl_label = "Gregory Curve Properties"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gregory"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.prop(obj, 'greg_bulge')
        layout.prop(obj, 'greg_tilt')
        layout.prop(obj, 'greg_shear')
        layout.prop(obj, 'greg_is_sharp')

    @classmethod    
    def poll(cls, context):
        obj = context.active_object
        if obj is None:
            return False
        if obj.greg_curve_settings.used_for_greg:
            if not obj.greg_is_sharp:
                collection = get_greg_collection(obj)
                phantom_curves = collection.greg_settings.phantom_curves
                for phantom_curve_id in obj.greg_curve_settings.phantom_curves_ids:
                    phantom_curve = phantom_curves[phantom_curve_id.name]
                    if not phantom_curve.conditional_sharp:
                        return True
        return False

class OBJECT_PT_greg_curve_properties1(bpy.types.Panel):
    bl_idname = "OBJECT_PT_greg_curve_propertirs1"
    bl_label = "Gregory Curve Properties 1"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gregory"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.prop(obj, 'greg_bulge1')
        layout.prop(obj, 'greg_tilt1')
        layout.prop(obj, 'greg_shear1')

    @classmethod    
    def poll(cls, context):
        obj = context.active_object
        if obj is None:
            return False
        if obj.greg_curve_settings.used_for_greg:
            if obj.greg_is_sharp:
                return True
            collection = get_greg_collection(obj)
            phantom_curves = collection.greg_settings.phantom_curves
            for phantom_curve_id in obj.greg_curve_settings.phantom_curves_ids:
                phantom_curve = phantom_curves[phantom_curve_id.name]
                if phantom_curve.conditional_sharp:
                    return True
        return False
        


class OBJECT_PT_greg_curve_properties2(bpy.types.Panel):
    bl_idname = "OBJECT_PT_greg_curve_propertirs2"
    bl_label = "Gregory Curve Properties 2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gregory"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.prop( obj, 'greg_bulge2')
        layout.prop( obj, 'greg_tilt2')
        layout.prop( obj, 'greg_shear2')

    @classmethod    
    def poll(cls, context):
        obj = context.active_object
        if obj is None:
            return False
        if obj.greg_curve_settings.used_for_greg:
            collection = get_greg_collection(obj)
            phantom_curves = collection.greg_settings.phantom_curves
            for phantom_curve_id in obj.greg_curve_settings.phantom_curves_ids:
                phantom_curve = phantom_curves[phantom_curve_id.name]
                if obj.greg_is_sharp or phantom_curve.conditional_sharp:
                    if len(phantom_curve.quads) == 2:
                        return True
        return False

class OBJECT_PT_greg_resolution(bpy.types.Panel):
    bl_idname = "OBJECT_PT_greg_resolution"
    bl_label = "Mesh Resolution"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gregory"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.prop(obj, 'greg_resolution')

    @classmethod    
    def poll(cls, context):
        obj = context.active_object
        if obj is None:
            return False
        return obj.greg_is_generated

class PrintItemInfo(bpy.types.Operator):
    """Gregory: print info about selected curve, arrow or empty"""
    bl_idname = "object.greg_print_info"
    bl_label = "Greg: print greg item info"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        obj = context.active_object
        collection = get_greg_collection(obj)
        if obj.greg_curve_settings.used_for_greg:
            settings = obj.greg_curve_settings
            print("******")
            print("Curve")
            print(settings.name)
            coll_curves = collection.greg_settings.curves
            is_in_collection = coll_curves.find(settings.name) != -1
            print("is in collection", is_in_collection)
            if is_in_collection:
                print("object name", coll_curves[settings.name].curve.name)
            print("end1 name", settings.end1_name)
            empty1_name = settings.end1_empty.greg_empty_settings.name
            print("end1 empty name", empty1_name)
            print("end2 name", settings.end2_name)
            empty2_name = settings.end2_empty.greg_empty_settings.name
            print("end2 empty name", empty2_name)
            for phantom_curve_el in settings.phantom_curves_ids:
                phantom_curve_name = phantom_curve_el.name
                phantom_curve = collection.greg_settings.phantom_curves[phantom_curve_name]
                print("phantom_curve name", phantom_curve.name)
                print("sharp", phantom_curve.conditional_sharp)
            print()
        elif obj.greg_empty_settings.used_for_greg:
            settings = obj.greg_empty_settings
            print("******")
            print("Empty")
            print(settings.name)
            coll_empties = collection.greg_settings.empties
            is_in_collection = coll_empties.find(settings.name) != -1
            print("is in collection", is_in_collection)
            if is_in_collection:
                print("object name", coll_empties[settings.name].empty.name)
            print("ends:")
            for i, end in enumerate(settings.curve_ends):
                print("end", i+1)
                print_end(end)
            print("arrows:")
            for i, arrow in enumerate(settings.coplanars):
                print("  arrow", i+1)
                print("\tarrow name", arrow.name)
                print("\tarrow object name", arrow.arrow.name)
            print()
        elif obj.greg_arrow_settings.used_for_greg:
            settings = obj.greg_arrow_settings
            print("******")
            print("Arrow")
            print(settings.name)
            coll_arrows = collection.greg_settings.arrows
            is_in_collection = coll_arrows.find(settings.name) != -1
            print("is in collection", is_in_collection)
            if is_in_collection:
                print("object name", coll_arrows[settings.name].arrow.name)
            print("ends")
            for i, basic_end in enumerate(settings.coplanars):
                print("  end", i+1)
                print("\tend name", basic_end.name)
                print("\tend curve name", basic_end.curve.greg_curve_settings.name)
                print("\tend i", basic_end.end)
            print()
        else:
            print("Not used for greg!")

        return {'FINISHED'}    

def add_print_info_func(self, context: bpy.types.Context):
    self.layout.operator(PrintItemInfo.bl_idname)


class UnsetCurveMirrorBridge(bpy.types.Operator):
    """Gregory: unset curve a bridge through mirror"""
    bl_idname = "object.unset_curve_mirror_bridge"
    bl_label = "Greg: unset curve bridge through mirror"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        if len(selected) != 1:
            return False
        obj = selected[0]
        return obj.greg_curve_settings.is_mirror_bridge

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        curve = context.selected_objects[0]
        curve_settings = curve.greg_curve_settings
        curve_settings.is_mirror_bridge = False
        curve_settings.bridge_mirror_object = None
        curve_settings.bridge_mirror_axis = 0
        if curve_settings.bridge_mirror_other_i == 0:
            other = curve_settings.end1_empty
            target = curve_settings.end2_empty
        elif curve_settings.bridge_mirror_other_i == 1:
            other = curve_settings.end2_empty
            target = curve_settings.end1_empty
        curve_settings.bridge_mirror_other_i = 0
        other_name = other.greg_empty_settings.name
        target_name = target.greg_empty_settings.name
        other.greg_empty_settings.mirror_bridge_other_names.remove(other.greg_empty_settings.mirror_bridge_other_names.find(target_name))
        target.greg_empty_settings.mirror_bridge_other_names.remove(target.greg_empty_settings.mirror_bridge_other_names.find(other_name))
        curve_name = curve_settings.name
        for constraint in other.constraints:
            if constraint.name == f"copy_location_{curve_name}":
                other.constraints.remove(constraint)
                break
        for constraint in other.constraints:
            if constraint.name == f"copy_rotation_{curve_name}":
                other.constraints.remove(constraint)
                break
        return {'FINISHED'}    

def add_unset_bridge_mirror_func(self, context: bpy.types.Context):
    self.layout.operator(UnsetCurveMirrorBridge.bl_idname)


def copy_mirrors_from_one_obj_to_another(obj_with_mirrors: bpy.types.Object,
                                         obj_without_mirrors: bpy.types.Object):
    i = 1
    for modifier in obj_with_mirrors.modifiers:
        if modifier.type == 'MIRROR':
            new_mirror = obj_without_mirrors.modifiers.new(f"mirror_{i}", 'MIRROR')
            new_mirror.mirror_object = modifier.mirror_object
            new_mirror.use_axis = modifier.use_axis
            i += 1


class GregExtrude(bpy.types.Operator):
    """Gregory: extrude curve"""
    bl_idname = "object.greg_extrude"
    bl_label = "Greg: extrude curve"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        empties = [obj for obj in selected if obj.greg_empty_settings.used_for_greg]
        if len(empties) != 1:
            return False
        empty = empties[0]
        if len(selected) > 2:
            return False
        candidates = [obj for obj in selected if obj != empty]
        if len(candidates) > 1:
            return False
        if len(candidates) == 1:
            if not candidates[0].greg_curve_settings.used_for_greg:
                return False
            curve = candidates[0]
            empties = [curve.greg_curve_settings.end1_empty, curve.greg_curve_settings.end2_empty]
            if empty not in empties:
                return False
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        selected = context.selected_objects
        empties = [obj for obj in selected if obj.greg_empty_settings.used_for_greg]
        empty = empties[0]
        collection = get_greg_collection(empty)
        candidates = [obj for obj in selected if obj != empty]
        for obj in selected:
            obj.select_set(False)
        co = empty.matrix_world.translation
        if len(candidates) == 1:
            curve = candidates[0]
            empties = [curve.greg_curve_settings.end1_empty, curve.greg_curve_settings.end2_empty]
            if empty == empties[0]:
                end_name = curve.greg_curve_settings.end1_name
                i = 0
            else:
                end_name = curve.greg_curve_settings.end2_name
                i = 1
            end = empty.greg_empty_settings.curve_ends[end_name]
            handle_left, handle_right = apply_hook_and_get_handles_from_end(end, curve, i, co)
        elif len(candidates) == 0:
            num_ends = len(empty.greg_empty_settings.curve_ends)
            if num_ends == 0:
                handle_left = mathutils.Vector((1, 0, 0))
                handle_right = mathutils.Vector((-1, 0, 0))
            elif num_ends == 1:
                end = empty.greg_empty_settings.curve_ends[0]
                curve = end.basic_end.curve
                i = end.basic_end.end
                handle_left, handle_right = apply_hook_and_get_handles_from_end(end, curve, i, co)
            else:
                for end in empty.greg_empty_settings.curve_ends:
                    apply_hook(end)
                vecs = extract_vectors_from_ends(empty.greg_empty_settings.curve_ends)
                handle_left, handle_right = get_handles_from_vecs_to_extrude(vecs)
        add_curve_by_extrude(empty, co, handle_left, handle_right, collection, context)
        return {'FINISHED'}    

def add_greg_extrude_func(self, context: bpy.types.Context):
    self.layout.operator(GregExtrude.bl_idname)

def apply_hook_and_get_handles_from_end(end: GregCurveEndItem,
                                        curve: bpy.types.Object,
                                        i: int,
                                        co: mathutils.Vector) -> Tuple[mathutils.Vector, mathutils.Vector]:
    apply_hook(end)
    add_hook(end)
    handle_left = curve.data.splines[0].bezier_points[i].handle_left - co
    handle_right = curve.data.splines[0].bezier_points[i].handle_right - co
    if i == 0:
        handle_right, handle_left = handle_left, handle_right
    return handle_left, handle_right

def add_curve_by_extrude(empty: bpy.types.Object,
                         co: mathutils.Vector,
                         handle_left: mathutils.Vector,
                         handle_right: mathutils.Vector,
                         collection: bpy.types.Collection,
                         context: bpy.types.Context):
    co_s = [co] * 2
    handles_left = [handle_left] * 2
    handles_right = [handle_right] * 2
    new_curve_obj, _ = add_curve_obj(collection, co_s, handles_left, handles_right)
    new_empty_obj = add_empty_obj(collection, co_s[0])
    old_end_name = add_curve_end(collection, empty, new_curve_obj, 0)
    new_end_name = add_curve_end(collection, new_empty_obj, new_curve_obj, 1)
    old_new_end = empty.greg_empty_settings.curve_ends[old_end_name]
    coplanar_collinear_add_one_end(empty, collection, old_new_end)
    add_hook(old_new_end)
    new_new_end = new_empty_obj.greg_empty_settings.curve_ends[new_end_name]
    add_hook(new_new_end)
    new_empty_obj.select_set(True)
    context.view_layer.objects.active = new_empty_obj
    bpy.ops.transform.translate('INVOKE_DEFAULT')

def get_handles_from_vecs_to_extrude(vecs: List[mathutils.Vector]) -> Tuple[mathutils.Vector, mathutils.Vector]:
    mean_length = sum([vec.length for vec in vecs]) / len(vecs)
    if verify_all_vecs_collinear(vecs):
        result_vec = get_perpendicular_vec(vecs[0])
    else:
        result_vec = vector_mean(vecs)
        if result_vec.length < mean_length * 0.3:
            result_vec = get_normal_vector(vecs)
    handle_left = result_vec.normalized() * mean_length
    return handle_left, -handle_left

def get_perpendicular_vec(vec: mathutils.Vector):
    def setone(lst, i):
        lst[i] = 1
        return lst
    xyz = [vec.cross(mathutils.Vector(setone([0, 0, 0], i))) for i in range(3)]
    return max(xyz, key=lambda x: x.length)

def verify_all_vecs_collinear(vecs: List[mathutils.Vector]):
    vec0 = vecs[0]
    for vec in vecs[1:]:
        if not are_collinear(vec0, vec):
            return False
    return True

def vector_mean(vecs: List[mathutils.Vector]):
    accumulator = mathutils.Vector((0,0,0))
    for vec in vecs:
        accumulator += vec
    return accumulator / len(vecs)

def merge_poll(context):
    if context.mode != "OBJECT":
        return False
    if len(context.selected_objects) != 2:
        return False
    for object in context.selected_objects:
        if not object.greg_empty_settings.used_for_greg:
            return False
    if get_greg_collection(context.selected_objects[0]) != get_greg_collection(context.selected_objects[1]):
        return False
    return True

def merge_empties(empty1, empty2):
    collection = get_greg_collection(empty1)
    for end in empty1.greg_empty_settings.curve_ends:
        apply_hook(end)
    for end in empty2.greg_empty_settings.curve_ends:
        apply_hook(end)
        curve_obj = end.basic_end.curve
        end_i = end.basic_end.end
        new_end_name = add_curve_end(collection, empty1, curve_obj, end_i)
        new_end = empty1.greg_empty_settings.curve_ends[new_end_name]
        coplanar_collinear_add_one_end(empty1, collection, new_end)
    for end in empty1.greg_empty_settings.curve_ends:
        add_hook(end)
    #delete old empty2
    empty2.greg_empty_settings.curve_ends.clear()
    arrows_to_delete = []
    for arrow in empty2.greg_empty_settings.coplanars:
        arrows_to_delete.append(arrow.arrow)
        collection.greg_settings.arrows.remove(collection.greg_settings.arrows.find(arrow.name))
    empty2.greg_empty_settings.coplanars.clear()
    objs = bpy.data.objects
    for obj in arrows_to_delete:
        objs.remove(obj, do_unlink=True)
    collection.greg_settings.empties.remove(collection.greg_settings.empties.find(empty2.greg_empty_settings.name))
    objs.remove(empty2, do_unlink=True)


class GregMergeAtCenter(bpy.types.Operator):
    """Gregory: merge two points at center"""
    bl_idname = "object.greg_merge_center"
    bl_label = "merge at center"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return merge_poll(context)

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        empty1, empty2 = context.selected_objects
        # merge at center
        center_location = (empty1.location + empty2.location) / 2
        empty1.location = center_location
        empty2.location = center_location
        merge_empties(empty1, empty2)
        return {'FINISHED'}

def get_first_second_from_two_selected(context: bpy.types.Context):
    objs = context.selected_objects
    last = context.active_object
    first = [obj for obj in objs if obj != last][0]
    return first, last

class GregMergeAtFirst(bpy.types.Operator):
    """Gregory: merge two points at first"""
    bl_idname = "object.greg_merge_first"
    bl_label = "merge at first"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return merge_poll(context)

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        empty1, empty2 = get_first_second_from_two_selected(context)
        # merge at first
        empty2.location = empty1.location
        merge_empties(empty1, empty2)
        return {'FINISHED'}

class GregMergeAtLast(bpy.types.Operator):
    """Gregory: merge two points at last"""
    bl_idname = "object.greg_merge_last"
    bl_label = "merge at last"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return merge_poll(context)

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        empty1, empty2 = get_first_second_from_two_selected(context)
        # merge at last
        empty1.location = empty2.location
        merge_empties(empty1, empty2)
        return {'FINISHED'} 

class GregMergeSub(bpy.types.Menu):
    bl_label = 'Greg: merge'
    bl_idname = 'VIEW3D_MT_merge_submenu'

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return merge_poll(context)

    def draw(self, context):
        layout = self.layout
        #layout.label("This is a submenu")
        layout.operator(GregMergeAtCenter.bl_idname)
        layout.operator(GregMergeAtFirst.bl_idname)
        layout.operator(GregMergeAtLast.bl_idname)

def add_greg_merge_func(self, context: bpy.types.Context):
    self.layout.menu(GregMergeSub.bl_idname)

class PrintDotInfo(bpy.types.Operator):
    """Gregory: print info about selected curve, arrow or empty"""
    bl_idname = "object.greg_print_dot"
    bl_label = "Print greg dot product"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        if len(selected) != 2:
            return False
        there_is_arrow = False
        there_is_curve = False
        for object in selected:
            if object.greg_arrow_settings.used_for_greg == True:
                there_is_arrow = True
            if object.greg_curve_settings.used_for_greg == True:
                there_is_curve = True
        if (not there_is_arrow) or (not there_is_curve):
            return False
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        selected = context.selected_objects
        for object in selected:
            if object.greg_arrow_settings.used_for_greg == True:
                arrow_obj = object
            if object.greg_curve_settings.used_for_greg == True:
                curve_obj = object
        empty_obj = arrow_obj.parent
        curve_settings = curve_obj.greg_curve_settings
        for j, empty in enumerate((curve_settings.end1_empty, curve_settings.end2_empty)):
            if empty.name == empty_obj.name:
                 i = j
        depsgraph = context.evaluated_depsgraph_get()
        p = curve_obj.evaluated_get(depsgraph).data.splines[0].bezier_points[i]
        co = p.co
        h = p.handle_right if i == 0 else p.handle_left
        hh = h - co
        mat = arrow_obj.matrix_world.copy()
        mat.invert()
        vec = mathutils.Vector((0,0,1)) @ mat
        print(vec.dot(hh), vec, hh)
        

        return {'FINISHED'}    

def add_print_dot_func(self, context: bpy.types.Context):
    self.layout.operator(PrintDotInfo.bl_idname)        

class SetNotPatch(bpy.types.Operator):
    """Gregory: set loop of curves not patch"""
    bl_idname = "object.greg_set_not_patch"
    bl_label = "Greg: set loop not patch"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        if len(selected) < 1 or len(selected) > 4:
            return False
        prev_collection = None
        for obj in context.selected_objects:
            if not obj.greg_curve_settings.used_for_greg:
                return False
            if prev_collection is None:
                prev_collection = get_greg_collection(obj)
            else:
                if get_greg_collection(obj) != prev_collection:
                    return False
        if len(selected) == 4:
            if not verify_is_closed_path_and_of_length_4(selected):
                return False
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        col = bpy.data.collections.new("NotPatchCollection")
        col.greg_is_not_face = True
        for obj in context.selected_objects:
            col.objects.link(obj)
        

        return {'FINISHED'}

def set_not_face_menu_func(self, context: bpy.types.Context):
    self.layout.operator(SetNotPatch.bl_idname)


def verify_empty_is_other_mirrored(empty: bpy.types.Object):
    for curve_end in empty.greg_empty_settings.curve_ends:
        curve_settings = curve_end.basic_end.curve.greg_curve_settings
        if curve_settings.is_mirror_bridge:
            if curve_settings.bridge_mirror_other_i == 0:
                if empty == curve_settings.end1_empty:
                    return True
            elif curve_settings.bridge_mirror_other_i == 1:
                if empty == curve_settings.end2_empty:
                    return True
    return False


class AddBezierCurve(bpy.types.Operator):
    """Gregory: add curve to structure"""     # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_add_curve_to_structure"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: add curve"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        if len(context.selected_objects) != 2:
            return False
        for obj in context.selected_objects:
            if obj.type != 'CURVE':
                return False
        counted = [obj for obj in context.selected_objects if obj.greg_curve_settings.used_for_greg]
        if len(counted) != 1:
            return False
        return True
    
    def execute(self, context: bpy.types.Context):
        in_structure = [obj for obj in context.selected_objects if obj.greg_curve_settings.used_for_greg][0]
        to_add = [obj for obj in context.selected_objects if obj != in_structure][0]
        #apply transforms
        mb = to_add.matrix_basis
        to_add.data.transform(mb)
        to_add.matrix_basis.identity()
        collection = get_greg_collection(in_structure)
        collection_empties = [e.empty for e in collection.greg_settings.empties if not verify_empty_is_other_mirrored(e.empty)]
        p_glist = PartialGlobalList()
        for spline in to_add.data.splines:
            for i in range(len(spline.bezier_points) - 1):
                p1 = spline.bezier_points[i]
                p2 = spline.bezier_points[i + 1]
                prev_ready = (i > 0)
                PartialCurve(p1, p2, collection_empties, p_glist, prev_ready)
            if spline.use_cyclic_u:
                p_last = spline.bezier_points[-1]
                p_first = spline.bezier_points[0]
                PartialCurve(p_last, p_first, collection_empties, p_glist, True)
        p_glist.add_curves_and_empties(collection)
        bpy.data.objects.remove(to_add, do_unlink=True)
        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def add_bezier_curve_menu_func(self, context: bpy.types.Context):
    self.layout.operator(AddBezierCurve.bl_idname)

class SetNotCoplanar(bpy.types.Operator):
    """Gregory: set not coplanar"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_set_not_coplanar"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: set not coplanar"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        if len(selected) < 2:
            return False
        curves = [obj for obj in selected if obj.greg_curve_settings.used_for_greg]
        if len(curves) < 1:
            return False
        empty_l = [obj for obj in selected if obj.greg_empty_settings.used_for_greg]
        if len(empty_l) != 1:
            return False
        empty = empty_l[0]
        connected_curves = [end.basic_end.curve for end in empty.greg_empty_settings.curve_ends]
        for curve in curves:
            if curve not in connected_curves:
                return False
        for curve in curves:
            empties = [curve.greg_curve_settings.end1_empty, curve.greg_curve_settings.end2_empty]
            if empty in empties:
                if empty == curve.greg_curve_settings.end1_empty:
                    end_name = curve.greg_curve_settings.end1_name
                else:
                    end_name = curve.greg_curve_settings.end2_name
                if not empty.greg_empty_settings.curve_ends[end_name].is_coplanar:
                    return False
            else:
                return False
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        selected = context.selected_objects
        curves = [obj for obj in selected if obj.greg_curve_settings.used_for_greg]
        empty = [obj for obj in selected if obj.greg_empty_settings.used_for_greg][0]
        for curve in curves:
            if empty == curve.greg_curve_settings.end1_empty:
                end_name = curve.greg_curve_settings.end1_name
            else:
                end_name = curve.greg_curve_settings.end2_name
            end = empty.greg_empty_settings.curve_ends[end_name]
            collection = get_greg_collection(curve)
            ends_to_remove_coplanar = [end]
            if end.is_colliniar:
                for basic_end in end.collinear_to:
                    collinear_end = empty.greg_empty_settings.curve_ends[basic_end.name]
                    ends_to_remove_coplanar.append(collinear_end)
            for end in ends_to_remove_coplanar:
                apply_hook(end)
                remove_coplanar(end, end_name, empty, collection)
                end.coplanar_vectors.clear()
                add_hook(end)

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def set_not_coplanar_menu_func(self, context: bpy.types.Context):
    self.layout.operator(SetNotCoplanar.bl_idname)

class SetCoplanar(bpy.types.Operator):
    """Gregory: set coplanar"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_set_coplanar"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: set coplanar"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        curves = [obj for obj in selected if obj.greg_curve_settings.used_for_greg]
        empties = [obj for obj in selected if obj.greg_empty_settings.used_for_greg]
        if len(empties) != 1:
            return False
        if len(curves) < 2:
            return False
        empty = empties[0]
        collection = get_greg_collection(empty)
        if len(curves) + 1 != len(selected):
            return False
        for curve in curves:
            if get_greg_collection(curve) != collection:
                return False
        connected_curves = [end.basic_end.curve for end in empty.greg_empty_settings.curve_ends]
        for curve in curves:
            if curve not in connected_curves:
                return False
        ends = get_ends_from_curves_connected_to_one_empty(curves, empty)
        if len(curves) == 2:
            if are_all_ends_not_coplanar(ends):
                return False
        if are_all_ends_not_coplanar(ends):
            return True
        else:
            one_coplanar = return_one_coplanar_plane_if_exists(ends)
            if one_coplanar is not None:
                return True
            else:
                res = return_normal_to_two_coplanaras_if_exist(ends)
                if res is not None:
                    return True
        return False

    def execute(self, context: bpy.types.Context):
        selected = context.selected_objects
        curves = [obj for obj in selected if obj.greg_curve_settings.used_for_greg]
        empty = [obj for obj in selected if obj.greg_empty_settings.used_for_greg][0]
        ends = get_ends_from_curves_connected_to_one_empty(curves, empty)
        collection = get_greg_collection(empty)
        if are_all_ends_not_coplanar(ends):
            assert len(ends) >= 3
            make_not_coplanar_ends_coplanar(ends, empty, collection)
        else:
            one_coplanar = return_one_coplanar_plane_if_exists(ends)
            if one_coplanar is not None:
                turn_to_defined_plane(ends, empty, collection, one_coplanar)
            else:
                res = return_normal_to_two_coplanaras_if_exist(ends)
                if res is not None:
                    not_coplanar_ends, normal_to_two_coplanars = res
                    vectors = extract_vectors_from_ends(not_coplanar_ends)
                    turn_ends_to_be_coplanar(not_coplanar_ends, vectors, normal_to_two_coplanars)
                    add_coplanar_arrow(ends, normal_to_two_coplanars, empty, collection)
        return {'FINISHED'}
    
def return_normal_to_two_coplanaras_if_exist(ends: List[GregCurveEndItem]) -> Optional[Tuple[List[GregCurveEndItem], mathutils.Vector]]:
    first_collinear_group = []
    first_arrow = None
    second_collinear_group = []
    second_arrow = None
    not_coplanar_ends = []
    for end in ends:
        if end.is_coplanar:
            if len(end.coplanar_vectors) == 1:
                if first_arrow is None:
                    first_arrow = end.coplanar_vectors[0]
                    first_collinear_group = [end]
                elif end.coplanar_vectors[0] == first_arrow:
                    if end.name in first_collinear_group[0].collinear_to:
                        first_collinear_group.append(end)
                    else:
                        return None
                elif second_arrow is None:
                    second_arrow = end.coplanar_vectors[0]
                    second_collinear_group = [end]
                elif end.coplanar_vectors[0] == second_arrow:
                    if end.name in second_collinear_group[0].collinear_to:
                        second_collinear_group.append(end)
                    else:
                        return None
                else:
                    return None
            else:
                return None
        else:
            not_coplanar_ends.append(end)
    if first_arrow is None or second_arrow is None:
        return None
    vec1, vec2 = extract_vectors_from_ends((first_collinear_group[0], second_collinear_group[0]))
    normal = vec1.cross(vec2)
    return not_coplanar_ends, normal




def get_vector_from_arrow_obj(arrow_obj: bpy.types.Object) -> mathutils.Vector:
    mat = arrow_obj.matrix_world.copy()
    mat.invert()
    return mathutils.Vector((0,0,1)) @ mat

def turn_to_defined_plane(ends: List[GregCurveEndItem], empty: bpy.types.Object, collection: bpy.types.Collection, coplanar: GregArrowItem):
    arrow_obj = coplanar.arrow
    coplanar_vector = get_vector_from_arrow_obj(arrow_obj)
    not_coplanar_ends = [end for end in ends if not end.is_coplanar]
    vectors = extract_vectors_from_ends(not_coplanar_ends)
    turn_ends_to_be_coplanar(not_coplanar_ends, vectors, coplanar_vector)
    for end in not_coplanar_ends:
        add_one_end_to_arrow(end, arrow_obj)
        add_hook(end)


def return_one_coplanar_plane_if_exists(ends: List[GregCurveEndItem]) -> Optional[GregArrowItem]:
    coplanar = None
    for end in ends:
        if end.is_coplanar:
            if len(end.coplanar_vectors) > 1:
                return None
            if coplanar is None:
                coplanar = end.coplanar_vectors[0]
            elif coplanar != end.coplanar_vectors[0]:
                return None
    return coplanar

def set_coplanar_menu_func(self, context: bpy.types.Context):
    self.layout.operator(SetCoplanar.bl_idname)

def get_ends_from_curves_connected_to_one_empty(curves: List[bpy.types.Object],
                                                empty: bpy.types.Object) -> List[GregCurveEndItem]:
    end_names = (curve.greg_curve_settings.end1_name
                 if empty == curve.greg_curve_settings.end1_empty
                 else curve.greg_curve_settings.end2_name
                 for curve in curves)
    ends = [empty.greg_empty_settings.curve_ends[name] for name in end_names]
    return ends

def are_all_ends_not_coplanar(ends: List[GregCurveEndItem]) -> bool:
    for end in ends:
        if end.is_coplanar:
            return False
    return True

def are_all_vectors_coplanar(vectors: List[mathutils.Vector]) -> bool:
    assert len(vectors) >= 3
    for i in range(len(vectors) - 2):
        v1, v2, v3 = vectors[i: i + 3]
        if not are_coplanar(v1, v2, v3):
            return False
    return True

def get_normal_vector(vectors: List[mathutils.Vector]) -> mathutils.Vector:
    xyz = np.array(vectors)
    u, sigma, v = np.linalg.svd(xyz)
    normal = mathutils.Vector(v[2])
    return normal

def get_normal_direction(vector: mathutils.Vector, normal: mathutils.Vector) -> mathutils.Vector:
    return vector - vector.project(normal)

def make_not_coplanar_ends_coplanar(ends: List[GregCurveEndItem],
                                    empty: bpy.types.Object,
                                    collection: bpy.types.Collection):
    vectors = extract_vectors_from_ends(ends)
    normal_vector = get_normal_vector(vectors)
    turn_ends_to_be_coplanar(ends, vectors, normal_vector)
    add_coplanar_arrow(ends, normal_vector, empty, collection)
    
def turn_ends_to_be_coplanar(ends: List[GregCurveEndItem], vectors: List[mathutils.Vector], normal_vector: mathutils.Vector):
    changed_vectors = [get_normal_direction(vector, normal_vector) for vector in vectors]
    for vector, end in zip(changed_vectors, ends):
        rotate_end_to_vec(vector, end.basic_end)

class SetCollinear(bpy.types.Operator):
    """Gregory: set collinear"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_set_collinear"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: set collinear"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        if len(selected) != 2:
            return False
        for obj in selected:
            if not obj.greg_curve_settings.used_for_greg:
                return False
        collection = get_greg_collection(selected[0])
        if collection != get_greg_collection(selected[1]):
            return False
        res = get_common_empty_of_two_curves_if_exists(selected[0], selected[1])
        if res is None:
            return False
        i1, i2, common_empty = res
        return not are_two_curves_collinear(selected[0], selected[1], i1, i2, common_empty)

    def execute(self, context: bpy.types.Context):
        selected = context.selected_objects
        curve1, curve2 = selected
        i1, i2, common_empty = get_common_empty_of_two_curves_if_exists(curve1, curve2)
        i_s = (i1, i2)
        end_names = [extract_end_name_from_curve_and_i(curve, i) for curve, i in zip(selected, i_s)]
        ends = [extract_end_from_name_and_empty(end_name, common_empty) for end_name in end_names]
        v1, v2 = extract_vectors_from_ends(ends)
        new_vecs = make_collinear(v1, v2)
        for vec, end in zip(new_vecs, ends):
            basic_end = end.basic_end
            rotate_end_to_vec(vec, basic_end)
            harmonize_ends(ends, i_s, common_empty)
        end_groups = [[end.basic_end] + list(end.collinear_to) for end in ends]
        for first_basic_ends, second_basic_ends in zip(end_groups, reversed(end_groups)):
            for first_basic_end in first_basic_ends:
                not_basic_end = extract_end_from_basic_end_and_empty(first_basic_end, common_empty)
                for second_basic_end in second_basic_ends:
                    added_basic_end = not_basic_end.collinear_to.add()
                    added_basic_end.curve = second_basic_end.curve
                    added_basic_end.end = second_basic_end.end
                    added_basic_end.name = second_basic_end.name
        first_end = extract_end_from_name_and_empty(end_names[0], common_empty)
        first_vec = extract_vectors_from_ends([first_end])[0]
        for basic_end in first_end.collinear_to:
            vec = extract_vector_from_basic_end(basic_end)
            new_vec = vec.project(first_vec)
            rotate_end_to_vec(new_vec, basic_end)
        return {'FINISHED'}

def harmonize_ends(ends: List[GregCurveEndItem],
                   i_s: Tuple[int, int],
                   common_empty: bpy.types.Object):
    # len(ends) must be 2
    vecs = extract_vectors_from_ends(ends)
    bezier_points = [end.basic_end.curve.data.splines[0].bezier_points[i] for end, i in zip(ends, i_s)]
    co = common_empty.matrix_world.translation
    for i, bezier_point, vec in zip(i_s, bezier_points, reversed(vecs)):
        if i == 0:
            bezier_point.handle_left = co + vec
        else:
            bezier_point.handle_right = co + vec
        bezier_point.handle_right_type == "ALIGNED"
        bezier_point.handle_left_type == "ALIGNED"

def set_collinear_menu_func(self, context: bpy.types.Context):
    self.layout.operator(SetCollinear.bl_idname)

class SetNotCollinear(bpy.types.Operator):
    """Gregory: set not collinear"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_set_not_collinear"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: set not collinear"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        selected = context.selected_objects
        if len(selected) < 2:
            return False
        for obj in selected:
            if not obj.greg_curve_settings.used_for_greg:
                return False
        collection = get_greg_collection(selected[0])
        for obj in selected[1:]:
            if collection != get_greg_collection(obj):
                return False
        res = get_common_empty_of_two_curves_if_exists(selected[0], selected[1])
        if res is None:
            return False
        i1, i2, common_empty = res
        if not are_two_curves_collinear(selected[0], selected[1], i1, i2, common_empty):
            return False
        for curve in selected[2:]:
            if not verify_curve_collinear_to_given(curve, selected[0], i1, common_empty):
                return False
        return True

    def execute(self, context: bpy.types.Context):
        selected = context.selected_objects
        i1, i2, common_empty = get_common_empty_of_two_curves_if_exists(selected[0], selected[1])
        if i1 == 0:
            end_name = selected[0].greg_curve_settings.end1_name
        else:
            end_name = selected[0].greg_curve_settings.end2_name
        end = extract_end_from_name_and_empty(end_name, common_empty)
        for basic_end in end.collinear_to:
            print("name", basic_end.name)
            full_end = extract_end_from_basic_end_and_empty(basic_end, common_empty)
            full_end.collinear_to.clear()
        end.collinear_to.clear()
        return {'FINISHED'}

def set_not_collinear_menu_func(self, context: bpy.types.Context):
    self.layout.operator(SetNotCollinear.bl_idname)


def extract_end_name_from_curve_and_empty(curve: bpy.types.Object, empty: bpy.types.Object) -> Optional[str]:
    if empty == curve.greg_curve_settings.end1_empty:
        end_name = curve.greg_curve_settings.end1_name
    elif empty == curve.greg_curve_settings.end2_empty:
        end_name = curve.greg_curve_settings.end2_name
    else:
        return None
    return end_name

def verify_curve_collinear_to_given(curve: bpy.types.Object,
                                    given: bpy.types.Object,
                                    i_given: int,
                                    common_empty: bpy.types.Object):
    end1_name = extract_end_name_from_curve_and_i(given, i_given)
    end1 = common_empty.greg_empty_settings.curve_ends[end1_name]
    end2_name = extract_end_name_from_curve_and_empty(curve, common_empty)
    if end2_name is None:
        return False
    return end2_name in end1.collinear_to
    

def get_common_empty_of_two_curves_if_exists(curve_obj1: bpy.types.Object,
                                             curve_obj2: bpy.types.Object) -> Optional[Tuple[int, int, bpy.types.Object]]:
    empty1_1 = curve_obj1.greg_curve_settings.end1_empty
    empty1_2 = curve_obj1.greg_curve_settings.end2_empty
    empty2_1 = curve_obj2.greg_curve_settings.end1_empty
    empty2_2 = curve_obj2.greg_curve_settings.end2_empty
    if empty1_1 == empty2_1:
        return 0, 0, empty1_1
    if empty1_1 == empty2_2:
        return 0, 1, empty1_1
    if empty1_2 == empty2_1:
        return 1, 0, empty1_2
    if empty1_2 == empty2_2:
        return 1, 1, empty1_2
    return None

def are_two_curves_collinear(curve_obj1: bpy.types.Object,
                             curve_obj2: bpy.types.Object,
                             i1: int,
                             i2: int,
                             common_empty: bpy.types.Object) -> bool:
    if i1 == 0:
        end1_name = curve_obj1.greg_curve_settings.end1_name
    else:
        end1_name = curve_obj1.greg_curve_settings.end2_name
    if i2 == 0:
        end2_name = curve_obj2.greg_curve_settings.end1_name
    else:
        end2_name = curve_obj2.greg_curve_settings.end2_name
    end1 = common_empty.greg_empty_settings.curve_ends[end1_name]
    return end2_name in end1.collinear_to
    



addon_keymaps = []

classes = (GregId, GregArrowItem, GregBasicEnd, GregArrow, GregCurveEndItem, GregEmptyItem, GregCurveItem, GregQuad,
           GregPhantomCurveEnd, GregPhantomCurve, GregPhantomBpoint, GregCollectionSettings, GregEmpty ,GregCurve,
           CreateCurvesCollection, CreateSurfacesBetweenCurves, SetNotPatch, PrintItemInfo, MakeCurveMirrorBridge,
           UnsetCurveMirrorBridge, PrintDotInfo, OBJECT_PT_greg_curve_properties, OBJECT_PT_greg_curve_properties1,
           OBJECT_PT_greg_curve_properties2, GregSubdivide, GregExtrude, GregMergeAtCenter, GregMergeAtFirst,
           GregMergeAtLast, GregMergeSub, AddBezierCurve, SetCoplanar, SetNotCoplanar, SetCollinear, SetNotCollinear,
           OBJECT_PT_greg_resolution)

functions_context_menu = (add_collection_menu_func, add_surface_menu_func, set_not_face_menu_func,
                          add_print_info_func, add_bridge_mirror_func, add_unset_bridge_mirror_func,
                          add_print_dot_func, add_greg_subdivide_func, add_greg_extrude_func, add_greg_merge_func,
                          add_bezier_curve_menu_func, set_coplanar_menu_func, set_not_coplanar_menu_func,
                          set_collinear_menu_func, set_not_collinear_menu_func)

def register():
    for my_class in classes:
        bpy.utils.register_class(my_class)

    for menu_func in functions_context_menu:
        bpy.types.VIEW3D_MT_object_context_menu.append(menu_func)
    
    bpy.types.Collection.greg_settings = bpy.props.PointerProperty(type=GregCollectionSettings)
    bpy.types.Object.greg_empty_settings = bpy.props.PointerProperty(type=GregEmpty)
    bpy.types.Object.greg_curve_settings = bpy.props.PointerProperty(type=GregCurve)
    bpy.types.Object.greg_arrow_settings = bpy.props.PointerProperty(type=GregArrow)
    bpy.types.Object.greg_bulge = bpy.props.FloatProperty(name="bulge", default=0, update = cb_update)
    bpy.types.Object.greg_shear = bpy.props.FloatProperty(name="shear", default=0, update = cb_update)
    bpy.types.Object.greg_tilt = bpy.props.FloatProperty(name="tilt", default=0, update = cb_update)
    bpy.types.Object.greg_bulge1 = bpy.props.FloatProperty(name="bulge side 1", default=0, update = cb_update)
    bpy.types.Object.greg_shear1 = bpy.props.FloatProperty(name="shear side 1", default=0, update = cb_update)
    bpy.types.Object.greg_tilt1 = bpy.props.FloatProperty(name="tilt side 1", default=0, update = cb_update)
    bpy.types.Object.greg_bulge2 = bpy.props.FloatProperty(name="bulge side 2", default=0, update = cb_update)
    bpy.types.Object.greg_shear2 = bpy.props.FloatProperty(name="shear side 2", default=0, update = cb_update)
    bpy.types.Object.greg_tilt2 = bpy.props.FloatProperty(name="tilt side 2", default=0, update = cb_update)
    bpy.types.Object.greg_is_sharp = bpy.props.BoolProperty(default=False)
    bpy.types.Collection.greg_is_not_face = bpy.props.BoolProperty(default=False)
    bpy.types.Object.greg_resolution = bpy.props.IntProperty(name="resolution (recreate mesh to update)", default=12)
    bpy.types.Object.greg_is_generated = bpy.props.BoolProperty(default=False)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = wm.keyconfigs.addon.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new(GregExtrude.bl_idname, type='E', value='PRESS', ctrl=False)
        addon_keymaps.append((km, kmi))

    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    
def unregister():

    # Remove the hotkey
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)

    del bpy.types.Collection.greg_settings
    del bpy.types.Object.greg_empty_settings
    del bpy.types.Object.greg_curve_settings
    del bpy.types.Object.greg_arrow_settings
    del bpy.types.Object.greg_bulge
    del bpy.types.Object.greg_shear
    del bpy.types.Object.greg_tilt
    del bpy.types.Object.greg_bulge1
    del bpy.types.Object.greg_shear1
    del bpy.types.Object.greg_tilt1
    del bpy.types.Object.greg_bulge2
    del bpy.types.Object.greg_shear2
    del bpy.types.Object.greg_tilt2
    del bpy.types.Object.greg_is_sharp
    del bpy.types.Collection.greg_is_not_face
    del bpy.types.Object.greg_resolution
    del bpy.types.Object.greg_is_generated

    for menu_func in functions_context_menu:
        bpy.types.VIEW3D_MT_object_context_menu.remove(menu_func)

    for my_class in reversed(classes):
        bpy.utils.unregister_class(my_class)


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()