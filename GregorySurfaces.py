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

bl_info = {
    "name": "Create Surfaces Between Curves",
    "blender": (3, 4, 0),
    "category": "Object",
}

from inspect import getouterframes, currentframe
from itertools import chain
import bpy
import bmesh
import mathutils
from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
import math

TH = 0.00001
TH2 = TH**2

#**************************************************************************

mode = [None]

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

def preserve_mirror_bridge(curve_obj):
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

def verify_curve_deleted_or_returned():
    for collection in bpy.data.collections:
        if collection.greg_settings.used_for_greg:
            verify_arrow_returned(collection)
            for setting in collection.greg_settings.curves:
                curve_obj = setting.curve
                if curve_obj:
                    if not bpy.context.scene.objects.get(curve_obj.name):
                        remove_curve_from_greg_structure(curve_obj, collection)
                        bpy.data.objects.remove(curve_obj, do_unlink=True)
                    else:
                        settings = curve_obj.greg_curve_settings
                        for i, (end_name, end_empty) in enumerate(((settings.end1_name, settings.end1_empty),
                                                                   (settings.end2_name, settings.end2_empty))):
                            if end_empty.greg_empty_settings.curve_ends.find(end_name) == -1:
                                repare_end(curve_obj, end_empty, end_name, i)
                                repare_hooks(end_empty)
                                
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

def add_end_to_arrow(arrow, end_name, i, curve):
    basic_end = arrow.greg_arrow_settings.coplanars.add()
    basic_end.name = end_name
    basic_end.end = i
    basic_end.curve = curve

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
    '''
    THIS DOESN'T WORK, BUT MAYBE IN THE FUTURE...
    depsgraph = bpy.context.evaluated_depsgraph_get()
    p = curve.evaluated_get(depsgraph).data.splines[0].bezier_points[i]
    co = p.co
    h = p.handle_right if i == 0 else p.handle_left
    hh = h - co
    for arrow_item in empty.greg_empty_settings.coplanars:
        arrow = arrow_item.arrow
        print("here1", end_name)
        if arrow.greg_arrow_settings.coplanars.find(end_name) == -1:
            print("here2", end_name)
            mat = arrow.matrix_world.copy()
            mat.invert()
            vec = mathutils.Vector((0,0,1)) @ mat
            if abs(vec.dot(hh)) < TH:
                print("here3", end_name)
                add_end_to_arrow(arrow, end_name, i, curve)
            else:
                print("not collin!", end_name, vec.dot(hh), vec, hh, i)
                for modifier in curve.modifiers:
                    print(modifier.name)'''

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

#**************************************************************************
try:
    #raise(ModuleNotFoundError)
    import numpy as np
    import numpy.typing as npt
    J_S = np.arange(4)                               # shape (4), [0, 1, 2, 3]
    COMBS = np.array([math.comb(3, j) for j in J_S]) # shape (4), [1, 3, 3, 1]
    class DependantsOfResolution_np:
        def __init__(self):
            self.nedges = 0
            self.v0: npt.NDArray[np.float64]
            self.u0: npt.NDArray[np.float64]
            self.v02: npt.NDArray[np.float64]
            self.u02: npt.NDArray[np.float64]
            self.uv_div: npt.NDArray[np.float64]
            self.berns: npt.NDArray[np.float64]
            self.berns2d: npt.NDArray[np.float64]
            self.dims1: npt.NDArray[np.float64]
            self.dims2: npt.NDArray[np.float64]
        
        def calculate_faces(self):
            faces: List[List[int]] = []
            for i in range(self.nedges-2):
                for j in range(self.nedges-2):
                    face = [i * (self.nedges - 1) + j]
                    face.append(i * (self.nedges - 1) + j + 1)
                    face.append((i + 1) * (self.nedges - 1) + j + 1)
                    face.append((i + 1) * (self.nedges - 1) + j)
                    faces.append(face)
            self.faces = np.array(faces)

        def conditional_update(self, nedges: int):
            if self.nedges != nedges:
                self.nedges = nedges
                self.update()

        def update(self):
            v = np.linspace(0, 1, self.nedges + 1) # shape      (N+1)
            u = np.expand_dims(v, 1)               # shape (N+1, 1  )
            self.v0 = v[1:self.nedges]             # shape      (N-1)
            self.u0 = u[1:self.nedges]             # shape (N-1, 1  )
            self.v02 = np.expand_dims(np.stack((self.v0, 1-self.v0), 1), axis=(1,3)) # shape      (N-1, 1, 2, 1)
            self.u02 = np.expand_dims(self.v02, axis=4)                              # shape (N-1, 1  , 2, 1, 1)
            self.uv_div = self.u02 + self.v02                                        # shape (N-1, N-1, 2, 2, 1)

            self.berns = np.expand_dims(COMBS * self.u0**J_S * (1-self.u0)**(3-J_S), axis=1) # shape      (N-1, 1, 4)
            berns_np_2 = np.expand_dims(self.berns, axis=3)                                  # shape (N-1, 1  , 4, 1)
            self.berns2 = np.expand_dims(self.berns*berns_np_2, axis=4)                      # shape (N-1, N-1, 4, 4, 1)

            # numpy magic to get products of all combinations of Bernstein coefficients

            self.dims1 = np.empty((self.nedges-1, self.nedges-1, 1, 2, 3))
            self.dims2 = np.empty((self.nedges-1, self.nedges-1, 4, 1, 3))
            self.calculate_faces()

    def calc_control_points_np(input1: List[List[mathutils.Vector]],
                               input2: List[List[mathutils.Vector]],
                               d: DependantsOfResolution_np):
        full = np.array(input1)                     # shape           (4, 4, 3)
        cp_u = full[1:3, 1:3, :]                    # shape           (2, 2, 3)
        cp_v = np.array(input2)                     # shape           (2, 2, 3)
        cp_central = (cp_u*d.v02 + cp_v*d.u02) / d.uv_div # shape (N-1, N-1, 2, 2, 3)

        top, _ = np.broadcast_arrays(full[0, 1:3], d.dims1)        # shape (N-1, N-1, 1, 2, 3)
        bottom, _ = np.broadcast_arrays(full[3, 1:3], d.dims1)     # shape (N-1, N-1, 1, 2, 3)
        res1 = np.concatenate((top, cp_central, bottom), axis=2) # shape (N-1, N-1, 4, 2, 3)
        left, _ = np.broadcast_arrays(np.expand_dims(full[:, 0], axis=1), d.dims2)  # shape (N-1, N-1, 4, 1, 3)
        right, _ = np.broadcast_arrays(np.expand_dims(full[:, 3], axis=1), d.dims2) # shape (N-1, N-1, 4, 1, 3)
        res2 = np.concatenate((left, res1, right), axis=3)  # shape (N-1, N-1, 4, 4, 3)
        return res2
    
    def calc_bezier_curve_np(k0: mathutils.Vector,
                          k1: mathutils.Vector,
                          k2: mathutils.Vector,
                          k3: mathutils.Vector,
                          d: DependantsOfResolution_np):
        ks = np.array([k0, k1, k2, k3]).T # shape (3, 4)
        return np.sum(d.berns * ks, 2) # shape (N-1, 3)
    
    def add_corner(glist: "GlobalList", coords: mathutils.Vector):
        if glist.verts is None:
            glist.verts = np.array([coords])
        else:
            assert isinstance(glist.verts, np.ndarray)
            glist.verts = np.vstack((glist.verts, np.array([coords])))
    
    def add_border(k0: mathutils.Vector,
                   k1: mathutils.Vector,
                   k2: mathutils.Vector,
                   k3: mathutils.Vector,
                   glist: "GlobalList",
                   d: "DependantsOfResolution | DependantsOfResolution_np"):
        assert isinstance(d, DependantsOfResolution_np)
        if glist.verts is None:
            init_point = 0
        else:
            init_point = len(glist.verts)
        border = list(range(init_point, init_point + d.nedges - 1))
        border_coords = calc_bezier_curve_np(k0, k1, k2, k3, d)
        if glist.verts is None:
            glist.verts = border_coords
        else:
            assert isinstance(glist.verts, np.ndarray)
            glist.verts = np.vstack((glist.verts, border_coords))
        return border
    
    def calc_quad_gregory_verts(kk, kk1, d):
        control_points = calc_control_points_np(kk, kk1, d)
        res: npt.NDArray[np.float64] = np.sum((d.berns2 * control_points), (2, 3)).reshape((d.nedges-1) * (d.nedges-1), 3)
        # numpy representation of the formula p(u,v) = sum_i_from_0_to_3(sum_j_from_0_to_3( k(i,j)*B(i,u)*B(j,v) ))
        return res

    def calc_gregory_surf(kk: List[List[mathutils.Vector]],
                          kk1: List[List[mathutils.Vector]],
                          d: "DependantsOfResolution_np | DependantsOfResolution",
                          border1: List[int],
                          border2: List[int],
                          border3: List[int],
                          border4: List[int],
                          corner1: int,
                          corner2: int,
                          corner3: int,
                          corner4: int,
                          glist: "GlobalList") -> None:
        assert isinstance(d, DependantsOfResolution_np)
        res = calc_quad_gregory_verts(kk, kk1, d)
        # numpy representation of the formula p(u,v) = sum_i_from_0_to_3(sum_j_from_0_to_3( k(i,j)*B(i,u)*B(j,v) ))
        if glist.verts is None:
            glist.verts = res
            num_points = 0
        else:
            assert isinstance(glist.verts, np.ndarray)
            num_points = len(glist.verts)
            glist.verts = np.vstack((glist.verts, res))
        this_faces = d.faces + num_points
        if glist.faces is None:
            glist.faces = this_faces
        else:
            assert isinstance(glist.faces, np.ndarray)
            glist.faces = np.vstack((glist.faces, this_faces))
        faces_border4: npt.NDArray[np.int64] = np.array([[border4[i+1], border4[i], this_faces[i*(d.nedges-2)][0], this_faces[i*(d.nedges-2)][3]] for i in range(d.nedges-2)])
        faces_border1: npt.NDArray[np.int64] = np.array([[border1[i], border1[i+1], this_faces[i][1], this_faces[i][0]] for i in range(d.nedges-2)])
        faces_border2: npt.NDArray[np.int64] = np.array([[border2[i], border2[i+1], this_faces[i*(d.nedges-2)+d.nedges-3][2], this_faces[i*(d.nedges-2)+d.nedges-3][1]] for i in range(d.nedges-2)])
        faces_border3: npt.NDArray[np.int64] = np.array([[border3[i+1], border3[i], this_faces[(d.nedges-2)*(d.nedges-3)+i][3], this_faces[(d.nedges-2)*(d.nedges-3)+i][2]] for i in range(d.nedges-2)])
        face_corner1: npt.NDArray[np.int64] = np.array([[corner1, border1[0], this_faces[0][0], border4[0]]])
        face_corner2: npt.NDArray[np.int64] = np.array([[corner2, border2[0], this_faces[d.nedges-3][1], border1[d.nedges-2]]])
        face_corner3: npt.NDArray[np.int64] = np.array([[corner3, border3[d.nedges-2], this_faces[(d.nedges-2)*(d.nedges-2)-1][2], border2[d.nedges-2]]])
        face_corner4: npt.NDArray[np.int64] = np.array([[corner4, border4[d.nedges-2], this_faces[(d.nedges-2)*(d.nedges-3)][3], border3[0]]])
        glist.faces = np.vstack((glist.faces, faces_border1, faces_border2, faces_border3, faces_border4, face_corner1, face_corner2, face_corner3, face_corner4))

    d = DependantsOfResolution_np()

except (ModuleNotFoundError, ImportError):
    from copy import deepcopy
    def bernstein(i: int, u: float):
        return math.comb(3, i) * u**i * (1-u)**(3-i)

    class DependantsOfResolution:
        def __init__(self):
            self.nedges = 0
            self.berns: List[List[float]]
            self.b2: List[List[List[List[float]]]]
            self.faces: List[List[int]]

        def conditional_update(self, nedges: int):
            if self.nedges != nedges:
                self.nedges = nedges
                self.berns = [[bernstein(i, k/nedges) for i in range(4)] for k in range(nedges + 1)]
                self.b2 = [[[[self.berns2_el(i,j,nu,nv) for j in range(4)] for i in range(4)]\
                               for nv in range(nedges + 1)] for nu in range(nedges + 1)]
                self.calculate_faces()

        def berns2_el(self, i: int, j: int, nu: int, nv:int):
            return self.berns[nu][i] * self.berns[nv][j]
        
        def calculate_faces(self):
            self.faces: List[List[int]] = []
            for i in range(self.nedges-2):
                for j in range(self.nedges-2):
                    face = [i * (self.nedges - 1) + j]
                    face.append(i * (self.nedges - 1) + j + 1)
                    face.append((i + 1) * (self.nedges - 1) + j + 1)
                    face.append((i + 1) * (self.nedges - 1) + j)
                    self.faces.append(face)

    def calc_control_points(kk: List[List[mathutils.Vector]],
                            kk1: List[List[mathutils.Vector]],
                            nu: int,
                            nv: int,
                            d: DependantsOfResolution):
        u = nu / d.nedges
        v = nv / d.nedges
        cp = deepcopy(kk)
        cp[1][1] = (u * kk1[0][0] + v * kk[1][1]) / (u + v)
        cp[2][1] = ((1 - u) * kk1[1][0] + v * kk[2][1]) / (1 - u + v)
        cp[1][2] = (u * kk1[0][1] + (1 - v) * kk[1][2]) / (1 - v + u)
        cp[2][2] = ((1 - u) * kk1[1][1] + (1 - v) * kk[2][2]) / (2 - u - v)
        return cp

    def calc_point(nu: int,
                   nv: int,
                   kk: List[List[mathutils.Vector]],
                   kk1: List[List[mathutils.Vector]],
                   d: DependantsOfResolution):
        el: List[float] = []
        control_points = calc_control_points(kk, kk1, nu, nv, d)
        for k in range(3):
            el.append(sum(sum(control_points[i][j][k] * d.b2[nu][nv][i][j] for j in range(4)) for i in range(4)))
        return mathutils.Vector(el)
    
    def augment_faces(cons: int, d: DependantsOfResolution):
        res: List[List[int]] = []
        for face in d.faces:
            res.append([])
            for vert in face:
                res[-1].append(vert + cons)
        return res
    
    def add_corner(glist: "GlobalList", coords: mathutils.Vector):
        if glist.verts is None:
            glist.verts = [coords]
        else:
            assert isinstance(glist.verts, list)
            glist.verts.append(coords)
    
    def add_border(k0: mathutils.Vector,
                   k1: mathutils.Vector,
                   k2: mathutils.Vector,
                   k3: mathutils.Vector,
                   glist: "GlobalList",
                   d: "DependantsOfResolution | DependantsOfResolution_np"):
        if glist.verts is None:
            glist.verts = []
        else:
            assert isinstance(glist.verts, list)
        init_point = len(glist.verts)
        border = list(range(init_point, init_point + d.nedges - 1))
        border_coords = calc_bezier_curve(k0, k1, k2, k3, d)
        glist.verts.extend(border_coords)
        return border
    
    def calc_bezier_curve(k0: mathutils.Vector,
                          k1: mathutils.Vector,
                          k2: mathutils.Vector,
                          k3: mathutils.Vector,
                          d: "DependantsOfResolution_np | DependantsOfResolution") -> "npt.NDArray[np.float64] | List[mathutils.Vector]":
        assert isinstance(d, DependantsOfResolution)
        points = [k0, k1, k2, k3]
        return [my_vector_sum([d.berns[k][i] * points[i] for i in range(4)]) for k in range(1, d.nedges)]

    def calc_quad_gregory_verts(kk, kk1, d):
        return [calc_point(nu, nv, kk, kk1, d) for nv in range(1, d.nedges) for nu in range(1, d.nedges)]

    def calc_gregory_surf(kk: List[List[mathutils.Vector]],
                          kk1: List[List[mathutils.Vector]],
                          d: "DependantsOfResolution_np | DependantsOfResolution",
                          border1: List[int],
                          border2: List[int],
                          border3: List[int],
                          border4: List[int],
                          corner1: int,
                          corner2: int,
                          corner3: int,
                          corner4: int,
                          glist: "GlobalList") -> None:
        assert isinstance(d, DependantsOfResolution)
        res = calc_quad_gregory_verts(kk, kk1, d)
        if glist.verts is None:
            glist.verts = []
        num_points = len(glist.verts)
        assert isinstance(glist.verts, list)
        glist.verts.extend(res)
        this_faces = augment_faces(num_points, d)
        if glist.faces is None:
            glist.faces = []
        assert isinstance(glist.faces, list)
        glist.faces.extend(this_faces)
        glist.faces.extend([[border4[i+1], border4[i], this_faces[i*(d.nedges-2)][0], this_faces[i*(d.nedges-2)][3]] for i in range(d.nedges-2)])
        glist.faces.extend([[border1[i], border1[i+1], this_faces[i][1], this_faces[i][0]] for i in range(d.nedges-2)])
        glist.faces.extend([[border2[i], border2[i+1], this_faces[i*(d.nedges-2)+d.nedges-3][2], this_faces[i*(d.nedges-2)+d.nedges-3][1]] for i in range(d.nedges-2)])
        glist.faces.extend([[border3[i+1], border3[i], this_faces[(d.nedges-2)*(d.nedges-3)+i][3], this_faces[(d.nedges-2)*(d.nedges-3)+i][2]] for i in range(d.nedges-2)])
        glist.faces.append([corner1, border1[0], this_faces[0][0], border4[0]])
        glist.faces.append([corner2, border2[0], this_faces[d.nedges-3][1], border1[d.nedges-2]])
        glist.faces.append([corner3, border3[d.nedges-2], this_faces[(d.nedges-2)*(d.nedges-2)-1][2], border2[d.nedges-2]])
        glist.faces.append([corner4, border4[d.nedges-2], this_faces[(d.nedges-2)*(d.nedges-3)][3], border3[0]])

    def my_vector_sum(li: List[mathutils.Vector]):
        s = li[0]
        for ve in li[1:]:
            s += ve
        return s

    def generate_bezier(p1: mathutils.Vector,
                        p2: mathutils.Vector,
                        h1: mathutils.Vector,
                        h2: mathutils.Vector,
                        d: DependantsOfResolution):
        assert isinstance(d, DependantsOfResolution)
        points = [p1, p1 + h1, p2 + h2, p2]
        return [my_vector_sum([d.berns[k][i] * points[i] for i in range(4)]) for k in range(d.nedges + 1)]

    d = DependantsOfResolution()

#**************************************************************************

def quad_edges_to_normal(co_a1: mathutils.Vector,
                         co_a2: mathutils.Vector,
                         co_b1: mathutils.Vector,
                         co_b2: mathutils.Vector):
    diff_a = co_a2 - co_a1
    diff_b = co_b2 - co_b1
    diff_a.normalize()
    diff_b.normalize()
    no = diff_a + diff_b
    no.normalize()
    return no

def quad_verts_to_barycentric_tri(co_a: mathutils.Vector,
                                  co_b: mathutils.Vector,
                                  co_a_next: mathutils.Vector,
                                  co_b_next: mathutils.Vector,
                                  co_a_prev: mathutils.Vector | None,
                                  co_b_prev: mathutils.Vector | None,
                                  is_flip: bool):
    tri = [co_a, co_b]
    no = quad_edges_to_normal(co_a, co_a_next, co_b, co_b_next)
    if co_a_prev is not None and co_b_prev is not None:
        no_t = quad_edges_to_normal(co_a_prev, co_a, co_b_prev, co_b)
        no += no_t
        no.normalize()
    if is_flip:
        no = -no
    d = (co_a - co_b).length
    no *= d
    tri.append((co_a + co_b)/2)
    tri[2] += no
    return tri

def normal_tri_v3(v1: mathutils.Vector,
                  v2: mathutils.Vector,
                  v3: mathutils.Vector):
    n1 = v1 - v2
    n2 = v2 - v3
    n = n1.cross(n2)
    n.normalize()
    return n

def ortho_basis_v3v3_v3(n: mathutils.Vector):
    eps = 1.192092896e-07
    f = n[0] * n[0] + n[1] * n[1]
    if f > eps:
        d = 1 / (f ** 0.5)
        r_n1 = [n[1] * d, -n[0] * d, 0]
        r_n2 = [-n[2] * r_n1[1],
                n[2] * r_n1[0],
                n[0] * r_n1[1] - n[1] * r_n1[0]]
    else:
        r_n1 = [-1 if n[2] < 0 else 1, 0, 0]
        r_n2 = [0, 1, 0]
    return r_n1, r_n2

def axis_dominant_v3_to_m3(normal: mathutils.Vector):
    r_n1, r_n2 = ortho_basis_v3v3_v3(normal)
    r_mat = mathutils.Matrix((r_n1, r_n2, normal))
    r_mat.transpose()
    return r_mat

def cross_tri_v2(v1: mathutils.Vector,
                 v2: mathutils.Vector,
                 v3: mathutils.Vector):
    return (v1[0] - v2[0]) * (v2[1] - v3[1]) + (v1[1] - v2[1]) * (v3[0] - v2[0])

def barycentric_weights_v2(v1: mathutils.Vector,
                           v2: mathutils.Vector,
                           v3: mathutils.Vector,
                           co: mathutils.Vector):
    w = mathutils.Vector((cross_tri_v2(v2, v3, co),
                          cross_tri_v2(v3, v1, co),
                          cross_tri_v2(v1, v2, co)))
    wtot = w[0] + w[1] + w[2]
    if wtot != 0:
        w *= (1 / wtot)
    else:
        w = mathutils.Vector((1/3, 1/3, 1/3))
    return w

def interp_v3_v3v3v3(v1: mathutils.Vector,
                     v2: mathutils.Vector,
                     v3: mathutils.Vector,
                     w: mathutils.Vector):
    return mathutils.Vector((
        v1[0] * w[0] + v2[0] * w[1] + v3[0] * w[2],
        v1[1] * w[0] + v2[1] * w[1] + v3[1] * w[2],
        v1[2] * w[0] + v2[2] * w[1] + v3[2] * w[2]))

def interp_v3_v3v3(a: mathutils.Vector,
                   b: mathutils.Vector,
                   t: float):
    s = 1 - t
    return s * a + t * b

def transform_point_by_tri_v3(pt_src: mathutils.Vector,
                              tri_tar_p1: mathutils.Vector,
                              tri_tar_p2: mathutils.Vector,
                              tri_tar_p3: mathutils.Vector,
                              tri_src_p1: mathutils.Vector,
                              tri_src_p2: mathutils.Vector,
                              tri_src_p3: mathutils.Vector):
    no_tar = normal_tri_v3(tri_tar_p1, tri_tar_p2, tri_tar_p3)
    no_src = normal_tri_v3(tri_src_p1, tri_src_p2, tri_src_p3)
    mat_src = axis_dominant_v3_to_m3(no_src)
    pt_src_xy = pt_src @ mat_src
    tri_xy_src = (tri_src_p1 @ mat_src,
                  tri_src_p2 @ mat_src,
                  tri_src_p3 @ mat_src)
    w_src = barycentric_weights_v2(tri_xy_src[0], tri_xy_src[1], tri_xy_src[2], pt_src_xy)
    pt_tar = interp_v3_v3v3v3(tri_tar_p1, tri_tar_p2, tri_tar_p3, w_src)
    area_tar = math.sqrt(mathutils.geometry.area_tri(tri_tar_p1, tri_tar_p2, tri_tar_p3))
    area_src = math.sqrt(mathutils.geometry.area_tri(tri_xy_src[0], tri_xy_src[1], tri_xy_src[2]))
    z_ofs_src = pt_src_xy[2] - tri_xy_src[0][2]
    return pt_tar + no_tar * (z_ofs_src / area_src) * area_tar

def XY(x: int, y: int, xtot: int):
    return x + y * xtot

def grid_fill(verts1: List[mathutils.Vector],
              verts2: List[mathutils.Vector],
              railverts1: List[mathutils.Vector],
              railverts2: List[mathutils.Vector]):
    xtot = len(verts1)
    ytot = len(railverts1)
    v0 = mathutils.Vector((0, 0, 0))
    v_grid = verts1 + ([v0] * xtot * (ytot-2)) + verts2
    for i in range(1, ytot-1):
        v_grid[xtot*i] = railverts1[i]
        v_grid[xtot*(i+1) - 1] = railverts2[i]
    tri_a = quad_verts_to_barycentric_tri(
        v_grid[XY(0, 0, xtot)],
        v_grid[XY(xtot-1, 0, xtot)],
        v_grid[XY(0, 1, xtot)],
        v_grid[XY(xtot-1, 1, xtot)],
        None,
        None,
        False
    )
    tri_b = quad_verts_to_barycentric_tri(
        v_grid[XY(0, ytot-1, xtot)],
        v_grid[XY(xtot-1, ytot-1, xtot)],
        v_grid[XY(0, ytot-2, xtot)],
        v_grid[XY(xtot-1, ytot-2, xtot)],
        None,
        None,
        True
    )
    for y in range(1, ytot-1):
        tri_t = quad_verts_to_barycentric_tri(
            v_grid[XY(0, y, xtot)],
            v_grid[XY(xtot-1, y, xtot)],
            v_grid[XY(0, y+1, xtot)],
            v_grid[XY(xtot-1, y+1, xtot)],
            v_grid[XY(0, y-1, xtot)],
            v_grid[XY(xtot-1, y-1, xtot)],
            False
        )
        for x in range(1, xtot-1):
            co_a = transform_point_by_tri_v3(v_grid[x],
                                             tri_t[0],
                                             tri_t[1],
                                             tri_t[2],
                                             tri_a[0],
                                             tri_a[1],
                                             tri_a[2])
            co_b = transform_point_by_tri_v3(v_grid[(xtot * ytot) + (x - xtot)],
                                             tri_t[0],
                                             tri_t[1],
                                             tri_t[2],
                                             tri_b[0],
                                             tri_b[1],
                                             tri_b[2])
            co = interp_v3_v3v3(co_a, co_b, y / (ytot - 1))
            v_grid[(y * xtot) + x] = co
    return v_grid

def are_coplanar(v1: mathutils.Vector, v2: mathutils.Vector, v3: mathutils.Vector):
    return abs(mathutils.Matrix((v1.normalized(), v2.normalized(), v3.normalized())).determinant()) < TH

def are_collinear(v1: mathutils.Vector, v2: mathutils.Vector):
    return (v1.normalized().cross(v2.normalized())).length < TH

#*******************************************************************************************

def get_coefs(e1: mathutils.Vector, e2: mathutils.Vector, x: mathutils.Vector):
    # x = a*e1 + b*e2, we search a and b, e1 and e2 and x are coplanar, e1 and e2 are not collinear
    e1e1 = e1.length_squared
    e2e2 = e2.length_squared
    e1e2 = e1.dot(e2)
    e1x = e1.dot(x)
    e2x = e2.dot(x)
    coef = 1/(e1e1*e2e2 - e1e2**2)
    a = (e2e2*e1x - e1e2*e2x) * coef
    b = (e1e1*e2x - e1e2*e1x) * coef
    return (a, b)

def same_coords(c1: mathutils.Vector, c2: mathutils.Vector) -> bool:
    return (c1-c2).length_squared < TH2

class StepRes(Enum):
    FINISHED = 1
    NOT_FINISHED = 2
    PART_FINISHED = 3

class BigPoint:
    def __init__(self, i: int, coords: mathutils.Vector):
        self.points: List[Point] = []
        self.i = i
        self.count = 0
        self.coords = coords
        self.created_curves: List[Tuple[str, int]] = []

    def add_point(self, point: "Point"):
        self.points.append(point)
        self.count += 1

    def add_vert(self, glist: "GlobalList"):
        add_corner(glist, self.coords)


class Point:
    def __init__(self,
                 i: int,
                 spline: "Spline",
                 bpoint: "BigPoint",
                 handle_left: mathutils.Vector,
                 handle_right: mathutils.Vector):
        self.bpoint: BigPoint = bpoint
        self.i: int = i
        self.prev_seg: Optional[Segment] = None
        self.post_seg: Optional[Segment] = None
        self.spline: Spline = spline
        self.handle_left = handle_left
        self.handle_right = handle_right


class Spline:
    def __init__(self, glist: "GlobalList"):
        self.points: List[Point] = []
        self.segments: List[Segment] = []
        self.glist = glist
        self.glist.add_spline(self)

    def add_point(self, coords: mathutils.Vector, handle_left: mathutils.Vector, handle_right: mathutils.Vector):
        i = 0
        added = False
        count = self.glist.get_count()
        bpoint = BigPoint(-1, mathutils.Vector((0, 0, 0))) # placeholder
        while not added:
            if i == count:
                bpoint = self.glist.create_bpoint(coords)
                added = True
            else:
                if same_coords(coords, self.glist.get_coords(i)):
                    bpoint = self.glist.get_bpoint(i)
                    added = True
            i += 1
        point = Point(i-1, self, bpoint, handle_left - coords, handle_right - coords)
        bpoint.add_point(point)
        self.points.append(point)
        p_num = len(self.points)
        if p_num > 1:
            seg = Segment(self.points[p_num - 2], point, self.glist)
            self.segments.append(seg)

    def round_spline(self):
        seg = Segment(self.points[-1], self.points[0], self.glist)
        self.segments.append(seg)


class Segment:
    def __init__(self, p1: Point, p2: Point, glist: "GlobalList"):
        self.p1 = p1
        self.p2 = p2
        self.p1.post_seg = self
        self.p2.prev_seg = self
        self.finished = False
        glist.add_segment(self)
        

    @staticmethod
    def iterate_direct(segs: "List[Segment]"):
        yield segs[0].p1
        for seg in segs:
            yield seg.p2

    @staticmethod
    def iterate_reversed(segs: "List[Segment]"):
        yield segs[-1].p2
        for seg in reversed(segs):
            yield seg.p1


def get_y_normalized(p: mathutils.Vector, x: mathutils.Vector):
    return (p - p.project(x)).normalized()

def calc_basis(init: mathutils.Vector,
               edge: mathutils.Vector,
               point: mathutils.Vector,
               handle1: mathutils.Vector,
               handle2: Optional[mathutils.Vector] = None,
               permut: bool = True):
    vec1 = edge - init
    vec2 = point - init
    if permut:
        return calc_basis_intern(vec1, handle1, vec2, handle2)
    return calc_basis_intern(vec1, vec2, handle1, handle2)

def calc_basis_intern(vec1: mathutils.Vector,
                      vec2: mathutils.Vector,
                      handle1: mathutils.Vector,
                      handle2: Optional[mathutils.Vector] = None):
    v0 = mathutils.Vector((0, 0, 0))
    x_vec = vec1.normalized()
    z_vec = x_vec.cross(handle1)
    if z_vec.length_squared > TH2:
        y_vec = get_y_normalized(handle1, x_vec)
        z_vec.normalize()
    else:
        z_vec = x_vec.cross(vec2)
        if (z_vec.length_squared > TH2):
            y_vec = get_y_normalized(vec2, x_vec)
            z_vec.normalize()
        else:
            if (handle2 is None):
                y_vec = v0
                z_vec = v0
            else:
                z_vec = x_vec.cross(handle2)
                if (z_vec.length_squared > TH2):
                    z_vec.normalize()
                    y_vec = get_y_normalized(handle2, x_vec)
                else:
                    y_vec = v0
                    z_vec = v0
    return mathutils.Matrix((x_vec, y_vec, z_vec))

def get_coords_from_vec_and_basis_matrix(v: mathutils.Vector, m: mathutils.Matrix) -> mathutils.Vector:
    res = m @ v
    assert isinstance(res, mathutils.Vector)
    return res

def get_vec_from_coords_and_basis_matrix(v: mathutils.Vector, m: mathutils.Matrix) -> mathutils.Vector:
    return v @ m

def make_collinear(v1: mathutils.Vector, v2: mathutils.Vector):
    dif = (v1 - v2).normalized()
    new_v1 = dif * v1.length
    new_v2 = -1 * dif * v2.length
    return new_v1, new_v2


class GlobalList:
    def __init__(self):
        self.reduced_points: List[mathutils.Vector] = []
        self.big_points: List[BigPoint] = []
        self.count = 0
        self.splines: List[Spline] = []
        self.segments: List[Segment] = []
        self.verts: Optional[List[mathutils.Vector] | npt.NDArray[np.float64]] = None
        self.faces: Optional[List[List[int]] | npt.NDArray[np.int64]] = None

    def create_bpoint(self, coords: mathutils.Vector):
        self.reduced_points.append(coords)
        bpoint = BigPoint(self.count, coords)
        self.big_points.append(bpoint)
        self.count += 1
        return bpoint

    def get_count(self):
        return self.count

    def get_coords(self, i: int):
        return self.reduced_points[i]

    def get_bpoint(self, i: int):
        return self.big_points[i]

    def add_spline(self, spline: Spline):
        self.splines.append(spline)

    def get_splines(self):
        return self.splines

    def add_segment(self, segment: Segment):
        self.segments.append(segment)

    def add_many_curves(self, name: str, parent_collection: bpy.types.Collection, context: bpy.types.Context):
        collection = bpy.data.collections.new(name)
        collection.greg_settings.used_for_greg = True
        parent_collection.children.link(collection)
        for segment in self.segments:
            co_s = [p.bpoint.coords for p in (segment.p1, segment.p2)]
            handles_left = [p.handle_left for p in (segment.p1, segment.p2)]
            handles_right = [p.handle_right for p in (segment.p1, segment.p2)]
            curve_obj, curve_prop = add_curve_obj(collection, co_s, handles_left, handles_right)
            for i, p in enumerate((segment.p1, segment.p2)):
                p.bpoint.created_curves.append((curve_prop.name, i))
        for bpoint in self.big_points:
            empty_obj = add_empty_obj(collection, bpoint.coords)
            for name, i in bpoint.created_curves:
                curve_obj = collection.greg_settings.curves[name].curve
                add_curve_end(collection, empty_obj, curve_obj, i)
            coplanar_collinear(empty_obj, collection)
            for end in empty_obj.greg_empty_settings.curve_ends:
                add_hook(end, context)

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

def check_ends_collinear(end1: "GregCurveEndItem", end2: "GregCurveEndItem"):
    ends = (end1, end2)
    handles = extract_vectors_from_ends(ends)
    if are_collinear(*handles):
        common_empty = end1.empty
        set_ends_collinear_to_one_another(end1, end2, common_empty)

def check_ends_coplanar(ends1, ends2, ends3, empty, collection: bpy.types.Collection):
    handles: List[mathutils.Vector] = []
    ends = (ends1[0], ends2[0], ends3[0])
    handles = extract_vectors_from_ends(ends)
    if are_coplanar(*handles):
        coplanar_vector = (handles[0].cross(handles[1])).normalized()
        return coplanar_vector
    return None

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
   
def get_next_id(collection: bpy.types.Collection):
    name = str(collection.greg_settings.max_id)
    collection.greg_settings.max_id += 1
    return name

def get_all_possible_hooks(end):
    curve_obj = end.basic_end.curve
    curve_name = curve_obj.greg_curve_settings.name
    empty_obj = end.empty
    empty_name = empty_obj.greg_empty_settings.name
    res = [f"hook_{curve_name}_{empty_name}"]
    for coplanar in empty_obj.greg_empty_settings.coplanars:
        res.append(f"hook_{curve_name}_{coplanar.name}")
    return res

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

def get_greg_collection(obj):
    for collection in obj.users_collection:
        if collection.greg_settings.used_for_greg:
            return collection

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
    #print("after remove")
    #print_structure(collection)
    #print("****************************")

#**************************************************************************
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

#**************************************************************************
logging = [False, False]
class NewGlobalList:
    def __init__(self, collection):
        self.collection = collection
        self.quads = []
        self.verts = None
        self.faces = None
        self.ids_counter = 0
        self.curves_dict: Dict[str, str] = {}

    def add_quads(self):
        for phantom_curve in self.collection.greg_settings.phantom_curves:
            logging[0] = (phantom_curve.source_curve.name == "greg_curve_obj.002")
            logging[1] = logging[0]
            if not phantom_curve.finished:
                if logging[0]:
                    print("here 0")
                self.work_with_curve(phantom_curve)

    def work_with_curve(self, phantom_curve):
        curves_verified = [phantom_curve]
        if logging[0]:
            print([qq.source_curve.name for qq in curves_verified])

        bpoints_verified = []
        initial_bpoint = self.collection.greg_settings.phantom_bpoints[phantom_curve.bpoint1_name]
        bpoint = self.collection.greg_settings.phantom_bpoints[phantom_curve.bpoint2_name]
        end0 = bpoint.ends[phantom_curve.end2_name]
        self.edge_step(initial_bpoint,
                       bpoint,
                       curves_verified,
                       bpoints_verified,
                       end0,
                       True)
        phantom_curve.finished = True

    def edge_step(self,
                  initial_bpoint,
                  bpoint,
                  curves_verified,
                  bpoints_verified,
                  end0,
                  first: bool) -> StepRes:
        if len(curves_verified) > 4 or (len(curves_verified) == 4 and initial_bpoint != bpoint):
            if logging[1]:
                print("here 1")
            return StepRes.NOT_FINISHED
        if initial_bpoint == bpoint:
            if logging[1]:
                print("here 2")
            if len(curves_verified) < 4:
                return StepRes.NOT_FINISHED
            if self.verify_and_init_quad(curves_verified, end0):
                if curves_verified[0].finished:
                    return StepRes.FINISHED
                return StepRes.PART_FINISHED
            return StepRes.NOT_FINISHED
        if bpoint in bpoints_verified:
            if logging[1]:
                print("here 3")
            return StepRes.NOT_FINISHED
        bpoints_verified.append(bpoint)
        for end in bpoint.ends:
            phantom_curve1 = self.collection.greg_settings.phantom_curves[end.curve_name]
            if logging[1]:
                print("here", phantom_curve1.source_curve.name)
            if not self.are_collinear(end0, end): # they are not collinear
                res = self.step_curve(initial_bpoint,
                                    phantom_curve1,
                                    end.curve_i,
                                    curves_verified,
                                    bpoints_verified,
                                    first)
                if res in (StepRes.FINISHED, StepRes.PART_FINISHED):
                    if logging[1]:
                        print("here 4")
                    bpoints_verified.pop()
                    return res
            else:
                if logging[1]:
                    print("here 5")

        bpoints_verified.pop()
        return StepRes.NOT_FINISHED
    
    def extract_handle_phantom(self, end: "GregPhantomCurveEnd"):
        curve = self.collection.greg_settings.phantom_curves[end.curve_name]
        bpoint = self.collection.greg_settings.phantom_bpoints[end.bpoint_name]
        handle = curve.handle1 if end.curve_i == 0 else curve.handle2
        return handle - bpoint.co
    
    def are_collinear(self, end1: "GregPhantomCurveEnd", end2: "GregPhantomCurveEnd"):
        v1, v2 = [self.extract_handle_phantom(end) for end in (end1, end2)]
        if logging[1]:
            print(v1, v2)
        return are_collinear(v1, v2)
        

    def step_curve(self,
                   initial_bpoint,
                   phantom_curve: "GregPhantomCurve",
                   end_i,
                   curves_verified,
                   bpoints_verified,
                   first) -> StepRes:
        if phantom_curve in curves_verified:
            if logging[1]:
                print("here 6")
        if not phantom_curve in curves_verified:
            if phantom_curve.finished:
                if logging[1]:
                    print("here 7")
            if not phantom_curve.finished:
                curves_verified.append(phantom_curve)
                if logging[0]:
                    print([qq.source_curve.name for qq in curves_verified])
                    #logging[1] = (phantom_curve.source_curve.name == 'greg_curve_obj.012')
                if end_i == 0:
                    bpoint = self.collection.greg_settings.phantom_bpoints[phantom_curve.bpoint2_name]
                    end0 = bpoint.ends[phantom_curve.end2_name]
                else:
                    bpoint = self.collection.greg_settings.phantom_bpoints[phantom_curve.bpoint1_name]
                    end0 = bpoint.ends[phantom_curve.end1_name]
                res = self.edge_step(initial_bpoint,
                                     bpoint,
                                     curves_verified,
                                     bpoints_verified,
                                     end0,
                                     False)
                curves_verified.pop()
                logging[1] = False
                if logging[0]:
                    print([qq.source_curve.name for qq in curves_verified])

                match res:
                    case StepRes.FINISHED: 
                        return StepRes.FINISHED
                    case StepRes.PART_FINISHED:
                        if not first:
                            return StepRes.PART_FINISHED
                    case StepRes.NOT_FINISHED:
                        return StepRes.NOT_FINISHED
        return StepRes.NOT_FINISHED
    
    def verify_and_init_quad(self, curves_verified: "GregPhantomCurve", end0):
        first_curve = curves_verified[0]
        first_bpoint = self.collection.greg_settings.phantom_bpoints[first_curve.bpoint1_name]
        first_vec = first_curve.handle1 - first_bpoint.co
        other_vec = self.extract_handle_phantom(end0)
        if are_collinear(first_vec, other_vec):
            return False
        set_faces = set([phantom_curve.source_curve.greg_curve_settings.name for phantom_curve in curves_verified])
        for phantom_curve in curves_verified:
            not_faces_sets = get_not_face_ids(phantom_curve.source_curve)
            for se in not_faces_sets:
                if se == set_faces:
                    return False
        quad = self.collection.greg_settings.quads.add()
        quad.name = "_".join(sorted(
            [phantom_curve.name for phantom_curve in curves_verified]
        ))
        for i, phantom_curve in enumerate(curves_verified):
            quad_name = phantom_curve.quads.add()
            quad_name.name = quad.name
            if len(phantom_curve.quads) == 2:
                phantom_curve.finished = True
            new_curve_item = quad.curves.add()
            new_curve_item.name = phantom_curve.name
            if i == 0:
                quad.dirs[0] = True
            elif i == 1:
                quad.dirs[1] = (phantom_curve.bpoint1_name == curves_verified[0].bpoint2_name)
            elif i == 2:
                if quad.dirs[1]:
                    quad.dirs[2] = (phantom_curve.bpoint2_name == curves_verified[1].bpoint2_name)
                else:
                    quad.dirs[2] = (phantom_curve.bpoint2_name == curves_verified[1].bpoint1_name)
            elif i == 3:
                quad.dirs[3] = (phantom_curve.bpoint1_name == curves_verified[0].bpoint1_name)
        if logging[0]:
            print("quad!")
        return True

    @staticmethod
    def calculate_midpoint(curve_obj):
        curve_settings = curve_obj.greg_curve_settings
        p0 = curve_settings.end1_empty.matrix_world.translation
        p1 = curve_obj.data.splines[0].bezier_points[0].handle_right
        p3 = curve_settings.end2_empty.matrix_world.translation
        p2 = curve_obj.data.splines[0].bezier_points[1].handle_left
        return 1/8 * (p0 + 3*p1 + 3*p2 + p3)
    
    def calculate_midpoint_phantom(self, phantom_curve):
        p0, p1, p2, p3 = NewGlobalList.extract_control_points(self.collection, phantom_curve)
        return 1/8 * (p0 + 3*p1 + 3*p2 + p3)
    
    def get_edge_control_points(self, quad, edge_num: int):
        curve_name = quad.curves[edge_num].name
        phantom_curve = self.collection.greg_settings.phantom_curves[curve_name]
        res = NewGlobalList.extract_control_points(self.collection, phantom_curve)
        if not quad.dirs[edge_num]:
            res.reverse()
        return res
    
    @staticmethod
    def get_edge_control_points_from_kk(quad, edge_num: int):
        if edge_num == 0:
            res = quad.kk[0]
        elif edge_num == 1:
            res = [quad.kk[0][3], quad.kk[1][3], quad.kk[2][3], quad.kk[3][3]]
        elif edge_num == 2:
            res = quad.kk[3]
        elif edge_num == 3:
            res = [quad.kk[0][0], quad.kk[1][0], quad.kk[2][0], quad.kk[3][0]]
        return (mathutils.Vector(elem) for elem in res)

    def calculate_coefs(self, quad):
        quad.kk[0][0], quad.kk[0][1], quad.kk[0][2], quad.kk[0][3] = self.get_edge_control_points(quad, 0)
        quad.kk[1][0], quad.kk[2][0], quad.kk[3][0] = self.get_edge_control_points(quad, 3)[1:]
        quad.kk[1][3], quad.kk[2][3], quad.kk[3][3] = self.get_edge_control_points(quad, 1)[1:]
        quad.kk[3][1], quad.kk[3][2] = self.get_edge_control_points(quad, 2)[1:3]

    def get_neighbour_quad(self, quad, i: int):
        curve_name = quad.curves[i].name
        phantom_curve = self.collection.greg_settings.phantom_curves[curve_name]
        if len(phantom_curve.quads) == 1:
            return None
        res = [other_quad for other_quad in phantom_curve.quads if quad.name != other_quad.name][0]
        res_quad = self.collection.greg_settings.quads[res.name]
        return res_quad
    
    @staticmethod
    def extract_a0_a3(quad, i: int):
        if i == 0:
            a0 = mathutils.Vector(quad.kk[1][0]) - mathutils.Vector(quad.kk[0][0])
            a3 = mathutils.Vector(quad.kk[1][3]) - mathutils.Vector(quad.kk[0][3])
        elif i == 1:
            a0 = mathutils.Vector(quad.kk[0][2]) - mathutils.Vector(quad.kk[0][3])
            a3 = mathutils.Vector(quad.kk[3][2]) - mathutils.Vector(quad.kk[3][3])
        elif i == 2:
            a0 = mathutils.Vector(quad.kk[2][0]) - mathutils.Vector(quad.kk[3][0])
            a3 = mathutils.Vector(quad.kk[2][3]) - mathutils.Vector(quad.kk[3][3])
        elif i == 3:
            a0 = mathutils.Vector(quad.kk[0][1]) - mathutils.Vector(quad.kk[0][0])
            a3 = mathutils.Vector(quad.kk[3][1]) - mathutils.Vector(quad.kk[3][0])
        else:
            raise ValueError("wrong i")
        return a0, a3

    def calculate_coefs_along_segment(self, quad, i: int):
        curve_name = quad.curves[i].name
        phantom_curve = self.collection.greg_settings.phantom_curves[curve_name]
        quad_i = phantom_curve.quads.find(quad.name)
        a0, a3 = NewGlobalList.extract_a0_a3(quad, i)
        p0, p1, p2, p3 = NewGlobalList.get_edge_control_points_from_kk(quad, i)
        neighbour_quad = self.get_neighbour_quad(quad, i)
        if neighbour_quad is None:
            a1, a2 = NewGlobalList.calculate_free_coefs(a0, a3, quad_i, phantom_curve, self.collection)
            phantom_curve.b1_finished = True
            phantom_curve.b2_finished = True
            phantom_curve.conditional_sharp = True
        else:
            s0 = p1 - p0
            s1 = p2 - p1
            s2 = p3 - p2
            neighbour_i = neighbour_quad.curves.find(quad.curves[i].name)
            bb0, bb2 = NewGlobalList.extract_a0_a3(neighbour_quad, neighbour_i)
            if quad.dirs[i] != neighbour_quad.dirs[neighbour_i]:
                bb0, bb2 = bb2, bb0
            b0 = (bb0 - a0).normalized()
            b2 = (bb2 - a3).normalized()
            for v in (a0, a3, b0, b2, s0, s2):
                if v.length < TH:
                    a1, a2 = NewGlobalList.calculate_free_coefs(a0, a3, quad_i, phantom_curve, self.collection)
                    phantom_curve.conditional_sharp = True
                    break
            else:
                if not are_coplanar(a0, b0, s0) or not are_coplanar(a3, s2, b2):
                    a1, a2 = NewGlobalList.calculate_free_coefs(a0, a3, quad_i, phantom_curve, self.collection)
                    phantom_curve.conditional_sharp = True
                elif b0.cross(s0).length < TH or b2.cross(s2).length < TH:
                    a1, a2 = NewGlobalList.calculate_free_coefs(a0, a3, quad_i, phantom_curve, self.collection)
                    phantom_curve.conditional_sharp = True
                else:
                    k0, h0 = get_coefs(b0, s0, a0)
                    k1, h1 = get_coefs(b2, s2, a3)
                    ve = self.calculate_ve(quad, i, neighbour_quad, neighbour_i)
                    b1 = NewGlobalList.calculate_b1(k0, k1, h0, h1, b0, b2, s0, s1, s2, a0, a3, ve)
                    prev_b1 = phantom_curve.b2 if quad_i == 0 else phantom_curve.b1
                    if prev_b1 == (0,0,0):
                        if quad_i == 0:
                            phantom_curve.b1_prop = b1
                        else:
                            phantom_curve.b2_prop = b1
                        return
                    else:
                        prev_b1 = mathutils.Vector(prev_b1)
                        b1 = (b1 - prev_b1)/2
                        if quad_i == 0:
                            phantom_curve.b1_prop = b1
                            phantom_curve.b2_prop = -b1
                        else:
                            phantom_curve.b2_prop = b1
                            phantom_curve.b1_prop = -b1
                        next_curve_no = NewGlobalList.get_curve_name_for_shear(quad,
                                                                               i,
                                                                               self.collection)
                        other_curve_no = NewGlobalList.get_curve_name_for_shear(neighbour_quad,
                                                                                neighbour_i,
                                                                                self.collection)
                        
                        if (next_curve_no > other_curve_no) or\
                           (next_curve_no == other_curve_no and quad_i == 1):
                            try:
                                phantom_curve.invert_shear[quad_i] = True
                            except IndexError:
                                print("quad_i", quad_i)
                        if phantom_curve.source_curve.greg_is_sharp:
                            a1, a2 = NewGlobalList.calculate_free_coefs(a0, a3, quad_i, phantom_curve, self.collection)
                        else:
                            b1_corrected = NewGlobalList.get_b1_corrected(b1, quad, i, p0, p1, p2, p3, phantom_curve, quad_i)
                            multiply0 = 2 * k0
                            add0 = k1 * b0 + 2 * h0 * s1 + h1 * s0
                            multiply1 = 2 * k1
                            add1 = k0 * b2 + h0 * s2 + 2 * h1 * s1
                            phantom_curve.coefs_multiply[quad_i][0] = multiply0
                            phantom_curve.coefs_add[quad_i][0] = add0
                            phantom_curve.coefs_multiply[quad_i][1] = multiply1
                            phantom_curve.coefs_add[quad_i][1] = add1
                        
                            a1 = 1/3 * (multiply0 * b1_corrected + add0)
                            a2 = 1/3 * (multiply1 * b1_corrected + add1)
        if i == 0:
            quad.kk[1][1] = a1 + p1
            quad.kk[1][2] = a2 + p2
        elif i == 1:
            quad.kk1[0][1] = a1 + p1
            quad.kk1[1][1] = a2 + p2
        elif i == 2:
            quad.kk[2][1] = a1 + p1
            quad.kk[2][2] = a2 + p2
        elif i == 3:
            quad.kk1[0][0] = a1 + p1
            quad.kk1[1][0] = a2 + p2
        if quad_i == 0:
            phantom_curve.b1_finished = True
        else:
            phantom_curve.b2_finished = True
    
    @staticmethod
    def get_b1_corrected(b1, quad, i, p0, p1, p2, p3, phantom_curve, quad_i):
        bulge = b1.normalized()
        if quad.dirs[i]:
            pp0, pp1, pp2, pp3 = p0, p1, p2, p3
        else:
            pp0, pp1, pp2, pp3 = p3, p2, p1, p0
        curve_vec = pp2 + pp3 - pp0 - pp1
        if are_collinear(curve_vec, bulge):
            curve_vec = pp3 - pp0
        shear = (curve_vec - curve_vec.project(bulge)).normalized()
        tilt = bulge.cross(shear)
        if phantom_curve.invert_shear[quad_i]:
            shear = - shear
        if phantom_curve.mirrored:
            tilt = -tilt
        curve_obj = phantom_curve.source_curve
        b_correction = bulge*curve_obj.greg_bulge +  shear*curve_obj.greg_shear + tilt*curve_obj.greg_tilt
        return b1 + b_correction

    @staticmethod
    def recalculate_coefs_along_segment(quad, i: int, collection):
        curve_name = quad.curves[i].name
        phantom_curve = collection.greg_settings.phantom_curves[curve_name]
        quad_i = phantom_curve.quads.find(quad.name)
        p0, p1, p2, p3 = NewGlobalList.get_edge_control_points_from_kk(quad, i)
        if phantom_curve.conditional_sharp or phantom_curve.source_curve.greg_is_sharp:
            a0, a3 = NewGlobalList.extract_a0_a3(quad, i)
            a1, a2 = NewGlobalList.calculate_free_coefs(a0, a3, quad_i, phantom_curve, collection)
        else:
            if quad_i == 0:
                b1 = phantom_curve.b1
            else:
                b1 = phantom_curve.b2
            b1_corrected = NewGlobalList.get_b1_corrected(b1, quad, i, p0, p1, p2, p3, phantom_curve, quad_i)
            multiply0 = phantom_curve.coefs_multiply[quad_i][0]
            add0 = mathutils.Vector(phantom_curve.coefs_add[quad_i][0])
            multiply1 = phantom_curve.coefs_multiply[quad_i][1]
            add1 = mathutils.Vector(phantom_curve.coefs_add[quad_i][1])
            a1 = 1/3 * (multiply0 * b1_corrected + add0)
            a2 = 1/3 * (multiply1 * b1_corrected + add1)
        if i == 0:
            quad.kk[1][1] = a1 + p1
            quad.kk[1][2] = a2 + p2
        elif i == 1:
            quad.kk1[0][1] = a1 + p1
            quad.kk1[1][1] = a2 + p2
        elif i == 2:
            quad.kk[2][1] = a1 + p1
            quad.kk[2][2] = a2 + p2
        elif i == 3:
            quad.kk1[0][0] = a1 + p1
            quad.kk1[1][0] = a2 + p2

    @staticmethod
    def get_curve_name_for_shear(quad, i, collection):
        if i == 0:
            if quad.dirs[i]:
                next_curve_name = quad.curves[1].name
            else:
                next_curve_name = quad.curves[3].name
        elif i == 1:
            if quad.dirs[i]:
                next_curve_name = quad.curves[2].name
            else:
                next_curve_name = quad.curves[0].name
        elif i == 2:
            if quad.dirs[i]:
                next_curve_name = quad.curves[1].name
            else:
                next_curve_name = quad.curves[3].name
        elif i == 3:
            if quad.dirs[i]:
                next_curve_name = quad.curves[2].name
            else:
                next_curve_name = quad.curves[0].name
        phantom_curve = collection.greg_settings.phantom_curves[next_curve_name]
        return int(phantom_curve.source_curve_name)

    @staticmethod
    def calculate_free_coefs(a0: mathutils.Vector, a3: mathutils.Vector, quad_i, phantom_curve, collection):
        curve_obj = phantom_curve.source_curve
        a_central = 0.15*(a0 + a3) #looks better when it's small
        bulge = a_central
        p0, p1, p2, p3 = NewGlobalList.extract_control_points(collection, phantom_curve)
        curve_vec = p2 + p3 - p0 - p1
        if are_collinear(curve_vec, bulge):
            curve_vec = p3 - p0
        shear = (curve_vec - curve_vec.project(bulge)).normalized() * a_central.length
        tilt = bulge.cross(shear).normalized() * a_central.length
        if phantom_curve.mirrored:
            tilt = -tilt

        if quad_i == 0:
            correction = bulge*curve_obj.greg_bulge1 +  shear*curve_obj.greg_shear1 + tilt*curve_obj.greg_tilt1
        else:
            correction = bulge*curve_obj.greg_bulge2 +  shear*curve_obj.greg_shear2 + tilt*curve_obj.greg_tilt2
        a_central_corrected = a_central + correction
        a1 = a0*1/3 + a_central_corrected*2/3
        a2 = a3*1/3 + a_central_corrected*2/3
        return a1, a2

    def calculate_ve(self, quad, i: int, neighbour, neighbour_i: int) -> mathutils.Vector:
        phantom_curve1 = self.collection.greg_settings.phantom_curves[quad.curves[(i+2) % 4].name]
        phantom_curve2 = self.collection.greg_settings.phantom_curves[neighbour.curves[(neighbour_i+2) % 4].name]
        p1 = self.calculate_midpoint_phantom(phantom_curve1)
        p2 = self.calculate_midpoint_phantom(phantom_curve2)
        ve = p2 - p1
        return ve

    @staticmethod
    def calculate_b1(k0: float,
                     k1: float,
                     h0: float,
                     h1: float,
                     b0: mathutils.Vector,
                     b2: mathutils.Vector,
                     s0: mathutils.Vector,
                     s1: mathutils.Vector,
                     s2: mathutils.Vector,
                     a0: mathutils.Vector,
                     a3: mathutils.Vector,
                     ve: mathutils.Vector):
        b1_ref = (b0 + b2) / 2
        #a1_ref = (a0.length + a3.length) / 2
        #ve = ve.normalized() * a1_ref
        ve = ve * 0.5
        res = (-8/3*ve - a0 - k1*b0 - 2*h0*s1 - h1*s0 - k0*b2 - h0*s2 - 2*h1*s1 - a3) / (2*(k0 + k1))
                    #res = res * (b0_ref.length_squared / res.dot(b0_ref))
        return (res + b1_ref) * 0.7
        #self.b1 = res

    def calculate_remaining_kk(self):
        for phantom_curve in self.collection.greg_settings.phantom_curves:
            if len(phantom_curve.quads) > 0:
                if phantom_curve.b1_finished == False:
                    quad_name = phantom_curve.quads[0].name
                    quad = self.collection.greg_settings.quads[quad_name]
                    quad_i = quad.curves.find(phantom_curve.name)
                    self.calculate_coefs_along_segment(quad, quad_i)
                if phantom_curve.b2_finished == False:
                    quad_name = phantom_curve.quads[1].name
                    quad = self.collection.greg_settings.quads[quad_name]
                    quad_i = quad.curves.find(phantom_curve.name)
                    self.calculate_coefs_along_segment(quad, quad_i)

    def calculate_kk(self):
        for quad in self.collection.greg_settings.quads:
            self.calculate_coefs(quad)
        for quad in self.collection.greg_settings.quads:
            for i in range(4):
                self.calculate_coefs_along_segment(quad, i)
        self.calculate_remaining_kk()

    def add_curves_and_bpoints(self):
        for curve_item in self.collection.greg_settings.curves:
            curve_obj = curve_item.curve
            self.add_curve_copy(curve_obj)
            for modifier in curve_obj.modifiers:
                if modifier.type == "MIRROR":
                    for axis in range(3):
                        if modifier.use_axis[axis]:
                            names = [item.name for item in curve_obj.greg_curve_settings.phantom_curves_ids]
                            for name in names:
                                phantom_curve = self.collection.greg_settings.phantom_curves[name]
                                self.add_mirror_from_phantome(phantom_curve, curve_obj, modifier.mirror_object, axis)


    
    def add_bpoint_if_needed(self, co, original_name, mirror_other_names):
        for phantom_bpoint in self.collection.greg_settings.phantom_bpoints:
            if phantom_bpoint.original_empty_name in (original_name, *mirror_other_names):
                if same_coords(co, phantom_bpoint.co):
                    return phantom_bpoint.name
        new_bpoint = self.collection.greg_settings.phantom_bpoints.add()
        new_bpoint.name = str(self.ids_counter)
        self.ids_counter += 1
        new_bpoint.co_prop = co
        new_bpoint.original_empty_name = original_name
        return new_bpoint.name
    
    def add_phantom_curve(self, co1, co2, handle1, handle2, original_curve_obj, mirrored):
        empty1_settings = original_curve_obj.greg_curve_settings.end1_empty.greg_empty_settings
        empty2_settings = original_curve_obj.greg_curve_settings.end2_empty.greg_empty_settings
        empty1_name = empty1_settings.name
        empty2_name = empty2_settings.name
        empty1_mirror_bridge_other_names = [it.name for it in empty1_settings.mirror_bridge_other_names]
        empty2_mirror_bridge_other_names = [it.name for it in empty2_settings.mirror_bridge_other_names]
        bpoint1_name = self.add_bpoint_if_needed(co1, empty1_name, empty1_mirror_bridge_other_names)
        bpoint2_name = self.add_bpoint_if_needed(co2, empty2_name, empty2_mirror_bridge_other_names)
        bpoint_names = (bpoint1_name, bpoint2_name)
        curve_key = f"{'.'.join(sorted(bpoint_names))}_{original_curve_obj.greg_curve_settings.name}"
        if curve_key in self.curves_dict:
            phantom_curve_name = self.curves_dict[curve_key]
            if phantom_curve_name not in original_curve_obj.greg_curve_settings.phantom_curves_ids[phantom_curve_name]:
                curve_item = original_curve_obj.greg_curve_settings.phantom_curves_ids.add()
                curve_item.name = phantom_curve_name
        else:
            phantom_curve = self.collection.greg_settings.phantom_curves.add()
            phantom_curve.name = str(self.ids_counter)
            self.curves_dict[curve_key] = phantom_curve.name
            self.ids_counter += 1
            phantom_curve.mirrored = mirrored
            phantom_curve.handle1_prop = handle1
            phantom_curve.handle2_prop = handle2
            phantom_curve.source_curve = original_curve_obj
            phantom_curve.source_curve_name = original_curve_obj.greg_curve_settings.name
            end_names = []
            for i in range(2):
                bpoint = self.collection.greg_settings.phantom_bpoints[bpoint_names[i]]
                if i == 0:
                    phantom_curve.bpoint1_name = bpoint.name
                else:
                    phantom_curve.bpoint2_name = bpoint.name
                end = bpoint.ends.add()
                end.name = str(self.ids_counter)
                self.ids_counter += 1
                end.curve_name = phantom_curve.name
                end.bpoint_name = bpoint.name
                end.curve_i = i
                end_names.append(end.name)
            phantom_curve.end1_name = end_names[0]
            phantom_curve.end2_name = end_names[1]
            curve_item = original_curve_obj.greg_curve_settings.phantom_curves_ids.add()
            curve_item.name = phantom_curve.name
                    
    def add_curve_copy(self, curve_obj):
        p1, p2 = curve_obj.data.splines[0].bezier_points
        self.add_phantom_curve(p1.co, p2.co, p1.handle_right, p2.handle_left, curve_obj, False)

    @staticmethod
    def compare_vecs_all_same(vecs1, vecs2):
        return (NewGlobalList.compare_vecs_all_same_direct(vecs1, vecs2) or
                NewGlobalList.compare_vecs_all_same_reverse(vecs1, vecs2))
    @staticmethod
    def compare_vecs_all_same_direct(vecs1, vecs2):
        for vec1, vec2 in zip(vecs1, vecs2):
            if not same_coords(vec1, vec2):
                return False
        return True
    @staticmethod
    def compare_vecs_all_same_reverse(vecs1, vecs2):
        vecs3 = [vecs2[1], vecs2[0], vecs2[3], vecs2[2]]
        for vec1, vec2 in zip(vecs1, vecs3):
            if not same_coords(vec1, vec2):
                return False
        return True
    
    def add_mirror_from_phantome(self, phantom_curve, curve_obj, mirror_obj, axis):
        co1, handle1, handle2, co2 = NewGlobalList.extract_control_points(self.collection, phantom_curve)
        vecs = (co1, co2, handle1, handle2)
        mirrored_vecs = [mirror_vec(vec, mirror_obj, axis) for vec in vecs]
        if not NewGlobalList.compare_vecs_all_same(vecs, mirrored_vecs):
            self.add_phantom_curve(*mirrored_vecs, curve_obj, not phantom_curve.mirrored)

    def prepare_for_greg(self):
        for empty_item in self.collection.greg_settings.empties:
            empty_obj = empty_item.empty
            for end in empty_obj.greg_empty_settings.curve_ends:
                apply_hook(end)
                add_hook(end)
        for curve_item in self.collection.greg_settings.curves:
            curve_obj = curve_item.curve
            curve_settings = curve_obj.greg_curve_settings
            curve_settings.phantom_curves_ids.clear()
        self.collection.greg_settings.phantom_curves.clear()
        self.collection.greg_settings.phantom_bpoints.clear()
        self.collection.greg_settings.quads.clear()
    
    @staticmethod
    def extract_control_points(collection, phantom_curve):
        bpoint1 = collection.greg_settings.phantom_bpoints[phantom_curve.bpoint1_name]
        bpoint2 = collection.greg_settings.phantom_bpoints[phantom_curve.bpoint2_name]
        k0 = bpoint1.co
        k3 = bpoint2.co
        k1 = phantom_curve.handle1
        k2 = phantom_curve.handle2
        return [k0, k1, k2, k3]
    
    def extract_vert(self, quad, curve_no, bpoint_no):
        phantom_curve = self.collection.greg_settings.phantom_curves[quad.curves[curve_no].name]
        if bpoint_no == 0:
            bpoint_name = phantom_curve.bpoint1_name
        elif bpoint_no == 1:
            bpoint_name = phantom_curve.bpoint2_name
        return self.collection.greg_settings.phantom_bpoints[bpoint_name].vert
    
    def extract_corner(self, quad, i: int):
        if i == 0:
            return self.extract_vert(quad, 0, 0)
        if i == 1:
            return self.extract_vert(quad, 0, 1)
        if quad.dirs[2]:
            if i == 2:
                return self.extract_vert(quad, 2, 1)
            if i == 3:
                return self.extract_vert(quad, 2, 0)
        else:
            if i == 2:
                return self.extract_vert(quad, 2, 0)
            if i == 3:
                return self.extract_vert(quad, 2, 1)
    
    def render_mesh(self, d: "DependantsOfResolution|DependantsOfResolution_np", name: str, context: bpy.types.Context):
        greg_settings = self.collection.greg_settings
        print("len(greg_settings.phantom_bpoints)", len(greg_settings.phantom_bpoints))
        print("len(greg_settings.phantom_curves)", len(greg_settings.phantom_curves))
        for i, bpoint in enumerate(greg_settings.phantom_bpoints):
            coords = bpoint.co
            add_corner(self, coords)
            bpoint.vert = i
        for phantom_curve in greg_settings.phantom_curves:
            verts = add_border(*NewGlobalList.extract_control_points(self.collection, phantom_curve), self, d)
            phantom_curve.first_vert = verts[0]
        for quad in greg_settings.quads:
            self.render_quad(quad, d)
        if self.verts is not None and self.faces is not None:
            if greg_settings.mesh_obj is None:
                mesh = bpy.data.meshes.new(name=name + "_Mesh")
                obj = bpy.data.objects.new(name + "_GeneratedMesh", mesh)
                context.collection.objects.link(obj)
                greg_settings.mesh_obj = obj
            else:
                obj = greg_settings.mesh_obj
                mesh = bpy.data.meshes.new(name=name + "_Mesh")
                old_mesh = obj.data
                obj.data = mesh
                bpy.data.meshes.remove(old_mesh)
            mesh.from_pydata(self.verts, [], self.faces)
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(mesh)
            bm.clear()
            mesh.update()
            bm.free()
            
            return obj
        return None

    
    def render_quad(self, quad, d):
        borders = []
        for i in range(4):
            curve_name = quad.curves[i].name
            phantom_curve = self.collection.greg_settings.phantom_curves[curve_name]
            init_point = phantom_curve.first_vert
            borders.append(list(range(init_point, init_point + d.nedges - 1)))
            if not quad.dirs[i]:
                borders[i] = list(reversed(borders[i]))
        corners = [self.extract_corner(quad, i) for i in range(4)]
        calc_gregory_surf(quad.kk, quad.kk1, d, *borders, *corners, self)
        quad.first_vert = len(self.verts) - ((d.nedges-1) * (d.nedges-1))
    
def render_existing_quad(quad, d, collection, i_s):
    mesh = collection.greg_settings.mesh_obj.data
    for i in i_s:
        NewGlobalList.recalculate_coefs_along_segment(quad, i, collection)
    coords = calc_quad_gregory_verts(quad.kk, quad.kk1, d)
    for i, vert_num in enumerate(range(quad.first_vert, quad.first_vert + (d.nedges-1)*(d.nedges-1))):
        mesh.vertices[vert_num].co = coords[i]



#**************************************************************************

class GregId(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")

class GregArrowItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")
    arrow: bpy.props.PointerProperty(type=bpy.types.Object)

class GregBasicEnd(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")
    curve: bpy.props.PointerProperty(type=bpy.types.Object)
    end: bpy.props.IntProperty(default=-1)

class GregArrow(bpy.types.PropertyGroup):
    used_for_greg: bpy.props.BoolProperty(default=False)
    name: bpy.props.StringProperty(default="")
    coplanars: bpy.props.CollectionProperty(type=GregBasicEnd)

class GregCurveEndItem(bpy.types.PropertyGroup):
    basic_end: bpy.props.PointerProperty(type=GregBasicEnd)
    empty: bpy.props.PointerProperty(type=bpy.types.Object)
    coplanar_vectors: bpy.props.CollectionProperty(type=GregArrowItem)
    collinear_to: bpy.props.CollectionProperty(type=GregBasicEnd)
    hook: bpy.props.StringProperty(default="")
    name: bpy.props.StringProperty(default="")
    @property
    def is_coplanar(self):
        return len(self.coplanar_vectors) > 0
    @property
    def is_collinear(self):
        return len(self.collinear_to) > 0

class GregEmptyItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")
    empty: bpy.props.PointerProperty(type=bpy.types.Object)

class GregCurveItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")
    curve: bpy.props.PointerProperty(type=bpy.types.Object)
    
class GregQuad(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")
    curves: bpy.props.CollectionProperty(type=GregId)
    kk: bpy.props.FloatVectorProperty(size=(4,4,3))
    kk1: bpy.props.FloatVectorProperty(size=(2,2,3))
    dirs: bpy.props.BoolVectorProperty(size=4)
    first_vert: bpy.props.IntProperty()

class GregPhantomCurveEnd(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")
    curve_name: bpy.props.StringProperty(default="")
    bpoint_name: bpy.props.StringProperty(default="")
    curve_i: bpy.props.IntProperty()

class GregPhantomCurve(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")
    handle1_prop: bpy.props.FloatVectorProperty()
    handle2_prop: bpy.props.FloatVectorProperty()
    end1_name: bpy.props.StringProperty(default="")
    end2_name: bpy.props.StringProperty(default="")
    bpoint1_name: bpy.props.StringProperty(default="")
    bpoint2_name: bpy.props.StringProperty(default="")
    quads: bpy.props.CollectionProperty(type=GregId)
    first_vert: bpy.props.IntProperty()
    finished: bpy.props.BoolProperty(default=False)
    b1_prop: bpy.props.FloatVectorProperty()
    b2_prop: bpy.props.FloatVectorProperty()
    b1_finished: bpy.props.BoolProperty(default=False)
    b2_finished: bpy.props.BoolProperty(default=False)
    source_curve: bpy.props.PointerProperty(type=bpy.types.Object)
    source_curve_name: bpy.props.StringProperty(default="")
    mirrored: bpy.props.BoolProperty(default=False)
    conditional_sharp: bpy.props.BoolProperty(default=False)
    coefs_multiply: bpy.props.FloatVectorProperty(size=(2, 2))
    coefs_add: bpy.props.FloatVectorProperty(size=(2, 2, 3))
    invert_shear: bpy.props.BoolVectorProperty(size=2)
    @property
    def handle1(self):
        return mathutils.Vector(self.handle1_prop)
    @property
    def handle2(self):
        return mathutils.Vector(self.handle2_prop)
    @property
    def b1(self):
        return mathutils.Vector(self.b1_prop)
    @property
    def b2(self):
        return mathutils.Vector(self.b2_prop)

class GregPhantomBpoint(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="default")
    co_prop: bpy.props.FloatVectorProperty()
    ends: bpy.props.CollectionProperty(type=GregPhantomCurveEnd)
    vert: bpy.props.IntProperty()
    original_empty_name: bpy.props.StringProperty(default="")
    @property
    def co(self):
        return mathutils.Vector(self.co_prop)

class GregCollectionSettings(bpy.types.PropertyGroup):
    used_for_greg: bpy.props.BoolProperty(default=False)
    empties: bpy.props.CollectionProperty(type=GregEmptyItem)
    curves: bpy.props.CollectionProperty(type=GregCurveItem)
    arrows: bpy.props.CollectionProperty(type=GregArrowItem)
    quads: bpy.props.CollectionProperty(type=GregQuad)
    phantom_curves: bpy.props.CollectionProperty(type=GregPhantomCurve)
    phantom_bpoints: bpy.props.CollectionProperty(type=GregPhantomBpoint)
    mesh_obj: bpy.props.PointerProperty(type=bpy.types.Object)
    max_id: bpy.props.IntProperty(default=0)

class GregEmpty(bpy.types.PropertyGroup):
    curve_ends: bpy.props.CollectionProperty(type=GregCurveEndItem)
    used_for_greg: bpy.props.BoolProperty(default=False)
    coplanars: bpy.props.CollectionProperty(type=GregArrowItem)
    name: bpy.props.StringProperty(default="")
    mirror_bridge_other_names: bpy.props.CollectionProperty(type=GregId)

class GregCurve(bpy.types.PropertyGroup):
    used_for_greg: bpy.props.BoolProperty(default=False)
    end1_empty: bpy.props.PointerProperty(type=bpy.types.Object)
    end2_empty: bpy.props.PointerProperty(type=bpy.types.Object)
    end1_name: bpy.props.StringProperty(default="")
    end2_name: bpy.props.StringProperty(default="")
    name: bpy.props.StringProperty(default="")
    phantom_curves_ids: bpy.props.CollectionProperty(type=GregId)
    is_mirror_bridge: bpy.props.BoolProperty(default=False)
    bridge_mirror_object: bpy.props.PointerProperty(type=bpy.types.Object)
    bridge_mirror_axis: bpy.props.IntProperty(default=0)
    bridge_mirror_other_i: bpy.props.IntProperty(default=0)

#**************************************************************************

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

'''def verify_is_closed_path_and_of_length_4(curves: List[bpy.types.Object]):
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
    return True''' # TODO

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

def cb_update(self, context):
    collection = get_greg_collection(self)
    quads_done = []
    self_name = self.greg_curve_settings.name
    for phantom_curve_el in self.greg_curve_settings.phantom_curves_ids:
        phantom_curve_name = phantom_curve_el.name
        phantom_curve = collection.greg_settings.phantom_curves[phantom_curve_name]
        for quad_id in phantom_curve.quads:
            if quad_id not in quads_done:
                quads_done.append(quad_id)
                quad = collection.greg_settings.quads[quad_id.name]
                i_s = []
                phantom_curves = collection.greg_settings.phantom_curves
                for i, curve_item in enumerate(quad.curves):
                    if phantom_curves[curve_item.name].source_curve_name == self_name:
                        i_s.append(i)
                render_existing_quad(quad, d, collection, i_s)
    return None

class OBJECT_PT_greg_curve_properties(bpy.types.Panel):
    bl_idname = "OBJECT_PT_greg_curve_propertirs"
    bl_label = "Gregory Curve Properties"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.prop( obj, 'greg_bulge')
        layout.prop( obj, 'greg_tilt')
        layout.prop( obj, 'greg_shear')

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
    bl_category = "Tool"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.prop( obj, 'greg_bulge1')
        layout.prop( obj, 'greg_tilt1')
        layout.prop( obj, 'greg_shear1')

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
    bl_category = "Tool"

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

class PrintItemInfo(bpy.types.Operator):
    """Gregory: print info about selected curve, arrow or empty"""
    bl_idname = "object.greg_print_info"
    bl_label = "Print greg item info"         # Display name in the interface.
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

class MakeCurveMirrorBridge(bpy.types.Operator):
    """Gregory: make curve a bridge through mirror"""
    bl_idname = "object.make_curve_mirror_bridge"
    bl_label = "Make curve bridge through mirror"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        obj = context.active_object
        if obj is None:
            return False
        if not obj.greg_curve_settings.used_for_greg:
            return False
        if obj.greg_curve_settings.is_mirror_bridge:
            return False
        selection = context.selected_objects
        if len(selection) != 2:
            return False
        other = [sel for sel in selection if sel != obj][0]
        if not other.greg_empty_settings.used_for_greg:
            return False
        if not (other in (obj.greg_curve_settings.end1_empty, obj.greg_curve_settings.end2_empty)):
            return False
        if check_curve_crosses_mirror(obj):
            return True
        return False

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        curve = context.active_object
        i, axis = check_curve_crosses_mirror(curve)
        mirror_obj = curve.modifiers[i].mirror_object
        target = [sel for sel in context.selected_objects if sel != curve][0]
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
    self.layout.operator(MakeCurveMirrorBridge.bl_idname)


class UnsetCurveMirrorBridge(bpy.types.Operator):
    """Gregory: unset curve a bridge through mirror"""
    bl_idname = "object.unset_curve_mirror_bridge"
    bl_label = "Unset curve bridge through mirror"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        obj = context.active_object
        if obj is None:
            return False
        if len(context.selected_objects) != 1:
            return False
        return obj.greg_curve_settings.is_mirror_bridge

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        curve = context.active_object
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


class GlobalForSubdivide:
    def __init__(self):
        self.curves: Dict[str, CurveForSubdivide] = {}
        self.points: Dict[int, PointForSubdivide] = {}
        self.max_id = 0
        self.borders: Dict[int, Dict[int, Tuple[int, List["CurveForSubdivide"], List["PointForSubdivide"]]]] = {}
        self.v_grid = None
        self.points_grid: List[List["MiddlePoint"]] = []
        self.edge_points: List[List["PointForSubdivide"]] = []
        self.corner_handles: List[Tuple[mathutils.Vector, mathutils.Vector]] = []
        self.outer_corner_handles: List[Tuple[mathutils.Vector, mathutils.Vector]] = []
        self.real_curves = []
        self.border0: List[CurveForSubdivide] = []
        self.border1: List[CurveForSubdivide] = []
        self.border2: List[CurveForSubdivide] = []
        self.border3: List[CurveForSubdivide] = []
        self.vertical: Optional[Tuple[Optional[bpy.types.Object], int]] = None
        self.horizontal: Optional[Tuple[Optional[bpy.types.Object], int]] = None
        self.diagonal1: Optional[Tuple[Optional[bpy.types.Object], int]] = None
        self.diagonal2: Optional[Tuple[Optional[bpy.types.Object], int]] = None
        self.quads: Dict[str, List[int]] = {}
        self.optimal_quad: List[int]
        self.xtot: int
        self.ytot: int
    
    def add_point_if_needed(self, empty_obj, co, triangles):
        for point in self.points.values():
            if point.empty == empty_obj or point.empty.greg_empty_settings.name in [it.name for it in empty_obj.greg_empty_settings.mirror_bridge_other_names]:
                if same_coords(point.co, co):
                    point.add_triangles(triangles)
                    return point
        point = PointForSubdivide(co, empty_obj, self.max_id, triangles)
        self.max_id += 1
        self.points[point.number] = point
        return point

    def add_curve(self,
                  empties: Tuple[bpy.types.Object, bpy.types.Object],
                  co_s: List[mathutils.Vector],
                  is_mirrored: bool,
                  original_curve: bpy.types.Object,
                  triangles_s: List[List[mathutils.Vector]],
                  mirror_sequence: Optional[List["MirrorSequenceItem"]] = None):
        points = [self.add_point_if_needed(empty, co, triangles) for empty, co, triangles in zip(empties, co_s, triangles_s)]
        p1, p2 = points
        key = GlobalForSubdivide.make_key(p1.number, p2.number)
        if key not in self.curves:
            curve = CurveForSubdivide(p1, p2, is_mirrored, original_curve, mirror_sequence)
            self.curves[key] = curve
            p1.curves[key] = (curve, 0)
            p2.curves[key] = (curve, 1)
            return curve
        return self.curves[key] #TODO verify it's correct

    @staticmethod
    def get_triangle_from_empty(empty: bpy.types.Object):
        x = mathutils.Vector((1, 0, 0))
        y = mathutils.Vector((0, 1, 0))
        z = mathutils.Vector((0, 0, 1))
        res = []
        for vec in (x, y, z):
            mat = empty.matrix_world.copy()
            mat.invert()
            vec_to_emp = vec @ mat
            res.append(vec_to_emp + empty.matrix_world.translation)
        return res

    
    def add_curve_and_mirrors(self, curve_obj):
        empties = (curve_obj.greg_curve_settings.end1_empty, curve_obj.greg_curve_settings.end2_empty)
        co_s = [empty.matrix_world.translation for empty in empties]
        triangles_s = [[GlobalForSubdivide.get_triangle_from_empty(empty)] for empty in empties]
        curves = [self.add_curve(empties, co_s, False, curve_obj, triangles_s)]
        for modifier in curve_obj.modifiers:
            if modifier.type == 'MIRROR':
                mirror_object = modifier.mirror_object
                for axis in range(3):
                    if modifier.use_axis[axis]:
                        new_curves = []
                        for curve in curves:
                            co_s = [p.co for p in curve.points]
                            new_triangles = [p.generate_new_triangles(mirror_object, axis) for p in curve.points]
                            new_cos = [mirror_vec(co, mirror_object, axis) for co in co_s]
                            new_mirror_sequence = curve.mirror_sequence.copy()
                            new_mirror_sequence.append(MirrorSequenceItem(mirror_object, axis))
                            new_curves.append(self.add_curve(empties, new_cos, True, curve_obj, new_triangles, new_mirror_sequence))
                        curves.extend(new_curves)

    def find_borders_step(self,
                          curves_sequence: List["CurveForSubdivide"], 
                          prev_point: "PointForSubdivide", 
                          next_curve: "CurveForSubdivide", 
                          point: "PointForSubdivide",
                          number: int,
                          all_corresponding_points: List["PointForSubdivide"],
                          points_sequence: List["PointForSubdivide"]):
        next_point = [p for p in next_curve.points if p != prev_point][0]
        if next_point not in points_sequence:
            points_sequence.append(next_point)
            if next_point in all_corresponding_points:
                if next_point == point:
                    print(1)
                    return False
                other_number = next_point.number
                self.borders[number][other_number] = (len(curves_sequence), curves_sequence.copy(), points_sequence.copy())
                if other_number not in self.borders:
                    self.borders[other_number] = {}
                self.borders[other_number][number] = (len(curves_sequence), list(reversed(curves_sequence)), list(reversed(points_sequence)))
            else:
                prev_point = next_point
                next_curves = [c[0] for c in next_point.curves.values() if c[0] not in curves_sequence]
                for next_curve  in next_curves:
                    curves_sequence.append(next_curve)
                    if not self.find_borders_step(curves_sequence, prev_point, next_curve, point, number, all_corresponding_points, points_sequence):
                        return False #error!
                    curves_sequence.pop()
            points_sequence.pop()
        return True

    
    def find_borders(self, empties):
        all_corresponding_points: List["PointForSubdivide"] = []
        for empty in empties:
            corresponding_points = [point for point in self.points.values() if point.empty == empty]
            if len(corresponding_points) == 0:
                print(0)
                return False
            all_corresponding_points.extend(corresponding_points)
        print("len(all_corresponding_points)", len(all_corresponding_points))
        for point in all_corresponding_points:
            number = point.number
            if number not in self.borders:
                self.borders[number] = {}
            #curves_visited = [value[1][0] for value in self.borders[number].values()]
            for curve, _ in point.curves.values():
                #if curve not in curves_visited:
                prev_point = point
                next_curve = curve
                curves_sequence = [next_curve]
                points_sequence = [prev_point]
                if not self.find_borders_step(curves_sequence, prev_point, next_curve, point, number, all_corresponding_points, points_sequence):
                    return False #error!
        print("self.borders", {key: {key1: value1[0] for key1, value1 in value.items()} for key, value in self.borders.items()})
        for point in self.borders.keys():
            stack = [point]
            if self.step(stack) == False:
                print(3)
                return False
        not_mirrored_max = 0
        optimal_quad = None
        for quad in self.quads.values():
            not_mirrored = GlobalForSubdivide.get_count_not_mirrored(quad, self.borders)
            if not_mirrored > not_mirrored_max:
                not_mirrored_max = not_mirrored
                optimal_quad = quad
        if optimal_quad is None:
            print(4)
            return False
        if not_mirrored_max != len([curve for curve in self.curves.values() if not curve.is_mirrored]):
            print(5)
            return False
        self.optimal_quad = optimal_quad
        self.border0 = self.borders[self.optimal_quad[0]][self.optimal_quad[1]][1]
        self.border1 = self.borders[self.optimal_quad[1]][self.optimal_quad[2]][1]
        self.border2 = self.borders[self.optimal_quad[3]][self.optimal_quad[2]][1]
        self.border3 = self.borders[self.optimal_quad[0]][self.optimal_quad[3]][1]
        self.edge_points = [self.borders[self.optimal_quad[0]][self.optimal_quad[1]][2],
                            self.borders[self.optimal_quad[1]][self.optimal_quad[2]][2],
                            self.borders[self.optimal_quad[3]][self.optimal_quad[2]][2],
                            self.borders[self.optimal_quad[0]][self.optimal_quad[3]][2]]
        return True



    @staticmethod
    def get_count_not_mirrored(quad, borders):
        counter = 0
        for i in range(4):
            curves_sequence = borders[quad[i]][quad[(i+1)%4]][1]
            for curve in curves_sequence:
                if not curve.is_mirrored:
                    counter += 1
        return counter
    
    @staticmethod
    def get_name_of_quad(stack):
        borders = stack.copy()
        borders.sort()
        borders = [str(b) for b in borders]
        return ".".join(borders)
    
    def step(self, stack):
        first_point = stack[0]
        for other_point in self.borders[stack[-1]].keys():
            points_list: List["PointForSubdivide"] = []
            for i in range(1, len(stack)):
                points_list.extend(self.borders[stack[i - 1]][stack[i]][2][1:-1])
            set_new = set(self.borders[stack[-1]][other_point][2][1:-1])
            if len(set_new.intersection(points_list)) == 0: # new and old don't use same curves
                if len(stack) < 4 and other_point not in stack:
                    stack.append(other_point)
                    if self.step(stack) == False:
                        return False
                    stack.pop()
                elif len(stack) == 4 and other_point == first_point:
                    if not GlobalForSubdivide.verify_correct_quad(stack, self.borders):
                        return False
                    name = GlobalForSubdivide.get_name_of_quad(stack)
                    if name not in self.quads:
                        self.quads[name] = stack.copy()
        return

    @staticmethod
    def verify_correct_quad(stack, borders):
        sides = [borders[stack[i]][stack[(i+1)%4]] for i in range(4)]
        if sides[0][0] != sides[2][0]:
            return False
        if sides[1][0] != sides[3][0]:
            return False
        return True

    @staticmethod
    def make_key(i1: int, i2: int):
        i_s = [i1, i2]
        i_s.sort()
        return f"{i_s[0]}_{i_s[1]}"
    
    @staticmethod
    def get_next_point(point: "PointForSubdivide", curve: "CurveForSubdivide"):
        return [p for p in curve.points if p != point][0]
    
    @staticmethod
    def mirror_with_sequence(vec, mirror_sequence):
        for item in mirror_sequence:
            vec = mirror_vec(vec, item.mirror_object, item.axis)
        return vec
    
    @staticmethod
    def get_handle_other_and_outer(point, curve_prev, curve_post):
        prev_other_i = None
        post_other_i = None
        bridge_mirror_vec_prev = None
        bridge_mirror_vec_post = None
        not_bridge = True
        if curve_prev.original_curve.greg_curve_settings.is_mirror_bridge:
            prev_other_i = curve_prev.original_curve.greg_curve_settings.bridge_mirror_other_i
        if curve_post.original_curve.greg_curve_settings.is_mirror_bridge:
            post_other_i = curve_post.original_curve.greg_curve_settings.bridge_mirror_other_i
        if (prev_other_i == 0 and curve_prev.points[0] == point) or\
           (prev_other_i == 1 and curve_prev.points[1] == point):
            not_bridge = False
            original_empty = curve_prev.points[(prev_other_i + 1) % 2].empty
            bridge_mirror_vec_prev = curve_prev.points[1].co - curve_prev.points[0].co
            bridge_mirror_pos_prev = (curve_prev.points[1].co + curve_prev.points[0].co) / 2
        if (post_other_i == 0 and curve_post.points[0] == point) or\
           (post_other_i == 1 and curve_post.points[1] == point):
            not_bridge = False
            original_empty = curve_post.points[(post_other_i + 1) % 2].empty
            bridge_mirror_vec_post = curve_post.points[1].co - curve_post.points[0].co
            bridge_mirror_pos_post = (curve_post.points[1].co + curve_post.points[0].co) / 2
        if not_bridge:
            original_empty = point.empty
        curves = [(end.basic_end.curve, end.basic_end.end)\
                  for end in original_empty.greg_empty_settings.curve_ends\
                  if end.basic_end.curve not in (curve_prev.original_curve, curve_post.original_curve)]
        if not curves:
            return [None, None]
        points = [(curve.data.splines[0].bezier_points[i], i) for curve, i in curves]
        handles = [point.handle_left if i == 0 else point.handle_right for point, i in points]
        alternative_handles = [point.handle_left if i == 1 else point.handle_right for point, i in points]
        res = []
        for hh in handles, alternative_handles:
            mirrored_handles_prev = [GlobalForSubdivide.mirror_with_sequence(handle, curve_prev.mirror_sequence) for handle in hh]
            if bridge_mirror_vec_prev is not None:
                mirrored_handles_prev = [mirror_vec_with_vec(h, bridge_mirror_vec_prev, bridge_mirror_pos_prev)
                                        for h in mirrored_handles_prev]
            mirrored_handles_post = [GlobalForSubdivide.mirror_with_sequence(handle, curve_post.mirror_sequence) for handle in hh]
            if bridge_mirror_vec_post is not None:
                mirrored_handles_post = [mirror_vec_with_vec(h, bridge_mirror_vec_post, bridge_mirror_pos_post)
                                        for h in mirrored_handles_post]
            mirrored_handles = mirrored_handles_post.copy()
            for handle1 in mirrored_handles_prev:
                add = True
                for handle2 in mirrored_handles_post:
                    if same_coords(handle1, handle2):
                        add = False
                        break
                if add:
                    mirrored_handles.append(handle1)
            total = mirrored_handles[0]
            for el in mirrored_handles[1:]:
                total += el
            mean_handle = total / len(mirrored_handles)
            res.append(mean_handle - point.co)
        return res

    @staticmethod
    def get_handles_left_right(point, curve_prev, curve_post):
        if curve_prev.points[0] == point:
            handle_left_point_init = curve_prev.original_curve.data.splines[0].bezier_points[0].handle_right
        elif curve_prev.points[1] == point:
            handle_left_point_init = curve_prev.original_curve.data.splines[0].bezier_points[1].handle_left
        handle_left_point = GlobalForSubdivide.mirror_with_sequence(handle_left_point_init, curve_prev.mirror_sequence)
        handle_left = handle_left_point - point.co
        if curve_post.points[0] == point:
            handle_right_point_init = curve_post.original_curve.data.splines[0].bezier_points[0].handle_right
        elif curve_post.points[1] == point:
            handle_right_point_init = curve_post.original_curve.data.splines[0].bezier_points[1].handle_left
        handle_right_point = GlobalForSubdivide.mirror_with_sequence(handle_right_point_init, curve_post.mirror_sequence)
        handle_right = handle_right_point - point.co
        return handle_left, handle_right

    @staticmethod
    def get_outer_handles_left_right(point, curve_prev, curve_post):
        if curve_prev.points[0] == point:
            handle_right_point_init = curve_prev.original_curve.data.splines[0].bezier_points[0].handle_left
        elif curve_prev.points[1] == point:
            handle_right_point_init = curve_prev.original_curve.data.splines[0].bezier_points[1].handle_right
        handle_right_point = GlobalForSubdivide.mirror_with_sequence(handle_right_point_init, curve_prev.mirror_sequence)
        handle_right = handle_right_point - point.co
        if curve_post.points[0] == point:
            handle_left_point_init = curve_post.original_curve.data.splines[0].bezier_points[0].handle_left
        elif curve_post.points[1] == point:
            handle_left_point_init = curve_post.original_curve.data.splines[0].bezier_points[1].handle_right
        handle_left_point = GlobalForSubdivide.mirror_with_sequence(handle_left_point_init, curve_post.mirror_sequence)
        handle_left = handle_left_point - point.co
        return handle_left, handle_right
    
    def fill_handles(self):
        for i in range(4):
            p0_number = self.optimal_quad[i]
            p1_number = self.optimal_quad[(i+1)%4]
            if i in (2, 3):
                p0_number, p1_number = p1_number, p0_number
            curves_len, curves, _ = self.borders[p0_number][p1_number]
            for j in range(curves_len - 1):
                point = self.edge_points[i][j+1]
                curve_prev = curves[j]
                curve_post = curves[j+1]
                point.handle_left, point.handle_right = GlobalForSubdivide.get_handles_left_right(point, curve_prev, curve_post)
                point.handle_other, point.handle_outer = GlobalForSubdivide.get_handle_other_and_outer(point, curve_prev, curve_post)

    @staticmethod
    def populate_handles(handles):
        i = 0
        count = 1
        while i < (len(handles) - 1):
            p1 = handles[i]
            count = 1
            p2 = handles[i + count]
            while p2 is None:
                count += 1
                p2 = handles[i + count]
            for j in range(i + 1, i + count):
                factor = j / count
                handles[j] = p1.lerp(p2, factor)
            i += count
        return handles
    
    def extract_coords(self, i):
        return [p.co for p in self.edge_points[i]]
    
    def extract_xtot_ytot(self):
        self.xtot = self.borders[self.optimal_quad[0]][self.optimal_quad[1]][0] + 1
        self.ytot = self.borders[self.optimal_quad[1]][self.optimal_quad[2]][0] + 1
    
    def extract_corner_handles(self):
        p30, p01, p12, p23 = [self.points[q] for q in self.optimal_quad]
        handle_left_1, handle_right_0 = GlobalForSubdivide.get_handles_left_right(p01, self.border0[-1], self.border1[0])
        handle_right_2, handle_right_1 = GlobalForSubdivide.get_handles_left_right(p12, self.border1[-1], self.border2[-1])
        handle_left_2, handle_right_3 = GlobalForSubdivide.get_handles_left_right(p23, self.border3[-1], self.border2[0])
        handle_left_0, handle_left_3 = GlobalForSubdivide.get_handles_left_right(p30, self.border3[0], self.border0[0])
        outer_handle_right_0, outer_handle_left_1 = GlobalForSubdivide.get_outer_handles_left_right(p01, self.border0[-1], self.border1[0])
        outer_handle_right_1, outer_handle_right_2 = GlobalForSubdivide.get_outer_handles_left_right(p12, self.border1[-1], self.border2[-1])
        outer_handle_right_3, outer_handle_left_2 = GlobalForSubdivide.get_outer_handles_left_right(p23, self.border3[-1], self.border2[0])
        outer_handle_left_3, outer_handle_left_0 = GlobalForSubdivide.get_outer_handles_left_right(p30, self.border3[0], self.border0[0])
        self.corner_handles.append((handle_left_0, handle_right_0))
        self.corner_handles.append((handle_left_1, handle_right_1))
        self.corner_handles.append((handle_left_2, handle_right_2))
        self.corner_handles.append((handle_left_3, handle_right_3))
        self.outer_corner_handles.append((outer_handle_left_0, outer_handle_right_0))
        self.outer_corner_handles.append((outer_handle_left_1, outer_handle_right_1))
        self.outer_corner_handles.append((outer_handle_left_2, outer_handle_right_2))
        self.outer_corner_handles.append((outer_handle_left_3, outer_handle_right_3))
    
    def extract_handles(self, i: int, outer=False):
        handles_len = self.xtot if i in (0, 2) else self.ytot
        handles = [None] * handles_len
        if outer:
            handles[0] = self.outer_corner_handles[i][0]
            handles[-1] = self.outer_corner_handles[i][1]
        else:
            handles[0] = self.corner_handles[i][0]
            handles[-1] = self.corner_handles[i][1]
        for j in range(1, handles_len - 1):
            point = self.edge_points[i][j]
            if outer:
                handles[j] = point.handle_outer
            else:
                handles[j] = point.handle_other
        return handles
    
    def get_handles_right_left(self, i, right):
        if right:
            if i == 0:
                first = self.corner_handles[3][0]
                last = -self.corner_handles[1][0]
            elif i == 1:
                first = self.corner_handles[0][1]
                last = -self.corner_handles[2][1]
            elif i == 2:
                first = self.corner_handles[3][1]
                last = -self.corner_handles[1][1]
            elif i == 3:
                first = self.corner_handles[0][0]
                last = -self.corner_handles[2][0]
        else:
            if i == 0:
                first = -self.corner_handles[3][0]
                last = self.corner_handles[1][0]
            elif i == 1:
                first = -self.corner_handles[0][1]
                last = self.corner_handles[2][1]
            elif i == 2:
                first = -self.corner_handles[3][1]
                last = self.corner_handles[1][1]
            elif i == 3:
                first = -self.corner_handles[0][0]
                last = self.corner_handles[2][0]
        middle = [p.handle_right if right else p.handle_left for p in self.edge_points[i][1:-1]]
        return [first] + middle + [last]
    
    def calc_init_coords(self,
                         i: int,
                         right: bool,
                         horisontal: bool,
                         h1: bool,
                         h: List[mathutils.Vector]):
        tot = self.ytot if horisontal else self.xtot
        second_coord = 0 if h1 else tot - 1
        first_coord1 = i - 1 if right else i + 1
        first_coord2 = i + 1 if right else i - 1
        coords1 = [first_coord1, second_coord]
        coords2 = [first_coord2, second_coord]
        coords3 = [i, second_coord]
        if not horisontal:
            for l in (coords1, coords2, coords3):
                l.reverse()
        for l in (coords1, coords2, coords3):
            l.append(self.xtot)
        matrix = calc_basis(self.v_grid[XY(*coords1)],
                            self.v_grid[XY(*coords2)],
                            self.v_grid[XY(*coords3)],
                            h[first_coord1],
                            h[i],
                            False)
        return get_coords_from_vec_and_basis_matrix(h[i], matrix)
    
    def handle_fin(self,
                   i: int,
                   j: int,
                   vh1: List[mathutils.Vector],
                   vh2: List[mathutils.Vector],
                   coords1: mathutils.Vector,
                   coords2: mathutils.Vector,
                   right: bool,
                   horisontal: bool):
        xtot = self.xtot if horisontal else self.ytot
        ytot = self.ytot if horisontal else self.xtot
        coord1 = i - 1 if right else i + 1
        coord2 = i + 1 if right else i - 1
        c1 = [coord1, j]
        c2 = [coord2, j]
        c3 = [i, j]
        h_side = vh1[j].lerp(-vh2[j], (i - 1)/(xtot - 1)) if right else -vh1[j].lerp(vh2[j], (i + 1)/(xtot - 1))
        if not horisontal:
            for c in (c1, c2, c3):
                c.reverse()
        for c in (c1, c2, c3):
            c.append(self.xtot)
        dest_m = calc_basis(self.v_grid[XY(*c1)],
                             self.v_grid[XY(*c2)],
                             self.v_grid[XY(*c3)],
                             h_side)
        coords_fin = coords1.lerp(coords2, j/(ytot - 1))
        return get_vec_from_coords_and_basis_matrix(coords_fin, dest_m)
    
    @staticmethod
    def verify_is_mirror(object_for_mirror, curve1, curve2):
        if curve1.original_curve == curve2.original_curve:
            co1 = [p.co for p in curve1.points]
            co2 = [p.co for p in curve2.points]
            for modifier in object_for_mirror.modifiers:
                if modifier.type == 'MIRROR':
                    obj = modifier.mirror_object
                    for axis in range(3):
                        if same_coords(mirror_vec(co1[0], obj, axis), co2[0]) and\
                        same_coords(mirror_vec(co1[1], obj, axis), co2[1]):
                            return obj, axis
        return None
    
    def symmetrize(self):
        if self.xtot > 2 and self.ytot > 2:
            object_for_mirror = self.border0[0].original_curve
            if self.xtot == self.ytot:
                self.diagonal1 = GlobalForSubdivide.verify_is_mirror(object_for_mirror, self.border0[0], self.border3[0])
                self.diagonal2 = GlobalForSubdivide.verify_is_mirror(object_for_mirror, self.border0[-1], self.border1[0])
            self.horizontal = GlobalForSubdivide.verify_is_mirror(object_for_mirror, self.border0[0], self.border2[0])
            self.vertical = GlobalForSubdivide.verify_is_mirror(object_for_mirror, self.border1[0], self.border3[0])
            if self.diagonal1:
                for x in range(1, self.xtot - 1):
                    point = self.points_grid[x][x]
                    GlobalForSubdivide.add_point_to_symmetrize(point, *self.diagonal1)
            if self.diagonal2:
                for x in range(1, self.xtot - 1):
                    y = self.xtot - x - 1
                    point = self.points_grid[y][x]
                    GlobalForSubdivide.add_point_to_symmetrize(point, *self.diagonal2)
            if self.horizontal and self.vertical and (not self.diagonal1) and (not self.diagonal2) and self.xtot % 2 == 0 and self.ytot % 2 == 0:
                x2 = self.xtot // 2
                x1 = x2 - 1
                y2 = self.ytot // 2
                y1 = y2 - 1
                p1 = self.points_grid[y1][x1]
                p2 = self.points_grid[y1][x2]
                p3 = self.points_grid[y2][x1]
                p4 = self.points_grid[y2][x2]
                p1.other_points_and_mirrors.add((p4, self.horizontal, self.vertical))
                p2.other_points_and_mirrors.add((p3, self.horizontal, self.vertical))
                p3.other_points_and_mirrors.add((p2, self.horizontal, self.vertical))
                p4.other_points_and_mirrors.add((p1, self.horizontal, self.vertical))
            if self.horizontal:
                if self.ytot % 2 == 1:
                    middle_y = self.ytot // 2
                    for x in range(1, self.xtot - 1):
                        point = self.points_grid[middle_y][x]
                        GlobalForSubdivide.add_point_to_symmetrize(point, *self.horizontal)
                else:
                    y2 = self.ytot // 2
                    y1 = y2 - 1
                    for x in range(1, self.xtot - 1):
                        point1 = self.points_grid[y1][x]
                        point2 = self.points_grid[y2][x]
                        point1.other_points_and_mirrors.add((point2, self.horizontal))
                        point2.other_points_and_mirrors.add((point1, self.horizontal))
            if self.vertical:
                if self.xtot % 2 == 1:
                    middle_x = self.xtot // 2
                    for y in range(1, self.ytot - 1):
                        point = self.points_grid[y][middle_x]
                        GlobalForSubdivide.add_point_to_symmetrize(point, *self.vertical)
                else:
                    x2 = self.xtot // 2
                    x1 = x2 - 1
                    for y in range(1, self.ytot - 1):
                        point1 = self.points_grid[y][x1]
                        point2 = self.points_grid[y][x2]
                        point1.other_points_and_mirrors.add((point2, self.vertical))
                        point2.other_points_and_mirrors.add((point1, self.vertical))
        for row in self.points_grid:
            for point in row:
                point.total_from_mirrored()
        for row in self.points_grid:
            for point in row:
                point.set_final_coords()

    @staticmethod
    def add_point_to_symmetrize(point: "MiddlePoint", obj, axis: int):
        new_co_s = []
        for co in point.mirrored_co_s:
            new_co_s.append(mirror_vec(co, obj, axis))
        for new_co in new_co_s:
            to_add = True
            for co in point.mirrored_co_s:
                if same_coords(new_co, co):
                    to_add = False
                    break
            if to_add:
                point.mirrored_co_s.append(new_co)
    
    def adjust_if_border_on_mirror(self, cross_hs: List[List[mathutils.Vector]]):
        print("len quads", len(self.quads))
        print("quads", self.quads)
        if len(self.quads) > 1:
            if self.xtot > 2 or self.ytot > 2:
                print("hore 1")
                obj_for_mirror = self.border0[0].original_curve # some curve. We suppose here that all curves have the same mirrors TODO maybe do better
                for modifier in obj_for_mirror.modifiers:
                    if modifier.type == 'MIRROR':
                        mirror_obj = modifier.mirror_object
                        if mirror_obj is not None:
                            mat = mirror_obj.matrix_world.copy()
                            mat.invert()
                            central = mirror_obj.matrix_world.translation
                        else:
                            central = mathutils.Vector((0, 0, 0))
                        for axis in range(3):
                            normal = mathutils.Vector((0, 0, 0))
                            normal[axis] = 1
                            if mirror_obj is not None:
                                normal = normal @ mat
                            for i in range(4):
                                for l in range(1, len(self.edge_points[i]) - 1):
                                    point = self.edge_points[i][l]
                                    print("hore 2")
                                    if len(point.triangles) > 1:
                                        print("hore 3")
                                        co = point.co
                                        triangles = point.triangles
                                        handle_left = point.handle_left
                                        handle_right = point.handle_right
                                        if are_collinear(handle_left, handle_right):
                                            print("hore 4")
                                            if same_coords(co, mirror_vec_with_vec(co, normal, central)):
                                                print("hore 5")
                                                if handle_left.dot(normal) < TH:
                                                    print("hore 6")
                                                    mirror_found = False
                                                    j = 0
                                                    while (not mirror_found) and j < (len(triangles) - 1):
                                                        triangle1 = triangles[j]
                                                        for triangle2 in triangles[j + 1:]:
                                                            print("triangle1")
                                                            for vec in triangle1:
                                                                print(vec)
                                                            print("triangle2")
                                                            for vec in triangle2:
                                                                print(vec)
                                                            print("****************************")
                                                            same = True
                                                            for k in range(3):
                                                                if not same_coords(triangle1[k], mirror_vec_with_vec(triangle2[k], normal, central)):
                                                                    same = False
                                                                    break
                                                            if same:
                                                                print("hore 7")
                                                                mirror_found = True
                                                                alternative = point.handle_left.cross(normal)
                                                                h = cross_hs[i][l]
                                                                new_h = (h - h.project(alternative)).normalized() * h.length
                                                                print("hore")
                                                                cross_hs[i][l] = new_h
                                                                break
                                                        j += 1
        return cross_hs
    
    def subdivide(self):
        self.extract_xtot_ytot()
        v1 = self.extract_coords(0)
        v2 = self.extract_coords(2)
        rv1 = self.extract_coords(3)
        rv2 = self.extract_coords(1)
        self.v_grid = grid_fill(v1, v2, rv1, rv2)
        for i in range(self.ytot):
            row = []
            for j in range(self.xtot):
                row.append(MiddlePoint(self.v_grid[XY(j, i, self.xtot)]))
            self.points_grid.append(row)
        self.symmetrize()
        self.extract_corner_handles()
        self.fill_handles()
        cross_hs = [GlobalForSubdivide.populate_handles(self.extract_handles(i)) for i in range(4)]
        cross_hs = self.adjust_if_border_on_mirror(cross_hs)
        outer_hs = [GlobalForSubdivide.populate_handles(self.extract_handles(i, True)) for i in range(4)]
        cross_h1 = cross_hs[0]
        cross_h2 = cross_hs[2]
        cross_vh1 = cross_hs[3]
        cross_vh2 = cross_hs[1]
        h_right_left = [
            (
                self.get_handles_right_left(i, True),
                self.get_handles_right_left(i, False)
            ) for i in range(4)]
        h1_right = h_right_left[0][0]
        h1_left = h_right_left[0][1]
        h2_right = h_right_left[2][0]
        h2_left = h_right_left[2][1]
        vh1_down = h_right_left[3][0]
        vh1_up = h_right_left[3][1]
        vh2_down = h_right_left[1][0]
        vh2_up = h_right_left[1][1]
        horisontal = True
        for i in range(1, self.xtot - 1):
            coords_right1 = self.calc_init_coords(i, True, horisontal, True, h1_right)
            coords_right2 = self.calc_init_coords(i, True, horisontal, False, h2_right)
            coords_left1 = self.calc_init_coords(i, False, horisontal, True, h1_left)
            coords_left2 = self.calc_init_coords(i, False, horisontal, False, h2_left)
            for j in range(1, self.ytot - 1):
                res_right = self.handle_fin(i, j, cross_vh1, cross_vh2, coords_right1, coords_right2, True, horisontal)
                res_left = self.handle_fin(i, j, cross_vh1, cross_vh2, coords_left1, coords_left2, False, horisontal)
                collinear_right, collinear_left = make_collinear(res_right, res_left)
                self.points_grid[j][i].handle_right = collinear_right
                self.points_grid[j][i].handle_left = collinear_left
        vertical = False
        for j in range(1, self.ytot - 1):
            coords_down1 = self.calc_init_coords(j, True, vertical, True, vh1_down)
            coords_down2 = self.calc_init_coords(j, True, vertical, False, vh2_down)
            coords_up1 = self.calc_init_coords(j, False, vertical, True, vh1_up)
            coords_up2 = self.calc_init_coords(j, False, vertical, False, vh2_up)
            for i in range(1, self.xtot - 1):
                res_down = self.handle_fin(j, i, cross_h1, cross_h2, coords_down1, coords_down2, True, vertical)
                res_up = self.handle_fin(j, i, cross_h1, cross_h2, coords_up1, coords_up2, False, vertical)
                collinear_down, collinear_up = make_collinear(res_down, res_up)
                self.points_grid[j][i].handle_down = collinear_down
                self.points_grid[j][i].handle_up = collinear_up
        for i in range(self.xtot):
            self.points_grid[0][i].handle_down = cross_h1[i]
            self.points_grid[0][i].handle_up = outer_hs[0][i]
            self.points_grid[self.ytot - 1][i].handle_down = outer_hs[2][i]
            self.points_grid[self.ytot - 1][i].handle_up = cross_h2[i]
        for i in range(self.ytot):
            self.points_grid[i][0].handle_right = cross_vh1[i]
            self.points_grid[i][0].handle_left = outer_hs[3][i]
            self.points_grid[i][self.xtot - 1].handle_right = outer_hs[1][i]
            self.points_grid[i][self.xtot - 1].handle_left = cross_vh2[i]
        self.adjust_midpoints_handles_mirrors()
    
    def adjust_midpoints_handles_mirrors(self):
        if self.horizontal:
            if self.ytot % 2 == 1:
                mirror_object, axis = self.horizontal
                normal = mathutils.Vector((0, 0, 0))
                normal[axis] = 1
                if mirror_object is not None:
                    mat = mirror_object.matrix_world.copy()
                    mat.invert()
                    normal = normal @ mat
                y_middle = self.ytot // 2
                point0 = self.points_grid[y_middle][0]
                point1 = self.points_grid[y_middle][-1]
                point0.handle_right -= point0.handle_right.project(normal)
                point0.handle_left -= point0.handle_left.project(normal)
                point1.handle_right -= point1.handle_right.project(normal)
                point1.handle_left -= point1.handle_left.project(normal)
                for x in range(1, self.xtot - 1):
                    point = self.points_grid[y_middle][x]
                    point.handle_left -= point.handle_left.project(normal)
                    point.handle_right -= point.handle_right.project(normal)
                    point.handle_up = point.handle_up.project(normal)
                    point.handle_down = point.handle_down.project(normal)
        if self.vertical:
            if self.xtot % 2 == 1:
                mirror_object, axis = self.vertical
                normal = mathutils.Vector((0, 0, 0))
                normal[axis] = 1
                if mirror_object is not None:
                    mat = mirror_object.matrix_world.copy()
                    mat.invert()
                    normal = normal @ mat
                x_middle = self.xtot // 2
                point0 = self.points_grid[0][x_middle]
                point1 = self.points_grid[-1][x_middle]
                point0.handle_down -= point0.handle_down.project(normal)
                point0.handle_up -= point0.handle_up.project(normal)
                point1.handle_down -= point1.handle_down.project(normal)
                point1.handle_up -= point1.handle_up.project(normal)
                for y in range(1, self.ytot - 1):
                    point = self.points_grid[y][x_middle]
                    point.handle_left = point.handle_left.project(normal)
                    point.handle_right = point.handle_right.project(normal)
                    point.handle_up -= point.handle_up.project(normal)
                    point.handle_down -= point.handle_down.project(normal)
        if self.diagonal1:
            for x in range(1, self.xtot - 1):
                point = self.points_grid[x][x]
                second = mirror_vec(-point.handle_up, *self.diagonal1)
                point.handle_left = point.handle_left.project(second)
                second = mirror_vec(-point.handle_down, *self.diagonal1)
                point.handle_right = point.handle_right.project(second)
        if self.diagonal2:
            for x in range(1, self.xtot - 1):
                point = self.points_grid[x][self.xtot - 1 - x]
                second = mirror_vec(-point.handle_up, *self.diagonal2)
                point.handle_right = point.handle_right.project(second)
                second = mirror_vec(-point.handle_down, *self.diagonal2)
                point.handle_left = point.handle_left.project(second)
  
    @staticmethod
    def verify_and_get_empty(curves, point):
        empty = None
        for curve in curves:
            if not curve.is_mirrored:
                if (not curve.original_curve.greg_curve_settings.is_mirror_bridge):
                    empty = point.empty
                else:
                    key = GlobalForSubdivide.make_key(curve.points[0].number, curve.points[1].number)
                    curve_i = point.curves[key][1]
                    if curve_i != curve.original_curve.greg_curve_settings.bridge_mirror_other_i:
                        empty = point.empty
        if empty is None:
            raise ValueError("something wrong with selection")
        return empty
    
    def is_real_side_part(self, side, part):
        borders = (self.border0, self.border1, self.border2, self.border3)
        index = -part
        return not borders[side][index].is_mirrored
    
    def generate_triangle(self, side, part): # only if xtot == ytot
        if side == 3:
            if part == 0:
                for y in range(1, self.ytot // 2):
                    for x in range(y):
                        yield ((x, y), (x + 1, y))
                for y in range(1, (self.ytot - 1) // 2):
                    for x in range(1, y + 1):
                        yield ((x, y), (x, y + 1))
            elif part == 1:
                for y in range((self.ytot + 1) // 2, self.ytot - 1):
                    for x in range(self.ytot - y - 1):
                        yield ((x, y), (x + 1, y))
                for y in range(self.ytot // 2, self.ytot - 2):
                    for x in range(1, self.ytot - y - 1):
                        yield ((x, y), (x, y + 1))
        elif side == 0:
            if part == 0:
                for x in range(1, self.xtot // 2):
                    for y in range(x):
                        yield ((x, y), (x, y + 1))
                for x in range(1, (self.xtot - 1) // 2):
                    for y in range(1, x + 1):
                        yield ((x, y), (x + 1, y))
            elif part == 1:
                for x in range((self.xtot + 1) // 2, self.xtot - 1):
                    for y in range(self.xtot - x - 1):
                        yield ((x, y), (x, y + 1))
                for x in range(self.xtot // 2, self.xtot - 2):
                    for y in range(1, self.xtot - x - 1):
                        yield ((x, y), (x + 1, y))
        elif side == 1:
            if part == 0:
                for y in range(1, self.ytot // 2):
                    for x in range(self.ytot - y - 1, self.ytot - 1):
                        yield ((x, y), (x + 1, y))
                for y in range(1, (self.ytot - 1) // 2):
                    for x in range(self.ytot - y - 1, self.ytot - 1):
                        yield ((x, y), (x, y + 1))
            elif part == 1:
                for y in range((self.ytot + 1) // 2, self.ytot - 1):
                    for x in range(self.ytot - y, self.ytot - 1):
                        yield ((x, y), (x + 1, y))
                for y in range(self.ytot // 2, self.ytot - 2):
                    for x in range(self.ytot - y + 1, self.ytot - 1):
                        yield ((x, y), (x, y + 1))
        elif side == 2:
            if part == 0:
                for x in range(1, self.xtot // 2):
                    for y in range(self.xtot - x - 1, self.xtot - 1):
                        yield ((x, y), (x, y + 1))
                for x in range(1, (self.xtot - 1) // 2):
                    for y in range(self.xtot - x - 1, self.xtot - 1):
                        yield ((x, y), (x + 1, y))
            elif part == 1:
                for x in range((self.xtot + 1) // 2, self.xtot - 1):
                    for y in range(self.xtot - x, self.xtot - 1):
                        yield ((x, y), (x, y + 1))
                for x in range(self.xtot // 2, self.xtot - 2):
                    for y in range(self.xtot - y + 1, self.xtot - 1):
                        yield ((x, y), (x + 1, y))

    
    def generate_strip(self, side):
        if side == 0:
            if self.xtot % 2 == 0:
                for y in range(1, self.ytot // 2):
                    yield ((self.xtot // 2 - 1, y), (self.xtot // 2, y))
            elif self.xtot % 2 == 1:
                for y in range((self.ytot - 1) // 2):
                    yield ((self.xtot // 2, y), (self.xtot // 2, y + 1))
        if side == 3:
            if self.ytot % 2 == 0:
                for x in range(1, self.xtot // 2):
                    yield ((x, self.ytot // 2 - 1), (x, self.ytot // 2))
            elif self.ytot % 2 == 1:
                for x in range((self.xtot - 1) // 2):
                    yield ((x, self.ytot // 2), (x + 1, self.ytot // 2))
        elif side == 1:
            if self.ytot % 2 == 0:
                for x in range((self.xtot + 1) // 2, self.xtot - 1):
                    yield((x, self.ytot // 2 - 1), (x, self.ytot // 2))
            elif self.ytot % 2 == 1:
                for x in range(self.xtot // 2, self.xtot - 1):
                    yield ((x, self.ytot // 2), (x + 1, self.ytot // 2))
        elif side == 2:
            if self.xtot % 2 == 0:
                for y in range((self.ytot + 1) // 2, self.ytot - 1):
                    yield((self.xtot // 2 - 1, y), (self.xtot // 2, y))
            elif self.xtot % 2 == 1:
                for y in range(self.ytot // 2, self.ytot - 1):
                    yield ((self.xtot // 2, y), (self.xtot // 2, y + 1))
    
    def generate_central(self):
        if self.xtot % 2 == 0 and self.ytot % 2 == 1:
            yield ((self.xtot // 2 - 1, self.ytot // 2), (self.xtot // 2, self.ytot // 2))
        elif self.xtot % 2 == 1 and self.ytot % 2 == 0:
            yield ((self.xtot // 2, self.ytot // 2 - 1), (self.xtot // 2, self.ytot // 2))
                
    def generate_rect(self, side, part): # if xtot != ytot 
        if (side == 0 and part == 0) or (side == 3 and part == 0):
            for x in range((self.xtot - 1) // 2):
                for y in range(1, self.ytot // 2):
                    yield ((x, y), (x + 1, y))
            for x in range(1, self.xtot // 2):
                for y in range((self.ytot - 1) // 2):
                    yield ((x, y), (x, y + 1))
        elif (side == 0 and part == 1) or (side == 1 and part == 0):
            for x in range(self.xtot // 2, self.xtot - 1):
                for y in range(1, self.ytot // 2):
                    yield ((x, y), (x + 1, y))
            for x in range((self.xtot + 1) // 2, self.xtot - 1):
                for y in range((self.ytot - 1) // 2):
                    yield ((x, y), (x, y + 1))
        elif (side == 2 and part == 0) or (side == 3 and part == 1):
            for y in range(self.ytot // 2, self.ytot - 1):
                for x in range(1, self.xtot // 2):
                    yield ((x, y), (x, y + 1))
            for y in range((self.ytot + 1) // 2, self.ytot - 1):
                for x in range((self.xtot - 1) // 2):
                    yield ((x, y), (x + 1, y))
        elif (side == 1 and part == 1) or (side == 2 and part == 1):
            for x in range(self.xtot // 2, self.xtot - 1):
                for y in range((self.ytot + 1) // 2, self.ytot - 1):
                    yield ((x, y), (x + 1, y))
            for y in range(self.ytot // 2, self.ytot - 1):
                for x in range((self.xtot + 1) // 2, self.xtot - 1):
                    yield ((x, y), (x, y + 1))

    def add_real_curve(self, xy1, xy2, empties_to_coplanar_collinear, collection, obj_for_mirror):
        x1, y1 = xy1
        x2, y2 = xy2
        is_border1 = False
        is_border2 = False
        empty1 = None
        empty2 = None
        middle_point1 = self.points_grid[y1][x1]
        middle_point2 = self.points_grid[y2][x2]
        if x1 == 0:
            curves_border = self.border3[y1 - 1], self.border3[y1]
            point = self.edge_points[3][y1]
            empty1 = GlobalForSubdivide.verify_and_get_empty(curves_border, point)
            is_border1 = True
        elif y1 == 0:
            curves_border = self.border0[x1 - 1], self.border0[x1]
            point = self.edge_points[0][x1]
            empty1 = GlobalForSubdivide.verify_and_get_empty(curves_border, point)
            is_border1 = True
        if x2 == self.xtot - 1:
            curves_border = self.border1[y1 - 1], self.border1[y1]
            point = self.edge_points[1][y1]
            empty2 = GlobalForSubdivide.verify_and_get_empty(curves_border, point)
            is_border2 = True
        elif y2 == self.ytot - 1:
            curves_border = self.border2[x1 - 1], self.border2[x1]
            point = self.edge_points[2][x1]
            empty2 = GlobalForSubdivide.verify_and_get_empty(curves_border, point)
            is_border2 = True
        if empty1 is None:
            empty1 = middle_point1.empty
        if empty2 is None:
            empty2 = middle_point2.empty
        if empty1 is None:
            empty1 = add_empty_obj(collection, middle_point1.co)
        if empty2 is None:
            empty2 = add_empty_obj(collection, middle_point2.co)
        middle_point1.empty = empty1
        middle_point2.empty = empty2
        co_s = (middle_point1.co, middle_point2.co)
        if x1 == x2:
            handles_left = (middle_point1.handle_up, middle_point2.handle_up)
            handles_right = (middle_point1.handle_down, middle_point2.handle_down)
        elif y1 == y2:
            handles_left = (middle_point1.handle_left, middle_point2.handle_left)
            handles_right = (middle_point1.handle_right, middle_point2.handle_right)
        curve_obj, _ = add_curve_obj(collection, co_s, handles_left, handles_right)
        i = 1
        for modifier in obj_for_mirror.modifiers:
            if modifier.type == 'MIRROR':
                new_mirror = curve_obj.modifiers.new(f"mirror_{i}", 'MIRROR')
                new_mirror.mirror_object = modifier.mirror_object
                new_mirror.use_axis = modifier.use_axis
                i += 1
        end1_name = add_curve_end(collection, empty1, curve_obj, 0)
        end2_name = add_curve_end(collection, empty2, curve_obj, 1)
        if is_border1:
            empties_to_coplanar_collinear.add((empty1, end1_name))
        else:
            empties_to_coplanar_collinear.add((empty1, None))
        if is_border2:
            empties_to_coplanar_collinear.add((empty2, end2_name))
        else:
            empties_to_coplanar_collinear.add((empty2, None))
        if x1 == (self.xtot // 2 - 1) and y2 == y1 and self.xtot % 2 == 0:
            res = check_curve_crosses_mirror(curve_obj, False)
            if res:
                i, axis = res
                mirror_obj = curve_obj.modifiers[i].mirror_object
                if (not self.border0[0].is_mirrored) or (not self.border2[0].is_mirrored):
                    target = empty1
                    other = empty2
                elif (not self.border0[-1].is_mirrored) or (not self.border2[-1].is_mirrored):
                    target = empty2
                    other = empty1
                _, __, target_point, other_point = make_curve_mirror_bridge(curve_obj, target, other, mirror_obj, axis)
                other_point.co = mirror_vec(target_point.co, mirror_obj, axis)
                other_point.handle_left = mirror_vec(target_point.handle_right, mirror_obj, axis)
                other_point.handle_right = mirror_vec(target_point.handle_left, mirror_obj, axis)
        elif y1 == (self.ytot // 2 - 1) and x2 == x1 and self.ytot % 2 == 0:
            res = check_curve_crosses_mirror(curve_obj, False)
            if res:
                i, axis = res
                mirror_obj = curve_obj.modifiers[i].mirror_object
                if (not self.border1[0].is_mirrored) or (not self.border3[0].is_mirrored):
                    target = empty1
                    other = empty2
                elif (not self.border1[-1].is_mirrored) or (not self.border3[-1].is_mirrored):
                    target = empty2
                    other = empty1
                _, __, target_point, other_point = make_curve_mirror_bridge(curve_obj, target, other, mirror_obj, axis)
                other_point.co = mirror_vec(target_point.co, mirror_obj, axis)
                other_point.handle_left = mirror_vec(target_point.handle_right, mirror_obj, axis)
                other_point.handle_right = mirror_vec(target_point.handle_left, mirror_obj, axis)
    
    
    def add_real_curves(self, collection, context):
        empties_to_coplanar_collinear = set()
        sequence_of_xys = [self.generate_central()]
        obj_for_mirror = self.border0[0].original_curve # some curve. We suppose here that all curves have the same mirrors TODO maybe do better
        if self.xtot == self.ytot:
            for side in range(4):
                for part in range(2):
                    if self.is_real_side_part(side, part):
                        sequence_of_xys.append(self.generate_triangle(side, part))
        else:
            for side in (0, 2):
                for part in range(2):
                    if self.is_real_side_part(side, part):
                        sequence_of_xys.append(self.generate_rect(side, part))
        for side in range(4):
            if self.is_real_side_part(side, 0) or\
            self.is_real_side_part(side, 1):
                sequence_of_xys.append(self.generate_strip(side))
        for xy1, xy2 in chain.from_iterable(sequence_of_xys):
            self.add_real_curve(xy1, xy2, empties_to_coplanar_collinear, collection, obj_for_mirror)
        for empty_obj, end_name in empties_to_coplanar_collinear:
            if end_name is not None:
                new_end = empty_obj.greg_empty_settings.curve_ends[end_name]
                coplanar_collinear_add_one_end(empty_obj, collection, new_end)
            else:
                coplanar_collinear(empty_obj, collection)
            for end in empty_obj.greg_empty_settings.curve_ends:
                add_hook(end, context)

    def adjust_free_outers(self):
        for x in range(1, self.xtot - 1):
            midpoint1 = self.points_grid[0][x]
            handle_up = midpoint1.handle_up
            handle_down = midpoint1.handle_down
            midpoint1.handle_up = -handle_down * handle_up.length / handle_down.length
            midpoint2 = self.points_grid[self.ytot - 1][x]
            handle_up = midpoint2.handle_up
            handle_down = midpoint2.handle_down
            midpoint2.handle_down = -handle_up * handle_down.length / handle_up.length
        for y in range(1, self.ytot - 1):
            midpoint1 = self.points_grid[y][0]
            handle_left = midpoint1.handle_left
            handle_right = midpoint1.handle_right
            midpoint1.handle_left = -handle_right * handle_left.length / handle_right.length
            midpoint2 = self.points_grid[y][self.xtot - 1]
            handle_left = midpoint2.handle_left
            handle_right = midpoint2.handle_right
            midpoint2.handle_right = -handle_left * handle_right.length / handle_left.length

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

class MiddlePoint:
    def __init__(self, co: mathutils.Vector):
        self.co = co
        self.handle_up: mathutils.Vector
        self.handle_down: mathutils.Vector
        self.handle_left: mathutils.Vector
        self.handle_right: mathutils.Vector
        self.empty = None
        self.mirrored_co_s = [co]
        self.other_points_and_mirrors = set()
        self.total = co.copy()
    
    def total_from_mirrored(self):
        if len(self.mirrored_co_s) > 1:
            self.total = self.mirrored_co_s[0].copy()
            for new_co in self.mirrored_co_s[1:]:
                self.total += new_co
            self.total /= len(self.mirrored_co_s)
    
    def set_final_coords(self):
        new_co_s = []
        for item in self.other_points_and_mirrors:
            if len(item) == 2:
                point, mirror = item
                new_co_s.append(mirror_vec(point.total, *mirror))
            elif len(item) == 3:
                point, mirror1, mirror2 = item
                new_co_s.append(mirror_vec(mirror_vec(point.total, *mirror1), *mirror2))
        new_total = self.total.copy()
        for new_co in new_co_s:
            new_total += new_co
        if new_co_s:
            new_total /= (len(new_co_s) + 1)
        self.co = new_total

class PointForSubdivide:
    def __init__(self, co: mathutils.Vector, empty: bpy.types.Object, number: int, triangles: List[List[mathutils.Vector]]):
        self.co: mathutils.Vector = co
        self.handle_right: Optional[mathutils.Vector] = None
        self.handle_left: Optional[mathutils.Vector] = None
        self.handle_other: Optional[mathutils.Vector] = None
        self.handle_outer: Optional[mathutils.Vector] = None
        self.empty: bpy.types.Object = empty
        self.curves: Dict[str, Tuple["CurveForSubdivide", int]] = {}
        self.number: int = number
        self.triangles: List[List[mathutils.Vector]] = triangles
    
    def add_triangles(self, triangles: List[mathutils.Vector]):
        for triangle in triangles:
            to_add = True
            for old_triangle in self.triangles:
                same = True
                for i in range(3):
                    if not same_coords(triangle[i], old_triangle[i]):
                        same = False
                        break
                if same:
                    to_add = False
                    break
            if to_add:
                self.triangles.append(triangle)

    def generate_new_triangles(self, mirror_obj: bpy.types.Object, axis: mathutils.Vector):
        return [[mirror_vec(vec, mirror_obj, axis) for vec in triangle] for triangle in self.triangles]

class CurveForSubdivide:
    def __init__(self, p1, p2, is_mirrored, original_curve, mirror_sequence: Optional[List["MirrorSequenceItem"]]=None):
        self.points: Tuple[PointForSubdivide, PointForSubdivide] = p1, p2
        self.is_mirrored: bool = is_mirrored
        self.original_curve: bpy.types.Object = original_curve
        if mirror_sequence is None:
            self.mirror_sequence: List["MirrorSequenceItem"] = []
        else:
            self.mirror_sequence = mirror_sequence

class MirrorSequenceItem:
    def __init__(self, mirror_object: bpy.types.Object, axis: int):
        self.mirror_object: bpy.types.Object = mirror_object
        self.axis: int = axis

class GregSubdivide(bpy.types.Operator):
    """Gregory: subdivide loop of curves"""
    bl_idname = "object.greg_sibdivide"
    bl_label = "Subdivide loop of curves"         # Display name in the interface.
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

        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        empties = []
        curves = []
        corners = []
        collection = get_greg_collection(context.selected_objects[0])
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
            raise ValueError("invalid corners or borders selected!")
        for empty_obj in empties:
            for end in empty_obj.greg_empty_settings.curve_ends:
                apply_hook(end)
                add_hook(end)
        g_list.subdivide()
        g_list.adjust_free_outers()
        g_list.add_real_curves(collection, context)
        return {'FINISHED'}    

def add_greg_subdivide_func(self, context: bpy.types.Context):
    self.layout.operator(GregSubdivide.bl_idname)


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
    handle_left = result_vec * mean_length / result_vec.length
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

class SetNotFace(bpy.types.Operator):
    """Gregory: set loop of curves not face"""
    bl_idname = "object.greg_set_not_face"
    bl_label = "greg: set loop not face"         # Display name in the interface.
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
        '''if not verify_is_closed_path_and_of_length_4(selected):
            return False''' # TODO
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        col = bpy.data.collections.new("NotFaceCollection")
        col.greg_is_not_face = True
        for obj in context.selected_objects:
            col.objects.link(obj)
        

        return {'FINISHED'}

def set_not_face_menu_func(self, context: bpy.types.Context):
    self.layout.operator(SetNotFace.bl_idname)

def get_not_face_ids(curve_obj: bpy.types.Object) -> List[Set[str]]:
    not_face_ids_groups = []
    for col in curve_obj.users_collection:
        if col.greg_is_not_face:
            not_face_ids = set([obj.greg_curve_settings.name for obj in col.objects])
            not_face_ids_groups.append(not_face_ids)
    return not_face_ids_groups

class CreateCurvesCollection(bpy.types.Operator):
    """Gregory: create curves"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_create_curves"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Create curves"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.

        glist = GlobalList()

        active = context.active_object
        mb = active.matrix_basis
        active.data.transform(mb)
        active.matrix_basis.identity()

        cur = active.data
        nedges = cur.resolution_u
        d.conditional_update(nedges)
        splines = cur.splines

        for s in splines:
            spline = Spline(glist)
            for p in s.bezier_points:
                spline.add_point(p.co, p.handle_left, p.handle_right)
            if s.use_cyclic_u:
                spline.round_spline()
        glist.add_many_curves(active.name, active.users_collection[0], context)

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def add_collection_menu_func(self, context: bpy.types.Context):
    self.layout.operator(CreateCurvesCollection.bl_idname)

class CreateSurfacesBetweenCurves(bpy.types.Operator):
    """Gregory: create surface"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_create_surfs"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Create surfaces"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        active = context.active_object
        collection = get_greg_collection(active)
        glist = NewGlobalList(collection)
        glist.prepare_for_greg()
        glist.add_curves_and_bpoints()
        glist.add_quads()
        glist.calculate_kk()
        glist.render_mesh(d, collection.name, context)

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def add_surface_menu_func(self, context: bpy.types.Context):
    self.layout.operator(CreateSurfacesBetweenCurves.bl_idname)

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

class PartialGlobalList:
    def __init__(self):
        self.bpoints: List["PartialBigPoint"] = []
        self.curves: List["PartialCurve"] = []
    
    def addBp(self, co: mathutils.Vector, collection_empties: List[bpy.types.Object]):
        for bp in self.bpoints:
            if same_coords(bp.co, co):
                return bp
        for e in collection_empties:
            if same_coords(e.matrix_world.translation, co):
                new_bp = PartialBigPoint(co, e)
                self.bpoints.append(new_bp)
                return new_bp
        new_bp = PartialBigPoint(co)
        self.bpoints.append(new_bp)
        return new_bp
    
    def add_curves_and_empties(self, collection):
        for bp in self.bpoints:
            if bp.Empty is None:
                empty = add_empty_obj(collection, bp.co)
                bp.new_empty = empty
        for curve in self.curves:
            curve_obj, _ = add_curve_obj(collection,
                          (curve.bp1.co, curve.bp2.co),
                          (curve.hl1, curve.hl2),
                          (curve.hr1, curve.hr2))
            for i, bp in enumerate((curve.bp1, curve.bp2)):
                if bp.Empty is None:
                    empty_obj = bp.new_empty
                else:
                    empty_obj = bp.Empty
                new_end_name = add_curve_end(collection, empty_obj, curve_obj, i)
                new_end = empty_obj.greg_empty_settings.curve_ends[new_end_name]
                coplanar_collinear_add_one_end(empty_obj, collection, new_end)
        for bp in self.bpoints:
            if bp.Empty is None:
                coplanar_collinear(bp.new_empty, collection)
                empty_obj = bp.new_empty
            else:
                empty_obj = bp.Empty
            for end in empty_obj.greg_empty_settings.curve_ends:
                add_hook(end)

class PartialCurve:
    def __init__(self,
                 bezier_point1: bpy.types.BezierSplinePoint,
                 bezier_point2: bpy.types.BezierSplinePoint,
                 collection_empties: List[bpy.types.Object],
                 p_glist: PartialGlobalList,
                 prev_ready=False):
        if prev_ready:
            self.bp1 = p_glist.curves[-1].bp2
        else:
            self.bp1 = p_glist.addBp(bezier_point1.co, collection_empties)
        self.bp2 = p_glist.addBp(bezier_point2.co, collection_empties)
        self.hr1 = bezier_point1.handle_right - bezier_point1.co
        self.hl1 = bezier_point1.handle_left - bezier_point1.co
        self.hr2 = bezier_point2.handle_right - bezier_point2.co
        self.hl2 = bezier_point2.handle_left - bezier_point2.co
        self.bp1.add_curve(self, 0)
        self.bp2.add_curve(self, 1)
        p_glist.curves.append(self)
        
        
        

class PartialBigPoint:
    def __init__(self, co: mathutils.Vector, empty: Optional[bpy.types.Object]=None):
        self.co = co
        self.Empty = empty
        self.curves: List[Tuple[PartialCurve, int]] = []
        self.new_empty: Optional[bpy.types.Object] = None
    
    def add_curve(self, curve: PartialCurve, i: int):
        self.curves.append((curve, i))

class AddBezierCurve(bpy.types.Operator):
    """Gregory: add curve to structure"""     # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_add_curve_to_structure"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: add curve"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    @classmethod
    def poll(cls, context: bpy.types.Context):
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

def set_collinear_menu_func(self, context: bpy.types.Context):
    self.layout.operator(SetCollinear.bl_idname)

class SetNotCollinear(bpy.types.Operator):
    """Gregory: set not collinear"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.greg_set_not_collinear"        # Unique identifier for bu: bpy.types.Contextttons and menu items to reference.
    bl_label = "Greg: set not collinear"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
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

def extract_end_from_name_and_empty(name: str, empty: bpy.types.Object):
    return empty.greg_empty_settings.curve_ends[name]

def extract_end_from_basic_end_and_empty(basic_end: GregBasicEnd, empty: bpy.types.Object) -> str:
    return extract_end_from_name_and_empty(basic_end.name, empty)

def extract_end_name_from_curve_and_empty(curve: bpy.types.Object, empty: bpy.types.Object) -> Optional[str]:
    if empty == curve.greg_curve_settings.end1_empty:
        end_name = curve.greg_curve_settings.end1_name
    elif empty == curve.greg_curve_settings.end2_empty:
        end_name = curve.greg_curve_settings.end2_name
    else:
        return None
    return end_name

def extract_end_name_from_curve_and_i(curve: bpy.types.Object, i: int):
    if i == 0:
        end_name = curve.greg_curve_settings.end1_name
    elif i == 1:
        end_name = curve.greg_curve_settings.end2_name
    else:
        raise Exception("I must be 0 or 1!")
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

def register():
    bpy.utils.register_class(GregId)
    bpy.utils.register_class(GregArrowItem)
    bpy.utils.register_class(GregBasicEnd)
    bpy.utils.register_class(GregArrow)
    bpy.utils.register_class(GregCurveEndItem)
    bpy.utils.register_class(GregEmptyItem)
    bpy.utils.register_class(GregCurveItem)
    bpy.utils.register_class(GregQuad)
    bpy.utils.register_class(GregPhantomCurveEnd)
    bpy.utils.register_class(GregPhantomCurve)
    bpy.utils.register_class(GregPhantomBpoint)
    bpy.utils.register_class(GregCollectionSettings)
    bpy.utils.register_class(GregEmpty)
    bpy.utils.register_class(GregCurve)

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

    bpy.utils.register_class(CreateCurvesCollection)
    bpy.types.VIEW3D_MT_object.append(add_collection_menu_func)
    bpy.utils.register_class(CreateSurfacesBetweenCurves)
    bpy.types.VIEW3D_MT_object.append(add_surface_menu_func)  # Adds the new operator to an existing menu.
    bpy.utils.register_class(SetNotFace)
    bpy.types.VIEW3D_MT_object_context_menu.append(set_not_face_menu_func)  # Adds the new operator to an existing menu.
    bpy.utils.register_class(PrintItemInfo)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_print_info_func)
    bpy.utils.register_class(MakeCurveMirrorBridge)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_bridge_mirror_func)
    bpy.utils.register_class(UnsetCurveMirrorBridge)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_unset_bridge_mirror_func)
    bpy.utils.register_class(PrintDotInfo)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_print_dot_func)
    bpy.utils.register_class(OBJECT_PT_greg_curve_properties)
    bpy.utils.register_class(OBJECT_PT_greg_curve_properties1)
    bpy.utils.register_class(OBJECT_PT_greg_curve_properties2)
    bpy.utils.register_class(GregSubdivide)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_greg_subdivide_func)
    bpy.utils.register_class(GregExtrude)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_greg_extrude_func)
    bpy.utils.register_class(GregMergeAtCenter)
    bpy.utils.register_class(GregMergeAtFirst)
    bpy.utils.register_class(GregMergeAtLast)
    bpy.utils.register_class(GregMergeSub)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_greg_merge_func)
    bpy.utils.register_class(AddBezierCurve)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_bezier_curve_menu_func)
    bpy.utils.register_class(SetCoplanar)
    bpy.types.VIEW3D_MT_object_context_menu.append(set_coplanar_menu_func)
    bpy.utils.register_class(SetNotCoplanar)
    bpy.types.VIEW3D_MT_object_context_menu.append(set_not_coplanar_menu_func)
    bpy.utils.register_class(SetCollinear)
    bpy.types.VIEW3D_MT_object_context_menu.append(set_collinear_menu_func)
    bpy.utils.register_class(SetNotCollinear)
    bpy.types.VIEW3D_MT_object_context_menu.append(set_not_collinear_menu_func)

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
    
    bpy.utils.unregister_class(SetCollinear)
    bpy.utils.unregister_class(SetNotCollinear)
    bpy.utils.unregister_class(SetCoplanar)
    bpy.utils.unregister_class(SetNotCoplanar)
    bpy.utils.unregister_class(AddBezierCurve)
    bpy.utils.unregister_class(GregMergeSub)
    bpy.utils.unregister_class(GregMergeAtLast)
    bpy.utils.unregister_class(GregMergeAtFirst)
    bpy.utils.unregister_class(GregMergeAtCenter)
    bpy.utils.unregister_class(GregExtrude)
    bpy.utils.unregister_class(GregSubdivide)
    bpy.utils.unregister_class(OBJECT_PT_greg_curve_properties2)
    bpy.utils.unregister_class(OBJECT_PT_greg_curve_properties1)
    bpy.utils.unregister_class(OBJECT_PT_greg_curve_properties)
    bpy.utils.unregister_class(PrintDotInfo)
    bpy.utils.unregister_class(UnsetCurveMirrorBridge)
    bpy.utils.unregister_class(MakeCurveMirrorBridge)
    bpy.utils.unregister_class(PrintItemInfo)
    bpy.utils.unregister_class(SetNotFace)
    bpy.utils.unregister_class(CreateSurfacesBetweenCurves)

    bpy.utils.unregister_class(GregCurve)
    bpy.utils.unregister_class(GregEmpty)
    bpy.utils.unregister_class(GregCollectionSettings)
    bpy.utils.unregister_class(GregPhantomBpoint)
    bpy.utils.unregister_class(GregPhantomCurve)
    bpy.utils.unregister_class(GregPhantomCurveEnd)
    bpy.utils.unregister_class(GregQuad)
    bpy.utils.unregister_class(GregCurveItem)
    bpy.utils.unregister_class(GregEmptyItem)
    bpy.utils.unregister_class(GregCurveEndItem)
    bpy.utils.unregister_class(GregArrow)
    bpy.utils.unregister_class(GregBasicEnd)
    bpy.utils.unregister_class(GregArrowItem)
    bpy.utils.unregister_class(GregId)


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()