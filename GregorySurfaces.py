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
import itertools
import bpy
import bmesh
import mathutils
from typing import List, Optional, Tuple, Callable
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
                                if end_empty.greg_empty_settings.name == "96":
                                    print("repare empty 96")
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
        setting.is_collinear = True
        c1 = setting.collinear_to.add()
        c1.name = collinear_end.name
        c1.curve = collinear_end.basic_end.curve
        c1.end = collinear_end.basic_end.end
        collinear_end.is_collinear = True
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
                end.is_coplanar = True


    



def preserve_points(curve_obj):
    # edit mode
    spline = curve_obj.data.splines[0]
    settings = curve_obj.greg_curve_settings
    for i, empty in enumerate((settings.end1_empty, settings.end2_empty)):
        if (spline.bezier_points[i].co - empty.location).length_squared > TH2:
            spline.bezier_points[i].co = empty.location

def rotate_end_to_vec(vec, basic_end):
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

def make_coplanar(v1: mathutils.Vector, v2: mathutils.Vector, v3: mathutils.Vector):
    v4 = v2 - v3
    normal = v1.cross(v4)
    new_v2 = v2 - v2.project(normal)
    new_v3 = v3 - v3.project(normal)
    return (new_v2, new_v3)

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
        self.quads: List[Quad | BigQuad] = []
        self.p1 = p1
        self.p2 = p2
        self.p1.post_seg = self
        self.p2.prev_seg = self
        self.finished = False
        self.verts: Optional[List[int]] = None
        glist.add_segment(self)
        self.b1: Optional[mathutils.Vector] = None

    def add_quad(self, quad: "Quad | BigQuad"):
        self.quads.append(quad)
        if len(self.quads) == 2:
            self.finished = True

    def calculate_midpoint(self):
        p0 = self.p1.bpoint.coords
        p1 = p0 + self.p1.handle_right
        p3 = self.p2.bpoint.coords
        p2 = p3 + self.p2.handle_left
        return 1/8 * (p0 + 3*p1 + 3*p2 + p3)
    
    def extract_control_points(self, glist: "GlobalList"):
        k0 = self.p1.bpoint.coords
        k3 = self.p2.bpoint.coords
        k1 = self.p1.handle_right + k0
        k2 = self.p2.handle_left + k3
        return k0, k1, k2, k3

    def calculate_verts(self, glist: "GlobalList", d: "DependantsOfResolution|DependantsOfResolution_np"):
        self.verts = add_border(*self.extract_control_points(glist), glist, d)
        

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

    def calculate_b1(self,
                     k0: float,
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
        self.b1 = (res + b1_ref) * 0.5
        #self.b1 = res
        return self.b1


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


class MiddlePoint:
    def __init__(self, co: mathutils.Vector):
        self.co = co
        self.handle_up: mathutils.Vector
        self.handle_down: mathutils.Vector
        self.handle_left: mathutils.Vector
        self.handle_right: mathutils.Vector


class BigQuad:
    def __init__(self,
                 edg1: List[Segment],
                 edg2: List[Segment],
                 edg3: List[Segment],
                 edg4: List[Segment],
                 dir1: bool,
                 dir2: bool,
                 dir3: bool,
                 dir4: bool,
                 glist: "GlobalList"):
        self.v_grid: List[mathutils.Vector]
        self.xtot = len(edg1) + 1
        self.ytot = len(edg2) + 1
        self.edges = [edg1, edg2, edg3, edg4]
        self.dirs = [dir1, dir2, dir3, dir4]
        self.points_grid: List[List[MiddlePoint]] = []
        self.corner_handles: List[Tuple[mathutils.Vector, mathutils.Vector]] = []
        for edg in self.edges:
            for segment in edg:
                segment.add_quad(self)
        glist.add_big_quad(self)

    @staticmethod
    def extract_first_handle(edge: List[Segment]):
        return edge[0].p1.handle_right

    @staticmethod
    def extract_last_handle(edge: List[Segment]):
        return edge[-1].p2.handle_left

    def extract_corner_handles(self):
        # edges[0]
        if self.dirs[1]:
            handle_right_0 = BigQuad.extract_first_handle(self.edges[1])
        else:
            handle_right_0 = BigQuad.extract_last_handle(self.edges[1])
        if self.dirs[3]:
            handle_left_0 = BigQuad.extract_first_handle(self.edges[3])
        else:
            handle_left_0 = BigQuad.extract_last_handle(self.edges[3])
        self.corner_handles.append((handle_left_0, handle_right_0))

        # edges[1]
        handle_left_1 = BigQuad.extract_last_handle(self.edges[0])
        if self.dirs[2]:
            handle_right_1 = BigQuad.extract_last_handle(self.edges[2])
        else:
            handle_right_1 = BigQuad.extract_first_handle(self.edges[2])
        if not self.dirs[1]:
            handle_left_1, handle_right_1 = handle_right_1, handle_left_1
        self.corner_handles.append((handle_left_1, handle_right_1))

        # edges[2]
        if self.dirs[1]:
            handle_right_2 = BigQuad.extract_last_handle(self.edges[1])
        else:
            handle_right_2 = BigQuad.extract_first_handle(self.edges[1])
        if self.dirs[3]:
            handle_left_2 = BigQuad.extract_last_handle(self.edges[3])
        else:
            handle_left_2 = BigQuad.extract_first_handle(self.edges[3])
        if not self.dirs[2]:
            handle_left_2, handle_right_2 = handle_right_2, handle_left_2
        self.corner_handles.append((handle_left_2, handle_right_2))

        # edges[3]
        handle_left_3 = BigQuad.extract_first_handle(self.edges[0])
        if self.dirs[2]:
            handle_right_3 = BigQuad.extract_first_handle(self.edges[2])
        else:
            handle_right_3 = BigQuad.extract_last_handle(self.edges[2])
        if not self.dirs[3]:
            handle_left_3, handle_right_3 = handle_right_3, handle_left_3
        self.corner_handles.append((handle_left_3, handle_right_3))

    @staticmethod
    def extract_coords(edg: List[Segment], dir: bool):
        if dir:
            res = [p.bpoint.coords for p in Segment.iterate_direct(edg)]
        else:
            res = [p.bpoint.coords for p in Segment.iterate_reversed(edg)]
        return res

    def extract_handles(self, edg: List[Segment], i: int):
        handles: List[Optional[mathutils.Vector]] = [None] * (len(edg) + 1)
        handles[0] = self.corner_handles[i][0]
        handles[-1] = self.corner_handles[i][1]
        for j, p in enumerate(Segment.iterate_direct(edg)):
            if j not in (0, len(edg)):
                if len(p.bpoint.points) != 1:
                    point_handles: List[mathutils.Vector] = []
                    for pp in p.bpoint.points:
                        if pp != p:
                            if pp.post_seg and not pp.prev_seg:
                                point_handles.append(pp.handle_left)
                            elif pp.prev_seg and not pp.post_seg:
                                point_handles.append(pp.handle_right)
                    if len(point_handles) != 0:
                        if len(point_handles) == 1:
                            handles[j] = point_handles[0]
                        else:
                            total = point_handles[0]
                            for h in point_handles[1:]:
                                total += h
                            handles[j] = total/len(point_handles)
        return handles

    @staticmethod
    def populate_handles(handles: List[Optional[mathutils.Vector]]) -> List[mathutils.Vector]:
        i = 0
        count = 1
        while i < (len(handles) - 1):
            p1 = handles[i]
            assert isinstance(p1, mathutils.Vector)
            count = 1
            p2 = handles[i + count]
            while p2 is None:
                count += 1
                p2 = handles[i + count]
            for j in range(i + 1, i + count):
                factor = j / count
                handles[j] = p1.lerp(p2, factor)
            i += count
        res_handles = [h for h in handles if h is not None]
        assert len(res_handles) == len(handles)
        return res_handles

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
    def add_point_to_curve(point: bpy.types.BezierSplinePoint,
                           co: mathutils.Vector,
                           handle_left: mathutils.Vector,
                           handle_right: mathutils.Vector):
        point.co = co
        point.handle_left = handle_left + co
        point.handle_right = handle_right + co
        point.handle_left_type = "ALIGNED"
        point.handle_right_type = "ALIGNED"

    def add_new_quads(self, glist: "GlobalList"):
        glist.big_quads.remove(self)
        for side in self.edges:
            for edg in side:
                edg.quads.remove(self)
        horisontal_splines: List[Spline] = []
        vertical_splines: List[Spline] = []
        for i in range(1, self.xtot - 1):
            spline = Spline(glist)
            vertical_splines.append(spline)
            for j in range(self.ytot):
                p = self.points_grid[j][i]
                spline.add_point(p.co, p.handle_up + p.co, p.handle_down + p.co)
        for j in range(1, self.ytot - 1):
            spline = Spline(glist)
            horisontal_splines.append(spline)
            for i in range(self.xtot):
                p = self.points_grid[j][i]
                spline.add_point(p.co, p.handle_left + p.co, p.handle_right + p.co)
        for i in range(self.xtot - 1):
            for j in range(self.ytot - 1):
                seg1 = horisontal_splines[j - 1].segments[i] if j > 0 else self.edges[0][i]
                dir1 = True
                if j < (self.ytot - 2):
                    seg3 = horisontal_splines[j].segments[i]
                    dir3 = True
                else:
                    dir3 = self.dirs[2]
                    if dir3:
                        seg3 = self.edges[2][i]
                    else:
                        seg3 = self.edges[2][-i - 1]
                if i > 0:
                    dir4 = True
                    seg4 = vertical_splines[i - 1].segments[j]
                else:
                    dir4 = self.dirs[3]
                    if dir4:
                        seg4 = self.edges[3][j]
                    else:
                        seg4 = self.edges[3][-j - 1]
                if i < (self.xtot - 2):
                    seg2 = vertical_splines[i].segments[j]
                    dir2 = True
                else:
                    dir2 = self.dirs[1]
                    if dir2:
                        seg2 = self.edges[1][j]
                    else:
                        seg2 = self.edges[1][-j - 1]
                Quad(seg1, seg2, seg3, seg4, dir1, dir2, dir3, dir4, glist)

    def subdivide(self, obj: bpy.types.ID, glist: "GlobalList"):
        v1 = BigQuad.extract_coords(self.edges[0], True)
        v2 = BigQuad.extract_coords(self.edges[2], self.dirs[2])
        rv1 = BigQuad.extract_coords(self.edges[3], self.dirs[3])
        rv2 = BigQuad.extract_coords(self.edges[1], self.dirs[1])
        self.v_grid = grid_fill(v1, v2, rv1, rv2)
        for i in range(self.ytot):
            row: List[MiddlePoint] = []
            for j in range(self.xtot):
                row.append(MiddlePoint(self.v_grid[XY(j, i, self.xtot)]))
            self.points_grid.append(row)
        self.extract_corner_handles()
        cross_hs = [BigQuad.populate_handles(self.extract_handles(edg, i)) for i, edg in enumerate(self.edges)]
        for i in range(4):
            if not self.dirs[i]:
                cross_hs[i].reverse()
        cross_h1 = cross_hs[0]
        cross_h2 = cross_hs[2]
        cross_vh1 = cross_hs[3]
        cross_vh2 = cross_hs[1]
        h_right_left = [
            (
                [p.handle_right for p in Segment.iterate_direct(edge)] if d else 
                [p.handle_left for p in Segment.iterate_reversed(edge)],
                [p.handle_left for p in Segment.iterate_direct(edge)] if d else
                [p.handle_right for p in Segment.iterate_reversed(edge)]
            ) for edge, d in zip(self.edges, self.dirs)]
        h1_right = h_right_left[0][0]
        h1_left = h_right_left[0][1]
        h2_right = h_right_left[2][0]
        h2_left= h_right_left[2][1]
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
            self.points_grid[0][i].handle_up = -cross_h1[i]
            self.points_grid[self.ytot - 1][i].handle_down = -cross_h2[i]
            self.points_grid[self.ytot - 1][i].handle_up = cross_h2[i]
        for i in range(self.ytot):
            self.points_grid[i][0].handle_right = cross_vh1[i]
            self.points_grid[i][0].handle_left = -cross_vh1[i]
            self.points_grid[i][self.xtot - 1].handle_right = -cross_vh2[i]
            self.points_grid[i][self.xtot - 1].handle_left = cross_vh2[i]
        for i in range(1, self.xtot - 1):
            spline = obj.data.splines.new("BEZIER")
            spline.bezier_points.add(self.ytot - 1)
            for j, point in zip(range(self.ytot), spline.bezier_points):
                p = self.points_grid[j][i]
                BigQuad.add_point_to_curve(point, p.co, p.handle_up, p.handle_down)
        for j in range(1, self.ytot - 1):
            spline = obj.data.splines.new("BEZIER")
            spline.bezier_points.add(self.xtot - 1)
            for i, point in zip(range(self.xtot), spline.bezier_points):
                p = self.points_grid[j][i]
                BigQuad.add_point_to_curve(point, p.co, p.handle_left, p.handle_right)
        self.add_new_quads(glist)


class Quad:
    @staticmethod
    def step_left(segments: List[Segment], s: Segment):
        p1 = s.p1
        prev_seg = p1.prev_seg
        if prev_seg is None:
            return None
        if prev_seg not in segments:
            return None
        return prev_seg

    @staticmethod
    def step_right(segments: List[Segment], s: Segment):
        p2 = s.p2
        post_seg = p2.post_seg
        if post_seg is None:
            return None
        if post_seg not in segments:
            return None
        return post_seg

    @staticmethod
    def move_along_spline(segments: List[Segment], s0: Segment):
        edg: List[Segment] = [s0]
        s = s0
        bps: List[BigPoint] = []
        while s is not None:
            s = Quad.step_right(segments, s)
            if s == s0:
                raise ValueError("cycling!")
            if s is not None:
                edg.append(s)
                bps.append(s.p1.bpoint)
        s = s0
        while s is not None:
            s = Quad.step_left(segments, s)
            if s == s0:
                raise ValueError("cycling!")
            if s is not None:
                edg = [s] + edg
                bps = [s.p2.bpoint] + bps
        return edg, bps

    @staticmethod
    def verify_segment_connects_point(s: Segment, p: Point) -> bool:
        for point in p.bpoint.points:
            if point in (s.p1, s.p2):
                return True
        return False

    @staticmethod
    def get_neighbours(segments: List[Segment], edg: List[Segment]) -> List[Segment]:
        edges = (edg[0], edg[-1])
        ret: List[Optional[Segment]] = [None, None]
        right_added = False
        left_added = False
        points = [edges[0].p1, edges[1].p2]
        for i in range(2):
            e = edges[i]
            found = False
            n = 0
            while not found:
                s = segments[n]
                if s == e:
                    found = True
                    if not left_added:
                        prev_seg = segments[n-1]
                        if prev_seg not in edg:
                            if Quad.verify_segment_connects_point(prev_seg, points[i]):
                                ret[i] = prev_seg
                                left_added = True
                    if not right_added:
                        nn = n + 1
                        if nn == len(segments):
                            nn = 0
                        next_seg = segments[nn]
                        if segments[nn] not in edg:
                            if Quad.verify_segment_connects_point(next_seg, points[i]):
                                ret[i] = segments[nn]
                                right_added = True
                n += 1
        ret2 = [seg for seg in ret if seg is not None]
        assert len(ret2) == 2
        return ret2

    @staticmethod
    def steps_dividing_edges_right(p: Point, n: int):
        for _ in range(n):
            p_post = p.post_seg
            if p_post is None:
                return None
            p = p_post.p2
        return p.bpoint

    @staticmethod
    def steps_dividing_edges_left(p: Point, n: int):
        for _ in range(n):
            p_prev = p.prev_seg
            if p_prev is None:
                return None
            p = p_prev.p1
        return p.bpoint

    @staticmethod
    def verify_dividing_edges(bp1: BigPoint, bp2: BigPoint, n: int):
        for p in bp1.points:
            s1 = Quad.steps_dividing_edges_right(p, n)
            if s1 is not None:
                if s1 == bp2:
                    return True
            s2 = Quad.steps_dividing_edges_left(p, n)
            if s2 is not None:
                if s2 == bp2:
                    return True
        return False

    @staticmethod
    def turn(p: Point):
        for pp in p.bpoint.points:
            if pp != p:
                yield pp

    @staticmethod
    def little_quads_step_left(p: Point):
        if p.prev_seg:
            return p.prev_seg.p1
        return None

    @staticmethod
    def little_quads_step_right(p: Point):
        if p.post_seg:
            return p.post_seg.p2
        return None

    @staticmethod
    def step(p: Point,
            func: Callable[[Point], Optional[Point]],
            s1: int,
            s2: int,
            bp_s1: BigPoint,
            bp_s2: BigPoint) -> bool | Point:
        res = func(p)
        if res:
            for pp in Quad.turn(res):
                for s, bp in zip((s1, s2), (bp_s1, bp_s2)):
                    if s != 0:
                        for f in (Quad.steps_dividing_edges_left, Quad.steps_dividing_edges_right):
                            if f(pp, s) == bp:
                                return True
            return res
        return False

    @staticmethod
    def steps(p: Point,
              n: int,
              end_bps_s1: List[BigPoint],
              end_bps_s2: List[BigPoint],
              s1: int,
              s2: int,
              func: Callable[[Point], Optional[Point]]) -> bool:
        res = p
        for i in range(n):
            if i != (n - 1) or (s1 != 0 and s2 != 0):
                res = Quad.step(res, func, s1, s2, end_bps_s1[i], end_bps_s2[i])
                if res in (True, False):
                    return res
        return False

    @staticmethod
    def verify_little_quads(p1: Point,
                            end_bps_s1: List[BigPoint],
                            end_bps_s2: List[BigPoint],
                            s1: int,
                            s2: int,
                            n: int):
        for p in Quad.turn(p1):
            for func in (Quad.little_quads_step_left, Quad.little_quads_step_right):
                if Quad.steps(p, n, end_bps_s1, end_bps_s2, s1, s2, func):
                    return True
        return False

    @staticmethod
    def get_dir_from_right(edg: List[Segment], segments: List[Segment]):
        bp = edg[-1].p2.bpoint
        for p in bp.points:
            post = p.post_seg
            if post is not None:
                if post in segments:
                    return True
        return False

    @staticmethod
    def get_dir_from_left(edg: List[Segment], segments:List[Segment]):
        bp = edg[0].p1.bpoint
        for p in bp.points:
            prev = p.prev_seg
            if prev is not None:
                if prev in segments:
                    return True
        return False

    @staticmethod
    def verify_div(b1: List[BigPoint], b2: List[BigPoint], dir1: bool, dir2: bool, n: int):
        for i in range(len(b1)):
            if dir1 == dir2:
                if Quad.verify_dividing_edges(b1[i], b2[i], n):
                    return True
            else:
                if Quad.verify_dividing_edges(b1[i], b2[-i-1], n):
                    return True
        return False

    @staticmethod
    def extract_points(edge: List[Segment], dir: bool) -> List[Point]:
        points: List[Point] = []
        if edge:
            if dir:
                points = list(Segment.iterate_direct(edge))
            else:
                points = list(Segment.iterate_reversed(edge))
        return points

    @staticmethod
    def verify_and_init(segments: List[Segment], glist: "GlobalList") -> bool:
        le = len(segments)
        if le < 4:
            return False
        if (le % 2) == 1:
            return False
        for segment in segments:
            if segment.finished:
                return False
        edg1, b1 = Quad.move_along_spline(segments, segments[0])
        ret = Quad.get_neighbours(segments, edg1)
        edg2, b2 = Quad.move_along_spline(segments, ret[1])
        edg4, b4 = Quad.move_along_spline(segments, ret[0])
        if len(edg2) != len(edg4):
            return False
        if le - len(edg2)*2 != len(edg1)*2:
            return False
        ret = Quad.get_neighbours(segments, edg2)
        if ret[0] not in edg1:
            seg = ret[0]
        else:
            seg = ret[1]
        edg3, b3 = Quad.move_along_spline(segments, seg)
        if le != len(edg1) + len(edg2) + len(edg3) + len(edg4):
            return False
        dir1 = True
        dir2 = Quad.get_dir_from_right(edg1, segments)
        dir4 = not Quad.get_dir_from_left(edg1, segments)
        if dir2:
            dir3 = not Quad.get_dir_from_right(edg2, segments)
        else:
            dir3 = Quad.get_dir_from_left(edg2, segments)
        if le == 4:
            Quad(edg1[0], edg2[0], edg3[0], edg4[0], dir1, dir2, dir3, dir4, glist)
            return True
        if Quad.verify_div(b1, b3, dir1, dir3, len(edg2)) or Quad.verify_div(b2, b4, dir2, dir4, len(edg1)):
            return False
        points1 = Quad.extract_points(edg1, True)
        points2 = Quad.extract_points(edg2, dir2)
        bpoints2 = [p.bpoint for p in points2]
        points3 = Quad.extract_points(edg3, dir3)
        points4 = Quad.extract_points(edg4, dir4)
        bpoints4 = [p.bpoint for p in points4]
        le_e1 = len(edg1)
        n = len(edg2)
        for i, p in enumerate(points1):
            if Quad.verify_little_quads(p, bpoints4[1:], bpoints2[1:], i, le_e1 - i, n):
                return False
        if Quad.steps(points1[0], n, bpoints2[1:], bpoints2[1:], le_e1, 0, Quad.little_quads_step_left):
            return False
        if Quad.steps(points1[-1], n, bpoints4[1:], bpoints4[1:], le_e1, 0, Quad.little_quads_step_right):
            return False
        bpoints2.reverse()
        bpoints4.reverse()
        for i, p in enumerate(points3):
            if Quad.verify_little_quads(p, bpoints4[1:], bpoints2[1:], i, le_e1 - i, n):
                return False
        if Quad.steps(points3[0], n, bpoints2[1:], bpoints2[1:], le_e1, 0,
                      Quad.little_quads_step_right if dir3 else Quad.little_quads_step_left):
            return False
        if Quad.steps(points3[-1], n, bpoints4[1:], bpoints4[1:], le_e1, 0,
                      Quad.little_quads_step_left if dir3 else Quad.little_quads_step_right):
            return False
        BigQuad(edg1, edg2, edg3, edg4, dir1, dir2, dir3, dir4, glist)
        return True

    def __init__(self,
                 seg1: Segment,
                 seg2: Segment,
                 seg3: Segment,
                 seg4: Segment,
                 dir1: bool,
                 dir2: bool,
                 dir3: bool,
                 dir4: bool,
                 glist: "GlobalList"):
        self.kk = [[mathutils.Vector((0,0,0)) for _ in range(4)] for __ in range(4)]
        self.kk1 = [[mathutils.Vector((0,0,0)) for _ in range(2)] for __ in range(2)]
        self.segments = [seg1, seg2, seg3, seg4]
        self.directions = [dir1, dir2, dir3, dir4]
        for seg in self.segments:
            seg.add_quad(self)
        glist.add_quad(self)

    def get_edge_control_points(self, edge_num: int):
        p1 = self.segments[edge_num].p1
        p4 = self.segments[edge_num].p2
        k1 = p1.bpoint.coords
        k2 = k1 + p1.handle_right
        k4 = p4.bpoint.coords
        k3 = k4 + p4.handle_left
        res = [k1, k2, k3, k4]
        if not self.directions[edge_num]:
            res.reverse()
        return res

    def calculate_coefs(self):
        self.kk[0][0], self.kk[0][1], self.kk[0][2], self.kk[0][3] = self.get_edge_control_points(0)
        self.kk[1][0], self.kk[2][0], self.kk[3][0] = self.get_edge_control_points(3)[1:]
        self.kk[1][3], self.kk[2][3], self.kk[3][3] = self.get_edge_control_points(1)[1:]
        self.kk[3][1], self.kk[3][2] = self.get_edge_control_points(2)[1:3]

    def extract_first_handle(self, i: int):
        if self.directions[i]:
            return self.segments[i].p1.handle_right
        return self.segments[i].p2.handle_left

    def extract_last_handle(self, i: int):
        if self.directions[i]:
            return self.segments[i].p2.handle_left
        return self.segments[i].p1.handle_right

    def get_neighbour_quad(self, i: int):
        seg = self.segments[i]
        if len(seg.quads) == 1:
            return None
        res = [quad for quad in seg.quads if quad != self][0]
        assert isinstance(res, Quad)
        return res

    def extract_a0_a3(self, i: int):
        if i == 0:
            a0 = self.extract_first_handle(3)
            a3 = self.extract_first_handle(1)
        elif i == 1:
            a0 = self.extract_last_handle(0)
            a3 = self.extract_last_handle(2)
        elif i == 2:
            a0 = self.extract_last_handle(3)
            a3 = self.extract_last_handle(1)
        elif i == 3:
            a0 = self.extract_first_handle(0)
            a3 = self.extract_first_handle(2)
        else:
            raise ValueError("wrong i")
        return a0, a3

    def calculate_coefs_along_segment(self, i: int):
        a0, a3 = self.extract_a0_a3(i)
        _, p1, p2, __ = self.get_edge_control_points(i)
        neighbour_quad = self.get_neighbour_quad(i)
        if neighbour_quad is None:
            a1, a2 = Quad.calculate_free_coefs(a0, a3)
        else:
            s0 = self.extract_first_handle(i)
            s1 = p2 - p1
            s2 = -1 * self.extract_last_handle(i)
            neighbour_i = neighbour_quad.segments.index(self.segments[i])
            bb0, bb2 = neighbour_quad.extract_a0_a3(neighbour_i)
            if self.directions[i] != neighbour_quad.directions[neighbour_i]:
                bb0, bb2 = bb2, bb0
            b0 = (bb0 - a0).normalized()
            b2 = (bb2 - a3).normalized()
            for v in (a0, a3, b0, b2, s0, s2):
                if v.length < TH:
                    a1, a2 = Quad.calculate_free_coefs(a0, a3)
                    break
            else:
                if not are_coplanar(a0, b0, s0) or not are_coplanar(a3, s2, b2):
                    a1, a2 = Quad.calculate_free_coefs(a0, a3)
                elif b0.cross(s0).length < TH or b2.cross(s2).length < TH:
                    a1, a2 = Quad.calculate_free_coefs(a0, a3)
                else:
                    k0, h0 = get_coefs(b0, s0, a0)
                    k1, h1 = get_coefs(b2, s2, a3)
                    prev_b1 = self.segments[i].b1
                    if prev_b1 is not None:
                        b1 = -1 * prev_b1
                    else:
                        ve = self.calculate_ve(i, neighbour_quad, neighbour_i)
                        b1 = self.segments[i].calculate_b1(k0, k1, h0, h1, b0, b2, s0, s1, s2, a0, a3, ve)
                    a1 = 1/3 * (2 * k0 * b1 + k1 * b0 + 2 * h0 * s1 + h1 * s0)
                    a2 = 1/3 * (k0 * b2 + 2 * k1 * b1 + h0 * s2 + 2 * h1 * s1)
        if i == 0:
            self.kk[1][1] = a1 + p1
            self.kk[1][2] = a2 + p2
        elif i == 1:
            self.kk1[0][1] = a1 + p1
            self.kk1[1][1] = a2 + p2
        elif i == 2:
            self.kk[2][1] = a1 + p1
            self.kk[2][2] = a2 + p2
        elif i == 3:
            self.kk1[0][0] = a1 + p1
            self.kk1[1][0] = a2 + p2
        return True

    @staticmethod
    def calculate_free_coefs(a0: mathutils.Vector, a3: mathutils.Vector):
        a1 = a0*2/3 + a3*1/3
        a2 = a0*1/3 + a3*2/3
        return a1, a2
    
    def extract_corner(self, i: int):
        if i == 0:
            return self.segments[0].p1.i
        if i == 1:
            return self.segments[0].p2.i
        if self.directions[2]:
            if i == 2:
                return self.segments[2].p2.i
            if i == 3:
                return self.segments[2].p1.i
        else:
            if i == 2:
                return self.segments[2].p1.i
            if i == 3:
                return self.segments[2].p2.i
    
    def render_quad(self, d: "DependantsOfResolution_np | DependantsOfResolution", glist: "GlobalList"):
        self.calculate_coefs()
        for i in range(4):
            self.calculate_coefs_along_segment(i)
        borders = [self.segments[i].verts for i in range(4)]
        for i in range(4):
            if not self.directions[i]:
                border = borders[i]
                assert isinstance(border, list)
                borders[i] = list(reversed(border))
        corners = [self.extract_corner(i) for i in range(4)]
        calc_gregory_surf(self.kk, self.kk1, d, *borders, *corners, glist)

    def calculate_ve(self, i: int, neighbour: "Quad", neighbour_i: int) -> mathutils.Vector:
        p1 = self.segments[(i+2) % 4].calculate_midpoint()
        p2 = neighbour.segments[(neighbour_i+2) % 4].calculate_midpoint()
        ve = p2 - p1
        return ve

class GlobalList:
    def __init__(self):
        self.reduced_points: List[mathutils.Vector] = []
        self.big_points: List[BigPoint] = []
        self.count = 0
        self.splines: List[Spline] = []
        self.quads: List[Quad] = []
        self.big_quads: List[BigQuad] = []
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

    def add_quad(self, quad: Quad):
        self.quads.append(quad)

    def add_big_quad(self, big_quad: BigQuad):
        self.big_quads.append(big_quad)

    def work_with_segment(self, segment: Segment):
        segments_verified = [segment]
        bpoints_verified: List[BigPoint] = []
        counter_splines = 1
        prev_spline = segment.p1.spline
        initial_spline = prev_spline
        initial_bpoint = segment.p1.bpoint
        self.edge_step(initial_bpoint, segment.p2, segments_verified, bpoints_verified, counter_splines, prev_spline, initial_spline, True)
        segment.finished = True

    def edge_step(self,
                  initial_bpoint: BigPoint,
                  point: Point,
                  segments_verified: List[Segment],
                  bpoints_verified: List[BigPoint],
                  counter_splines: int,
                  prev_spline: Spline,
                  initial_spline: Spline,
                  first: bool) -> StepRes:
        bpoint = point.bpoint
        spline = point.spline
        if spline != prev_spline:
            counter_splines += 1
            first = False
            if counter_splines > 5:
                return StepRes.NOT_FINISHED
            if counter_splines == 5 and spline != initial_spline:
                return StepRes.NOT_FINISHED
        if initial_bpoint == bpoint:
            if counter_splines < 4:
                return StepRes.NOT_FINISHED
            if Quad.verify_and_init(segments_verified, self):
                if segments_verified[0].finished:
                    return StepRes.FINISHED
                return StepRes.PART_FINISHED
        if bpoint in bpoints_verified:
            return StepRes.NOT_FINISHED
        bpoints_verified.append(bpoint)
        for point in bpoint.points:
            segment1 = point.prev_seg
            res = self.step_segment(initial_bpoint,
                                    segment1,
                                    True,
                                    segments_verified,
                                    bpoints_verified,
                                    counter_splines,
                                    spline,
                                    initial_spline,
                                    first)
            if res in (StepRes.FINISHED, StepRes.PART_FINISHED):
                bpoints_verified.pop()
                return res
            segment2 = point.post_seg
            res = self.step_segment(initial_bpoint,
                                    segment2,
                                    False,
                                    segments_verified,
                                    bpoints_verified,
                                    counter_splines,
                                    spline,
                                    initial_spline,
                                    first)
            if res in (StepRes.FINISHED, StepRes.PART_FINISHED):
                bpoints_verified.pop()
                return res
        bpoints_verified.pop()
        return StepRes.NOT_FINISHED

    def step_segment(self,
                     initial_bpoint: BigPoint,
                     segment: Optional[Segment],
                     is_p1: bool,
                     sv: List[Segment],
                     bv: List[BigPoint],
                     counter_splines: int,
                     spline: Spline,
                     initial_spline: Spline,
                     first: bool) -> StepRes:
        if segment is not None:
            if not segment in sv:
                if not segment.finished:
                    sv.append(segment)
                    point = segment.p1 if is_p1 else segment.p2
                    res = self.edge_step(initial_bpoint, point, sv, bv, counter_splines, spline, initial_spline, first)
                    sv.pop()
                    match res:
                        case StepRes.FINISHED: 
                            return StepRes.FINISHED
                        case StepRes.PART_FINISHED:
                            if not first:
                                return StepRes.PART_FINISHED
                        case StepRes.NOT_FINISHED:
                            return StepRes.NOT_FINISHED
        return StepRes.NOT_FINISHED

    def add_quads(self):
        for segment in self.segments:
            if not segment.finished:
                self.work_with_segment(segment)

    def subdivide_quads(self, obj: bpy.types.ID):
        for bq in self.big_quads:
            bq.subdivide(obj, self)

    def render_mesh(self, d: "DependantsOfResolution|DependantsOfResolution_np", name: str, context: bpy.types.Context):
        for bpoint in self.big_points:
            bpoint.add_vert(self)
        for segment in self.segments:
            segment.calculate_verts(self, d)
        for quad in self.quads:
            quad.render_quad(d, self)
        if self.verts is not None and self.faces is not None:
            mesh = bpy.data.meshes.new(name=name + "_Mesh")
            mesh.from_pydata(self.verts, [], self.faces)
            obj = bpy.data.objects.new(name + "_GeneratedMesh", mesh)
            context.collection.objects.link(obj)
            return obj
        return None

    def add_many_curves(self, name: str, parent_collection: bpy.types.Collection, context: bpy.types.Context):
        collection = bpy.data.collections.new(name)
        collection.greg_settings.used_for_greg = True
        parent_collection.children.link(collection)
        for segment in self.segments:
            curve = bpy.data.curves.new(name="greg_curve", type="CURVE")
            curve.dimensions = "3D"
            spline = curve.splines.new("BEZIER")
            spline.bezier_points.add(1)
            for i, p in enumerate((segment.p1, segment.p2)):
                spline.bezier_points[i].co = p.bpoint.coords
                spline.bezier_points[i].handle_left = p.bpoint.coords + p.handle_left
                spline.bezier_points[i].handle_right = p.bpoint.coords + p.handle_right
                spline.bezier_points[i].handle_left_type = "ALIGNED"
                spline.bezier_points[i].handle_right_type = "ALIGNED"
            curve_obj = bpy.data.objects.new(name="greg_curve_obj", object_data=curve)
            curve_obj.greg_curve_settings.used_for_greg = True
            curve_obj.greg_curve_settings.name = get_next_id(collection)
            curve_prop = collection.greg_settings.curves.add()
            curve_prop.name = curve_obj.greg_curve_settings.name
            curve_prop.curve = curve_obj
            for i, p in enumerate((segment.p1, segment.p2)):
                p.bpoint.created_curves.append((curve_prop.name, i))
            collection.objects.link(curve_obj)
        for bpoint in self.big_points:
            empty_obj = bpy.data.objects.new("greg_empty", None)
            empty_obj.location = bpoint.coords
            empty_obj.greg_empty_settings.used_for_greg = True
            empty_obj.greg_empty_settings.name = get_next_id(collection)
            empty_prop = collection.greg_settings.empties.add()
            empty_prop.name = empty_obj.greg_empty_settings.name
            empty_prop.empty = empty_obj
            collection.objects.link(empty_obj)
            for name, i in bpoint.created_curves:
                curve_obj = collection.greg_settings.curves[name].curve
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

def extract_handles(ends):
    handles: List[mathutils.Vector] = []
    for end in ends:
        spline = end.basic_end.curve.data.splines[0]
        i = end.basic_end.end
        point = spline.bezier_points[i]
        co = point.co
        if i == 0:
            ha = point.handle_right
        else:
            ha = point.handle_left
        handles.append(ha - co)
    return handles

def check_ends_collinear(end1, end2):
    ends = (end1, end2)
    handles = extract_handles(ends)
    if are_collinear(*handles):
        for i in range(2):
            this_end = ends[i]
            other_end = ends[1-i]
            this_end.is_collinear = True
            basic_end = this_end.collinear_to.add()
            basic_end.curve = other_end.basic_end.curve
            basic_end.end = other_end.basic_end.end
            basic_end.name = other_end.basic_end.name

def check_ends_coplanar(ends1, ends2, ends3, empty, collection: bpy.types.Collection):
    handles: List[mathutils.Vector] = []
    ends = (ends1[0], ends2[0], ends3[0])
    handles = extract_handles(ends)
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
            end.is_coplanar = True
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
        print("some ends", [end.name for end in empty.greg_empty_settings.curve_ends])
        end = empty.greg_empty_settings.curve_ends[end_name]
        apply_hook(end)
        empty_name = empty.greg_empty_settings.name
        if end.is_collinear:
            for other_basic_end in end.collinear_to:
                other_end = empty.greg_empty_settings.curve_ends[other_basic_end.name]
                other_end.collinear_to.remove(other_end.collinear_to.find(end_name))
                if len(other_end.collinear_to) == 0:
                    other_end.is_collinear = False
        if end.is_coplanar:
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
                            other_end.is_coplanar = False
                            if other_end.name != end_name:
                                apply_hook(other_end)
                                add_hook(other_end)
                    bpy.data.objects.remove(arrow, do_unlink=True)
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

def copy_transforms(active: bpy.types.Object, new: bpy.types.Object):
    new.rotation_euler = active.rotation_euler
    new.location = active.location
    new.scale = active.scale

#**************************************************************************

def mirror_vec(vec, mirror_object, axis): #axis: 0 - x, 1 - y, 2 - z
    basic_vec = mathutils.Vector((0, 0, 0))
    basic_vec[axis] = 1
    if mirror_object is not None:
        mat = mirror_object.matrix_world.copy()
        mat.invert()
        mirror = basic_vec @ mat
        reflected = vec.reflect(mirror)
        delta_vec = mirror_object.location.project(mirror)
        return reflected + 2*delta_vec
    else:
        return vec.reflect(basic_vec)

#**************************************************************************

class NewGlobalList:
    def __init__(self, collection):
        self.collection = collection
        self.quads = []
        self.verts = None
        self.faces = None
        self.ids_counter = 0

    def add_quads(self):
        for phantom_curve in self.collection.greg_settings.phantom_curves:
            if not phantom_curve.finished:
                self.work_with_curve(phantom_curve)

    def work_with_curve(self, phantom_curve):
        curves_verified = [phantom_curve]
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
        first = False
        if len(curves_verified) > 4:
            return StepRes.NOT_FINISHED
        if initial_bpoint.name == bpoint.name:
            if len(curves_verified) < 4:
                return StepRes.NOT_FINISHED
            if self.verify_and_init_quad(curves_verified, end0):
                if curves_verified[0].finished:
                    return StepRes.FINISHED
                return StepRes.PART_FINISHED
            return StepRes.NOT_FINISHED
        if bpoint in bpoints_verified:
            return StepRes.NOT_FINISHED
        bpoints_verified.append(bpoint)
        for end in bpoint.ends:
            phantom_curve1 = self.collection.greg_settings.phantom_curves[end.curve_name]
            if not self.are_collinear(end0, end): # they are not collinear
                res = self.step_curve(initial_bpoint,
                                    phantom_curve1,
                                    end.curve_i,
                                    curves_verified,
                                    bpoints_verified,
                                    first)
                if res in (StepRes.FINISHED, StepRes.PART_FINISHED):
                    bpoints_verified.pop()
                    return res
        bpoints_verified.pop()
        return StepRes.NOT_FINISHED
    
    def extract_handle_phantom(self, end):
        curve = self.collection.greg_settings.phantom_curves[end.curve_name]
        bpoint = self.collection.greg_settings.phantom_bpoints[end.bpoint_name]
        handle = curve.handle1 if end.curve_i == 0 else curve.handle2
        return handle - bpoint.co
    
    def are_collinear(self, end1, end2):
        v1, v2 = [self.extract_handle_phantom(end) for end in (end1, end2)]
        return are_collinear(v1, v2)
        

    def step_curve(self,
                   initial_bpoint,
                   phantom_curve,
                   end_i,
                   curves_verified,
                   bpoints_verified,
                   first) -> StepRes:
        if not phantom_curve in curves_verified:
            if not phantom_curve.finished:
                curves_verified.append(phantom_curve)
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
                                     first)
                curves_verified.pop()
                match res:
                    case StepRes.FINISHED: 
                        return StepRes.FINISHED
                    case StepRes.PART_FINISHED:
                        if not first:
                            return StepRes.PART_FINISHED
                    case StepRes.NOT_FINISHED:
                        return StepRes.NOT_FINISHED
        return StepRes.NOT_FINISHED
    
    def verify_and_init_quad(self, curves_verified, end0):
        first_curve = curves_verified[0]
        first_bpoint = self.collection.greg_settings.phantom_bpoints[first_curve.bpoint1_name]
        first_vec = first_curve.handle1 - first_bpoint.co
        other_vec = self.extract_handle_phantom(end0)
        if are_collinear(first_vec, other_vec):
            return False

        sets = [set(el.name for el in phantom_curve.source_curve.greg_curve_settings.not_face_ids)
                for phantom_curve in curves_verified]
        common = set.intersection(*sets)
        if common: # there is a common not_face
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

    @staticmethod
    def calculate_midpoint(curve_obj):
        curve_settings = curve_obj.greg_curve_settings
        p0 = curve_settings.end1_empty.location
        p1 = curve_obj.data.splines[0].bezier_points[0].handle_right
        p3 = curve_settings.end2_empty.location
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
    
    '''@staticmethod
    def extract_first_handle(quad, i: int):
        if quad.dirs[i]:
            point = quad.curves[i].curve.data.splines[0].bezier_points[0]
            return point.handle_right - point.co
        point = quad.curves[i].curve.data.splines[0].bezier_points[1]
        return point.handle_left - point.co

    @staticmethod
    def extract_last_handle(quad, i: int):
        if quad.dirs[i]:
            point = quad.curves[i].curve.data.splines[0].bezier_points[1]
            return point.handle_left - point.co
        point = quad.curves[i].curve.data.splines[0].bezier_points[0]
        return point.handle_right - point.co'''

    '''@staticmethod
    def extract_a0_a3(quad, i: int):
        if i == 0:
            a0 = NewGlobalList.extract_first_handle(quad, 3)
            a3 = NewGlobalList.extract_first_handle(quad, 1)
        elif i == 1:
            a0 = NewGlobalList.extract_last_handle(quad, 0)
            a3 = NewGlobalList.extract_last_handle(quad, 2)
        elif i == 2:
            a0 = NewGlobalList.extract_last_handle(quad, 3)
            a3 = NewGlobalList.extract_last_handle(quad, 1)
        elif i == 3:
            a0 = NewGlobalList.extract_first_handle(quad, 0)
            a3 = NewGlobalList.extract_first_handle(quad, 2)
        else:
            raise ValueError("wrong i")
        return a0, a3'''
    
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
                            phantom_curve.invert_shear[quad_i] = True
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


    
    def add_bpoint_if_needed(self, co, original_name):
        for phantom_bpoint in self.collection.greg_settings.phantom_bpoints:
            if phantom_bpoint.original_empty_name == original_name:
                if same_coords(co, phantom_bpoint.co):
                    return phantom_bpoint
        new_bpoint = self.collection.greg_settings.phantom_bpoints.add()
        new_bpoint.name = str(self.ids_counter)
        self.ids_counter += 1
        new_bpoint.co_prop = co
        new_bpoint.original_empty_name = original_name
        return new_bpoint # DANGER! Do not create next bpoint while this is saved somewhere in Blender
    
    def add_phantom_curve(self, co1, co2, handle1, handle2, original_curve_obj, mirrored):
        empty1_name = original_curve_obj.greg_curve_settings.end1_empty.greg_empty_settings.name
        empty2_name = original_curve_obj.greg_curve_settings.end2_empty.greg_empty_settings.name
        phantom_curve = self.collection.greg_settings.phantom_curves.add()
        phantom_curve.name = str(self.ids_counter)
        self.ids_counter += 1
        phantom_curve.mirrored = mirrored
        phantom_curve.handle1_prop = handle1
        phantom_curve.handle2_prop = handle2
        phantom_curve.source_curve = original_curve_obj
        phantom_curve.source_curve_name = original_curve_obj.greg_curve_settings.name
        end_names = []
        for i in range(2):
            if i == 0:
                bpoint = self.add_bpoint_if_needed(co1, empty1_name)
                phantom_curve.bpoint1_name = bpoint.name
            else:
                bpoint = self.add_bpoint_if_needed(co2, empty2_name)
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
    is_coplanar: bpy.props.BoolProperty(default=False)
    coplanar_vectors: bpy.props.CollectionProperty(type=GregArrowItem)
    is_collinear: bpy.props.BoolProperty(default=False)
    collinear_to: bpy.props.CollectionProperty(type=GregBasicEnd)
    hook: bpy.props.StringProperty(default="")
    name: bpy.props.StringProperty(default="")

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

class GregCurve(bpy.types.PropertyGroup):
    used_for_greg: bpy.props.BoolProperty(default=False)
    end1_empty: bpy.props.PointerProperty(type=bpy.types.Object)
    end2_empty: bpy.props.PointerProperty(type=bpy.types.Object)
    end1_name: bpy.props.StringProperty(default="")
    end2_name: bpy.props.StringProperty(default="")
    name: bpy.props.StringProperty(default="")
    not_face_ids: bpy.props.CollectionProperty(type=GregId)
    phantom_curves_ids: bpy.props.CollectionProperty(type=GregId)
    is_mirror_bridge: bpy.props.BoolProperty(default=False)

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

def verify_is_closed_path(curves):
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
    return True

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

def check_curve_crosses_mirror(obj):
    p1, p2 = None, None
    for i, modifier in enumerate(obj.modifiers):
        if modifier.type == 'MIRROR':
            if p1 is None:
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
                        pos = mirror_object.location
                    else:
                        pos = mathutils.Vector((0,0,0))
                    q1 = normal.dot(pos - p1)
                    q2 = normal.dot(pos - p2)
                    if (q1 < -TH and q2 > TH) or (q1 > TH and q2 < -TH):
                        return i, axis
    return False

def add_mirror_empty_constraints(empty, target, mirror_object, axis):
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
        add_mirror_empty_constraints(other, target, mirror_obj, axis)
        if target == curve.greg_curve_settings.end1_empty:
            target_end_name = curve.greg_curve_settings.end1_name
            other_end_name = curve.greg_curve_settings.end2_name
            target_point = curve.data.splines[0].bezier_points[0]
            other_point = curve.data.splines[0].bezier_points[1]
        elif target == curve.greg_curve_settings.end2_empty:
            target_end_name = curve.greg_curve_settings.end2_name
            other_end_name = curve.greg_curve_settings.end1_name
            target_point = curve.data.splines[0].bezier_points[1]
            other_point = curve.data.splines[0].bezier_points[0]
        end1 = target.greg_empty_settings.curve_ends[target_end_name]
        end2 = other.greg_empty_settings.curve_ends[other_end_name]
        apply_hook(end1)
        apply_hook(end2)
        other_point.handle_left = mirror_vec(target_point.handle_right, mirror_obj, axis)
        other_point.handle_right = mirror_vec(target_point.handle_left, mirror_obj, axis)
        add_hook(end1)
        add_hook(end2)
        curve.greg_curve_settings.is_mirror_bridge = True
        return {'FINISHED'}    

def add_bridge_mirror_func(self, context: bpy.types.Context):
    self.layout.operator(MakeCurveMirrorBridge.bl_idname)
    
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
    bl_label = "Set loop not face"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.mode != "OBJECT":
            return False
        slen = len(context.selected_objects)
        if slen < 4 and slen % 2 != 0:
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
        if not verify_is_closed_path(context.selected_objects):
            return False
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        collection = get_greg_collection(context.selected_objects[0])
        id = get_next_id(collection)
        for curve in context.selected_objects:
            id_setting = curve.greg_curve_settings.not_face_ids.add()
            id_setting.name = id
        

        return {'FINISHED'}

def set_not_face_menu_func(self, context: bpy.types.Context):
    self.layout.operator(SetNotFace.bl_idname)

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
        '''new = glist.render_mesh(d, active.name, context)
        if new is not None:
            copy_transforms(active, new)'''

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
        new = glist.render_mesh(d, collection.name, context)
        '''new = glist.render_mesh(d, active.name, context)
        if new is not None:
            copy_transforms(active, new)'''

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def add_surface_menu_func(self, context: bpy.types.Context):
    self.layout.operator(CreateSurfacesBetweenCurves.bl_idname)

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

    bpy.utils.register_class(CreateCurvesCollection)
    bpy.types.VIEW3D_MT_object.append(add_collection_menu_func)
    bpy.utils.register_class(CreateSurfacesBetweenCurves)
    bpy.types.VIEW3D_MT_object.append(add_surface_menu_func)  # Adds the new operator to an existing menu.
    bpy.utils.register_class(SetNotFace)
    bpy.types.VIEW3D_MT_object_context_menu.append(set_not_face_menu_func)  # Adds the new operator to an existing menu.
    bpy.utils.register_class(PrintItemInfo)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_print_info_func)
    bpy.utils.register_class(MakeCurveMirrorBridge)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_bridge_mirror_func)  # Adds the new operator to an existing menu.
    bpy.utils.register_class(PrintDotInfo)
    bpy.types.VIEW3D_MT_object_context_menu.append(add_print_dot_func)
    bpy.utils.register_class(OBJECT_PT_greg_curve_properties)
    bpy.utils.register_class(OBJECT_PT_greg_curve_properties1)
    bpy.utils.register_class(OBJECT_PT_greg_curve_properties2)

    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    
def unregister():
    bpy.utils.unregister_class(OBJECT_PT_greg_curve_properties2)
    bpy.utils.unregister_class(OBJECT_PT_greg_curve_properties1)
    bpy.utils.unregister_class(OBJECT_PT_greg_curve_properties)
    bpy.utils.unregister_class(PrintDotInfo)
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

