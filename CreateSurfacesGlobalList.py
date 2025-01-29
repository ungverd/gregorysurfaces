from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

import numpy as np
import numpy.typing as npt

import bpy
import mathutils
import bmesh

from .DependantsOfResolution import DependantsOfResolution_np
from .commons import are_collinear, TH, are_coplanar, same_coords, mirror_vec, apply_hook, add_hook
from .commons import get_coefs
from .propertyGroups import GregPhantomCurveEnd, GregPhantomCurve, GregQuad
from .calculate_quad import calc_bs

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

def add_corner(glist: "CreateSurfacesGlobalList", coords: mathutils.Vector):
    if glist.verts is None:
        glist.verts = np.array([coords])
    else:
        glist.verts = np.vstack((glist.verts, np.array([coords])))

def add_border(k0: mathutils.Vector,
                k1: mathutils.Vector,
                k2: mathutils.Vector,
                k3: mathutils.Vector,
                glist: "CreateSurfacesGlobalList",
                d: "DependantsOfResolution_np"):
    if glist.verts is None:
        init_point = 0
    else:
        init_point = len(glist.verts)
    border = list(range(init_point, init_point + d.nedges - 1))
    border_coords = calc_bezier_curve_np(k0, k1, k2, k3, d)
    if glist.verts is None:
        glist.verts = border_coords
    else:
        glist.verts = np.vstack((glist.verts, border_coords))
    return border

def calc_quad_gregory_verts(kk, kk1, d):
    control_points = calc_control_points_np(kk, kk1, d)
    res: npt.NDArray[np.float64] = np.sum((d.berns2 * control_points), (2, 3)).reshape((d.nedges-1) * (d.nedges-1), 3)
    # numpy representation of the formula p(u,v) = sum_i_from_0_to_3(sum_j_from_0_to_3( k(i,j)*B(i,u)*B(j,v) ))
    return res

def calc_gregory_surf(kk: List[List[mathutils.Vector]],
                        kk1: List[List[mathutils.Vector]],
                        d: "DependantsOfResolution_np",
                        border1: List[int],
                        border2: List[int],
                        border3: List[int],
                        border4: List[int],
                        corner1: int,
                        corner2: int,
                        corner3: int,
                        corner4: int,
                        glist: "CreateSurfacesGlobalList") -> None:
    res = calc_quad_gregory_verts(kk, kk1, d)
    # numpy representation of the formula p(u,v) = sum_i_from_0_to_3(sum_j_from_0_to_3( k(i,j)*B(i,u)*B(j,v) ))
    if glist.verts is None:
        glist.verts = res
        num_points = 0
    else:
        num_points = len(glist.verts)
        glist.verts = np.vstack((glist.verts, res))
    this_faces = d.faces + num_points
    if glist.faces is None:
        glist.faces = this_faces
    else:
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

class TemporaryQuad:
    def __init__(self,
                 name: str,
                 collection: bpy.types.Collection):
        self.name = name
        self.collection = collection
        self.ps: List[Optional[mathutils.Vector]] = [None] * 4
        self.eps: List[Optional[mathutils.Vector]] = [None] * 4
        self.ems: List[Optional[mathutils.Vector]] = [None] * 4
        self.curves: List[GregPhantomCurve] = []
        self.cps: List[Optional[mathutils.Vector]] = [None] * 4
        self.cms: List[Optional[mathutils.Vector]] = [None] * 4
        self.edges_to_calculate: List[bool] = [False] * 4
        self.desired_directions: List[Optional[mathutils.Vector]] = [None] * 4
        self.dirs_to_set: List[bool] = [False] * 4
        self.bs: List[Optional[mathutils.Vector]] = [None] * 4
    
    def set_curve(self,
                  curve_i: int,
                  curve_dir: bool,
                  curve: GregPhantomCurve):
        dir_to_set = (curve_dir == (curve_i in (0, 1)))
        self.dirs_to_set[curve_i] = dir_to_set
        control_points = CreateSurfacesGlobalList.extract_control_points(self.collection, curve)
        if not dir_to_set:
            control_points = list(reversed(control_points))
        self.ps[curve_i] = control_points[0]
        self.eps[curve_i] = control_points[1]
        self.ems[(curve_i + 1) % 4] = control_points[2]
        self.curves.append(curve)
    
    def set_cs(self,
               bb0: mathutils.Vector,
               bb2: mathutils.Vector,
               i: int):
        if i in (2, 3):
            bb2, bb0 = bb0, bb2
        self.cps[i] = self.ps[i] + bb0
        self.cms[(i + 1) % 4] = self.ps[(i + 1) % 4] + bb2

    def calculate_bs(self):
        print("names:")
        for curve in self.curves:
            print(curve.source_curve.name)
        if True in self.edges_to_calculate:
            desired_directions = [dir for dir, flag in zip(self.desired_directions, self.edges_to_calculate) if flag]
            bs = calc_bs(self.ps, self.ems, self.eps, self.cms, self.cps, self.edges_to_calculate, desired_directions)
            counter = 0
            for i in range(4):
                if self.edges_to_calculate[i]:
                    self.bs[i] = bs[counter]
                    counter += 1

class StepRes(Enum):
    FINISHED = 1
    NOT_FINISHED = 2
    PART_FINISHED = 3

class CreateSurfacesGlobalList:
    def __init__(self, collection):
        self.collection = collection
        self.quads_dict: Dict[str, TemporaryQuad] = {}
        self.verts = None
        self.faces = None
        self.ids_counter = 0
        self.curves_dict: Dict[str, str] = {}

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
        if len(curves_verified) > 4 or (len(curves_verified) == 4 and initial_bpoint != bpoint):
            return StepRes.NOT_FINISHED
        if initial_bpoint == bpoint:
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
    
    def extract_handle_phantom(self, end: "GregPhantomCurveEnd"):
        curve = self.collection.greg_settings.phantom_curves[end.curve_name]
        bpoint = self.collection.greg_settings.phantom_bpoints[end.bpoint_name]
        handle = curve.handle1 if end.curve_i == 0 else curve.handle2
        return handle - bpoint.co
    
    def are_collinear(self, end1: "GregPhantomCurveEnd", end2: "GregPhantomCurveEnd"):
        v1, v2 = [self.extract_handle_phantom(end) for end in (end1, end2)]
        return are_collinear(v1, v2)
        

    def step_curve(self,
                   initial_bpoint,
                   phantom_curve: "GregPhantomCurve",
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
                                     False)
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
        temporary_quad = TemporaryQuad(quad.name, self.collection)
        self.quads_dict[quad.name] = temporary_quad
        for i, phantom_curve in enumerate(curves_verified):
            quad_name = phantom_curve.quads.add()
            quad_name.name = quad.name
            if len(phantom_curve.quads) == 2:
                phantom_curve.finished = True
            new_curve_item = quad.curves.add()
            new_curve_item.name = phantom_curve.name
            if i == 0:
                dir = True
            elif i == 1:
                dir = (phantom_curve.bpoint1_name == curves_verified[0].bpoint2_name)
            elif i == 2:
                if quad.dirs[1]:
                    dir = (phantom_curve.bpoint2_name == curves_verified[1].bpoint2_name)
                else:
                    dir = (phantom_curve.bpoint2_name == curves_verified[1].bpoint1_name)
            elif i == 3:
                dir = (phantom_curve.bpoint1_name == curves_verified[0].bpoint1_name)
            quad.dirs[i] = dir
            temporary_quad.set_curve(i, dir, phantom_curve)
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
        p0, p1, p2, p3 = CreateSurfacesGlobalList.extract_control_points(self.collection, phantom_curve)
        return 1/8 * (p0 + 3*p1 + 3*p2 + p3)
    
    def get_edge_control_points(self, quad, edge_num: int):
        curve_name = quad.curves[edge_num].name
        phantom_curve = self.collection.greg_settings.phantom_curves[curve_name]
        res = CreateSurfacesGlobalList.extract_control_points(self.collection, phantom_curve)
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

    def init_coefs_calculation_along_segment(self,
                                             quad: GregQuad,
                                             i: int):
        temporary_quad = self.quads_dict[quad.name]
        curve_name = quad.curves[i].name
        phantom_curve = self.collection.greg_settings.phantom_curves[curve_name]
        if phantom_curve.source_curve.greg_is_sharp or phantom_curve.conditional_sharp:
            return
        a0, a3 = CreateSurfacesGlobalList.extract_a0_a3(quad, i)
        p0, p1, p2, p3 = CreateSurfacesGlobalList.get_edge_control_points_from_kk(quad, i)
        neighbour_quad = self.get_neighbour_quad(quad, i)
        if neighbour_quad is None:
            phantom_curve.conditional_sharp = True
        else:
            s0 = p1 - p0
            s2 = p3 - p2
            neighbour_i = neighbour_quad.curves.find(quad.curves[i].name)
            bb0, bb2 = CreateSurfacesGlobalList.extract_a0_a3(neighbour_quad, neighbour_i)
            if quad.dirs[i] != neighbour_quad.dirs[neighbour_i]:
                bb0, bb2 = bb2, bb0
            b0 = (bb0 - a0).normalized()
            b2 = (bb2 - a3).normalized()
            for v in (a0, a3, b0, b2, s0, s2):
                if v.length < TH:
                    phantom_curve.conditional_sharp = True
                    break
            else:
                if not are_coplanar(a0, b0, s0) or not are_coplanar(a3, s2, b2):
                    phantom_curve.conditional_sharp = True
                elif b0.cross(s0).length < TH or b2.cross(s2).length < TH:
                    phantom_curve.conditional_sharp = True
                else:
                    temporary_quad.edges_to_calculate[i] = True
                    temporary_quad.set_cs(bb0, bb2, i)
                    ve = self.calculate_ve(quad, i, neighbour_quad, neighbour_i)
                    temporary_quad.desired_directions[i] = -ve

    @staticmethod
    def align_bs(b1: mathutils.Vector,
                 b2: mathutils.Vector) -> Tuple[mathutils.Vector, mathutils.Vector]:
        length = min(b1.length, b2.length)
        if length < TH:
            res = (b1 - b2) / 2
        else:
            print(b1, b2)
            direction = b1.normalized().slerp(-b2.normalized(), 0.5)
            res = direction * length
        return res, -res
    
    def finalise_coefs_calculation_along_segment(self,
                                                 phantom_curve: GregPhantomCurve):
        quad_names = []
        quads = []
        i_s = []
        for quad_id in phantom_curve.quads:
            quad_name = quad_id.name
            quad = self.collection.greg_settings.quads[quad_name]
            i = quad.curves.find(phantom_curve.name)
            quad_names.append(quad_name)
            quads.append(quad)
            i_s.append(i)
        if phantom_curve.source_curve.greg_is_sharp or phantom_curve.conditional_sharp:
            for quad, i, quad_i in zip(quads, i_s, range(2)):
                a0, a3 = CreateSurfacesGlobalList.extract_a0_a3(quad, i)
                a1, a2 = CreateSurfacesGlobalList.calculate_free_coefs(a0, a3, quad_i, phantom_curve, self.collection)
                CreateSurfacesGlobalList.modify_kk(quad, i, a1, a2)
        else:
            b1s = []
            for quad_name, i in zip(quad_names, i_s):
                temporary_quad = self.quads_dict[quad_name]
                b1s.append(temporary_quad.bs[i])
            b1, b2 = CreateSurfacesGlobalList.align_bs(b1s[0], b1s[1])
            phantom_curve.b1_prop = b1
            phantom_curve.b2_prop = b2
            for quad_name, quad, i, b1, quad_i in zip(quad_names, quads, i_s, (b1, b2), range(2)):
                a0, a3 = CreateSurfacesGlobalList.extract_a0_a3(quad, i)
                p0, p1, p2, p3 = CreateSurfacesGlobalList.get_edge_control_points_from_kk(quad, i)
                neighbour_quad = self.get_neighbour_quad(quad, i)
                neighbour_i = neighbour_quad.curves.find(quad.curves[i].name)
                s0 = p1 - p0
                s1 = p2 - p1
                s2 = p3 - p2
                bb0, bb2 = CreateSurfacesGlobalList.extract_a0_a3(neighbour_quad, neighbour_i)
                if quad.dirs[i] != neighbour_quad.dirs[neighbour_i]:
                    bb0, bb2 = bb2, bb0
                b0 = (bb0 - a0).normalized()
                b2 = (bb2 - a3).normalized()
                k0, h0 = get_coefs(b0, s0, a0)
                k1, h1 = get_coefs(b2, s2, a3)
                
                next_curve_no = CreateSurfacesGlobalList.get_curve_name_for_shear(quad,
                                                                    i,
                                                                    self.collection)
                other_curve_no = CreateSurfacesGlobalList.get_curve_name_for_shear(neighbour_quad,
                                                                        neighbour_i,
                                                                        self.collection)
                if (next_curve_no > other_curve_no) or\
                (next_curve_no == other_curve_no and quad_i == 1):
                    try:
                        phantom_curve.invert_shear[quad_i] = True
                    except IndexError:
                        print("quad_i", quad_i)
                b1_corrected = CreateSurfacesGlobalList.get_b1_corrected(b1, quad, i, p0, p1, p2, p3, phantom_curve, quad_i)
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
                CreateSurfacesGlobalList.modify_kk(quad, i, a1, a2)
    
    @staticmethod
    def modify_kk(quad: GregQuad,
                  i: int,
                  a1: mathutils.Vector,
                  a2: mathutils.Vector):
        _, p1, p2, __ = CreateSurfacesGlobalList.get_edge_control_points_from_kk(quad, i)
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
        p0, p1, p2, p3 = CreateSurfacesGlobalList.get_edge_control_points_from_kk(quad, i)
        if phantom_curve.conditional_sharp or phantom_curve.source_curve.greg_is_sharp:
            a0, a3 = CreateSurfacesGlobalList.extract_a0_a3(quad, i)
            a1, a2 = CreateSurfacesGlobalList.calculate_free_coefs(a0, a3, quad_i, phantom_curve, collection)
        else:
            if quad_i == 0:
                b1 = phantom_curve.b1
            else:
                b1 = phantom_curve.b2
            b1_corrected = CreateSurfacesGlobalList.get_b1_corrected(b1, quad, i, p0, p1, p2, p3, phantom_curve, quad_i)
            multiply0 = phantom_curve.coefs_multiply[quad_i][0]
            add0 = mathutils.Vector(phantom_curve.coefs_add[quad_i][0])
            multiply1 = phantom_curve.coefs_multiply[quad_i][1]
            add1 = mathutils.Vector(phantom_curve.coefs_add[quad_i][1])
            a1 = 1/3 * (multiply0 * b1_corrected + add0)
            a2 = 1/3 * (multiply1 * b1_corrected + add1)
        CreateSurfacesGlobalList.modify_kk(quad, i, a1, a2)

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
        a_central = 0.5*(a0 + a3)
        bulge = a_central
        p0, p1, p2, p3 = CreateSurfacesGlobalList.extract_control_points(collection, phantom_curve)
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


    def calculate_kk(self):
        for quad in self.collection.greg_settings.quads:
            self.calculate_coefs(quad)
        for quad in self.collection.greg_settings.quads:
            for i in range(4):
                self.init_coefs_calculation_along_segment(quad, i)
        for temporal_quad in self.quads_dict.values():
            temporal_quad.calculate_bs()
        for phantom_curve in self.collection.greg_settings.phantom_curves:
            if len(phantom_curve.quads) > 0:
                self.finalise_coefs_calculation_along_segment(phantom_curve)

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
        return (CreateSurfacesGlobalList.compare_vecs_all_same_direct(vecs1, vecs2) or
                CreateSurfacesGlobalList.compare_vecs_all_same_reverse(vecs1, vecs2))
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
        co1, handle1, handle2, co2 = CreateSurfacesGlobalList.extract_control_points(self.collection, phantom_curve)
        vecs = (co1, co2, handle1, handle2)
        mirrored_vecs = [mirror_vec(vec, mirror_obj, axis) for vec in vecs]
        if not CreateSurfacesGlobalList.compare_vecs_all_same(vecs, mirrored_vecs):
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
    
    def render_mesh(self, d: "DependantsOfResolution_np", name: str):
        greg_settings = self.collection.greg_settings
        print("len(greg_settings.phantom_bpoints)", len(greg_settings.phantom_bpoints))
        print("len(greg_settings.phantom_curves)", len(greg_settings.phantom_curves))
        for i, bpoint in enumerate(greg_settings.phantom_bpoints):
            coords = bpoint.co
            add_corner(self, coords)
            bpoint.vert = i
        for phantom_curve in greg_settings.phantom_curves:
            verts = add_border(*CreateSurfacesGlobalList.extract_control_points(self.collection, phantom_curve), self, d)
            phantom_curve.first_vert = verts[0]
        for quad in greg_settings.quads:
            self.render_quad(quad, d)
        if self.verts is not None and self.faces is not None:
            if greg_settings.mesh_obj is None:
                mesh = bpy.data.meshes.new(name=name + "_Mesh")
                obj = bpy.data.objects.new(name + "_GeneratedMesh", mesh)
                obj.greg_is_generated = True
                self.collection.objects.link(obj)
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
        CreateSurfacesGlobalList.recalculate_coefs_along_segment(quad, i, collection)
    coords = calc_quad_gregory_verts(quad.kk, quad.kk1, d)
    for i, vert_num in enumerate(range(quad.first_vert, quad.first_vert + (d.nedges-1)*(d.nedges-1))):
        mesh.vertices[vert_num].co = coords[i]
    
def get_not_face_ids(curve_obj: bpy.types.Object) -> List[Set[str]]:
    not_face_ids_groups = []
    for col in curve_obj.users_collection:
        if col.greg_is_not_face:
            not_face_ids = set([obj.greg_curve_settings.name for obj in col.objects])
            not_face_ids_groups.append(not_face_ids)
    return not_face_ids_groups
