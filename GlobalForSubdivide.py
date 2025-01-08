from itertools import chain
from typing import Dict, List, Tuple, Optional

import bpy
import mathutils

from .propertyGroups import GregCurveEndItem
from .commons import same_coords, mirror_vec, mirror_vec_with_vec, are_collinear, add_hook
from .commons import copy_mirrors_from_one_obj_to_another, TH, TH2, add_curve_end, add_curve_obj, add_empty_obj
from .commons import get_approx_bezier_length, make_collinear, check_curve_crosses_mirror
from .commons import make_curve_mirror_bridge, coplanar_collinear, coplanar_collinear_add_one_end
from .commons import extract_end_name_from_curve_and_i, are_coplanar
from .GridFill import XY, grid_fill

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

    
    def add_curve_and_mirrors(self, curve_obj: bpy.types.Object):
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
        return optimal_quad

    def add_optimal_quad(self, optimal_quad: List[int]):
        self.optimal_quad = optimal_quad
        self.border0 = self.borders[self.optimal_quad[0]][self.optimal_quad[1]][1]
        self.border1 = self.borders[self.optimal_quad[1]][self.optimal_quad[2]][1]
        self.border2 = self.borders[self.optimal_quad[3]][self.optimal_quad[2]][1]
        self.border3 = self.borders[self.optimal_quad[0]][self.optimal_quad[3]][1]
        self.edge_points = [self.borders[self.optimal_quad[0]][self.optimal_quad[1]][2],
                            self.borders[self.optimal_quad[1]][self.optimal_quad[2]][2],
                            self.borders[self.optimal_quad[3]][self.optimal_quad[2]][2],
                            self.borders[self.optimal_quad[0]][self.optimal_quad[3]][2]]


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
    def get_handle_other_and_outer(point: "PointForSubdivide", curve_prev: "CurveForSubdivide", curve_post: "CurveForSubdivide"):
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
    def get_handles_left_right(point: "PointForSubdivide",
                               curve_prev: "CurveForSubdivide",
                               curve_post: "CurveForSubdivide") -> Tuple[mathutils.Vector, mathutils.Vector]:
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
    def get_outer_handles_left_right(point: "PointForSubdivide",
                                     curve_prev: "CurveForSubdivide",
                                     curve_post: "CurveForSubdivide") -> Tuple[mathutils.Vector, mathutils.Vector]:
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
    def populate_handles(border: List["CurveForSubdivide"],
                         handles: List[Optional[mathutils.Vector]]) -> List[mathutils.Vector]:
        i = 0
        count = 1
        while i < (len(handles) - 1):
            p1 = handles[i]
            count = 1
            p2 = handles[i + count]
            while p2 is None:
                count += 1
                p2 = handles[i + count]
            curves = border[i:i + count]
            curves_lengths = [curve.get_approx_length() for curve in curves]
            total_length = sum(curves_lengths)
            now_length = 0
            for j, curve_length in zip(range(i + 1, i + count), curves_lengths[:-1]):
                now_length += curve_length
                factor = now_length / total_length
                handles[j] = p1.lerp(p2, factor)
            i += count
        return handles
    
    def extract_coords(self, i: int):
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
    
    def get_handles_right_left(self, i: int, right: bool):
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
    
    def adjust_cross_handles_lengths(self,
                                     i: int, #must be 0, 1, 2, or 3 
                                     border: List["CurveForSubdivide"],
                                     handles: List[Optional[mathutils.Vector]]):
        edge_points = self.edge_points[i]
        other_edge_points = self.edge_points[(i + 2) % 4]
        curves_lengths = [curve.get_approx_length() for curve in border]
        total_length = sum(curves_lengths)
        corner_handles_lengths = [handles[ind].length for ind in (0, -1)]
        corner_edges_lengths = [(edge_points[ind].co - other_edge_points[ind].co).length for ind in (0, -1)]
        handles_adjusted_lengths = [handle_len / edge_len for handle_len, edge_len in zip(corner_handles_lengths, corner_edges_lengths)]
        this_length = 0
        for curve_length, i in zip(curves_lengths[:-1], range(1, len(handles)-1)):
            this_length += curve_length
            handle = handles[i]
            if handle is not None:
                mixing_factor = this_length / total_length
                this_edge_len = (edge_points[i].co - other_edge_points[i].co).length
                this_handle_length = (handles_adjusted_lengths[1] * mixing_factor +
                                      handles_adjusted_lengths[0] * (1 - mixing_factor)) * this_edge_len
                handles[i] = handle.normalized() * this_handle_length

    
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
        borders = (self.border0, self.border1, self.border2, self.border3)
        cross_hs_candidates = [self.extract_handles(i) for i in range(4)]
        for i, (cross_hs_candidate, border) in enumerate(zip(cross_hs_candidates, borders)):
            self.adjust_cross_handles_lengths(i, border, cross_hs_candidate)
        cross_hs = [GlobalForSubdivide.populate_handles(border, cross_hs_candidates[i]) for i, border in enumerate(borders)]
        cross_hs = self.adjust_if_border_on_mirror(cross_hs)
        outer_hs = [GlobalForSubdivide.populate_handles(border, self.extract_handles(i, True)) for i, border in enumerate(borders)]
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
        copy_mirrors_from_one_obj_to_another(obj_for_mirror, curve_obj)
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

    def is_collinear(self, corner: int, is_horizontal: bool) -> bool:
        # corner must be 0, 1, 2, or 3
        corner_point_num = self.optimal_quad[corner]
        corner_point = self.points[corner_point_num]
        other_corner = (corner + 1) % 4 if (is_horizontal and (corner % 2 == 0) or 
                                            ((not is_horizontal) and (corner % 2 == 1))) else (corner - 1) % 4
        # if horizontal: 0->1, 1->0, 2->3, 3->2
        # if vertical: 0->3, 3->0, 1->2, 2->1
        border = self.extract_border_from_two_corner_nums(corner, other_corner)
        curve = border[1][0]
        if curve.points[0] == corner_point:
            end = 0
        elif curve.points[1] == corner_point:
            end = 1
        else:
            raise Exception("Curve is not connected to the point!")
        end_name = extract_end_name_from_curve_and_i(curve.original_curve, end)
        end: GregCurveEndItem = corner_point.empty.greg_empty_settings.curve_ends[end_name]
        return end.is_collinear
    
    def extract_border_from_two_corner_nums(self, corner1: int, corner2: int):
        # corners must be 0, 1, 2, or 3
        point_nums = [self.optimal_quad[corner] for corner in (corner1, corner2)]
        return self.borders[point_nums[0]][point_nums[1]]
    
    def adjust_free_outers_border(self, is_horizontal: bool, is_first: bool):
        if is_horizontal:
            range_max = self.xtot - 1
            if is_first:
                corners = (0, 1)
            else:
                corners = (3, 2)
        else:
            range_max = self.ytot - 1
            if is_first:
                corners = (0, 3)
            else:
                corners = (1, 2)
        border = self.extract_border_from_two_corner_nums(*corners)
        is_horizontal_to_cross = not(is_horizontal)
        if self.is_collinear(corners[0], is_horizontal_to_cross) and self.is_collinear(corners[1], is_horizontal_to_cross):
            for coord in range(1, range_max):
                original_empty = border[2][coord].empty
                if len(original_empty.greg_empty_settings.curve_ends) == 2:
                    if is_horizontal:
                        if is_first:
                            midpoint = self.points_grid[0][coord]
                            handle_up = midpoint.handle_up
                            handle_down = midpoint.handle_down
                            midpoint.handle_up = -handle_down.normalized() * handle_up.length
                        else:
                            midpoint = self.points_grid[self.ytot - 1][coord]
                            handle_up = midpoint.handle_up
                            handle_down = midpoint.handle_down
                            midpoint.handle_down = -handle_up.normalized() * handle_down.length
                    else:
                        if is_first:
                            midpoint = self.points_grid[coord][0]
                            handle_left = midpoint.handle_left
                            handle_right = midpoint.handle_right
                            midpoint.handle_left = -handle_right.normalized() * handle_left.length
                        else:
                            midpoint = self.points_grid[coord][self.xtot - 1]
                            handle_left = midpoint.handle_left
                            handle_right = midpoint.handle_right
                            midpoint.handle_right = -handle_left.normalized() * handle_right.length
        else:
            for coord in range(1, range_max):
                point_for_subdivide = border[2][coord]
                original_empty = point_for_subdivide.empty
                if len(original_empty.greg_empty_settings.curve_ends) == 2:
                    two_handles = (point_for_subdivide.handle_left, point_for_subdivide.handle_left)
                    if is_horizontal:
                        if is_first:
                            midpoint = self.points_grid[0][coord]
                            handle_to_correct = midpoint.handle_up
                            three_handles = (midpoint.handle_down, *two_handles)
                            res = GlobalForSubdivide.make_vector_coplanar_to_three_if_possible(handle_to_correct, three_handles)
                            if res is not None:
                                midpoint.handle_up = res
                        else:
                            midpoint = self.points_grid[self.ytot - 1][coord]
                            handle_to_correct = midpoint.handle_down
                            three_handles = (midpoint.handle_up, *two_handles)
                            res = GlobalForSubdivide.make_vector_coplanar_to_three_if_possible(handle_to_correct, three_handles)
                            if res is not None:
                                midpoint.handle_down = res
                    else:
                        if is_first:
                            midpoint = self.points_grid[coord][0]
                            handle_to_correct = midpoint.handle_left
                            three_handles = (midpoint.handle_right, *two_handles)
                            res = GlobalForSubdivide.make_vector_coplanar_to_three_if_possible(handle_to_correct, three_handles)
                            if res is not None:
                                midpoint.handle_left = res
                        else:
                            midpoint = self.points_grid[coord][self.xtot - 1]
                            handle_to_correct = midpoint.handle_right
                            three_handles = (midpoint.handle_left, *two_handles)
                            res = GlobalForSubdivide.make_vector_coplanar_to_three_if_possible(handle_to_correct, three_handles)
                            if res is not None:
                                midpoint.handle_right = res
    
    @staticmethod
    def make_vector_coplanar_to_three_if_possible(handle_to_correct: mathutils.Vector,
                                                  three_handles: Tuple[mathutils.Vector,
                                                                       mathutils.Vector,
                                                                       mathutils.Vector]) -> Optional[mathutils.Vector]:
        if are_coplanar(*three_handles):
            perpendicular = three_handles[0].cross(three_handles[1])
            if perpendicular.length < TH:
                perpendicular = three_handles[0].cross(three_handles[2])
            corrected_handle = (handle_to_correct - handle_to_correct.project(perpendicular))
            corrected_handle = corrected_handle.normalized() * handle_to_correct.length
            return corrected_handle
        return None
    
    def adjust_free_outers(self):
        for is_horizontal in (True, False):
            for is_first in (True, False):
                self.adjust_free_outers_border(is_horizontal, is_first)


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
    
    def get_approx_length(self):
        bezier_points = self.original_curve.data.splines[0].bezier_points
        co_s = [point.co for point in bezier_points]
        handles = (bezier_points[0].handle_right - co_s[0], bezier_points[1].handle_left - co_s[1])
        return get_approx_bezier_length(co_s, handles)

class MirrorSequenceItem:
    def __init__(self, mirror_object: bpy.types.Object, axis: int):
        self.mirror_object: bpy.types.Object = mirror_object
        self.axis: int = axis

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
