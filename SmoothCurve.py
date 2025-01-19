from typing import Tuple, List, Optional
from itertools import chain

import numpy as np
import numpy.typing as npt

import bpy
import mathutils

INIT_STEP = 0.3
NUM_OF_ITERATIONS = 20

from .commons import extract_end_from_name_and_empty, apply_hook, extract_vectors_from_ends
from .commons import add_hook, set_properties_for_collinears
from .commons import turn_all_collinears
from .propertyGroups import GregCurveEndItem

def get_angles(ve: mathutils.Vector) -> Tuple[float, float]:
    r = ve.length
    th = np.arccos(ve.z / r)
    xy = np.sqrt(ve.x**2 + ve.y**2)
    if xy == 0:
        ph = 0
    else:
        ph = np.sign(ve.y) * np.arccos(ve.x / xy)
    return th, ph

class VecPoint:
    def __init__(self,
                 co: mathutils.Vector,
                 handle_right: mathutils.Vector,
                 handle_left: mathutils.Vector):
        self.co = co
        self.r1 = handle_left.length
        self.r2 = handle_right.length
        vec = handle_right - handle_left
        self.th, self.ph = get_angles(vec)
        self.unit = self.get_unit()
    
    def set_th_ph(self, th, ph):
        self.th = th
        self.ph = ph
        self.unit = self.get_unit()
    
    def get_unit(self) -> mathutils.Vector:
        x = np.sin(self.th) * np.cos(self.ph)
        y = np.sin(self.th) * np.sin(self.ph)
        z = np.cos(self.th)
        return mathutils.Vector((x, y, z))
    
    def get_prev_co(self) -> mathutils.Vector:
        return self.co - self.r1*self.unit

    def get_post_co(self) -> mathutils.Vector:
        return self.co + self.r2*self.unit
    
    def get_diff_th_term(self, r: float, co: mathutils.Vector) -> float:
        unit = self.unit
        rel_vec = co + r*unit
        numx = rel_vec.x * np.cos(self.th) * np.cos(self.ph)
        numy = rel_vec.y * np.cos(self.th) * np.sin(self.ph)
        numz = rel_vec.z * np.sin(self.th)
        return r * (numx + numy + numz) / rel_vec.length
    
    def get_diff_ph_term(self, r: float, co: mathutils.Vector) -> float:
        unit = self.unit
        rel_vec = co + r*unit
        numx = -rel_vec.x * np.sin(self.th) * np.sin(self.ph)
        numy = rel_vec.y * np.sin(self.th) * np.cos(self.ph)
        return r * (numx + numy) / rel_vec.length
    
    def get_diff_th(self, co1: mathutils.Vector, co2: mathutils.Vector) -> float:
        a = co1 - self.co
        b = co2 - self.co
        return self.get_diff_th_term(self.r1, a) + self.get_diff_th_term(-self.r2, b)

    def get_diff_ph(self, co1: mathutils.Vector, co2: mathutils.Vector) -> float:
        a = co1 - self.co
        b = co2 - self.co
        return self.get_diff_ph_term(self.r1, a) + self.get_diff_ph_term(-self.r2, b)

def le(co_start: mathutils.Vector, co_end: mathutils.Vector, points: List[VecPoint]) -> float:
    total = 0
    prev = co_start
    for point in points:
        total += (point.get_prev_co() - prev).length
        prev = point.get_post_co()
    total += len(co_end - prev)
    return total

def set_angles(points: List[VecPoint],
               x: npt.NDArray):
    for i, point in enumerate(points):
        th = x[2 * i]
        ph = x[2 * i + 1]
        point.set_th_ph(th, ph)

def get_all_angles(points: List[VecPoint]):
    return np.array(list(chain(*((p.th, p.ph) for p in points))))

def get_all_diffs(co_start: mathutils.Vector, co_end: mathutils.Vector, points: List[VecPoint]):
    prev = co_start
    post = points[1].get_prev_co()
    res = []
    for i, point in enumerate(points):
        if i + 1 == len(points):
            post = co_end
        else:
            post = points[i + 1].get_prev_co()
        res.extend([point.get_diff_th(prev, post), point.get_diff_ph(prev, post)])
        prev = point.get_post_co()
    return np.array(res)

def gradient_descent_step(points:List[VecPoint],
                          step: Optional[float] = None,
                          co_start: Optional[mathutils.Vector] = None,
                          co_end: Optional[mathutils.Vector] = None,
                          xnm1: Optional[npt.NDArray] = None,
                          dxnm1: Optional[npt.NDArray] = None):
    x = get_all_angles(points)
    if co_start is None:
        co_start = points[-1].get_post_co()
    if co_end is None:
        co_start = points[0].get_prev_co()
    dx = get_all_diffs(co_start, co_end, points)
    if step is None:
        step = get_step(x, xnm1, dx, dxnm1)
    xnp1 = x - step*dx
    set_angles(points, xnp1)
    return x, dx

def gradient_descent_steps(points: List[VecPoint],
                           num_steps: int,
                           init_step: float,
                           co_start: Optional[mathutils.Vector] = None,
                           co_end: Optional[mathutils.Vector] = None):

    xnm1, dxnm1 = gradient_descent_step(points, step=init_step, co_start=co_start, co_end=co_end)
    for _ in range(num_steps):
        xnm1, dxnm1 = gradient_descent_step(points, co_start=co_start, co_end=co_end, xnm1=xnm1, dxnm1=dxnm1)


def get_step(x, xnm1, dx, dxnm1):
    diff_d = dx - dxnm1
    return np.dot(x - xnm1, diff_d)/np.dot(diff_d, diff_d)

def do_steps(curve0: bpy.types.Object,
             selected: List[bpy.types.Object],
             i: int,
             direct: bool):
    curves: List[CurveAndDirection] = []
    res: bool | None | CurveAndFreeI = True #placeholder
    last_curve = None
    curve = curve0
    while res is not None and last_curve != curve0:
        res = move_forward(curve, selected, i)
        if res == False:
            return False
        elif res is not None:
            curve = res.curve
            last_curve = res.curve
            i = res.free_i
            dir_this_curve = (i == 1)
            dir_to_return = (dir_this_curve == direct)
            curve_with_direction = CurveAndDirection(curve, dir_to_return)
            curves.append(curve_with_direction)
    return curves

def return_line_or_circle_from_selected_if_possible(selected: List[bpy.types.Object]):
    curve0 = selected[0]
    res = do_steps(curve0, selected, 1, True)
    if res == False:
        return False
    if len(res) > 0:
        if res[-1] == curve0:
            cyclic = True
            return res, cyclic
    cyclic = False
    res2 = do_steps(curve0, selected, 0, False)
    if res2 == False:
        return False
    first = CurveAndDirection(curve0, True)
    print([c.curve.name for c in chain(reversed(res2), [first], res)])
    return list(chain(reversed(res2), [first], res)), cyclic



def get_neighbors_with_direction(curve: bpy.types.Object,
                                 selected: List[bpy.types.Object],
                                 i: int):
    if i == 0:
        empty = curve.greg_curve_settings.end1_empty
    else:
        empty = curve.greg_curve_settings.end2_empty
    return get_selected_neighbour_curves(curve, empty, selected)

def move_forward(curve: bpy.types.Object,
                 selected: List[bpy.types.Object],
                 i: int):
    curves_neighbors = get_neighbors_with_direction(curve, selected, i)
    if len(curves_neighbors) == 1:
        return curves_neighbors[0]
    elif len(curves_neighbors) == 0:
        return None
    else:
        return False

class CurveAndFreeI:
    def __init__(self, curve: bpy.types.Object, free_i: int):
        self.curve = curve
        self.free_i = free_i
    
class CurveAndDirection:
    def __init__(self, curve: bpy.types.Object, direction: bool):
        self.curve = curve
        self.direction = direction

def get_all_neighbour_curves(curve: bpy.types.Object,
                             empty: bpy.types.Object):
    ends = empty.greg_empty_settings.curve_ends
    return [CurveAndFreeI(end.basic_end.curve, 1-end.basic_end.end)
            for end in ends if end.basic_end.curve != curve]

def get_selected_neighbour_curves(curve: bpy.types.Object,
                                  empty: bpy.types.Object, 
                                  selected: List[bpy.types.Object]):
    curves_neighbors = get_all_neighbour_curves(curve, empty)
    return [curve_and_free_i for curve_and_free_i in curves_neighbors
            if curve_and_free_i.curve in selected]

def get_empty_and_ends(curve_and_dir1: CurveAndDirection,
                       curve_and_dir2: CurveAndDirection):
    gs1 = curve_and_dir1.curve.greg_curve_settings
    empty = gs1.end2_empty if curve_and_dir1.direction else gs1.end1_empty
    gs2 = curve_and_dir2.curve.greg_curve_settings
    end1_name = gs1.end2_name if curve_and_dir1.direction else gs1.end1_name
    end2_name = gs2.end1_name if curve_and_dir2.direction else gs2.end2_name
    print("dir1", curve_and_dir1.direction)
    print("dir2", curve_and_dir2.direction)
    print("empty", empty.greg_empty_settings.name)
    print("empty ends", [end.name for end in empty.greg_empty_settings.curve_ends])
    print("end1_1_name", gs1.end1_name)
    print("end1_2_name", gs1.end2_name)
    print("end2_1_name", gs2.end1_name)
    print("end2_2_name", gs2.end2_name)
    print("end1_name", end1_name)
    print("end2_name", end2_name)
    ends = [extract_end_from_name_and_empty(end_name, empty) for end_name in (end1_name, end2_name)]
    return empty, ends



def apply_handles_from_points_to_curves(points: List[VecPoint],
                                        curves: List[CurveAndDirection],
                                        is_cyclic: bool,
                                        context: bpy.types.Context):
    if is_cyclic:
        curves.append(curves[0])
    for i, point in enumerate(points):
        curve_and_dir1 = curves[i]
        curve_and_dir2 = curves[i + 1]
        apply_handles_point_two_curves(point, curve_and_dir1, curve_and_dir2, context)

def apply_handles_point_two_curves(point: VecPoint,
                                   curve1: CurveAndDirection,
                                   curve2: CurveAndDirection,
                                   context: bpy.types.Context):
    handle1 = point.get_prev_co()
    handle2 = point.get_post_co()
    curves = (curve1, curve2)
    splines = [curve.curve.data.splines[0] for curve in curves]
    directions = [curve.direction for curve in curves]
    directions[1] = not directions[1]
    handles = (handle1, handle2)
    for spline, dir, (h1, h2) in zip(splines, directions, (handles, reversed(handles))):
        if dir:
            spline.bezier_points[1].handle_left = h1
            spline.bezier_points[1].handle_right = h2
        else:
            spline.bezier_points[0].handle_left = h2
            spline.bezier_points[0].handle_right = h1
    ends: List[GregCurveEndItem] = []
    for curve_and_dir, dir in zip(curves, directions):
        gs = curve_and_dir.curve.greg_curve_settings
        if dir:
            empty = gs.end2_empty
            name = gs.end2_name
        else:
            empty = gs.end1_empty
            name = gs.end1_name
        end = extract_end_from_name_and_empty(name, empty)
        ends.append(end)
    if ends[0].name not in ends[1].collinear_to:
        set_properties_for_collinears(ends, empty)
    for end in ends:
        turn_all_collinears(end)
        add_hook(end, context)
    

class OBJECT_OT_smooth_curves(bpy.types.Operator):
    """Gregory: smooth curves by recalculating handles"""
    bl_idname = "object.greg_smooth_curves"
    bl_label = "Greg: smooth curves"         # Display name in the interface.
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
        res = return_line_or_circle_from_selected_if_possible(selected)
        if res == False:
            return False
        return True

    def execute(self, context: bpy.types.Context):        # execute() is called when running the operator.
        selected = context.selected_objects
        line, is_cyclic = return_line_or_circle_from_selected_if_possible(selected)
        if is_cyclic:
            pairs_of_curves = zip(line, chain(line[1:], [line[0]]))
            co_start = None
            co_end = None
        else:
            pairs_of_curves = zip(line[:-1], line[1:])
            first_last_curves_with_directions = (line[0], line[-1])
            gss = (obj.curve.greg_curve_settings for obj in first_last_curves_with_directions)
            directions = [obj.direction for obj in first_last_curves_with_directions]
            directions[1] = not directions[1]
            names_empties = [(gs.end1_name, gs.end1_empty) if dir else (gs.end2_name, gs.end2_empty) for gs, dir in zip(gss, directions)]
            ends = [extract_end_from_name_and_empty(*name_and_empty) for name_and_empty in names_empties]
            for end in ends:
                apply_hook(end, context)
                add_hook(end, context)
            co_s = (name_and_empty[1].matrix_world.translation for name_and_empty in names_empties)
            handles = extract_vectors_from_ends(ends)
            co_start, co_end = (co + handle for co, handle in zip(co_s, handles))
        points = []
        for i, (curve_and_dir1, curve_and_dir2) in enumerate(pairs_of_curves):
            empty, ends = get_empty_and_ends(curve_and_dir1, curve_and_dir2)
            for end in ends:
                apply_hook(end, context)
            handles = extract_vectors_from_ends(ends)
            co = empty.matrix_world.translation
            points.append(VecPoint(co, handles[1], handles[0]))
        gradient_descent_steps(points, NUM_OF_ITERATIONS, INIT_STEP, co_start, co_end)
        apply_handles_from_points_to_curves(points, line, is_cyclic, context)
        return {'FINISHED'}

def add_smooth_curve_func(self, context: bpy.types.Context):
    self.layout.operator(OBJECT_OT_smooth_curves.bl_idname)
