from typing import List, Optional, Tuple

import bpy
import mathutils

from .commons import same_coords, add_empty_obj, add_curve_end, add_curve_obj, coplanar_collinear_add_one_end
from .commons import coplanar_collinear, add_hook

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