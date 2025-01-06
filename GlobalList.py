from typing import List, Optional, Tuple
from enum import Enum
from uuid import uuid4

import numpy as np
import numpy.typing as npt

import bpy
import mathutils

from .commons import same_coords, add_hook, add_curve_obj, add_empty_obj, add_curve_end, coplanar_collinear
from .commons import copy_mirrors_from_one_obj_to_another
from .numpyCalculations import add_corner

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

class GlobalList:
    def __init__(self):
        self.reduced_points: List[mathutils.Vector] = []
        self.big_points: List[BigPoint] = []
        self.count = 0
        self.splines: List[Spline] = []
        self.segments: List[Segment] = []
        self.verts: Optional[npt.NDArray[np.float64]] = None
        self.faces: Optional[npt.NDArray[np.int64]] = None

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

    def add_many_curves(self,
                        name: str,
                        source_object: bpy.types.Object,
                        parent_collection: bpy.types.Collection,
                        context: bpy.types.Context):
        collection = bpy.data.collections.new(name)
        collection.greg_settings.used_for_greg = True
        collection.greg_settings.name = generate_collection_name()
        parent_collection.children.link(collection)
        curves_to_copy_mirrors = []
        for segment in self.segments:
            co_s = [p.bpoint.coords for p in (segment.p1, segment.p2)]
            handles_left = [p.handle_left for p in (segment.p1, segment.p2)]
            handles_right = [p.handle_right for p in (segment.p1, segment.p2)]
            curve_obj, curve_prop = add_curve_obj(collection, co_s, handles_left, handles_right)
            curves_to_copy_mirrors.append(curve_obj)
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
        for curve_obj in curves_to_copy_mirrors:
            copy_mirrors_from_one_obj_to_another(source_object, curve_obj)

def generate_collection_name():
    return str(uuid4())