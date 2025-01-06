import bpy
import mathutils

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
    name: bpy.props.StringProperty(default="")
    nedges: bpy.props.IntProperty(default=12)
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