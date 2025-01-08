from .commons import get_greg_collection
from .DependantsOfResolution import dependants_of_resolution_dict, DependantsOfResolution_np
from .CreateSurfacesGlobalList import render_existing_quad

def cb_update(self, context):
    collection = get_greg_collection(self)
    d = dependants_of_resolution_dict.get(collection.greg_settings.name)
    if d is None:
        d = DependantsOfResolution_np()
        d.conditional_update(collection.greg_settings.nedges)
        dependants_of_resolution_dict[collection.greg_settings.name] = d
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