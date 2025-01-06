from typing import List

import numpy as np
import numpy.typing as npt

import mathutils

from .DependantsOfResolution import DependantsOfResolution_np
from .GlobalList import GlobalList

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
        glist.verts = np.vstack((glist.verts, np.array([coords])))

def add_border(k0: mathutils.Vector,
                k1: mathutils.Vector,
                k2: mathutils.Vector,
                k3: mathutils.Vector,
                glist: "GlobalList",
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
                        glist: "GlobalList") -> None:
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
