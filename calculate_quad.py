from typing import List, Optional
from itertools import chain

import numpy as np
import numpy.typing as npt

import mathutils

from .commons import get_angles

'''# to delete
def get_angles(ve: mathutils.Vector) -> Tuple[float, float]:
    r = ve.length
    th = np.arccos(ve.z / r)
    xy = np.sqrt(ve.x**2 + ve.y**2)
    if xy == 0:
        ph = 0
    else:
        ph = np.sign(ve.y) * np.arccos(ve.x / xy)
    return th, ph

#######'''

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

def get_d1(f0p: mathutils.Vector,
           f1m: mathutils.Vector,
           w2: mathutils.Vector,
           w4: mathutils.Vector) -> mathutils.Vector:
    return -3*w2 + 3*w4 + 1.125*f0p + 1.125*f1m

def get_d2(w2: List[mathutils.Vector],
           w3: List[mathutils.Vector],
           w4: List[mathutils.Vector],
           fms: List[mathutils.Vector],
           fps: List[mathutils.Vector],
           i: int,
           j: int):
    return 6*w2[j] + 6*w3[j] - 12*w4[j]\
          + 4.5*fms[i]\
          - 9.0*fps[i]\
          - 9.0*fms[(i + 1) % 4]\
          + 4.5*fps[(i + 1) % 4]\
          + 1.5*fms[(i + 2) % 4]\
          + 0.75*fps[(i + 2) % 4]\
          + 0.75*fms[(i + 3) % 4]\
          + 1.5*fps[(i + 3) % 4]

def get_f0p_f1m(b1: mathutils.Vector,
                w5: mathutils.Vector,
                w6: mathutils.Vector,
                e0p: mathutils.Vector,
                e1m: mathutils.Vector,
                k0: float,
                k1: float):
    f0p = e0p + 1/3 * (2*k0*b1 + w5)
    f1m = e1m + 1/3 * (2*k1*b1 + w6)
    return f0p, f1m

def dfp_db(k0: List[float],
           j: int):
    return 2/3 * k0[j]

def dfm_db(k1: List[float],
           i: int,
           js: List[int]):
    return 2/3 * k1[js[(i - 1) % 4]]

def dd1_df():
    return 1.125

def dd2_dfm(i_d2: int, i_fm: int) -> float:
    if i_d2 == i_fm:
        return 4.5
    elif ((i_d2 + 1) % 4) == i_fm:
        return -9
    elif ((i_d2 + 2) % 4) == i_fm:
        return 1.5
    elif ((i_d2 + 3) % 4) == i_fm:
        return 0.75
    else:
        raise Exception(f"not correct inputs! i_d2 is {i_d2} and i_fm is {i_fm}")

def dd2_dfp(i_d2: int, i_fp: int):
    if i_d2 == i_fp:
        return -9
    elif ((i_d2 + 1) % 4) == i_fp:
        return 4.5
    elif ((i_d2 + 2) % 4) == i_fp:
        return 0.75
    elif ((i_d2 + 3) % 4) == i_fp:
        return 1.5
    else:
        raise Exception(f"not correct inputs! i_d2 is {i_d2} and i_fp is {i_fp}")

def dcurv_d1(d1: mathutils.Vector,
             d2: mathutils.Vector,
             xyz: int, # xyz must be 0, 1 or 2
             dcrossdot: float,
             d1dot: float,
             d1_denom: float) -> float:
    x = xyz
    y = (xyz + 1) % 3
    z = (xyz + 2) % 3
    d1x = d1[x]
    d1y = d1[y]
    d1z = d1[z]
    d2x = d2[x]
    d2y = d2[y]
    d2z = d2[z]

    return (-3*d1x*dcrossdot + (d2y*(d1x*d2y - d1y*d2x) + d2z*(d1x*d2z - d1z*d2x))*d1dot) / d1_denom

def dcurv_d2(d1: mathutils.Vector,
             d2: mathutils.Vector,
             xyz: int, # xyz must be 0, 1 or 2
             d2_denom: float) -> float:
    x = xyz
    y = (xyz + 1) % 3
    z = (xyz + 2) % 3
    d1x = d1[x]
    d1y = d1[y]
    d1z = d1[z]
    d2x = d2[x]
    d2y = d2[y]
    d2z = d2[z]
    return (d1y*(d1y*d2x - d1x*d2y) + d1z*(d1z*d2x - d1x*d2z)) / d2_denom

def get_curv(d1: mathutils.Vector,
             d2: mathutils.Vector):
    return (d1.cross(d2)).length / (d1.length ** 3)


def get_d1s_d2s(bs: List[mathutils.Vector],
                fps_predefined: List[Optional[mathutils.Vector]],
                fms_predefined: List[Optional[mathutils.Vector]],
                js: List[int],
                edges_to_calculate: List[bool],
                w2: List[mathutils.Vector],
                w3: List[mathutils.Vector],
                w4: List[mathutils.Vector],
                w5: List[mathutils.Vector],
                w6: List[mathutils.Vector],
                ems: List[mathutils.Vector],
                eps: List[mathutils.Vector],
                k0s: List[float],
                k1s: List[float]):
    fps: List[Optional[mathutils.Vector]] = fps_predefined[:]
    fms: List[Optional[mathutils.Vector]] = fms_predefined[:]
    d1s: List[mathutils.Vector] = []
    for i in range(4):
        if edges_to_calculate[i]:
            j = js[i]
            f0p, f1m, = get_f0p_f1m(bs[j],
                                    w5[j],
                                    w6[j],
                                    eps[i],
                                    ems[(i + 1) % 4],
                                    k0s[j],
                                    k1s[j])
            fps[i] = f0p
            fms[(i + 1) % 4] = f1m
            d1s.append(get_d1(f0p, f1m, w2[j], w4[j]))
    
    d2s: List[mathutils.Vector] = []
    for i in range(4):
        if edges_to_calculate[i]:
            j = js[i]
            d2s.append(get_d2(w2, w3, w4, fms, fps, i, j))
    return d1s, d2s
    

def direct(bs: List[mathutils.Vector],
           w2: List[mathutils.Vector],
           w3: List[mathutils.Vector],
           w4: List[mathutils.Vector],
           w5: List[mathutils.Vector],
           w6: List[mathutils.Vector],
           ems: List[mathutils.Vector],
           eps: List[mathutils.Vector],
           k0s: List[float],
           k1s: List[float],
           angle_matrices: List[mathutils.Matrix],
           ideal_res: npt.NDArray,
           js: List[int],
           edges_to_calculate: List[float],
           fps_predefined: List[Optional[mathutils.Vector]],
           fms_predefined: List[Optional[mathutils.Vector]]):
    d1s, d2s = get_d1s_d2s(bs, fps_predefined, fms_predefined, js, edges_to_calculate, w2, w3, w4, w5, w6, ems, eps, k0s, k1s)
    ths: List[float] = []
    phs: List[float] = []
    curvs: List[float] = []
    for i in range(len(d1s)):
        th, ph = get_angles(angle_matrices[i] @ d1s[i])
        ths.append(th)
        phs.append(ph)
        curvs.append(get_curv(d1s[i], d2s[i]))
    direct_res = np.array(list(chain(ths, phs, curvs))).T - ideal_res
    return d1s, d2s, direct_res

def dd_db(k0: List[float],
          k1: List[float],
          js: List[int],
          edges_to_calculate: List[bool]):
    dd1_df_v = dd1_df()
    dd1_dbs: List[float] = []
    dd2_dbs: List[List[float]] = []
    for i_b in range(4):
        if edges_to_calculate[i_b]:
            i_fm = (i_b + 1) % 4
            i_fp = i_b
            j_b = js[i_b]
            dfm_db_v = dfm_db(k1, i_b, js)
            dfp_db_v = dfp_db(k0, j_b)
            dd1_db = dd1_df_v*(dfm_db_v + dfp_db_v)
            dd1_dbs.append(dd1_db)
            dd2_dbs.append([])
            for i_curv in range(4):
                if edges_to_calculate[i_curv]:
                    dd2_dfm_v = dd2_dfm(i_curv, i_fm)
                    dd2_dfp_v = dd2_dfp(i_curv, i_fp)
                    dd2_db = dd2_dfm_v*dfm_db_v + dd2_dfp_v*dfp_db_v
                    dd2_dbs[-1].append(dd2_db)
    return dd1_dbs, dd2_dbs

def dth_ph_dd1(d1: mathutils.Vector):
    x = d1.x
    y = d1.y
    z = d1.z
    xysq = x**2 + y**2
    xy = np.sqrt(xysq)
    rsq = d1.length_squared
    denom = xy * rsq
    dth_dx = x * z / denom
    dth_dy = y * z / denom
    dth_dz = - xy / rsq
    dph_dx = -y / xysq
    dph_dy = x / xysq
    dph_dz = 0
    dth_dd1 = mathutils.Vector((dth_dx, dth_dy, dth_dz))
    dph_dd1 = mathutils.Vector((dph_dx, dph_dy, dph_dz))
    return dth_dd1, dph_dd1

def deriv_th_ph(d1s: List[mathutils.Vector],
                angle_matrices: List[mathutils.Matrix],
                angle_matrices_T: List[mathutils.Matrix],
                dd1_dbs: List[float]):
    number_of_bs = len(d1s)
    res = [[0] * number_of_bs*3 for _ in range(number_of_bs*2)]
    for i in range(len(d1s)):
        d1_converted = angle_matrices[i] @ d1s[i]
        dth_dd1, dph_dd1 = dth_ph_dd1(d1_converted)
        dth_dd1_v = angle_matrices_T[i] @ dth_dd1
        dph_dd1_v = angle_matrices_T[i] @ dph_dd1
        dth_db = dth_dd1_v * dd1_dbs[i]
        dph_db = dph_dd1_v * dd1_dbs[i]
        res[i][i * 3] = dth_db.x
        res[i][i * 3 + 1] = dth_db.y
        res[i][i * 3 + 2] = dth_db.z
        res[number_of_bs + i][i * 3] = dph_db.x
        res[number_of_bs + i][i * 3 + 1] = dph_db.y
        res[number_of_bs + i][i * 3 + 2] = dph_db.z
    return np.array(res)

def deriv_curv(d1s: List[mathutils.Vector],
               d2s: List[mathutils.Vector],
               dd1_dbs: List[float],
               dd2_dbs: List[List[float]]):
    number_of_bs = len(d1s)
    res: List[List[float]] = [[0]*number_of_bs*3 for _ in range(number_of_bs)]
    for i_curv in range(number_of_bs):
        d1 = d1s[i_curv]
        d2 = d2s[i_curv]
        dcross = d1.cross(d2)
        dcrossdot = dcross.dot(dcross)
        d1dot = d1.dot(d1)
        sqrt_dcrossdot = np.sqrt(dcrossdot)
        d1_denom = (d1dot**(5/2) * sqrt_dcrossdot)
        d2_denom = (d1dot**(3/2) * sqrt_dcrossdot)
        for xyz in range(3):
            dcurv_d1_v = dcurv_d1(d1, d2, xyz, dcrossdot, d1dot, d1_denom)
            dcurv_d2_v = dcurv_d2(d1, d2, xyz, d2_denom)
            for i_b in range(number_of_bs):
                dd2_db = dd2_dbs[i_b][i_curv]
                if i_curv == i_b:
                    dd1_db = dd1_dbs[i_b]
                    res[i_curv][i_b*3 + xyz] = dcurv_d1_v*dd1_db + dcurv_d2_v*dd2_db
                else:
                    res[i_curv][i_b*3 + xyz] = dcurv_d2_v*dd2_db
    return np.array(res)

def jacobian_step(state: npt.NDArray,
                  jacobian: npt.NDArray,
                  direct_res: npt.NDArray):
    new_state = state - np.linalg.inv(jacobian)@direct_res
    return new_state

def get_coordinate_systems(starter_d1s: List[mathutils.Vector],
                           desired_directions: List[mathutils.Vector],
                           js: List[int],
                           edges_to_calculate: List[bool]):
    coord_matrices: List[mathutils.Matrix] = []
    ths_ideal = []
    phs_ideal = []
    for i in range(4):
        if edges_to_calculate[i]:
            j = js[i]
            dir1 = starter_d1s[j].normalized()
            dir2 = desired_directions[j].normalized()
            destination = dir1.slerp(dir2, 0.5).normalized()
            x = destination.slerp(dir1, 0.5).normalized()
            z = dir1.cross(destination).normalized()
            y = -x.cross(z).normalized()
            mat = mathutils.Matrix((x, y, z))
            coord_matrices.append(mat)
            th_ideal, ph_ideal = get_angles(mat @ destination)
            print("th_ideal, ph_ideal", i, th_ideal, ph_ideal)
            ths_ideal.append(th_ideal)
            phs_ideal.append(ph_ideal)
    return coord_matrices, ths_ideal, phs_ideal

def get_state_from_bs(bs: List[mathutils.Vector]):
    res = list(chain(*((b.x, b.y, b.z) for b in bs)))
    return np.array(res).T

def get_bs_from_state(state: npt.NDArray):
    len_bs = len(state) // 3
    return [mathutils.Vector(state[i*3: i*3 + 3].T) for i in range(len_bs)]

def calc_curv_bezier(p0: mathutils.Vector,
                     p1: mathutils.Vector,
                     p2: mathutils.Vector):
    d1 = 3 * (p1 - p0)
    d2 = 6 * (p0 + p2 - 2*p1)
    curv = get_curv(d1, d2)
    return curv

def calc_desired_curvature(ps: List[mathutils.Vector],
                           ems: List[mathutils.Vector],
                           eps: List[mathutils.Vector],
                           i: int):
    p0 = ps[i]
    p1 = ems[i]
    p2 = eps[(i + 3) % 4]
    curv1 = calc_curv_bezier(p0, p1, p2)
    p0 = ps[(i + 1) % 4]
    p1 = eps[(i + 1) % 4]
    p2 = ems[(i + 2) % 4]
    curv2 = calc_curv_bezier(p0, p1, p2)
    return (curv1 + curv2) / 2

def calc_ideal_curvatires(desired_curvatures: List[float],
                          d1s: List[mathutils.Vector],
                          d2s: List[mathutils.Vector]) -> List[float]:
    now_curvs = [get_curv(d1, d2) for d1, d2 in zip(d1s, d2s)]
    return [min(nc, (dc + nc) / 2) for dc, nc in zip(desired_curvatures, now_curvs)]
    #return [(dc + nc) / 2 for dc, nc in zip(desired_curvatures, now_curvs)]


def calc_bs(ps: List[mathutils.Vector],
            ems: List[mathutils.Vector],
            eps: List[mathutils.Vector],
            cms: List[Optional[mathutils.Vector]],
            cps: List[Optional[mathutils.Vector]],
            edges_to_calculate: List[bool],
            desired_directions: List[mathutils.Vector]): # end directions and curvature radii
                                              # will be average between initial and desired
    w2: List[mathutils.Vector] = []
    w3: List[mathutils.Vector] = []
    w4: List[mathutils.Vector] = []
    w5: List[mathutils.Vector] = []
    w6: List[mathutils.Vector] = []
    k0s: List[float] = []
    k1s: List[float] = []
    bs = []
    fms_predefined = [None] * 4
    fps_predefined = [None] * 4
    js = [0] * 4
    counter = 0
    for i in range(4):
        a0 = ems[i] - ps[i]
        a3 = eps[(i + 1) % 4] - ps[(i + 1) % 4]
        if edges_to_calculate[i]:
            js[i] = counter
            counter += 1
            b0_init = cps[i] - ps[i]
            b2_init = cms[(i + 1) % 4] - ps[(i + 1) % 4]
            s0 = eps[i] - ps[i]
            s1 = ems[(i + 1) % 4] - eps[i]
            s2 = ps[(i + 1) % 4] - ems[(i + 1) % 4]
            b0 = (b0_init - a0).normalized()
            b2 = (b2_init - a3).normalized()
            w2.append(0.375*eps[i] + 0.375*ems[(i + 1) % 4] + 0.125*ps[i] + 0.125*ps[(i + 1) % 4])
            w3.append(0.125*ems[(i + 2) % 4] + 0.125*eps[(i + 3) % 4])
            w4.append(0.125*ems[i] + 0.125*eps[(i + 1) % 4])
            bs.append((b0 + b2) / 2)
            k0, h0 = get_coefs(b0, s0, a0)
            k1, h1 = get_coefs(b2, s2, a3)
            k0s.append(k0)
            k1s.append(k1)
            w5.append(k1*b0 + 2*h0*s1 + h1*s0)
            w6.append(k0*b2 + h0*s2 + 2*h1*s1)
        else:
            fps_predefined[i] = eps[i] + (a0*2 + a3) / 3
            fms_predefined[(i + 1) % 4] = ems[(i + 1) % 4] + (a0 + a3*2) / 3
    d1s, d2s = get_d1s_d2s(bs, fps_predefined, fms_predefined, js, edges_to_calculate, w2, w3, w4, w5, w6, ems, eps, k0s, k1s)
    dd1_dbs, dd2_dbs = dd_db(k0s, k1s, js, edges_to_calculate)
    angle_matrices, ths_ideal, phs_ideal = get_coordinate_systems(d1s, desired_directions, js, edges_to_calculate)
    desired_curvatires = [calc_desired_curvature(ps, ems, eps, i) for i in range(4) if edges_to_calculate[i]]
    print("desired_curvatires", desired_curvatires)
    ideal_curvatures = calc_ideal_curvatires(desired_curvatires, d1s, d2s)
    print("ideal_curvatires", ideal_curvatures)
    ideal_res = np.array(list(chain(ths_ideal, phs_ideal, ideal_curvatures))).T
    angle_matrices_T = [mat.transposed() for mat in angle_matrices]
    state = get_state_from_bs(bs)
    for i in range(20):
        d1s, d2s, direct_res = direct(bs, w2, w3, w4, w5, w6, ems, eps, k0s, k1s,
                                      angle_matrices, ideal_res, js, edges_to_calculate, fps_predefined, fms_predefined)
        jac_th_ph = deriv_th_ph(d1s, angle_matrices, angle_matrices_T, dd1_dbs)
        jac_curv = deriv_curv(d1s, d2s, dd1_dbs, dd2_dbs)
        print("direct_res", direct_res)
        jacobian = np.vstack([jac_th_ph, jac_curv])
        state = jacobian_step(state, jacobian, direct_res)
        bs = get_bs_from_state(state)
    return bs

################
# test
def test():
    def get_e0_e1(p0, p1):
        up = mathutils.Vector((0,0,1/3))
        ep = (p0*2 + p1) / 3 + up
        em = (p0 + p1*2) / 3 + up
        return ep, em

    ps = [mathutils.Vector((i, j, 0)) for i , j in zip((0,0,1,1), (0,1,1,0))]
    print("ps", ps)
    eps = [None] * 4
    ems = [None] * 4
    for i in range(4):
        p0 = ps[i]
        p1 = ps[(i + 1) % 4]
        ep, em = get_e0_e1(p0, p1)
        eps[i] = ep
        ems[(i + 1) % 4] = em
    cs = [2*p - (ep + em) / 2 for p, ep, em in zip(ps, eps, ems)]
    print("cs", cs)
    cps = [None] + cs[1:]
    cms = [cs[0]] + [None] + cs[2:]
    desired_directions = [mathutils.Vector((0,0,1))] * 3
    edges_to_calculate = [False, True, True, True]
    bs = calc_bs(ps, ems, eps, cms, cps, edges_to_calculate, desired_directions)



