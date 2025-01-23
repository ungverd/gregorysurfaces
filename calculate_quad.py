from typing import List, Tuple, Optional

import numpy as np

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
           i: int):
    return 6*w2[i] + 6*w3[i] - 12*w4[i]\
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
          i: int):
    return 2/3 * k0[i]

def dfm_db(k1: List[float],
          i: int):
    return 2/3 * k1[(i - 1) % 4]

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

def direct(bs: List[mathutils.Vector],
           w2: List[mathutils.Vector],
           w3: List[mathutils.Vector],
           w4: List[mathutils.Vector],
           w5: List[mathutils.Vector],
           w6: List[mathutils.Vector],
           ems: List[mathutils.Vector],
           eps: List[mathutils.Vector],
           k0s: List[float],
           k1s: List[float]):
    fps: List[Optional[mathutils.Vector]] = [None] * 4
    fms: List[Optional[mathutils.Vector]] = [None] * 4
    d1s: List[mathutils.Vector] = []
    for i in range(4):
        f0p, f1m, = get_f0p_f1m(bs[i],
                                w5[i],
                                w6[i],
                                eps[i],
                                ems[(i + 1) % 4],
                                k0s[i],
                                k1s[i])
        fps[i] = f0p
        fms[(i + 1) % 4] = f1m
        d1s.append(get_d1(f0p, f1m, w2[i], w4[i]))
    
    d2s: List[mathutils.Vector] = []
    for i in range(4):
        d2s.append(get_d2(w2, w3, w4, fms, fps, i))
    ths: List[float] = []
    phs: List[float] = []
    curvs: List[float] = []
    for i in range[4]:
        th, ph = get_angles(d1s[i])
        ths.append(th)
        phs.append(ph)
        curvs.append(get_curv(d1s[i], d2s[i]))
    return ths, phs, curvs

def dd_db(k0: List[float], k1: List[float]):
    dd1_df_v = dd1_df()
    dd1_dbs = []
    dd2_dbs = []
    for i_b in range(4):
        i_fm = (i_b + 1) % 4
        i_fp = i_b
        dfm_db_v = dfm_db(k1, i_b)
        dfp_db_v = dfp_db(k0, i_b)
        dd1_db = dd1_df_v*(dfm_db_v + dfp_db_v)
        dd1_dbs.append(dd1_db)
        dd2_dbs.append([])
        for i_curv in range(4):
            dd2_dfm_v = dd2_dfm(i_curv, i_fm)
            dd2_dfp_v = dd2_dfp(i_curv, i_fp)
            dd2_db = dd2_dfm_v*dfm_db_v + dd2_dfp_v*dfp_db_v
            dd2_dbs[-1].append(dd2_db)
    return dd1_dbs, dd2_dbs

def deriv_th(d1s: List[mathutils.Vector],):

def deriv_ph(d1s: List[mathutils.Vector],):

def deriv_curv(d1s: List[mathutils.Vector],
               d2s: List[mathutils.Vector],
               k0: List[float],
               k1: List[float]
              dd1_dbs,
              dd2_dbs):
    res: List[List[float]] = [[0]*4 for _ in range(12)]
    for i_curv in range(4):
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
            for i_b in range(4):
                if i_curv == i_b:
                    dd1_db = dd1_dbs[i_b]
                    res[i_curv][i_b*3 + xyz] = dcurv_d1_v*dd1_db + dcurv_d2_v*dd2_db
                else:
                    res[i_curv][i_b*3 + xyz] = dcurv_d2_v*dd2_db
    return res

def jacobian_step(state, jacobian, direct_res):
    new_state = state - np.linalg.inv(jacobian)@direct_res

def calc_bs(ps: List[mathutils.Vector],
            ems: List[mathutils.Vector],
            eps: List[mathutils.Vector],
            cms: List[mathutils.Vector],
            cps: List[mathutils.Vector], 
            desired_directions: List[Tuple[float, float]], 
            desired_curvatures: List[float]):
    w2: List[mathutils.Vector] = []
    w3: List[mathutils.Vector] = []
    w4: List[mathutils.Vector] = []
    w5: List[mathutils.Vector] = []
    w6: List[mathutils.Vector] = []
    k0s: List[float] = []
    k1s: List[float] = []
    for i in range(4):
        w2.append(0.375*eps[i] + 0.375*ems[(i + 1) % 4] + 0.125*ps[i] + 0.125*ps[(i + 1) % 4])
        w3.append(0.125*ems[(i + 2) % 4] + 0.125*eps[(i + 3) % 4])
        w4.append(0.125*ems[i] + 0.125*eps[(i + 1) % 4])
        b0 = cps[i] - ps[i]
        b2 = cms[(i + 1) % 4] - ps[(i + 1) % 4]
        s0 = eps[i] - ps[i]
        s1 = ems[(i + 1) % 4] - eps[i]
        s2 = ps[(i + 1) % 4] - ems[(i + 1) % 4]
        a0 = ems[i] - ps[i]
        a3 = eps[(i + 1) % 4] - ps[(i + 1) % 4]
        k0, h0 = get_coefs(b0, s0, a0)
        k1, h1 = get_coefs(b2, s2, a3)
        k0s.append(k0)
        k1s.append(k1)
        w5.append(k1*b0 + 2*h0*s1 + h1*s0)
        w6.append(k0*b2 + h0*s2 + 2*h1*s1)
