import numpy as np
import pandas as pd
import INS_MECH_CLASS


def CrossProduct(a, b):
    c = np.zeros((3, 1))
    c[0, 0] = a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0]
    c[1, 0] = a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0]
    c[2, 0] = a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0]
    return c


def GetRM(a, e2, lat):
    return a * (1 - e2) / np.power(1 - e2 * np.sin(lat) * np.sin(lat), 1.5)


def GetRN(a, e2, lat):
    return a / np.sqrt(1 - e2 * np.sin(lat) * np.sin(lat))


def GetW_en(r, v, n, m):
    return np.array(
        [[v[1, 0] / (n + r[2, 0])], [-v[0, 0] / (m + r[2, 0])], [-v[1, 0] * np.tan(r[0, 0]) / (n + r[2, 0])]])


def GetW_ie(w_e, r):
    return w_e * np.array([[np.cos(r[0, 0])], [0], [-np.sin(r[0, 0])]])


def NormalGravity(latitude, he):
    a1 = 9.7803267715
    a2 = 0.0052790414
    a3 = 0.0000232718
    a4 = -0.000003087691089
    a5 = 0.000000004397731
    a6 = 0.000000000000721
    s2 = np.sin(latitude) * np.sin(latitude)
    s4 = s2 * s2
    return a1 * (1 + a2 * s2 + a3 * s4) + (a4 + a5 * s2) * he + a6 * he * he


def cp_form(v):
    V = np.zeros((3, 3))
    V[0, 1] = -v[2, 0]
    V[0, 2] = v[1, 0]
    V[1, 0] = v[2, 0]
    V[1, 2] = -v[0, 0]
    V[2, 0] = -v[1, 0]
    V[2, 1] = v[0, 0]
    return V


def pos2dcm(lat, lon):  # refrence to 4-32
    s_lat = np.sin(lat)
    c_lat = np.cos(lat)
    s_lon = np.sin(lon)
    c_lon = np.cos(lon)
    C_ne = np.array(
        [-s_lat * c_lon, -s_lon, -c_lat * c_lon, -s_lat * s_lon, c_lon, -c_lat * s_lon, c_lat, 0.0, -s_lat]).reshape(
        3, 3)
    return C_ne


def dcm2pos(C_ne):
    lat = -np.arctan2(C_ne[2, 2], C_ne[2, 0])
    lon = -np.arctan2(C_ne[0, 1], C_ne[1, 1])
    return [lat, lon]


def rvec2quat(rot_vec):  ##refrence to 3-48
    mag2 = rot_vec[0, 0] * rot_vec[0, 0] + rot_vec[1, 0] * rot_vec[1, 0] + rot_vec[2, 0] * rot_vec[2, 0]
    if mag2 < np.pi * np.pi:
        mag2 = 0.25 * mag2
        c = 1.0 - mag2 / 2.0 * (1.0 - mag2 / 12.0 * (1.0 - mag2 / 30.0))
        s = 1.0 - mag2 / 6.0 * (1.0 - mag2 / 20.0 * (1.0 - mag2 / 42.0))
        q = np.array([[c], [s * 0.5 * rot_vec[0, 0]], [s * 0.5 * rot_vec[1, 0]], [s * 0.5 * rot_vec[2, 0]]])
    else:
        mag = np.sqrt(mag2)
        s_mag = np.sin(mag / 2)

        q = np.array(
            [[np.cos(mag / 2.0)], [rot_vec[0, 0] * s_mag / mag], [rot_vec[1, 0] * s_mag / mag],
             [rot_vec[2, 0] * s_mag / mag]])

        if q[0, 0] < 0:
            q = -q
    return q


def quat2rvec(q):  # refrence to 3-48
    if q[0, 0] == 0:
        rot_vec = np.array(np.pi * [[q[1, 0]], [q[2, 0]], [q[3, 0]]])
        return rot_vec

    if q[0, 0] < 0:
        q = -q
    mag2 = np.arctan(np.sqrt(q[1, 0] * q[1, 0] + q[2, 0] * q[2, 0] + q[3, 0] * q[3, 0]) / q[0, 0])
    f = np.sin(mag2) / mag2 / 2
    rot_vec = q[1:4, 0] / f
    # mag2 = (q[1, 0] * q[1, 0] + q[2, 0] * q[2, 0] + q[3, 0] * q[3, 0]) / (q[0, 0]*q[0, 0])
    # f = 1 - mag2 / 6.0 * (1 - mag2 / 20.0 * (1 - mag2 / 42))
    # f = 0.5 * f
    # rot_vec = q[1:4, 0] / f
    return rot_vec


def dcm2quat(C):  # refrence to 3-47
    Tr = np.trace(C)
    q = np.zeros((4, 1))
    p = np.zeros(4)
    p[0] = 1 + Tr
    p[1] = 1 + 2 * C[0, 0] - Tr
    p[2] = 1 + 2 * C[1, 1] - Tr
    p[3] = 1 + 2 * C[2, 2] - Tr
    index = np.argmax(p)
    if index == 0:
        q[0, 0] = 0.5 * np.sqrt(p[0])
        q[1, 0] = (C[2, 1] - C[1, 2]) / q[0, 0] / 4
        q[2, 0] = (C[0, 2] - C[2, 0]) / q[0, 0] / 4
        q[3, 0] = (C[1, 0] - C[0, 1]) / q[0, 0] / 4
    elif index == 1:
        q[1, 0] = 0.5 * np.sqrt(p[1])
        q[0, 0] = (C[2, 1] - C[1, 2]) / q[1, 0] / 4
        q[2, 0] = (C[1, 0] + C[0, 1]) / q[1, 0] / 4
        q[3, 0] = (C[0, 2] + C[2, 0]) / q[1, 0] / 4
    elif index == 2:
        q[2, 0] = 0.5 * np.sqrt(p[2])
        q[0, 0] = (C[0, 2] - C[2, 0]) / q[2, 0] / 4
        q[1, 0] = (C[1, 0] + C[0, 1]) / q[2, 0] / 4
        q[3, 0] = (C[2, 1] + C[1, 2]) / q[2, 0] / 4
    else:
        q[3, 0] = 0.5 * np.sqrt(p[3])
        q[0, 0] = (C[1, 0] - C[0, 1]) / q[3, 0] / 4
        q[1, 0] = (C[0, 2] + C[2, 0]) / q[3, 0] / 4
        q[2, 0] = (C[2, 1] + C[1, 2]) / q[3, 0] / 4
    if q[0, 0] < 0:
        q = -q
    return q


def quat2dcm(q):  # refrence to 3-46
    C_ne = np.zeros((3, 3))
    C_ne[0, 0] = q[0, 0] * q[0, 0] + q[1, 0] * q[1, 0] - q[2, 0] * q[2, 0] - q[3, 0] * q[3, 0]
    C_ne[0, 1] = 2 * (q[1, 0] * q[2, 0] - q[0, 0] * q[3, 0])
    C_ne[0, 2] = 2 * (q[1, 0] * q[3, 0] + q[0, 0] * q[2, 0])
    C_ne[1, 0] = 2 * (q[1, 0] * q[2, 0] + q[0, 0] * q[3, 0])
    C_ne[1, 1] = q[0, 0] * q[0, 0] - q[1, 0] * q[1, 0] + q[2, 0] * q[2, 0] - q[3, 0] * q[3, 0]
    C_ne[1, 2] = 2 * (q[2, 0] * q[3, 0] - q[0, 0] * q[1, 0])
    C_ne[2, 0] = 2 * (q[1, 0] * q[3, 0] - q[0, 0] * q[2, 0])
    C_ne[2, 1] = 2 * (q[2, 0] * q[3, 0] + q[0, 0] * q[1, 0])
    C_ne[2, 2] = q[0, 0] * q[0, 0] - q[1, 0] * q[1, 0] - q[2, 0] * q[2, 0] + q[3, 0] * q[3, 0]
    return C_ne


def quat2pos(q_ne):
    lat = -2 * np.arctan(q_ne[2, 0] / q_ne[0, 0]) - np.pi / 2
    lon = 2 * np.arctan2(q_ne[3, 0], q_ne[0, 0])
    return [lat, lon]


def pos2quat(lat, lon):
    s1 = np.sin(lon / 2)
    c1 = np.cos(lon / 2)
    s2 = np.sin(-np.pi / 4 - lat / 2)
    c2 = np.cos(-np.pi / 4 - lat / 2)
    q_ne = np.array([[c1 * c2], [-s1 * s2], [c1 * s2], [c2 * s1]])
    return q_ne


def qmul(q1, q2):
    q = np.zeros((4, 1))
    q = np.mat(q)
    q[0, 0] = q1[0, 0] * q2[0, 0] - q1[1, 0] * q2[1, 0] - q1[2, 0] * q2[2, 0] - q1[3, 0] * q2[3, 0]
    q[1, 0] = q1[0, 0] * q2[1, 0] + q1[1, 0] * q2[0, 0] + q1[2, 0] * q2[3, 0] - q1[3, 0] * q2[2, 0]
    q[2, 0] = q1[0, 0] * q2[2, 0] + q1[2, 0] * q2[0, 0] + q1[3, 0] * q2[1, 0] - q1[1, 0] * q2[3, 0]
    q[3, 0] = q1[0, 0] * q2[3, 0] + q1[3, 0] * q2[0, 0] + q1[1, 0] * q2[2, 0] - q1[2, 0] * q2[1, 0]
    # q = np.mat([q1[0, 0] * q2[0, 0] - q1[1, 0] * q2[1, 0] - q1[2, 0] * q2[2, 0] - q1[3, 0] * q2[3, 0],
    #             q1[0, 0] * q2[1, 0] + q1[1, 0] * q2[0, 0] + q1[2, 0] * q2[3, 0] - q1[3, 0] * q2[2, 0],
    #             q1[0, 0] * q2[2, 0] + q1[2, 0] * q2[0, 0] + q1[3, 0] * q2[1, 0] - q1[1, 0] * q2[3, 0],
    #             q1[0, 0] * q2[3, 0] + q1[3, 0] * q2[0, 0] + q1[1, 0] * q2[2, 0] - q1[2, 0] * q2[1, 0]]).T
    if q[0, 0] < 0.0:
        q = -q
    return q


def norm_quat(q):  # refrence to 7-28
    e = (q.T * q - 1) / 2
    e = e[0, 0]
    q_n = (1 - e) * q
    return q_n


def dist_ang(ang1, ang2):
    ang = ang2 - ang1
    if ang > np.pi:
        ang = ang - 2 * np.pi
    elif ang < -np.pi:
        ang = ang + 2 * np.pi
    return ang


def dpos2rvec(lat, delta_lat, delta_lon):
    return np.array([[delta_lon * np.cos(lat)], [-delta_lat], [-delta_lon * np.sin(lat)]])


def euler2dcm(roll, pitch, heading):
    C_bn = np.zeros((3, 3))
    cr = np.cos(roll)
    cp = np.cos(pitch)
    ch = np.cos(heading)
    sr = np.sin(roll)
    sp = np.sin(pitch)
    sh = np.sin(heading)

    C_bn[0, 0] = cp * ch
    C_bn[0, 1] = -cr * sh + sr * sp * ch
    C_bn[0, 2] = sr * sh + cr * sp * ch

    C_bn[1, 0] = cp * sh
    C_bn[1, 1] = cr * ch + sr * sp * sh
    C_bn[1, 2] = -sr * ch + cr * sp * sh

    C_bn[2, 0] = - sp
    C_bn[2, 1] = sr * cp
    C_bn[2, 2] = cr * cp
    return C_bn


def dcm2euler(C_bn):  # refrence to 3-34
    roll = 0
    pitch = np.arctan(-C_bn[2, 0] / (np.sqrt(C_bn[2, 1] * C_bn[2, 1] + C_bn[2, 2] * C_bn[2, 2])))
    heading = 0

    if C_bn[2, 0] <= -0.999:
        roll = np.NaN
        heading = np.arctan2((C_bn[1, 2] - C_bn[0, 1]), (C_bn[0, 2] + C_bn[1, 1]))
    elif C_bn[2, 0] >= 0.999:
        roll = np.NaN
        heading = np.pi + np.arctan2((C_bn[1, 2] + C_bn[0, 1]), (C_bn[0, 2] - C_bn[1, 1]))
    else:
        roll = np.arctan2(C_bn[2, 1], C_bn[2, 2])
        heading = np.arctan2(C_bn[1, 0], C_bn[0, 0])

    return np.array([[roll], [pitch], [heading]])


def qconj(qi):
    q = np.copy(qi)
    q[1:4, 0] = -q[1:4, 0]
    return q


# meas_prev = previous   [t gyro_x gyro_y gyro_z accel_x accel_y accel_z]
# meas_cur = current measurements, units [s, rad, m/s]
# EM = Earth model
# nav.r = position [lat; lon; h]
# nav.v = velocity [Vn; Ve; Vd]
# nav.dv_n = velocity increment in navigation frame
# nav.q_bn = quaternion from the b-frame to the n-frame;
# nav.q_ne = quaternion from the n-frame to the e-frame;
# par previous k-2 par1 k-1
# par.Rm = radius of curvature in meridian;
# par.Rn = radius of curvature in prime vertical;
# par.w_ie
# par.w_en
# par.g
# par.f_n = specific force
#  - OUTPUT -
# nav1 = updated navigation solution
# dv_n1 = current velocity increment

def INS_MECH_CS(meas_pre, meas_cur, nav, par):
    nav1 = INS_MECH_CLASS.Nav()
    EM = INS_MECH_CLASS.Earth_Model()
    nav.t = meas_cur[0, 0]
    par.Rm = GetRM(EM.a, EM.e2, nav.r[0, 0])
    par.Rn = GetRN(EM.a, EM.e2, nav.r[0, 0])

    dt = meas_cur[0, 0] - meas_pre[0, 0]

    beta = CrossProduct(meas_pre[1:4], meas_cur[1:4]) / 12.0
    scul = CrossProduct(meas_cur[1:4], meas_cur[4:7]) / 2.0 + \
           CrossProduct(meas_pre[1:4], meas_cur[4:7]) / 12 + \
           CrossProduct(meas_pre[4:7], meas_cur[1:4]) / 12

    # 1、速度更新
    # （1）高度和经纬度的预测

    # mid_r = np.zeros((3, 1))
    # mid_r[2, 0] = nav.r[2, 0] - 0.5 * mid_v[2, 0] * dt
    # par.Rm = ins.GetRM(EM.a, EM.e2, nav.r[0, 0])
    # mid_r[0, 0] = nav.r[0, 0] + 0.5 * mid_v[0, 0] * dt / (par.Rm + mid_r[2, 0])
    # par.Rn = ins.GetRN(EM.a, EM.e2, mid_r[0, 0])
    # mid_r[1, 0] = nav.r[1, 0] + 0.5 * mid_v[1, 0] * dt / ((par.Rn + mid_r[2, 0]) * np.cos(mid_r[0, 0]))

    mid_r = nav.r.copy()
    mid_r[2, 0] = nav.r[2, 0] - 0.5 * nav.v[2, 0] * dt
    d_lat = 0.5 * nav.v[0, 0] * dt / (par.Rm + mid_r[2, 0])
    d_lon = 0.5 * nav.v[1, 0] * dt / (par.Rn + mid_r[2, 0]) / np.cos(mid_r[0, 0])
    d_theta = dpos2rvec(nav.r[0, 0], d_lat, d_lon)
    mid_q = qmul(nav.q_ne, rvec2quat(d_theta))
    [mid_r[0, 0], mid_r[1, 0]] = quat2pos(mid_q)

    par.g = np.array([[0], [0], [NormalGravity(mid_r[0, 0], mid_r[2, 0])]])
    mid_v = nav.v + nav.dv_n * 0.5  # 用上一时刻的速度增量和速度，求出当前k和k-1的中间时刻的速度
    # (2)
    par.w_ie = GetW_ie(EM.w_e, mid_r)
    par.w_en = GetW_en(mid_r, mid_v, par.Rn, par.Rm)

    # （3）

    zeta = (par.w_ie + par.w_en) * dt

    Cn = np.eye(3) - cp_form(zeta) / 2
    dv_f_n = (Cn.dot(nav.C_bn)).dot(meas_cur[4:7] + scul)
    par.f_n = dv_f_n / dt

    dv_g_cor = (par.g - CrossProduct(2 * par.w_ie + par.w_en, mid_v)) * dt
    nav1.dv_n = dv_f_n + dv_g_cor
    nav1.v = nav.v + nav1.dv_n
    # 2、位置更新
    mid_v = 0.5 * (nav1.v + nav.v)

    par.w_en = GetW_en(mid_r, mid_v, par.Rn, par.Rm)
    zeta = (par.w_en + par.w_ie) * dt
    qn = rvec2quat(zeta)  # q_nk_nk-1
    xi = np.array([[0], [0], [-EM.w_e * dt]])
    qe = rvec2quat(xi)
    nav1.q_ne = qmul(qe, qmul(nav.q_ne, qn))
    nav1.q_ne = norm_quat(nav1.q_ne)
    [nav1.r[0, 0], nav1.r[1, 0]] = quat2pos(nav1.q_ne)
    nav1.r[2, 0] = nav.r[2, 0] - mid_v[2, 0] * dt

    # 3、姿态更新
    # (1)
    q = rvec2quat(meas_cur[1:4] + beta)  # coning
    nav1.q_bn = qmul(nav.q_bn, q)

    mid_r = np.array([[nav1.r[0, 0] + 0.5 * dist_ang(nav1.r[0, 0], nav.r[0, 0])],
                      [nav1.r[1, 0] + 0.5 * dist_ang(nav1.r[1, 0], nav.r[1, 0])],
                      [0.5 * (nav1.r[2, 0] + nav.r[2, 0])]])

    par.w_ie = GetW_ie(EM.w_e, mid_r)
    par.w_en = GetW_en(mid_r, mid_v, par.Rn, par.Rm)
    zeta = (par.w_en + par.w_ie) * dt
    # （2）
    q = rvec2quat(-zeta)  # q_nk-1_nk
    nav1.q_bn = qmul(q, nav1.q_bn)
    nav1.q_bn = norm_quat(nav1.q_bn)
    nav1.C_bn = quat2dcm(nav1.q_bn)
    return nav1
