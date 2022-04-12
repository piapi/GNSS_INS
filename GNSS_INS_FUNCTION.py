import numpy as np
import INS_MECH_FUNCTION as ins
import INS_MECH_CLASS

# IMU常参数
ARW = 0.003 * np.pi / 180 / 60.0  # 角度随机游走 deg/sqrt(hr) to rad/sqrt(s)
VRW = 0.03 / 60.0  # 速度随机游走 m/sqrt(hr) to m/sqrt(s)
gbStd = 0.027 * np.pi / 180 / 3600.0  # 陀螺零偏标准差 deg/hr to rad/s
abStd = 15.0 * 1e-5  # 加表零偏标准差 mGal to M/S^2
gbT = 4.0 * 3600.0  # 陀螺零偏相关时间 4hr to 4 * 3600s
abT = 4.0 * 3600.0  # 加表零偏相关时间 4hr to 4 * 3600s
gsStd = 300 * 1e-6  # 陀螺比例因子标准差 ppm to 1
asStd = 300 * 1e-6  # 加表比例因子标准差 ppm to 1
gsT = 4.0 * 3600.0  # 陀螺比例因子相关时间 4hr to 4 * 3600s
asT = 4.0 * 3600.0  # 加表比例因子相关时间 4hr to 4 * 3600s
# 导航的参数标准差
pos_std_n = 0.005
pos_std_e = 0.004
pos_std_d = 0.008
vel_std_n = 0.003
vel_std_e = 0.004
vel_std_d = 0.004
att_std_n = 0.003 * np.pi / 180
att_std_e = 0.003 * np.pi / 180
att_std_d = 0.023 * np.pi / 180
# 臂杆
lever = np.mat([0.136, -0.301, -0.184]).T

a = 6378137.0
we = 7.2921151467E-5
e2 = 0.0066943799901413156


def DR(R_M, R_N, h, lat):  # NED2BLH
    return np.diag([R_M + h, (R_N + h) * np.cos(lat), -1])


#
def getPhi(ins_prev, mear_cur, mear_prev):  # sk cur skp pre
    r = ins_prev.r.copy()
    v = ins_prev.v.copy()
    vD = v[2, 0]
    vE = v[1, 0]
    vN = v[0, 0]

    phi = r[0, 0]
    lamda = r[1, 0]
    h = r[2, 0]

    # theta = fnc.dcm2euler(ins_prev.C_bn)

    # dtheta = theta - fnc.dcm2euler(ins_prev.C_bn)
    dt = mear_cur[0, 0] - mear_prev[0, 0]

    fb = mear_prev[4:7, :] / dt
    # fbx = fnc.cp_form(fb)
    w_ib_b = mear_prev[1:4, :] / dt
    gp = ins.NormalGravity(phi, h)

    Rm = ins.GetRM(a, e2, phi)
    Rn = ins.GetRN(a, e2, phi)

    w_ie_n = ins.GetW_ie(we, r)
    w_en_n = ins.GetW_en(r, v, Rn, Rm)
    w_in_n = w_ie_n + w_en_n
    # w_in_nx = fnc.cp_form(w_in_n)

    F = np.zeros((21, 21))

    RMh = Rm + h
    RNh = Rn + h
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    tan_phi = np.tan(phi)
    sec_phi = 1 / cos_phi

    Frr = np.mat([[-vD / RMh, 0, vN / RMh],
                  [vE * tan_phi / RNh, -(vD + vN * tan_phi) / RNh, vE / RNh],
                  [0, 0, 0]])
    Frv = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    Fvr = np.mat(
        [[-2 * vE * we * cos_phi / RMh - vE * vE * sec_phi * sec_phi / RMh / RNh, 0,
          vN * vD / RMh / RMh - vE * vE * tan_phi / RNh / RNh],
         [2 * we * (vN * cos_phi - vD * sin_phi) / RMh + vN * vE * sec_phi * sec_phi / RMh / RNh, 0,
          (vE * vD + vN * vE * tan_phi) / RNh / RNh],
         [2 * we * vE * sin_phi / RMh, 0,
          -vE * vE / (RNh * RNh) - vN * vN / (RMh * RMh) + 2 * gp / (np.sqrt(Rm * Rn) + h)]])
    Fvv = np.mat([[vD / RMh, -2 * (we * sin_phi + vE * tan_phi / RNh), vN / RMh],
                  [2 * we * sin_phi + vE * tan_phi / RNh, (vD + vN * tan_phi) / RNh, 2 * we * cos_phi + vE / RNh],
                  [-2 * vN / RMh, -2 * (we * cos_phi + vE / RNh), 0]])
    Ffr = np.mat([[-we * sin_phi / RMh, 0, vE / (RNh * RNh)],
                  [0, 0, -vN / (RMh * RMh)],
                  [-we * cos_phi / RMh - vE * sec_phi * sec_phi / RMh / RNh, 0, -vE * tan_phi / (RNh * RNh)]])
    Ffv = np.mat([[0, 1 / RNh, 0],
                  [-1 / RMh, 0, 0],
                  [0, -tan_phi / RNh, 0]])
    Cbn = ins_prev.C_bn.copy()

    F[0:3, 0:3] = Frr
    F[0:3, 3:6] = Frv
    F[3:6, 0:3] = Fvr
    F[3:6, 3:6] = Fvv
    F[3:6, 6:9] = ins.cp_form(Cbn.dot(fb))
    F[3:6, 12:15] = Cbn
    F[3:6, 18:21] = Cbn.dot(np.diag([fb[0, 0], fb[1, 0], fb[2, 0]]))
    F[6:9, 0:3] = Ffr
    F[6:9, 3:6] = Ffv
    F[6:9, 6:9] = -ins.cp_form(w_in_n)
    F[6:9, 9:12] = -Cbn
    F[6:9, 15:18] = -Cbn.dot(np.diag([w_ib_b[0, 0], w_ib_b[1, 0], w_ib_b[2, 0]]))
    F[9:12, 9:12] = np.diag([-1 / gbT, -1 / gbT, -1 / gbT])
    F[12:15, 12:15] = np.diag([-1 / abT, -1 / abT, -1 / abT])
    F[15:18, 15:18] = np.diag([-1 / gsT, -1 / gsT, -1 / gsT])
    F[18:21, 18:21] = np.diag([-1 / asT, -1 / asT, -1 / asT])
    I_21 = np.diag([1] * 21)
    return I_21 + F * dt






def getG(Cbn):
    G = np.zeros((21, 18))
    G[3:, 0:] = np.diag([1] * 18)
    return G


P0 = np.diag([pos_std_n * pos_std_n, pos_std_e * pos_std_e, pos_std_d * pos_std_d,
              vel_std_n * vel_std_n, vel_std_e * vel_std_e, vel_std_d * vel_std_d,
              att_std_n * att_std_n, att_std_e * att_std_e, att_std_d * att_std_d,
              gbStd * gbStd, gbStd * gbStd, gbStd * gbStd,
              abStd * abStd, abStd * abStd, abStd * abStd,
              gsStd * gsStd, gsStd * gsStd, gsStd * gsStd,
              asStd * asStd, asStd * asStd, asStd * asStd])


# Q0 = np.diag([VRW * VRW, VRW * VRW, VRW * VRW,
#               ARW * ARW, ARW * ARW, ARW * ARW,
#               2 * gbStd * gbStd / gbT, 2 * gbStd * gbStd / gbT, 2 * gbStd * gbStd / gbT,
#               2 * abStd * abStd / abT, 2 * abStd * abStd / abT, 2 * abStd * abStd / abT,
#               2 * gsStd * gsStd / gsT, 2 * gsStd * gsStd / gsT, 2 * gsStd * gsStd / gsT,
#               2 * asStd * asStd / asT, 2 * asStd * asStd / asT, 2 * asStd * asStd / asT])


def INS_MECH(meas_pre, meas_cur, nav, par, imu_paramters):
    dt = meas_cur[0, 0] - meas_pre[0, 0]
    bg = imu_paramters[:, 0]
    ba = imu_paramters[:, 1]

    sg = np.diag([1/(1+imu_paramters[0, 2]), 1/(1+imu_paramters[1, 2]), 1/(1+imu_paramters[2, 2])])
    sa = np.diag([1/(1+imu_paramters[0, 3]), 1/(1+imu_paramters[1, 3]), 1/(1+imu_paramters[2, 3])])
    I_3 = np.diag([1] * 3)
    meas_cur[1:4, 0] = sg@(meas_cur[1:4, 0] - bg * dt)
    meas_cur[4:7, 0] = sa@(meas_cur[4:7, 0] - ba * dt)

    nav1 = ins.INS_MECH_CS(meas_pre, meas_cur, nav, par)
    return nav1


def predict(xk_1, pk_1, ins_state_cur, ins_state_pre, mear_cur, mear_prev):  # ins_state惯导当前状态 ins_state_pre前一时刻
    phi = getPhi(ins_state_pre, mear_cur, mear_prev)
    dt = ins_state_cur.t - ins_state_pre.t
    xk = phi.dot(xk_1)
    Q0 = np.diag([
        VRW * VRW, VRW * VRW, VRW * VRW,
        ARW * ARW, ARW * ARW, ARW * ARW,
        2 * gbStd * gbStd / gbT, 2 * gbStd * gbStd / gbT, 2 * gbStd * gbStd / gbT,
        2 * abStd * abStd / abT, 2 * abStd * abStd / abT, 2 * abStd * abStd / abT,
        2 * gsStd * gsStd / gsT, 2 * gsStd * gsStd / gsT, 2 * gsStd * gsStd / gsT,
        2 * asStd * asStd / asT, 2 * asStd * asStd / asT, 2 * asStd * asStd / asT])
    G_k_1 = getG(ins_state_pre.C_bn)
    G_k = getG(ins_state_cur.C_bn)
    Qk_1 = (phi @ G_k_1 @ Q0 @ G_k_1.T @ phi.T + G_k @ Q0 @ G_k.T) * dt / 2.0

    pk = phi.dot(pk_1).dot(phi.T) + Qk_1
    return xk, pk


def update(Z, x_k_k_1, p_k_k_1, ins_state_cur):
    Cbn = ins_state_cur.C_bn.copy()
    Rk = np.diag([Z[0, 4] * Z[0, 4], Z[0, 5] * Z[0, 5], Z[0, 6] * Z[0, 6]])
    Hk = np.zeros((3, 21))
    Hk[:, 0:3] = np.diag([1] * 3)

    Hk[:, 6:9] = ins.cp_form(Cbn.dot(lever))
    nr = np.mat([np.random.normal(0, Rk[0, 0]),
                 np.random.normal(0, Rk[1, 1]),
                 np.random.normal(0, Rk[2, 2])]).T
    r = ins_state_cur.r.copy()

    Rm = ins.GetRM(a, e2, r[0, 0])
    Rn = ins.GetRN(a, e2, r[0, 0])

    Dr = DR(Rm, Rn, r[2, 0], r[0, 0])
    rimu = r + np.linalg.inv(Dr).dot(Cbn).dot(lever)  # ins ->gnss
    rgnss = Z[0, 1:4].T - np.linalg.inv(Dr).dot(nr)
    # rgnss = Z[0, 1:4].T
    Zk = Dr.dot(rimu - rgnss)

    Kk = p_k_k_1.dot(Hk.T).dot(np.linalg.inv((Hk.dot(p_k_k_1).dot(Hk.T) + Rk)))  # Kk21*3

    xk = x_k_k_1 + Kk.dot(Zk - Hk.dot(x_k_k_1))

    I_21 = np.diag([1] * 21)
    pk = (I_21 - Kk.dot(Hk)).dot(p_k_k_1).dot((I_21 - Kk.dot(Hk)).T) + Kk.dot(Rk).dot(Kk.T)

    return xk, pk


def state_back(xk, ins_state_cur):  # 位置、速度、姿态反馈
    xk_reset = xk.copy()
    # 位置反馈
    lat = ins_state_cur.r[0, 0]
    # lot = ins_state_cur.r[1, 0]
    h = ins_state_cur.r[2, 0]
    Rm = ins.GetRM(a, e2, lat)
    Rn = ins.GetRN(a, e2, lat)
    correct = np.linalg.inv(DR(Rm, Rn, h, lat)).dot(xk[0:3, :])
    ins_state_cur.r[0:3, 0] = ins_state_cur.r[0:3, 0] - correct
    ins_state_cur.q_ne = ins.pos2quat(ins_state_cur.r[0, 0], ins_state_cur.r[1, 0])
    ins_state_cur.q_ne = ins.norm_quat(ins_state_cur.q_ne)
    # 姿态反馈
    Ctp = np.diag([1, 1, 1]) - ins.cp_form(xk[6:9, :])
    Cpt = np.linalg.inv(Ctp)
    Cbn = ins_state_cur.C_bn.copy()
    ins_state_cur.C_bn = Cpt.dot(Cbn)
    ins_state_cur.q_bn = ins.dcm2quat(ins_state_cur.C_bn)
    ins_state_cur.q_bn = ins.norm_quat(ins_state_cur.q_bn)
    xk_reset[0:9, :] = np.zeros((9, 1))
    # 速度反馈
    ins_state_cur.v[0:3, 0] = Cpt.dot(ins_state_cur.v[0:3, 0])
    ins_state_cur.v[0:3, 0] = ins_state_cur.v[0:3, 0] - xk[3:6, :]
    ins_state_cur.dv_n = Cpt.dot(ins_state_cur.dv_n) - xk[3:6, :]

    return ins_state_cur, xk_reset


def bs_back(xk, imu_paramters):  # bg, ba, sg, sa 矫正加速度计和陀螺仪的零偏和比例因子
    I = np.ones((3, 1))
    xk_reset = xk.copy()
    paramters_pre = imu_paramters.copy()
    imu_paramters[:, 0] = paramters_pre[:, 0] + np.multiply(I + paramters_pre[:, 1], xk[9:12, :])
    imu_paramters[:, 1] = np.multiply((I + paramters_pre[:, 1]), (I + xk[15:18, :])) - I
    imu_paramters[:, 2] = paramters_pre[:, 2] + np.multiply((I + paramters_pre[:, 3]), xk[12:15, :])
    imu_paramters[:, 3] = np.multiply((I + paramters_pre[:, 3]), (I + xk[18:21, :])) - I
    # for i in range(12):
    #     paramters_pre[:, i] = paramters_pre[:, i] + xk[9 + i,:]
    xk_reset[9:21, :] = np.zeros((12, 1))

    return imu_paramters, xk_reset


def inserpolate(mear_cur, mear_prev, gnssDate):
    t = mear_cur[0, 0] - mear_prev[0, 0]
    t1 = gnssDate[0, 0] - mear_prev[0, 0]
    t2 = mear_cur[0, 0] - gnssDate[0, 0]
    mear_gnss = mear_cur.copy()
    mear_k = mear_cur.copy()
    mear_gnss[0, 0] = gnssDate[0, 0]
    mear_gnss[1:7, :] = t1 / t * mear_cur[1:7, :]
    mear_k[1:7, 0] = t2 / t * mear_cur[1:7, :]
    return mear_gnss, mear_k
