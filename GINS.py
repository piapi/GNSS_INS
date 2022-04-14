import struct

import numpy as np
import INS_MECH_FUNCTION as ins
import file_test.readfile as rd
import os

D2R = np.pi / 180
# IMU常参数
ARW = 0.003 * D2R / 60.0  # 角度随机游走 deg/sqrt(hr) to rad/sqrt(s)
VRW = 0.03 / 60.0  # 速度随机游走 m/sqrt(hr) to m/sqrt(s)
gbStd = 0.027 * D2R / 3600.0  # 陀螺零偏标准差 deg/hr to rad/s
abStd = 15.0 * 1e-5  # 加表零偏标准差 mGal to M/S^2
Tgb = 4.0 * 3600.0  # 陀螺零偏相关时间 4hr to 4 * 3600s
Tab = 4.0 * 3600.0  # 加表零偏相关时间 4hr to 4 * 3600s
gsStd = 300 * 1e-6  # 陀螺比例因子标准差 ppm to 1
asStd = 300 * 1e-6  # 加表比例因子标准差 ppm to 1
Tgs = 4.0 * 3600.0  # 陀螺比例因子相关时间 4hr to 4 * 3600s
Tas = 4.0 * 3600.0  # 加表比例因子相关时间 4hr to 4 * 3600s
# 导航的参数标准差
pos_std_n = 0.005
pos_std_e = 0.004
pos_std_d = 0.008
vel_std_n = 0.003
vel_std_e = 0.004
vel_std_d = 0.004
att_std_n = 0.003 * D2R
att_std_e = 0.003 * D2R
att_std_d = 0.023 * D2R
# 臂杆
lever = np.array([[0.136], [-0.301], [-0.184]])

a = 6378137.0
we = 7.2921151467E-5
e2 = 0.0066943799901413156


def getPhi(nav, meas_cur, meas_prev):
    F = np.zeros((21, 21))
    r = nav.r.copy()
    v = nav.v.copy()
    Cbn = nav.C_bn.copy()
    vN = v[0, 0]
    vE = v[1, 0]
    vD = v[2, 0]

    lat = r[0, 0]
    lot = r[1, 0]
    h = r[2, 0]

    RM = ins.GetRM(a, e2, lat)
    RN = ins.GetRN(a, e2, lat)
    RMh = RM + h
    RNh = RN + h
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    tan_lat = np.tan(lat)
    sec_lat = 1 / cos_lat

    dt = meas_cur[0, 0] - meas_prev[0, 0]
    fb = meas_prev[4:7] / dt
    wib_b = meas_prev[1:4] / dt
    gp = ins.NormalGravity(lat, h)

    wie_n = ins.GetW_ie(we, r)
    wen_n = ins.GetW_en(r, v, RN, RM)
    win_n = wie_n + wen_n

    Frr = np.array([[-vD / RMh, 0, vN / RMh],
                    [vE * tan_lat / RNh, -(vD + vN * tan_lat) / RNh, vE / RNh],
                    [0, 0, 0]])
    Frv = np.eye(3)

    Fvr = np.array([[-2 * vE * we * cos_lat / RMh - (vE ** 2) * (sec_lat ** 2) / (RMh * RNh), 0,
                     vN * vD / (RMh ** 2) - (vE ** 2) * tan_lat / (RNh ** 2)],
                    [2 * we * (vN * cos_lat - vD * sin_lat) / RMh + vN * vE * sec_lat * sec_lat / (RMh * RNh), 0,
                     (vE * vD + vN * vE * tan_lat) / (RNh ** 2)],
                    [2 * we * vE * sin_lat / RMh, 0,
                     -(vE ** 2) / (RNh ** 2) - (vN ** 2) / (RMh ** 2) + 2 * gp / (h + np.sqrt(RM * RN))]])
    Fvv = np.array([[(vD / RMh), -2 * (we * sin_lat + vE * tan_lat / RNh), vN / RMh],
                    [2 * we * sin_lat + vE * tan_lat / RNh, (vD + vN * tan_lat) / RNh, 2 * we * cos_lat + vE / RNh],
                    [-2 * vN / RMh, -2 * (we * cos_lat + vE / RNh), 0]])
    Fpr = np.array([[-we * sin_lat / RMh, 0, vE / (RNh ** 2)],
                    [0, 0, -vN / (RMh ** 2)],
                    [-we * cos_lat / RMh - vE * sec_lat * sec_lat / (RMh * RNh), 0, -vE * tan_lat / (RNh ** 2)]])
    Fpv = np.array([[0, 1 / RNh, 0],
                    [-1 / RMh, 0, 0],
                    [0, -tan_lat / RNh, 0]])

    F[0:3, 0:3] = Frr
    F[0:3, 3:6] = Frv
    F[3:6, 0:3] = Fvr
    F[3:6, 3:6] = Fvv
    F[3:6, 6:9] = ins.cp_form(Cbn @ fb)
    F[3:6, 12:15] = Cbn
    F[3:6, 18:21] = Cbn @ np.diag([fb[0, 0], fb[1, 0], fb[2, 0]])
    F[6:9, 0:3] = Fpr
    F[6:9, 3:6] = Fpv
    F[6:9, 6:9] = -ins.cp_form(win_n)
    F[6:9, 9:12] = -Cbn
    F[6:9, 15:18] = -Cbn @ np.diag([wib_b[0, 0], wib_b[1, 0], wib_b[2, 0]])
    F[9:12, 9:12] = np.eye(3) / -Tgb
    F[12:15, 12:15] = np.eye(3) / -Tab
    F[15:18, 15:18] = np.eye(3) / -Tgs
    F[18:21, 18:21] = np.eye(3) / -Tas

    I_21 = np.eye(21)

    return I_21 + F * dt


q = np.diag([VRW * VRW, VRW * VRW, VRW * VRW,
             ARW * ARW, ARW * ARW, ARW * ARW,
             2 * gbStd * gbStd / Tgb, 2 * gbStd * gbStd / Tgb, 2 * gbStd * gbStd / Tgb,
             2 * abStd * abStd / Tab, 2 * abStd * abStd / Tab, 2 * abStd * abStd / Tab,
             2 * gsStd * gsStd / Tgs, 2 * gsStd * gsStd / Tgs, 2 * gsStd * gsStd / Tgs,
             2 * asStd * asStd / Tas, 2 * asStd * asStd / Tas, 2 * asStd * asStd / Tas])


def getG(Cbn):
    temp = np.eye(18)
    G = np.zeros((21, 18))
    temp[0:3, 0:3] = Cbn @ temp[0:3, 0:3]
    temp[3:6, 3:6] = Cbn @ temp[3:6, 3:6]

    G[3:, :] += temp
    return G


def DR(RM, RN, h, lat):  # NED2BLH
    return np.diag([RM + h, (RN + h) * np.cos(lat), -1])


def ins_mech(nav_prev, par, meas_cur, meas_prev, imu_parameters):
    I_3 = np.eye(3)
    dt = meas_cur[0, 0] - meas_prev[0, 0]
    sg = np.diag(imu_parameters[6:9, 0])
    sa = np.diag(imu_parameters[9:12, 0])
    meas_cur[1:4] = np.linalg.inv(I_3 + sg) @ (meas_cur[1:4] - imu_parameters[0:3] * dt)
    meas_cur[4:7] = np.linalg.inv(I_3 + sa) @ (meas_cur[4:7] - imu_parameters[3:6] * dt)
    nav_cur = ins.INS_MECH_CS(meas_prev, meas_cur, nav_prev, par)
    return meas_cur, nav_cur


def predict(xk_1, pk_1, nav_cur, nav_prev, meas_cur, meas_prev):
    q = np.diag([VRW * VRW, VRW * VRW, VRW * VRW,
                 ARW * ARW, ARW * ARW, ARW * ARW,
                 2 * gbStd * gbStd / Tgb, 2 * gbStd * gbStd / Tgb, 2 * gbStd * gbStd / Tgb,
                 2 * abStd * abStd / Tab, 2 * abStd * abStd / Tab, 2 * abStd * abStd / Tab,
                 2 * gsStd * gsStd / Tgs, 2 * gsStd * gsStd / Tgs, 2 * gsStd * gsStd / Tgs,
                 2 * asStd * asStd / Tas, 2 * asStd * asStd / Tas, 2 * asStd * asStd / Tas])

    phi = getPhi(nav_prev, meas_cur, meas_prev)
    xk_p = phi @ xk_1
    dt = meas_cur[0, 0] - meas_prev[0, 0]
    Gk = getG(nav_cur.C_bn)
    Gk_1 = getG(nav_prev.C_bn)

    Qk = (phi @ Gk_1 @ q @ Gk_1.T @ phi.T + Gk @ q @ Gk.T) * dt / 2.0

    pk_p = phi @ pk_1 @ phi.T + Qk
    return xk_p, pk_p


def updata(meas_gnss, nav_cur, xk_p, pk_p):
    r_gnss = meas_gnss[1:4]
    R = np.diag([meas_gnss[4, 0] ** 2, meas_gnss[5, 0] ** 2, meas_gnss[6, 0] ** 2])
    RM = ins.GetRM(a, e2, nav_cur.r[0, 0])
    RN = ins.GetRN(a, e2, nav_cur.r[0, 0])
    Dr = DR(RM, RN, nav_cur.r[2, 0], nav_cur.r[0, 0])
    r_imu = nav_cur.r + np.linalg.inv(Dr) @ nav_cur.C_bn @ lever
    zk = Dr @ (r_imu - r_gnss)
    H = np.zeros((3, 21))
    H[:, 0:3] = np.eye(3)
    H[:, 6:9] = ins.cp_form(nav_cur.C_bn @ lever)

    K = pk_p @ H.T @ (np.linalg.inv((H @ pk_p @ H.T + R)))  # Kk21*3
    xk = xk_p + K @ (zk - H @ xk_p)
    I_21 = np.eye(21)
    pk = (I_21 - K @ H) @ pk_p @ (I_21 - K @ H).T + K @ R @ K.T

    return xk, pk


# 状态反馈
def pva_back(xk, nav_cur):
    # 位置反馈
    RM = ins.GetRM(a, e2, nav_cur.r[0, 0])
    RN = ins.GetRN(a, e2, nav_cur.r[0, 0])
    Dr = DR(RM, RN, nav_cur.r[2, 0], nav_cur.r[0, 0])
    dp = np.linalg.inv(Dr) @ xk[0:3]
    nav_cur.r += - dp
    nav_cur.q_ne = ins.pos2quat(nav_cur.r[0, 0], nav_cur.r[1, 0])
    nav_cur.q_ne = ins.norm_quat(nav_cur.q_ne)
    # 姿态反馈
    Ctp = np.eye(3) - ins.cp_form(xk[6:9])
    Cpt = np.linalg.inv(Ctp)
    Cbn = nav_cur.C_bn
    nav_cur.C_bn = Cpt @ Cbn
    nav_cur.q_bn = ins.dcm2quat(nav_cur.C_bn)
    # nav_cur.q_bn = ins.norm_quat(nav_cur.q_bn)
    # 速度反馈
    nav_cur.v += -xk[3:6]
    nav_cur.dv_n += - xk[3:6]
    return nav_cur


# IMU误差反馈
def imu_back(xk, imu_paramters):
    I_3 = np.ones((3, 1))
    imu_paramters[0:3] = imu_paramters[0:3] + np.multiply((I_3 + imu_paramters[6:9]), xk[9:12])
    imu_paramters[3:6] = imu_paramters[3:6] + np.multiply((I_3 + imu_paramters[9:12]), xk[12:15])
    imu_paramters[6:9] = np.multiply((I_3 + imu_paramters[6:9]), (I_3 + xk[15:18])) - I_3
    imu_paramters[9:12] = np.multiply((I_3 + imu_paramters[9:12]), (I_3 + xk[18:21])) - I_3
    # imu_paramters += xk[9:21]
    return imu_paramters


# 误差反馈
def error_back(xk, nav_cur, imu_parameters):
    nav_cur = pva_back(xk, nav_cur)
    imu_parameters = imu_back(xk, imu_parameters)
    return np.zeros((21, 1)), nav_cur, imu_parameters


# GNSS观测数据，内插
def insert(t_gnss, meas_cur, t_prev):
    t = meas_cur[0, 0] - t_prev
    t1 = t_gnss - t_prev
    t2 = meas_cur[0, 0] - t_gnss
    mear_gnss = np.zeros(meas_cur.shape)
    mear_next = meas_cur.copy()
    mear_gnss[0, 0] = t_gnss
    mear_gnss[1:] = meas_cur[1:] * t1 / t
    mear_next[1:] = meas_cur[1:] * t2 / t
    return mear_gnss, mear_next


P0 = np.diag([pos_std_n ** 2, pos_std_e ** 2, pos_std_d ** 2,
              vel_std_n ** 2, vel_std_e ** 2, vel_std_d ** 2,
              att_std_n ** 2, att_std_e ** 2, att_std_d ** 2,
              gbStd ** 2, gbStd ** 2, gbStd ** 2,
              abStd ** 2, abStd ** 2, abStd ** 2,
              gsStd ** 2, gsStd ** 2, gsStd ** 2,
              asStd ** 2, asStd ** 2, asStd ** 2])


def gins(gnss_file, imu_file, start_time, init_nav):
    # 直接读取gnss的数据
    RTK = rd.read_file(gnss_file)
    i = 0
    while True:
        if RTK[i, 0] >= start_time:
            break
        i += 1
    RTK = RTK[i:, :]
    RTK[:, 1:3] = RTK[:, 1:3] * D2R

    # 对导航状态初始化
    meas_cur = np.array([[0.0], [0], [0], [0], [0], [0], [0]])
    meas_prev = np.array([[0.0], [0], [0], [0], [0], [0], [0]])
    nav = init_nav.copy()
    temp = [2022, start_time,
            nav.r[0, 0] / D2R, nav.r[1, 0] / D2R, nav.r[2, 0],
            nav.v[0, 0], nav.v[1, 0], nav.v[2, 0],
            (ins.dcm2euler(nav.C_bn).T / D2R)[0, 0],
            (ins.dcm2euler(nav.C_bn).T / D2R)[0, 1],
            (ins.dcm2euler(nav.C_bn).T / D2R)[0, 2]
            ]
    i_rtk = 1
    pk = P0.copy()
    xk = np.zeros((21, 1))
    import INS_MECH_CLASS
    par = INS_MECH_CLASS.Par()
    # 参考
    index = 0
    times = 640000
    imu_parameters = np.zeros((12, 1))
    f = open("./data/GINS.nav", "a")  # 利用追加模式,参数从w替换为a即可
    f.truncate(0)  # 清空文件后操作
    f.write('{}\t{}\t{}'
            '\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(temp[0], temp[1],
                                                temp[2], temp[3], temp[4],
                                                temp[5], temp[6], temp[7],
                                                temp[8], temp[9], temp[10]))
    binfile = open(imu_file, 'rb')  # 打开二进制文件
    size = os.path.getsize(imu_file)  # 获得文件大小
    for i in range(size):
        data = binfile.read(8)  # 每次输出8个字节
        num = struct.unpack('d', data)
        meas_cur[i % 7, 0] = num[0]
        if (i + 1) % 7 == 0 and meas_cur[0, 0] <= start_time:
            meas_prev[:] = meas_cur[:]
        if (i + 1) % 7 == 0 and meas_cur[0, 0] > start_time:
            if meas_cur[0, 0] < RTK[i_rtk, 0] or i_rtk >= RTK.shape[0]:
                # 预测
                meas_cur, nav1 = ins_mech(nav, par, meas_cur, meas_prev, imu_parameters)
                xk, pk = predict(xk, pk, nav1, nav, meas_cur, meas_prev)
            elif meas_cur[0, 0] == RTK[i_rtk, 0]:
                # 预测
                meas_cur, nav1 = ins_mech(nav, par, meas_cur, meas_prev, imu_parameters)
                xk, pk = predict(xk, pk, nav1, nav, meas_cur, meas_prev)
                # 更新
                xk, pk = updata(RTK[i_rtk].T, nav1, xk, pk)
                # 反馈
                xk, nav1, imu_parameters = error_back(xk, nav1, imu_parameters)
                i_rtk += 1
            elif meas_cur[0, 0] > RTK[i_rtk, 0] > meas_prev[0, 0]:
                # 内插
                meas_gnss, meas_cur = insert(RTK[i_rtk, 0], meas_cur, meas_prev[0, 0])
                # 预测
                meas_gnss, nav1 = ins_mech(nav, par, meas_gnss, meas_prev, imu_parameters)
                xk, pk = predict(xk, pk, nav1, nav, meas_gnss, meas_prev)
                # 更新
                xk, pk = updata(RTK[i_rtk].T, nav1, xk, pk)
                # 反馈
                xk, nav1, imu_parameters = error_back(xk, nav1, imu_parameters)
                # 预测
                nav = nav1.copy()
                meas_cur, nav1 = ins_mech(nav, par, meas_cur, meas_gnss, imu_parameters)
                xk, pk = predict(xk, pk, nav1, nav, meas_cur, meas_gnss)
                i_rtk += 1
            meas_prev[:] = meas_cur[:]
            nav = nav1.copy()
            temp = [2022, meas_cur[0, 0],
                    nav.r[0, 0] / D2R, nav.r[1, 0] / D2R, nav.r[2, 0],
                    nav.v[0, 0], nav.v[1, 0], nav.v[2, 0],
                    (ins.dcm2euler(nav.C_bn).T / D2R)[0, 0],
                    (ins.dcm2euler(nav.C_bn).T / D2R)[0, 1],
                    (ins.dcm2euler(nav.C_bn).T / D2R)[0, 2]
                    ]
            f.write('{}\t{}\t{}'
                    '\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(temp[0], temp[1],
                                                                temp[2], temp[3], temp[4],
                                                                temp[5], temp[6], temp[7],
                                                                temp[8], temp[9], temp[10]))
            if times / (index + 1) == 5:
                print('20%')
            elif times / (index + 1) == 2:
                print('50%')
            elif times / (index + 1.0) == 1.25:
                print('80%')
            index += 1
            if index == times:
                break
    f.close()
