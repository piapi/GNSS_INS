import GINS
import numpy as np
import INS_MECH_CLASS as cla
import INS_MECH_FUNCTION as ins
import file_test.readfile as rd
import matplotlib.pyplot as plt

D2R = np.pi / 180

Cbn = ins.euler2dcm(0.854 * D2R, -2.0345 * D2R, 185.696 * D2R)
par = cla.Par()
nav = cla.Nav(
    r=np.array([[np.pi * 30.444787369 / 180], [np.pi * 114.471863247 / 180], [20.910]]), C_bn=Cbn,
    v=np.array([[0.0], [-0.0], [0.0]]))

nav.q_bn = ins.dcm2quat(Cbn)
nav.q_ne = ins.pos2quat(nav.r[0, 0], nav.r[1, 0])
gnss_file = './data/GNSS_RTK.txt'
imu_file = './data/A15_imu.bin'

stat_time = 456300.0
GINS.gins(gnss_file, imu_file, stat_time, nav)
fusion = rd.read_file('./data/GINS.txt', 1)
ref = rd.read_file('./data/truth.nav')

for i in range(fusion.shape[0]):
    if fusion[i, -1] < 0:
        fusion[i, -1] += 360
    elif fusion[i, -1] > 360:
        fusion[i, -1] -= 360

ylabel = ['lat/deg', 'lot/deg', 'altitude/m', 'Vx/m/s', 'Vy/m/s', 'Vz/m/s', 'roll/deg', 'pitch/deg', 'heading/deg']
title = ['lat_compare', 'lot_compare', 'altitude_compare', 'Vx_compare', 'Vy_compare', 'Vz_compare', 'roll_compare',
         'pitch_compare', 'heading_compare']

for i in range(9):
    fig_i, ax = plt.subplots(figsize=(12, 8))

    ax.plot(fusion[:, 0], fusion[:, i + 1], 'r', label='fusion')
    ax.plot(ref[:, 1], ref[:, i + 2], 'b', label='ref')
    # ax.plot(res_imu[0:times - 2, 0], res_imu[0:times - 2, i + 1] - res_ref[1:times, i + 1], 'y', label='Compare')

    ax.legend(loc=2)
    ax.set_xlabel('time/s')
    ax.set_ylabel(ylabel[i])
    ax.set_title(title[i])
plt.show()
