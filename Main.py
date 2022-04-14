import GINS
import numpy as np
import INS_MECH_CLASS as cla
import INS_MECH_FUNCTION as ins
import file_test.readfile as rd
import plot_nav

D2R = np.pi / 180

Cbn = ins.euler2dcm(0.85252308 * D2R, -2.03396435 * D2R, 185.69625052 * D2R)
par = cla.Par()
nav = cla.Nav(
    r=np.array([[np.pi * 30.4447873696 / 180], [np.pi * 114.4718632476 / 180], [20.910]]), C_bn=Cbn,
    v=np.array([[0.0], [-0.0], [0.0]]))

nav.q_bn = ins.dcm2quat(Cbn)
nav.q_ne = ins.pos2quat(nav.r[0, 0], nav.r[1, 0])
gnss_file = './data/GNSS_RTK.txt'
imu_file = './data/A15_imu.bin'

stat_time = 456300.0

# GINS.gins(gnss_file, imu_file, stat_time, nav)

fusion = rd.read_file('./data/GINS.nav')
ref = rd.read_file('./data/truth.nav')

for i in range(fusion.shape[0]):
    if fusion[i, -1] < 0:
        fusion[i, -1] += 360
    elif fusion[i, -1] > 360:
        fusion[i, -1] -= 360

leng = int(fusion[-1, 1] - fusion[0, 1])

sample_fusion = np.zeros((leng, fusion.shape[1]))
sample_ref = np.zeros((leng, ref.shape[1]))

i = 0
j = 0
while i < fusion.shape[0] and j < leng:
    if abs(fusion[i, 1] - stat_time - j) < 1.0 / 800:
        sample_fusion[j, :] = fusion[i, :]
        j += 1
    i += 1

i = 0
j = 0
while i < ref.shape[0] and j < leng:
    if abs(ref[i, 1] - stat_time - j) < 1.0 / 800:
        sample_ref[j, :] = ref[i, :]
        j += 1
    i += 1

a = 6378137.0
e2 = 0.0066943799901413156

def BLH2NED(blh):
    blh[0] = blh[0] * D2R
    blh[1] = blh[1] * D2R
    h = blh[2]
    RN = a / np.sqrt(1 - e2 * np.sin(blh[0]) * np.sin(blh[0]))
    RM = a * (1 - e2) / np.power(np.sqrt(1 - e2 * np.sin(blh[0]) * np.sin(blh[0])), 3)
    ned = np.diag([RM + h, (RN + h) * np.cos(blh[0]), -1]) @ blh

    return ned

for i in range(leng):
    sample_ref[i, 2:5] = BLH2NED(sample_ref[i, 2:5])
    sample_fusion[i, 2:5] = BLH2NED(sample_fusion[i, 2:5])

title = ['N_compare', 'E_compare', 'altitude_compare', 'Vx_compare', 'Vy_compare', 'Vz_compare', 'roll_compare',
         'pitch_compare', 'heading_compare']


def rms(x):
    s = 0
    for i in range(x.shape[0]):
        s += x[i] * x[i]

    return np.sqrt(s / x.shape[0])


error1 = sample_ref[:, 2:] - sample_fusion[:, 2:]
for i in range(leng):
    error1[i, -1] = 0 if abs(error1[i, -1]) > 300 else error1[i, -1]

for i in range(error1.shape[1]):
    print(title[i], rms(error1[:, i]))

plot_nav.plot_pva(sample_fusion, sample_ref)
plot_nav.plot_pva_error(sample_fusion[:, 1], error1)
