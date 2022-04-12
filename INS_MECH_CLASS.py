import numpy as np


class Nav:
    def __init__(self, t=0.0, r=np.zeros((3, 1)), v=np.zeros((3, 1)), dv_n=np.zeros((3, 1)),
                 q_bn=np.zeros((4, 1)), q_ne=np.zeros((4, 1)), C_bn=np.zeros((3, 3))):
        self.t = t
        self.r = r
        self.v = v
        self.dv_n = dv_n
        self.q_bn = q_bn
        self.q_ne = q_ne
        self.C_bn = C_bn

    def copy(self):
        nav = Nav(self.t, self.r, self.v, self.dv_n, self.q_bn, self.q_ne, self.C_bn)
        return nav


class Earth_Model:
    def __init__(self):
        self.a = 6378137.0
        self.b = 6356752.3142451793
        self.f = 0.0033528106647474805
        self.w_e = 7.2921151467E-5
        self.e2 = 0.0066943799901413156
        self.GM = 398600441800000.00


# EM = Earth_Model()


class Par:
    def __init__(self):
        self.Rm = 0
        self.Rn = 0
        self.w_ie = np.zeros((3, 1))
        self.w_en = np.zeros((3, 1))
        self.g = np.zeros((3, 1))
        self.f_n = np.zeros((3, 1))
        self.w_b = np.zeros((3, 1))

    def copy(self):
        par = Par()
        par.Rm = self.Rm
        par.Rn = self.Rn
        par.w_ie = self.w_ie
        par.w_en = self.w_en
        par.g = self.g
        par.f_n = self.f_n
        return par
