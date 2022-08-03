from plasma_header import *
from interface_header import *

from extmath import IntTabulated, IntTabulatedLog, LQInterpolate, LQInterpolate2D, Spline, Spline2D

import math
import numpy as np
from numba.experimental import jitclass
import numba as nb

bigNeg = -9.2559631349317831e+61


# specArr = [
#     ('N_intervals', nb.int64),
#     ('E_x', nb.float64[:]),
#     ('logscale', nb.float64[:]),
#     ('nb', nb.float64),
#
#     ('EPS_mu0', nb.float64),
#     ('PK_on', nb.int32),
#     ('NE', nb.int64),
#     ('Nmu', nb.int64),
#
#     ('E_arr', nb.float64[:]),
#     ('mu_arr', nb.float64[:]),
#     ('f_avg', nb.float64[:]),
#     ('f_arr', nb.float64[:]),
#
#     ('log_on', nb.int32),
#     ('iso_on', nb.int32),
#     ('LQ_on', nb.int32),
#
#     ('S1', nb.pyobject),
#     ('S2', nb.pyobject)
#
# ]


# @jitclass(specArr)
class Arr_DF:
    """An object to handle the behaviors of the array distribution function"""

    def __init__(self, Lparms=None, E=None, mu=None, f=None, OK=None, empty=None):

        self.N_intervals = 0
        self.E_x = np.full(10, bigNeg)
        self.logscale = np.full(9, bigNeg)
        self.nb = 0.0
        self.EPS_mu0 = 0.0
        self.PK_on = 0

        self.NE = 0
        self.Nmu = 0

        self.E_arr = np.array([0.0])
        self.mu_arr = np.array([0.0])
        self.f_avg = np.array([0.0])
        self.f_arr = np.array([0.0])

        self.log_on = 0
        self.iso_on = 0
        self.LQ_on = 0

        self.S1 = np.array([None])
        self.S2 = np.array([None])

        OK[0] = 1

        if Lparms[i_NE] < 3 or Lparms[i_Nmu] < 3:
            empty[0] = 1
        else:
            self.NE = Lparms[i_NE]
            self.Nmu = Lparms[i_Nmu]

            self.log_on = (Lparms[i_logkey] == 0)

            self.iso_on = 0
            self.PK_on = 0

            if Lparms[i_PKkey] == 1:
                self.iso_on = 1
            if Lparms[i_PKkey] == 2:
                self.iso_on = 1
                self.PK_on = 1

            self.LQ_on = Lparms[i_splinekey] != 0

            self.E_arr = np.full(self.NE, bigNeg)

            for i in range(0, self.NE):
                q = E[i] * eV * 1e6
                if self.log_on and q <= 0:
                    OK[0] = 0
                    break
                else:
                    self.E_arr[i] = math.log(q) if self.log_on else q

            if OK[0]:

                for i in range(1, self.NE):
                    if self.E_arr[i] <= self.E_arr[i - 1]:
                        OK[0] = 0
                        break

                if OK[0]:
                    self.N_intervals = 1
                    self.E_x[0] = math.exp(self.E_arr[0]) if self.log_on else self.E_arr[0]
                    self.E_x[1] = math.exp(self.E_arr[self.NE - 1]) if self.log_on else self.E_arr[self.NE - 1]
                    self.logscale[0] = self.log_on

                    self.mu_arr = np.full(self.Nmu, bigNeg)

                    self.mu_arr = mu

                    for j in range(1, self.Nmu):
                        if self.mu_arr[j] <= self.mu_arr[j - 1]:
                            OK[0] = 0
                            break

                    if OK[0]:
                        if self.iso_on:
                            self.f_avg = np.full(self.NE, bigNeg)
                        self.mu_prof = np.full(self.Nmu, bigNeg)

                        self.f_arr = np.full((self.NE, self.Nmu), bigNeg)

                        for i in range(0, self.NE):
                            for j in range(0, self.Nmu):
                                q = f[i + j * self.NE] / eV / 1e6
                                if self.log_on and q <= 0:
                                    OK[0] = 0
                                    break
                                else:
                                    self.f_arr[i][j] = math.log(q) if self.log_on else q
                                    self.mu_prof[j] = q

                            if not (OK[0]):
                                break
                            if self.iso_on:
                                self.f_avg[i] = IntTabulated(self.mu_arr, self.mu_prof, self.Nmu) / (
                                                self.mu_arr[self.Nmu - 1] - self.mu_arr[0])
                                if self.log_on:
                                    self.f_avg[i] = math.log(self.f_avg[i])

                        if OK[0]:
                            self.nb = self.IntegratedF()

                            if self.nb < 0 or not (np.isfinite(self.nb)):
                                OK[0] = 0
                            empty[0] = (self.nb == 0)

                            if OK[0] and not (empty[0]):
                                if not self.LQ_on:
                                    if self.iso_on:
                                        self.S1[0] = Spline(self.NE, self.E_arr, self.f_avg)
                                    else:
                                        self.S2[0] = Spline2D(self.NE, self.Nmu, self.E_arr, self.mu_arr, self.f_arr)

                                self.EPS_mu0 = 1e-3

                                if not self.iso_on:
                                    gmax = np.array([0.0])
                                    g1 = np.array([bigNeg])

                                    for i in range(0, self.NE):
                                        for j in range(0, self.Nmu):
                                            self.FE(math.exp(self.E_arr[i]) if self.log_on else self.E_arr[i],
                                                    self.mu_arr[j], 0, 0, 0, g1[0], 0)
                                            gmax[0] = gmax[0] if (gmax[0] > math.fabs(g1[0])) else math.fabs(g1[0])

                                    if gmax[0] != 0.0:
                                        self.EPS_mu0 = self.EPS_mu0 if (self.EPS_mu0 < (1.0 / gmax[0] / 30)) else \
                                                       (1.0 / gmax[0] / 30)

    def IntegratedF(self):
        a = np.full(self.Nmu, bigNeg)
        np.full(self.NE, bigNeg)

        for j in range(0, self.Nmu):
            Eprof = self.f_arr[:, j]

            a[j] = IntTabulatedLog(self.E_arr, Eprof, self.NE) if self.log_on else IntTabulated(self.E_arr, Eprof,
                   self.NE)

        res = 2.0 * math.pi * IntTabulated(self.mu_arr, a, self.Nmu)

        return res

    def Fp(self, p, p_z, p_n, f, df_dp, df_dalpha):
        G = math.sqrt(1.0 + (p / mc)**2)
        E = mc2 * (G - 1.0)

        if self.iso_on:
            fE = np.array([bigNeg])
            dfE_dE = np.array([bigNeg])

            if self.LQ_on:
                LQInterpolate(math.log(E) if self.log_on else E, self.NE, self.E_arr, self.f_avg, fE, dfE_dE)
            else:
                self.S1[0].Interpolate(math.log(E) if self.log_on else E, fE, dfE_dE)

            if self.log_on:
                fE = math.exp(fE)
                dfE_dE *= (fE / E)

            f[0] = fE / (p * me * G)
            df_dp[0] = (dfE_dE - fE * G * me / (p ** 2) * (1.0 + ((p / G / mc) ** 2))) / ((me * G) ** 2)
            df_dalpha[0] = 0.0

        else:
            mu = p_z / p if (p > 0.0) else 0.0
            if mu > 1.0:
                mu = 1.0
            if mu < (-1.0):
                mu = -1.0
            sa = p_n / p if (p > 0.0) else 1.0

            fE = np.array([bigNeg])
            dfE_dE = np.array([bigNeg])
            dfE_dmu = np.array([bigNeg])
            if not self.LQ_on:
                self.S2[0].Interpolate(math.log(E) if self.log_on else E, mu, fE, dfE_dE, dfE_dmu, 0)
            else:
                LQInterpolate2D(math.log(E) if self.log_on else E, mu, self.NE, self.Nmu, self.E_arr, self.mu_arr,
                                self.f_arr, fE, dfE_dE, dfE_dmu, 0)

            if self.log_on:
                fE[0] = math.exp(fE[0])
                dfE_dE[0] *= (fE[0] / E)
                dfE_dmu[0] *= fE[0]

            f[0] = fE[0] / (p * me * G)
            df_dp[0] = (dfE_dE[0] - fE[0] * G * me / (p ** 2) * (1.0 + ((p / G / mc) ** 2))) / ((me * G) ** 2)
            df_dalpha[0] = -dfE_dmu[0] * sa / (p * me * G)

    def FE(self, E, mu, f, df_dE, df_dmu, g1, g2):

        if self.iso_on:
            fE = np.array([bigNeg])
            dfE_dE = np.array([bigNeg])

            if not (f[0]):
                g1[0] = 0.0
            else:
                if self.LQ_on:
                    LQInterpolate(math.log(E) if self.log_on else E, self.NE, self.E_arr, self.f_avg, fE, dfE_dE)
                else:
                    self.S1[0].Interpolate(math.log(E) if self.log_on else E, fE, dfE_dE)

                if self.log_on:
                    f[0] = math.exp(fE)
                    df_dE[0] = dfE_dE * (f[0] / E)
                else:
                    f[0] = fE
                    df_dE[0] = dfE_dE

                df_dmu[0] = 0.0
                g1[0] = 0.0
                g2[0] = 0.0

        else:
            fE = np.array([bigNeg])
            dfE_dE = np.array([bigNeg])
            dfE_dmu = np.array([bigNeg])
            d2fE_dmu2 = np.array([bigNeg])

            if not (f[0]):
                if self.LQ_on:
                    LQInterpolate2D(math.log(E) if self.log_on else E, mu, self.NE, self.Nmu, self.E_arr,
                                    self.mu_arr, self.f_arr, 0 if self.log_on else fE, 0, dfE_dmu, 0)
                else:
                    self.S2[0].Interpolate(math.log(E) if self.log_on else E, mu, 0 if self.log_on else fE, 0, dfE_dmu,
                                           0)
                g1[0] = dfE_dmu[0] if self.log_on else (dfE_dmu[0] / fE[0])

            else:
                if self.LQ_on:
                    LQInterpolate2D(math.log(E) if self.log_on else E, mu, self.NE, self.Nmu, self.E_arr,
                                    self.mu_arr, self.f_arr, fE, dfE_dE, dfE_dmu, d2fE_dmu2)
                else:
                    self.S2[0].Interpolate(math.log(E) if self.log_on else E, mu, fE, dfE_dE, dfE_dmu, d2fE_dmu2)

                if self.log_on:
                    f[0] = math.exp(fE[0])
                    df_dE[0] = dfE_dE[0] * (f[0] / E)
                    df_dmu[0] = dfE_dmu[0] * (f[0])
                    g1[0] = dfE_dmu[0]
                    g2[0] = d2fE_dmu2[0] + (g1[0])**2
                else:
                    f[0] = fE[0]
                    df_dE[0] = dfE_dE[0]
                    df_dmu[0] = dfE_dmu[0]
                    g1[0] = dfE_dmu[0] / fE[0]
                    g2[0] = d2fE_dmu2[0] / fE[0]
