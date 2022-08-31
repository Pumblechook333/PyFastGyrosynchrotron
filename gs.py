import time
from plasma_header import *
from plasma import EmWave
from extmath import FindBesselJ, FindBesselJ_WH, qromb, BrentRoot, SecantRoot, trapzdLog, qrombLog, trapzd
import numpy as np
from numba.experimental import jitclass
from numba import njit
import numba as nb
import math

# USED TO INSTANCE ALL NUMERICAL POINTERS
bigNeg = -9.2559631349317831e+61


# specCashStr = [
#     ('Q', nb.float64[:]),
#     ('f', nb.float64[:]),
#     ('df_dp', nb.float64[:]),
#     ('df_dalpha', nb.float64[:])
# ]


# @jitclass(specCashStr)
class CacheStruct:
    """A structure which is capable of accepting and storing information between separate instances of
       Gyrosynchotron Integral (GSI)"""

    def __init__(self):
        self.Q = np.array([bigNeg])
        self.f = np.array([bigNeg])
        self.df_dp = np.array([bigNeg])
        self.df_dalpha = np.array([bigNeg])


# NOTE Extracting this math provided a 10s overall slowdown to the code.
# @njit
# def F_init(p_zi, N_z, x):
#     Gi = p_zi * N_z / mc + x
#     p2i = (mc ** 2) * ((Gi ** 2) - 1.0)
#     pi = math.sqrt(p2i) if (p2i > 0.0) else 0.0
#     p_n2i = p2i - (p_zi ** 2)
#     p_ni = math.sqrt(p_n2i) if (p_n2i > 0.0) else 0.0
#     cai = p_zi / pi if (p2i > 0.0) else 0.0
#     sai = p_ni / pi if (p2i > 0.0) else 1.0
#     betai = pi / Gi / mc
#     beta_zi = p_zi / Gi / mc
#     beta_ni = p_ni / Gi / mc
#
#     return Gi, pi, p_ni, cai, sai, betai, beta_zi, beta_ni


# specGSI = [
#     ('w', nb.pyobject),
#     ('df', nb.pyobject),
#     ('ExactBessel', nb.int32),
#
#     ('s', nb.int64),
#     ('mode', nb.int32),
#     ('x', nb.float64),
#
#     ('Cache', nb.pyobject),
#     ('rom_count', nb.int32),
#     ('rom_n', nb.int32),
#     ('CacheSize', nb.int32)
# ]


# @jitclass(specGSI) Note inner functions lifted
class GSIntegrand:
    """An object that will describe the behavior of the integration of Gyrosynchotron emission"""

    def __init__(self):
        self.w = np.array([None])
        self.df = np.array([None])
        self.ExactBessel = 0

        self.s = 0
        self.mode = 0
        self.x = 0.0

        self.Cache = np.array([None])
        self.rom_count = 0
        self.rom_n = 0
        self.CacheSize = 0

    def F(self, p_z):
        """The F function belonging to a GS Integrand object"""
        # G, p, p_n, ca, sa, beta, beta_z, beta_n = F_init(p_z, self.w[0].N_z, self.x)

        G = p_z * self.w[0].N_z / mc + self.x
        p2 = (mc ** 2) * ((G ** 2) - 1.0)
        p = math.sqrt(p2) if (p2 > 0.0) else 0.0
        p_n2 = p2 - (p_z ** 2)
        p_n = math.sqrt(p_n2) if (p_n2 > 0.0) else 0.0
        ca = p_z / p if (p2 > 0.0) else 0.0
        sa = p_n / p if (p2 > 0.0) else 1.0
        beta = p / G / mc
        beta_z = p_z / G / mc
        beta_n = p_n / G / mc

        Q = np.array([bigNeg])
        f = np.array([bigNeg])
        df_dp = np.array([bigNeg])
        df_dalpha = np.array([bigNeg])

        i = self.rom_count
        self.rom_count += 1

        if self.rom_count > self.rom_n:
            lamb = self.w[0].y * p_n / mc * self.w[0].N * self.w[0].st

            if lamb != 0.0:
                Js = np.array([bigNeg])
                Js1 = np.array([bigNeg])

                if self.ExactBessel:
                    FindBesselJ(lamb, self.s, Js, Js1)
                else:
                    FindBesselJ_WH(lamb, self.s, Js, Js1)

                Q[0] = ((self.w[0].T * (self.w[0].ct - self.w[0].N * beta_z) + self.w[0].L * self.w[0].st) / (
                        self.w[0].N * beta_n * self.w[0].st) * Js[0] + Js1[0]) ** 2

            elif self.s == 1:
                Q[0] = (((self.w[0].T * (self.w[0].ct - self.w[0].N * beta_z) + self.w[0].L * self.w[0].st) * G *
                         self.w[0].y + 1.0) / 2) ** 2
            else:
                Q[0] = 0.0

            self.df[0].Fp(p, p_z, p_n, f, df_dp, df_dalpha)

            if i >= self.CacheSize:
                self.CacheSize += 64
                moreSpace = np.array([CacheStruct() for _ in range(64)])
                self.Cache = np.append(self.Cache, moreSpace)

            self.Cache[i].Q[0] = Q[0]
            self.Cache[i].f[0] = f[0]
            self.Cache[i].df_dp[0] = df_dp[0]
            self.Cache[i].df_dalpha[0] = df_dalpha[0]
            self.rom_n = self.rom_count
        else:
            Q[0] = self.Cache[i].Q[0]
            f[0] = self.Cache[i].f[0]
            df_dp[0] = self.Cache[i].df_dp[0]
            df_dalpha[0] = self.Cache[i].df_dalpha[0]

        Q[0] *= (G * mc * sa * (p_n * df_dp + (ca - self.w[0].N_z * beta) * df_dalpha) if self.mode else f * (p_n ** 2))

        return Q[0]


CLK_TCK = 1000


@njit
def EGpParam(Emi, Ema, Gmi, Gma, pmi, pma, ra, rma):
    if ra > 0:
        Emi[0] *= (1.0 + 1e-10)
    if ra < (rma - 1):
        Ema[0] *= (1.0 - 1e-10)

    Gmi[0] = Emi[0] / mc2 + 1.0  # minimal Lorentz factor
    Gma[0] = Ema[0] / mc2 + 1.0  # maximal Lorentz factor
    pmi[0] = mc * math.sqrt((Gmi[0] ** 2) - 1.0)  # minimal impulse
    pma[0] = mc * math.sqrt((Gma[0] ** 2) - 1.0)  # maximal impulse


# @njit Note loops lifted
def GS_jk(w, df, ExactBessel, j, k):
    """Calculates one pair of j and k values of solar electrons for a specified set of wave functions"""

    ERR_s = 1e-5
    S_MIN = 1
    S_MAX = 1000000
    T_MAX = 3600.0
    ERR_i = 1e-5

    if not w[0].Valid:
        j[0] = 0.0
        k[0] = 1e100

    else:
        j[0] = 0.0
        k[0] = 0.0

        if w[0].nu_B > 0.0:
            gsi = np.array([GSIntegrand()])
            gsi[0].ExactBessel = ExactBessel
            gsi[0].w[0] = w[0]
            gsi[0].df[0] = df

            gsi[0].CacheSize = 65
            gsi[0].Cache = np.array([CacheStruct() for _ in range(gsi[0].CacheSize)])

            timeout = 0
            s_out = 0
            t0 = time.clock()

            for r in range(0, df.N_intervals):
                if not timeout and not s_out:
                    j_loc = 0.0
                    k_loc = 0.0

                    # calculating the distribution function parameters:
                    E_min = df.E_x[r]
                    E_max = df.E_x[r + 1]

                    # EGpParam(E_min, E_max, G_min, G_max, p_min, p_max, r, df.N_intervals)

                    if r > 0:
                        E_min *= (1.0 + 1e-10)
                    if r < (df.N_intervals - 1):
                        E_max *= (1.0 - 1e-10)

                    G_min = E_min / mc2 + 1.0  # minimal Lorentz factor
                    G_max = E_max / mc2 + 1.0  # maximal Lorentz factor
                    p_min = mc * math.sqrt((G_min ** 2) - 1.0)  # minimal impulse
                    p_max = mc * math.sqrt((G_max ** 2) - 1.0)  # maximal impulse

                    p_out = 0
                    j_done = 0
                    k_done = 0
                    vfinite = 1

                    gsi[0].s = S_MIN

                    while True:

                        gsi[0].x = w[0].nu_B / w[0].nu * gsi[0].s
                        R2 = (w[0].N_z ** 2) + (gsi[0].x - 1.0) * (gsi[0].x + 1.0)
                        if R2 > 0.0:
                            R = math.sqrt(R2)
                            p_z1 = mc * (w[0].N_z * gsi[0].x - R) / (1.0 - (w[0].N_z ** 2))
                            p_z2 = mc * (w[0].N_z * gsi[0].x + R) / (1.0 - (w[0].N_z ** 2))

                            M = 1
                            if p_z1 > p_max or p_z2 < (-p_max) or (p_z1 > (-p_min) and p_z2 < p_min):
                                M = 0
                            if p_z1 < (-p_max) and p_z2 > p_max:
                                M = 0
                                if gsi[0].x > 1:
                                    p_out = 1

                            if M:
                                pzx = mc / w[0].N_z * (G_min - gsi[0].x)
                                if (math.fabs(pzx)) < p_min:
                                    if w[0].N_z > 0:
                                        p_z1 = pzx
                                    else:
                                        p_z2 = pzx

                                pzx = mc / w[0].N_z * (G_max - gsi[0].x)

                                if (math.fabs(pzx)) < p_max:
                                    if w[0].N_z > 0:
                                        p_z2 = pzx
                                    else:
                                        p_z1 = pzx

                                err = np.array([bigNeg])
                                gsi[0].rom_n = 0

                                if not j_done:
                                    gsi[0].mode = 0
                                    gsi[0].rom_count = 0

                                    q = qromb(gsi, p_z1, p_z2, ERR_i, err)

                                    j_loc += q
                                    if j_loc != 0.0:
                                        if (math.fabs(q / j_loc)) < ERR_s:
                                            j_done = 1
                                    if not (np.isfinite(q)):
                                        vfinite = 0

                                if not k_done:
                                    gsi[0].mode = 1
                                    gsi[0].rom_count = 0

                                    q = qromb(gsi, p_z1, p_z2, ERR_i, err)

                                    k_loc += q
                                    if k_loc != 0.0:
                                        if (math.fabs(q / k_loc)) < ERR_s:
                                            k_done = 1
                                    if not (np.isfinite(q)):
                                        vfinite = 0

                        if gsi[0].s >= S_MAX:
                            s_out = 1
                        if ((time.clock() - t0) / CLK_TCK) > T_MAX:
                            timeout = 1

                        gsi[0].s += 1

                        if not(not p_out and not (j_done and k_done) and not s_out and not timeout and vfinite):
                            break

                    j[0] += j_loc
                    k[0] += k_loc

            j[0] *= (4.0 * ((math.pi * e) ** 2) / c * w[0].N * w[0].nu / (1.0 + (w[0].T ** 2)))
            k[0] *= (-4.0 * ((math.pi * e) ** 2) / w[0].N / w[0].nu / (1.0 + (w[0].T ** 2)))


specCashStrA = [
    ('Q', nb.float64[:]),
    ('f', nb.float64[:]),
    ('R', nb.float64[:])
]


@jitclass(specCashStrA)
class CacheStructApprox:
    """A structure which is capable of accepting and storing information between separate instances of the approximation
       of the Gyrosynchotron Integral (GSI)"""
    def __init__(self):
        self.f = np.array([bigNeg])
        self.Q = np.array([bigNeg])
        self.R = np.array([bigNeg])


# specMuSol = [
#     ('gsi', nb.pyobject)
# ]
#
#
# @jitclass(specMuSol) Note does not need
class MuSolveFunction:
    """Container for a modified version of a GS Integrand object that will be used to find mu and mu0 for the
        j and k calculations performed using the approximated bessel calculation."""
    def __init__(self):
        self.gsi = np.array([None])

    def F(self, mu):
        h = self.gsi[0].H1(mu)
        return h


# specGSIA = [
#     ('w', nb.pyobject),
#     ('df', nb.pyobject),
#
#     ('Q_on', nb.int32),
#     ('Q_corr', nb.int32),
#     ('mode', nb.int32),
#
#     ('mu_list', nb.float64[:]),
#     ('lnQ1_list', nb.float64[:]),
#     ('mflag', nb.int32),
#
#
#     ('Cache', nb.pyobject),
#     ('rom_count', nb.int32),
#     ('rom_n', nb.int32),
#     ('CacheSize', nb.int32),
#
#     ('E_loc', nb.float64),
#     ('beta_loc', nb.float64),
#     ('G_loc', nb.float64),
#     ('p_loc', nb.float64),
# ]


# @jitclass(specGSIA) Note does not need
class GSIntegrandApprox:
    """A modified version of the object meant to handle Gyrosynchotron j and k calculations based on a formula of
        approximated bessel functions. Optimized for speed over accuracy after a certain threshold of complexity
        slowdown."""
    def __init__(self):
        self.w = np.array([None])
        self.df = np.array([None])

        self.Q_on = 0
        self.Qcorr = 0
        self.mode = 0

        self.mu_list = np.full(2, bigNeg)
        self.lnQ1_list = np.full(2, bigNeg)
        self.mflag = 0

        self.Cache = np.array([None])

        self.rom_count = 0
        self.rom_n = 0
        self.CacheSize = 0

        self.E_loc = 0.0
        self.beta_loc = 0.0
        self.G_loc = 0.0
        self.p_loc = 0.0

    def H1(self, mu):
        g1 = np.array([bigNeg])

        self.df[0].FE(self.E_loc, mu, 0, 0, 0, g1, 0)

        nbct = self.w[0].N * self.beta_loc * self.w[0].ct
        nbmct1 = 1.0 - nbct * mu
        sa2 = 1.0 - (mu ** 2)
        sa = math.sqrt(sa2)
        x = self.w[0].N * self.beta_loc * self.w[0].st * sa / nbmct1
        s1mx2 = math.sqrt(1.0 - (x ** 2))
        lnZ = s1mx2 + math.log(x / (1.0 + s1mx2))

        lnQ1 = 0.0
        if self.Qcorr:
            s1mx2_3 = s1mx2 * s1mx2 * s1mx2
            s = self.G_loc * self.w[0].y * nbmct1
            a6 = s1mx2_3 + 0.503297 / s
            b16 = s1mx2_3 + 1.193000 / s
            b2 = (1.0 - 0.2 * pow(s, -2.0 / 3))
            ab = pow(a6 * b16, 1.0 / 6) * b2
            xi = 3.0 * (x ** 2) * s1mx2 * (self.w[0].N_z * self.beta_loc - mu) / sa2
            eta = self.w[0].N_z * self.beta_loc / s
            lamb = self.G_loc * self.w[0].y / (6.0 * s)
            a_1a = lamb * (0.503297 * eta - xi) / a6
            b_1b = lamb * (1.193000 * eta - xi) / b16 + 4.0 * lamb * self.beta_loc * self.w[0].N_z * (b2 - 1.0) / b2
            lnQ1 = 2.0 * (ab * (a_1a + b_1b) * nbmct1 - ab * self.w[0].N_z * self.beta_loc - self.w[0].T * self.w[0].N
                          * self.beta_loc) / (self.w[0].T * (self.w[0].ct - self.w[0].N * self.beta_loc * mu)
                          + self.w[0].L * self.w[0].st + ab * nbmct1) - 2.0 * a_1a + self.w[0].N_z \
                          * self.beta_loc / nbmct1

            self.mu_list[self.mflag] = mu
            self.lnQ1_list[self.mflag] = lnQ1
            self.mflag ^= 1

        return g1[0] + 2.0 * self.G_loc * self.w[0].y * ((nbct - mu) / sa2 * s1mx2 - nbct * lnZ) + lnQ1

    def Find_mu0(self, mu0, lnQ2):

        msf = np.array([MuSolveFunction()])
        msf[0].gsi[0] = self

        lnQ2[0] = 0.0

        self.Qcorr = 0
        mu0[0] = BrentRoot(msf, -1.0 + 1e-5, 1.0 - 1e-5, self.df[0].EPS_mu0)

        if np.isfinite(mu0[0]) and self.Q_on:
            self.mflag = 0
            Qfound = 0

            self.Qcorr = 1
            mu1 = SecantRoot(msf, mu0[0], mu0[0] - 1e-4 * (1.0 if (mu0[0] > 0) else (-1.0 if (mu0[0] < 0) else 0)), self.df[0].EPS_mu0)

            if np.isfinite(mu1) and (math.fabs(mu1)) < 1.0:
                mu0[0] = mu1
                Qfound = 1
            else:
                mu1 = BrentRoot(msf, -1.0 + 1e-5, mu0[0], self.df[0].EPS_mu0)

                if np.isfinite(mu1):
                    mu0[0] = mu1
                    Qfound = 1
                else:
                    mu1 = BrentRoot(msf, mu0[0], 1.0 - 1e-5, self.df[0].EPS_mu0)
                    if np.isfinite(mu1):
                        mu0[0] = mu1
                        Qfound = 1

            if Qfound:
                lnQ2[0] = (self.lnQ1_list[1] - self.lnQ1_list[0]) / (self.mu_list[1] - self.mu_list[0])

    def F(self, E):
        if E == 0.0:
            return 0.0
        else:
            f = np.array([bigNeg])
            Q = np.array([bigNeg])
            R = np.array([bigNeg])

            i = self.rom_count
            self.rom_count += 1

            if self.rom_count > self.rom_n:
                self.E_loc = E
                self.G_loc = E / mc2 + 1.0
                self.beta_loc = math.sqrt((self.G_loc ** 2) - 1.0) / self.G_loc
                self.p_loc = self.beta_loc * self.G_loc * mc

                mu0 = np.array([bigNeg])
                lnQ2 = np.array([bigNeg])
                if self.df[0].PK_on:
                    mu0[0] = self.w[0].N * self.beta_loc * self.w[0].ct
                    lnQ2[0] = 0
                else:
                    self.Find_mu0(mu0, lnQ2)

                if np.isfinite(mu0[0]):
                    df_dE = np.array([bigNeg])
                    df_dmu = np.array([bigNeg])
                    g1 = np.array([bigNeg])
                    g2 = np.array([bigNeg])
                    self.df[0].FE(E, mu0[0], f, df_dE, df_dmu, g1, g2)

                    nbct = self.w[0].N * self.beta_loc * self.w[0].ct
                    nbctm = nbct - mu0[0]
                    nbmct1 = 1.0 - nbct * mu0[0]
                    sa2 = 1.0 - (mu0[0] ** 2)
                    sa = math.sqrt(sa2)
                    x = self.w[0].N * self.beta_loc * self.w[0].st * sa / nbmct1
                    s1mx2 = math.sqrt(1.0 - (x ** 2))
                    s1mx2_3 = s1mx2 * s1mx2 * s1mx2
                    s = self.G_loc * self.w[0].y * nbmct1
                    a = pow(s1mx2_3 + 0.503297 / s, 1.0 / 6)
                    b = pow(s1mx2_3 + 1.193000 / s, 1.0 / 6) * (1.0 - 0.2 * pow(s, -2.0 / 3))

                    Q[0] = ((self.w[0].T * (self.w[0].ct - self.w[0].N * self.beta_loc * mu0[0]) + self.w[0].L
                               * self.w[0].st + a * b * nbmct1) ** 2) / (a ** 2) / nbmct1

                    lnZ = s1mx2 + math.log(x / (1.0 + s1mx2))
                    Z = math.exp(2.0 * s * lnZ)

                    H2 = g2 - (g1 ** 2) - 2.0 * self.G_loc * self.w[0].y * s1mx2 / sa2 * (1.0 + ((self.w[0].N
                            * self.beta_loc * self.w[0].st * nbctm / nbmct1) ** 2) / nbmct1 / (1.0 - (x ** 2)) - 2.0
                            * mu0[0] * nbctm / sa2 + nbct * nbctm / nbmct1) + lnQ2[0]

                    LpFactor = math.sqrt(-2.0 * math.pi / H2)
                    if not (np.isfinite(LpFactor)):
                        LpFactor = 0

                    Q[0] *= (Z * LpFactor)

                    R[0] = df_dE - (1.0 + (self.beta_loc ** 2)) / (c * self.p_loc * self.beta_loc) * f[0] + nbctm / \
                           (c * self.p_loc * self.beta_loc) * df_dmu

                else:
                    f[0] = 0.0
                    Q[0] = 0.0
                    R[0] = 0.0

                if i >= self.CacheSize:
                    self.CacheSize += 64

                    moreSpace = np.array([CacheStructApprox() for _ in range(64)])
                    self.Cache = np.append(self.Cache, moreSpace)

                self.Cache[i].f[0] = f[0]
                self.Cache[i].Q[0] = Q[0]
                self.Cache[i].R[0] = R[0]
                self.rom_n = self.rom_count

            else:
                f[0] = self.Cache[i].f[0]
                Q[0] = self.Cache[i].Q[0]
                R[0] = self.Cache[i].R[0]

            return Q[0] * R[0] if self.mode else Q[0] * f[0]


# @njit Note does not need jit
def GS_jk_approx(w, df, Npoints, Q_on, j, k):
    """Performs the approximated calculations for j and k based on approximated bessel functions."""
    ERR_i = 1e-5

    if not w[0].Valid:
        j[0] = 0.0
        k[0] = 1e100
    else:
        j[0] = 0.0
        k[0] = 0.0

        if w[0].nu_B > 0.0:
            gsi = np.array([GSIntegrandApprox()])
            gsi[0].w[0] = w[0]
            gsi[0].df[0] = df
            gsi[0].Q_on = Q_on

            gsi[0].CacheSize = 65
            gsi[0].Cache = np.array([CacheStructApprox() for _ in range(gsi[0].CacheSize)])

            for r in range(0, df.N_intervals):
                jk_loc = np.full(2, bigNeg)
                err = np.array([bigNeg])
                gsi[0].rom_n = 0

                for gsi[0].mode in range(0, 2):
                    gsi[0].rom_count = 0

                    if df.logscale[r]:
                        val = (trapzdLog(gsi, df.E_x[r], df.E_x[r + 1], Npoints) if (Npoints >= 1)
                                else (qrombLog(gsi, df.E_x[r], df.E_x[r + 1], ERR_i, err)))
                    else:
                        val = (trapzd(gsi, df.E_x[r], df.E_x[r + 1], Npoints) if (Npoints >= 1)
                                else (qromb(gsi, df.E_x[r], df.E_x[r + 1], ERR_i, err)))

                    jk_loc[gsi[0].mode] = val

                j[0] += jk_loc[0]
                k[0] += jk_loc[1]

            j[0] *= (2.0 * math.pi * (e ** 2) * w[0].nu / c / w[0].N / (1.0 + (w[0].T ** 2)) / (w[0].st ** 2))
            k[0] *= (-2.0 * math.pi * (e ** 2) * c / (w[0].N ** 2) / w[0].N / w[0].nu / (1.0 + (w[0].T ** 2)) /
                     (w[0].st ** 2))


# @njit Note does not need jit
def GS_jk_mDF(w, df, ExactBessel, j, k):
    """Calls the exact GS_jk calculation module on every provided df element, and returns the associated j and k
       values for each."""

    j[0] = 0
    k[0] = 0

    df_loc = df

    while df_loc[0]:
        j_loc = np.array([bigNeg])
        k_loc = np.array([bigNeg])

        GS_jk(w, df_loc[0], ExactBessel, j_loc, k_loc)

        j[0] += j_loc[0]
        k[0] += k_loc[0]

        df_loc = np.roll(df_loc, 1)


# @njit Note does not need jit
def GS_jk_approx_mDF(w, df, Npoints, Q_on, j, k):
    """Calls the approximated GS_jk calculation module on every provided df element, and returns the associated j and k
           values for each."""

    j[0] = 0
    k[0] = 0

    df_loc = df

    while df_loc[0]:
        j_loc = np.array([bigNeg])
        k_loc = np.array([bigNeg])

        GS_jk_approx(w, df_loc[0], Npoints, Q_on, j_loc, k_loc)

        j[0] += j_loc[0]
        k[0] += k_loc[0]

        df_loc = np.roll(df_loc, 1)


# @njit Note does not need jit
def Find_jk_GS(df, nu, Nnu, mode, theta, nu_p, nu_B, nu_cr, nu_cr_WH, Npoints, Q_on, m_on, j, k):
    """Function which handles the organization and filling of the j and k arrays. Performs post-processing on the arrays
       based on provided modification parameters.

       Handles optimization of the Gyrosynchotron calculations by switching between algorithms using exact and
       approximated bessel calculations.

       Calculations to be performed on j and k values corresponding to a value of nu lower than the critical point
       (nu_cr) are found exactly, while j and k calculations corresponding to a higher nu are handled approximately."""

    # t1 = t2 = 0

    tempj = np.full(np.shape(j), bigNeg)
    tempk = np.full(np.shape(k), bigNeg)

    tempj[:] = j
    tempk[:] = k

    for i in range(0, Nnu):
        tempj = np.roll(tempj, -i)
        tempk = np.roll(tempk, -i)

        w = np.array([EmWave(nu[i], theta, mode, nu_p, nu_B, 1, 0)])

        # Time elapsed between GS_jk run
        # td = t2 - t1

        if nu[i] > nu_cr:
            # # Debugging Purposes
            # print("Performing GS_jk_approx_mDF ", i, " / ", Nnu, " || Time Elapsed: ", td, " s")
            print("Performing GS_jk_approx_mDF ", i, " / ", Nnu)

            # t1 = time.perf_counter()

            GS_jk_approx_mDF(w, df, Npoints, Q_on, tempj, tempk)
        else:
            # # Debugging Purposes
            # print("Performing GS_JK_mDF ", i, " / ", Nnu, " || Time Elapsed: ", td, " s")
            print("Performing GS_JK_mDF ", i, " / ", Nnu)

            # t1 = time.perf_counter()

            GS_jk_mDF(w, df, nu[i] < nu_cr_WH, tempj, tempk)

        # t2 = time.perf_counter()

        tempj = np.roll(tempj, i)
        tempk = np.roll(tempk, i)

    if m_on:
        if nu_cr != nu_cr_WH and nu[0] < nu_cr < nu[Nnu - 1]:
            i0 = 0

            for i0 in range(1, Nnu):
                if nu[i0] > nu_cr:
                    break

            if tempj[i0] != 0.0 and tempk[i0] != 0.0:
                w = np.array([EmWave(nu[i0], theta, mode, nu_p, nu_B, 1, 0)])

                j0 = np.array([bigNeg])
                k0 = np.array([bigNeg])

                GS_jk_mDF(w, df, nu[i0] < nu_cr_WH, j0, k0)

                mj = j0 / tempj[i0]
                mk = k0 / tempk[i0]

                for i in range(i0, Nnu):
                    tempj[i] *= mj
                    tempk[i] *= mk

        if nu_cr >= nu_cr_WH > nu[0] and nu[Nnu - 1] > nu_cr_WH:
            i0 = 0

            for i0 in range(1, Nnu):
                if nu[i0] > nu_cr_WH:
                    break

            if tempj[i0] != 0.0 and tempk[i0] != 0.0:
                w = np.array([EmWave(nu[i0], theta, mode, nu_p, nu_B, 1, 0)])

                j0 = np.array([bigNeg])
                k0 = np.array([bigNeg])

                GS_jk_mDF(w, df, 1, j0, k0)

                mj = j0[0] / tempj[i0]
                mk = k0[0] / tempk[i0]

                for i in range(i0, Nnu):
                    tempj[i] *= mj
                    tempk[i] *= mk

    if nu_cr < 0.0 and nu_cr_WH > 0.0:
        w = np.array([EmWave(nu_cr_WH, theta, mode, nu_p, nu_B, 1, 0)])
        if w[0].Valid:
            ja = np.array([bigNeg])
            ka = np.array([bigNeg])
            GS_jk_approx_mDF(w, df, Npoints, Q_on, ja, ka)

            if ja[0] != 0.0 and ka[0] != 0.0:
                je = np.array([bigNeg])
                ke = np.array([bigNeg])
                GS_jk_mDF(w, df, 1, je, ke)
                mj = je[0] / ja[0]
                mk = ke[0] / ka[0]

                tempj *= mj
                tempk *= mk

    j[:] = tempj
    k[:] = tempk

