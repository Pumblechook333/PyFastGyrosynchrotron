from extmath import ExpBesselK, qromb, Erf

from interface_header import *
from plasma_header import *

import math
import numpy as np
from numba.experimental import jitclass
import numba as nb

bigNeg = -9.2559631349317831e+61

FFF = 0  # free-free only
FFF1 = 1  # same as free-free only
THM = 2  # relativistic thermal distribution
PLW = 3  # power-law distribution on energy
DPL = 4  # double power-law distribution on energy
TNT = 5  # thermal + nonthermal on energy
KAP = 6  # kappa-distribution
PLP = 7  # power-law distribution on impulse module
PLG = 8  # power-law distribution on relativistic factor
TNP = 9  # thermal + nonthermal on impulse module
TNG = 10  # thermal + nonthermal on relativistic factor
TPL = 11  # isotropic maxwellian + power-law on energy
TDP = 12  # isotropic maxwellian + double power law

ISO = 0  # isotropic distribution
ISO1 = 1  # same as isotropic
ELC = 2  # exponential (on pitch-angle cosine) distribution
GAU = 3  # gaussian (on pitch-angle cosine) distribution
GAB = 4  # directed gaussian beam
SGA = 5  # directed gaussian beam with 4th power term


specTHM = [
    ('nb', nb.float64),
    ('N_intervals', nb.int64),
    ('E_x', nb.float64[:]),
    ('logscale', nb.float64[:]),

    ('A', nb.float64),
    ('theta', nb.float64)
]


# The following analytical built-in electron distribution functions are represented as a product of the energy
# (or momentum) distribution function 'Fp' and the angular distribution function 'FE'
@jitclass(specTHM)
class THMdf:  # 2
    """The Energy distribution function associated with a relativistic thermal distribution."""

    def __init__(self, Parms=None, OK=None, empty=None, Emax=None):

        self.nb = 0.0
        self.N_intervals = 0
        self.E_x = np.full(10, bigNeg)
        self.logscale = np.full(9, bigNeg)

        self.A = 0.0
        self.theta = 0.0

        T0 = math.fabs(Parms[i_T0])

        self.theta = kB * T0 / mc2
        self.A = Parms[i_n0] / (2.0 * math.pi * self.theta * ExpBesselK(2, 1.0 / self.theta))

        self.N_intervals = 3

        self.E_x[0] = 0.0
        self.E_x[1] = kB * T0
        self.E_x[2] = kB * T0 * 3
        self.E_x[3] = kB * T0 * 710
        self.logscale[0] = 0
        self.logscale[1] = 0
        self.logscale[2] = 1

        if Emax <= self.E_x[1]:
            self.N_intervals = 1
            self.E_x[1] = Emax

        elif Emax <= self.E_x[2]:
            self.N_intervals = 2
            self.E_x[2] = Emax

        elif Emax <= self.E_x[3]:
            self.E_x[3] = Emax

        OK[0] = np.isfinite(self.A) != 0 and self.A >= 0
        empty[0] = (self.A == 0.0)
        self.nb = 0.0

    def FE(self, E, f, f_E):

        G = E / mc2 + 1.0
        p = mc * math.sqrt((G ** 2) - 1.0)

        fp = np.array([bigNeg])
        dfp_dp = np.array([bigNeg])

        self.Fp(p, fp, dfp_dp)

        f[0] = fp[0] * p * me * G
        f_E[0] = ((me * G) ** 2) * (dfp_dp[0] + fp[0] / p) + fp * p / (c * c)

    def Fp(self, p, f, f_p):
        G = math.sqrt(1.0 + ((p / mc) ** 2))
        f[0] = self.A / (mc * mc * mc) * math.exp((1.0 - G) / self.theta)
        f_p[0] = -(f[0]) * p / G / self.theta / (mc * mc)


specPLW = [
    ('nb', nb.float64),
    ('N_intervals', nb.int64),
    ('E_x', nb.float64[:]),
    ('logscale', nb.float64[:]),

    ('A', nb.float64),
    ('delta', nb.float64)
]


@jitclass(specPLW)
class PLWdf:  # 3
    """The Single Power Law Distribution of nonthermal electrons over kinetic energy."""

    def __init__(self, Parms=None, OK=None, empty=None):

        self.nb = 0.0
        self.N_intervals = 0
        self.E_x = np.full(10, bigNeg)
        self.logscale = np.full(9, bigNeg)

        self.A = 0.0
        self.delta = 0.0

        E1 = Parms[i_Emin] * eV * 1e6
        E2 = Parms[i_Emax] * eV * 1e6
        self.delta = Parms[i_delta1]
        self.nb = Parms[i_nb]

        self.A = self.nb / (2.0 * math.pi) * (self.delta - 1.0) / (pow(E1, 1.0 - self.delta) - pow(E2, 1.0 - self.delta))

        self.N_intervals = 1
        self.E_x[0] = E1
        self.E_x[1] = E2
        self.logscale[0] = 1

        OK[0] = np.isfinite(self.A) != 0 and self.A >= 0.0 and E2 > E1
        empty[0] = (self.nb == 0.0)

    def FE(self, E, f, f_E):
        f[0] = self.A * math.pow(E, -self.delta)
        f_E[0] = (-self.delta / E) * (f[0])

    def Fp(self, p, f, f_p):
        G = math.sqrt(1.0 + ((p / mc) ** 2))
        E = mc2 * (G - 1.0)

        fE = np.array([bigNeg])
        dfE_dE = np.array([bigNeg])
        self.FE(E, fE, dfE_dE)

        f[0] = fE[0] / (p * me * G)
        f_p[0] = (dfE_dE[0] - fE[0] * G * me / (p ** 2) * (1.0 + ((p / G / mc) ** 2))) / ((me * G) ** 2)


specDPL = [
    ('nb', nb.float64),
    ('N_intervals', nb.int64),
    ('E_x', nb.float64[:]),
    ('logscale', nb.float64[:]),

    ('Ebr', nb.float64),
    ('A1', nb.float64),
    ('A2', nb.float64),
    ('delta1', nb.float64),
    ('delta2', nb.float64)
]


@jitclass(specDPL)
class DPLdf:  # 4
    """Double Power-Law Distribution of electrons over energy; for when the electron spectrum consists of a high-energy
       and a low-energy part.

       Two distributions: double power-law (Emin < E <= Ebreak) or broken power-law (Ebreak <= E < Emax)"""

    def __init__(self, Parms=None, OK=None, empty=None):

        self.nb = 0.0
        self.N_intervals = 0
        self.E_x = np.full(10, bigNeg)
        self.logscale = np.full(9, bigNeg)

        self.Ebr = 0.0
        self.A1 = 0.0
        self.A2 = 0.0
        self.delta1 = 0.0
        self.delta2 = 0.0

        E1 = Parms[i_Emin] * eV * 1e6
        E2 = Parms[i_Emax] * eV * 1e6
        self.Ebr = Parms[i_Ebreak] * eV * 1e6
        self.delta1 = Parms[i_delta1]
        self.delta2 = Parms[i_delta2]
        self.nb = Parms[i_nb]

        self.A1 = self.nb / (2.0 * math.pi) / ((pow(E1, 1.0 - self.delta1) - pow(self.Ebr, 1.0 - self.delta1)) /
                  (self.delta1 - 1.0) + pow(self.Ebr, self.delta2 - self.delta1) * (pow(self.Ebr, 1.0 - self.delta2) -
                  pow(E2, 1.0 - self.delta2)) / (self.delta2 - 1.0))

        self.A2 = self.A1 * pow(self.Ebr, self.delta2 - self.delta1)

        self.N_intervals = 2
        self.E_x[0] = E1
        self.E_x[1] = self.Ebr
        self.E_x[2] = E2
        self.logscale[0] = 1
        self.logscale[1] = 1

        OK[0] = np.isfinite(self.A1) != 0 and self.A1 >= 0.0 and np.isfinite(
            self.A2) != 0 and self.A2 >= 0.0 and E2 > self.Ebr > E1
        empty[0] = (self.nb == 0.0)

    def FE(self, E, f, f_E):
        if E < self.Ebr:
            f[0] = self.A1 * pow(E, -self.delta1)
            f_E[0] = (-self.delta1 / E) * (f[0])
        else:
            f[0] = self.A2 * pow(E, -self.delta2)
            f_E[0] = (-self.delta2 / E) * (f[0])

    def Fp(self, p, f, f_p):
        G = math.sqrt(1.0 + ((p / mc) ** 2))
        E = mc2 * (G - 1.0)

        fE = np.array([bigNeg])
        dfE_dE = np.array([bigNeg])
        self.FE(E, fE, dfE_dE)

        f[0] = fE / (p * me * G)
        f_p[0] = (dfE_dE - fE * G * me / (p ** 2) * (1.0 + ((p / G / mc) ** 2))) / ((me * G) ** 2)


specKapInt = [
    ('kappa_m32_theta', nb.float64),
    ('kappa_pi', nb.float64)
]


@jitclass(specKapInt)
class KappaIntegrand:
    """Calculation kernel for the Kappa distribution function (KAPdf)"""

    def __init__(self):
        self.kappa_m32_theta = 0.0
        self.kappa_p1 = 0.0

    def F(self, G):
        return G * math.sqrt((G ** 2) - 1.0) * pow(1.0 + (G - 1.0) / self.kappa_m32_theta, -self.kappa_p1)


specKAP = [
    ('nb', nb.float64),
    ('N_intervals', nb.int64),
    ('E_x', nb.float64[:]),
    ('logscale', nb.float64[:]),

    ('A', nb.float64),
    ('kappa_m32_theta', nb.float64),
    ('kappa_pi', nb.float64)

]


@jitclass(specKAP)
class KAPdf:
    """Kappa distribution (transition from thermal distribution to a non-thermal tail) used to quantify particle
       distributions in the interplanetary plasma."""

    def __init__(self, Parms=None, OK=None, empty=None):

        self.nb = 0.0
        self.N_intervals = 0
        self.E_x = np.full(10, bigNeg)
        self.logscale = np.full(9, bigNeg)

        self.A = 0.0
        self.kappa_p1 = 0.0
        self.kappa_m32_theta = 0.0

        T0 = math.fabs(Parms[i_T0])
        kappa = Parms[i_epskappa]
        E_max = Parms[i_Emax] * eV * 1e6

        theta = kB * T0 / mc2
        self.kappa_p1 = kappa + 1.0
        self.kappa_m32_theta = (kappa - 1.5) * theta

        G_max = E_max / mc2 + 1.0
        ki = KappaIntegrand()
        ki.kappa_m32_theta = self.kappa_m32_theta
        ki.kappa_p1 = self.kappa_p1
        err = 0

        self.A = Parms[i_n0] / (2.0 * math.pi) / qromb(ki, 1.0, G_max, 1e-6, err)

        self.E_x[0] = 0.0
        if E_max <= (kB * T0):
            self.N_intervals = 1
            self.E_x[1] = E_max
            self.logscale[0] = 0
        elif E_max <= (kB * T0 * 3):
            self.N_intervals = 2
            self.E_x[1] = kB * T0
            self.E_x[2] = E_max
            self.logscale[0] = self.logscale[1] = 0
        else:
            self.N_intervals = 3
            self.E_x[1] = kB * T0
            self.E_x[2] = kB * T0 * 3
            self.E_x[3] = E_max
            self.logscale[0] = self.logscale[1] = 0
            self.logscale[2] = (self.E_x[3] / self.E_x[2] > 5.0)

        OK[0] = np.isfinite(self.A) != 0 and self.A >= 0.0
        empty[0] = (self.A == 0.0)
        self.nb = 0.0

    def FE(self, E, f, f_E):
        G = E / mc2 + 1.0
        p = mc * math.sqrt((G ** 2) - 1.0)

        fp = np.array([bigNeg])
        dfp_dp = np.array([bigNeg])
        self.Fp(p, fp, dfp_dp)

        f[0] = fp * p * me * G
        f_E[0] = ((me * G) ** 2) * (dfp_dp + fp / p) + fp * p / (c * c)

    def Fp(self, p, f, f_p):
        G = math.sqrt(1.0 + ((p / mc) ** 2))
        D = 1.0 + (G - 1.0) / self.kappa_m32_theta
        f[0] = self.A / (mc * mc * mc) * pow(D, -self.kappa_p1)
        f_p[0] = -self.kappa_p1 * (f[0]) / D / self.kappa_m32_theta * p / G / (mc * mc)


specPLP = [
    ('nb', nb.float64),
    ('N_intervals', nb.int64),
    ('E_x', nb.float64[:]),
    ('logscale', nb.float64[:]),

    ('A', nb.float64),
    ('delta', nb.float64)
]


@jitclass(specPLP)
class PLPdf:
    """Power-Law distribution of the non-thermal electrons over the absolute value of their momentum."""

    def __init__(self, Parms=None, OK=None, empty=None):

        self.nb = 0.0
        self.N_intervals = 0
        self.E_x = np.full(10, bigNeg)
        self.logscale = np.full(9, bigNeg)

        self.A = 0.0
        self.delta = 0.0

        E1 = Parms[i_Emin] * eV * 1e6
        E2 = Parms[i_Emax] * eV * 1e6
        self.delta = Parms[i_delta1]
        self.nb = Parms[i_nb]

        p1 = mc * math.sqrt(((E1 / mc2 + 1.0) ** 2) - 1.0)
        p2 = mc * math.sqrt(((E2 / mc2 + 1.0) ** 2) - 1.0)

        self.A = self.nb / (2.0 * math.pi) * (self.delta - 3.0) / (pow(p1, 3.0 - self.delta) - pow(p2, 3.0 -
                 self.delta))

        self.N_intervals = 1
        self.E_x[0] = E1
        self.E_x[1] = E2
        self.logscale[0] = 1

        OK[0] = np.isfinite(self.A) != 0 and self.A >= 0.0 and E2 > E1
        empty[0] = (self.nb == 0.0)

    def FE(self, E, f, f_E):
        G = E / mc2 + 1.0
        p = mc * math.sqrt((G ** 2) - 1.0)

        fp = np.array([bigNeg])
        dfp_dp = np.array([bigNeg])
        self.Fp(p, fp, dfp_dp)

        f[0] = fp[0] * p * me * G
        f_E[0] = ((me * G) ** 2) * (dfp_dp[0] + fp / p) + fp * p / (c * c)

    def Fp(self, p, f, f_p):
        f[0] = self.A * pow(p, -self.delta)
        f_p[0] = -self.delta * (f[0]) / p


specPLG = [
    ('nb', nb.float64),
    ('N_intervals', nb.int64),
    ('E_x', nb.float64[:]),
    ('logscale', nb.float64[:]),

    ('A', nb.float64),
    ('delta', nb.float64)
]


@jitclass(specPLG)
class PLGdf:
    """Power-law distribution of the non-thermal electrons over their Lorentz factor"""

    def __init__(self, Parms=None, OK=None, empty=None):

        self.nb = 0.0
        self.N_intervals = 0
        self.E_x = np.full(10, bigNeg)
        self.logscale = np.full(9, bigNeg)

        self.A = 0.0
        self.delta = 0.0

        E1 = Parms[i_Emin] * eV * 1e6
        E2 = Parms[i_Emax] * eV * 1e6
        self.delta = Parms[i_delta1]
        self.nb = Parms[i_nb]

        G1 = E1 / mc2 + 1.0
        G2 = E2 / mc2 + 1.0

        self.A = self.nb / (2.0 * math.pi) * (self.delta - 1.0) / (
                pow(G1, 1.0 - self.delta) - pow(G2, 1.0 - self.delta))

        self.N_intervals = 1
        self.E_x[0] = E1
        self.E_x[1] = E2
        self.logscale[0] = 1

        OK[0] = np.isfinite(self.A) != 0 and self.A >= 0.0 and E2 > E1
        empty[0] = (self.nb == 0.0)

    def FG(self, G, f, f_G):
        f[0] = self.A * pow(G, -self.delta)
        f_G[0] = -self.delta * (f[0]) / G

    def FE(self, E, f, f_E):
        G = E / mc2 + 1.0

        fG = np.array([bigNeg])
        dfG_dG = np.array([bigNeg])
        self.FG(G, fG, dfG_dG)

        f[0] = fG[0] / mc2
        f_E[0] = dfG_dG[0] / (mc2 * mc2)

    def Fp(self, p, f, f_p):
        G = math.sqrt(1.0 + ((p / mc) ** 2))

        fG = np.array([bigNeg])
        dfG_dG = np.array([bigNeg])
        self.FG(G, fG, dfG_dG)

        f[0] = fG[0] / (p * G * mc * mc)
        f_p[0] = (dfG_dG[0] - fG[0] / G * (((G * mc / p) ** 2) + 1.0)) / ((G * mc * mc) ** 2)


specISO = [
    ('EPS_mu0', nb.float64)
]


@jitclass(specISO)
class ISOdf:  # 0,1
    """Describes an electron distribution which exists isotropically, independent of pitch-angle."""

    def __init__(self, OK=None):
        self.EPS_mu0 = 1e-3

        OK[0] = 1

    def Falpha(self, mu, sa, f, f_alpha):
        f[0] = 0.5
        f_alpha[0] = 0.0

    def Fmu(self, mu, f, f_mu, g1, g2):
        f[0] = 0.5
        f_mu[0] = 0.0
        if g1:
            g1[0] = 0.0
        if g2:
            g2[0] = 0.0

    def g1short(self, mu):
        return 0.0


specELC = [
    ('B', nb.float64),
    ('alpha_c', nb.float64),
    ('mu_c', nb.float64),
    ('dmu', nb.float64),

    ('EPS_mu0', nb.float64)
]


@jitclass(specELC)
class ELCdf:  # 2
    """Describes a loss-cone electron distribution with a symmetric exponential boundary."""

    def __init__(self, Parms=None, OK=None):
        self.B = 0.0
        self.alpha_c = 0.0
        self.mu_c = 0.0
        self.dmu = 0.0

        self.alpha_c = Parms[i_alphaC] * math.pi / 180
        self.dmu = Parms[i_dmu]

        self.mu_c = math.fabs(math.cos(self.alpha_c))
        self.B = 0.5 / (self.mu_c + self.dmu - self.dmu * math.exp((self.mu_c - 1.0) / self.dmu))

        self.EPS_mu0 = 1e-3
        self.EPS_mu0 = self.EPS_mu0 if (self.EPS_mu0 < (self.dmu / 30)) else (self.dmu / 30)
        OK[0] = np.isfinite(self.B) != 0 and self.B > 0.0

    def Falpha(self, mu, sa, f, f_alpha):
        self.Fmu(mu, f, f_alpha, 0, 0)
        f_alpha[0] *= (-sa)

    def Fmu(self, mu, f, f_mu, g1, g2):
        amu = math.fabs(mu)

        if amu < self.mu_c:
            f[0] = self.B
            f_mu[0] = 0.0
            if g1:
                g1[0] = 0.0
            if g2:
                g2[0] = 0.0
        else:
            f[0] = self.B * math.exp(-(amu - self.mu_c) / self.dmu)
            g1loc = -(1.0 if (mu > 0) else (-1.0 if (mu < 0) else 0)) / self.dmu
            f_mu[0] = f[0] * g1loc

            if g1:
                g1[0] = g1loc
            if g2:
                g2[0] = 1.0 / (self.dmu ** 2)

    def g1short(self, mu):
        amu = math.fabs(mu)

        if amu < self.mu_c:
            return 0.0
        else:
            return -(1.0 if (mu > 0) else (-1.0 if (mu < 0) else 0)) / self.dmu


specGAU = [
    ('B', nb.float64),
    ('alpha_c', nb.float64),
    ('mu_c', nb.float64),
    ('dmu', nb.float64),

    ('EPS_mu0', nb.float64)
]


#@jitclass(specGAU)
class GAUdf:
    """Describes the Gaussian distribution of electrons over pitch angle."""

    def __init__(self, Parms=None, OK=None):
        self.B = 0.0
        self.alpha_c = 0.0
        self.mu_c = 0.0
        self.dmu = 0.0

        self.alpha_c = Parms[i_alphaC] * math.pi / 180
        self.dmu = Parms[i_dmu]

        self.mu_c = math.fabs(math.cos(self.alpha_c))
        self.B = 0.5 / (self.mu_c + self.dmu * math.sqrt(math.pi) / 2 * Erf((1.0 - self.mu_c) / self.dmu))

        self.EPS_mu0 = 1e-3
        self.EPS_mu0 = self.EPS_mu0 if (self.EPS_mu0 < ((self.dmu ** 2) / 30)) else ((self.dmu ** 2) / 30)

        OK[0] = np.isfinite(self.B) != 0 and self.B > 0.0

    def Falpha(self, mu, sa, f, f_alpha):
        self.Fmu(mu, f, f_alpha, 0, 0)
        f_alpha[0] *= -sa

    def Fmu(self, mu, f, f_mu, g1, g2):
        amu = math.fabs(mu)

        if amu < self.mu_c:
            f[0] = self.B
            f_mu[0] = 0.0

            # Note change to account for g1, g2 = 0
            if g1 != 0:
                g1[0] = 0.0
            if g2 != 0:
                g2[0] = 0.0
        else:
            f[0] = self.B * math.exp(-(((amu - self.mu_c) / self.dmu) ** 2))
            g1loc = -2.0 * (amu - self.mu_c) / (self.dmu ** 2) * (1.0 if (mu > 0) else (-1.0 if (mu < 0) else 0))
            f_mu[0] = f[0] * g1loc
            if g1:
                g1[0] = g1loc
            if g2:
                g2[0] = 4.0 * (((amu - self.mu_c) / (self.dmu ** 2)) ** 2) - 2.0 / (self.dmu ** 2)

    def g1short(self, mu):
        amu = math.fabs(mu)
        if amu < self.mu_c:
            return 0.0
        else:
            return -2.0 * (amu - self.mu_c) / (self.dmu ** 2) * (1.0 if (mu > 0) else (-1.0 if (mu < 0) else 0))


specGAB = [
    ('B', nb.float64),
    ('alpha_c', nb.float64),
    ('mu_c', nb.float64),
    ('dmu', nb.float64),

    ('EPS_mu0', nb.float64)
]


@jitclass(specGAB)
class GABdf:
    """Distribution function for a directed Gaussian beam."""

    def __init__(self, Parms=None, OK=None):
        self.B = 0.0
        self.alpha_c = 0.0
        self.mu_c = 0.0
        self.dmu = 0.0

        self.alpha_c = Parms[i_alphaC] * math.pi / 180
        self.dmu = Parms[i_dmu]

        self.mu_c = math.cos(self.alpha_c)
        self.B = 2.0 / (math.sqrt(math.pi) * self.dmu) / (Erf((1.0 - self.mu_c) / self.dmu) + Erf((1.0 + self.mu_c) /
                 self.dmu))

        self.EPS_mu0 = 1e-3
        self.EPS_mu0 = self.EPS_mu0 if (self.EPS_mu0 < ((self.dmu ** 2) / 30)) else ((self.dmu ** 2) / 30)

        OK[0] = np.isfinite(self.B) != 0 and self.B > 0.0

    def Falpha(self, mu, sa, f, f_alpha):
        self.Fmu(mu, f, f_alpha, 0, 0)
        f_alpha[0] *= (-sa)

    def Fmu(self, mu, f, f_mu, g1, g2):
        f[0] = self.B * math.exp(-(((mu - self.mu_c) / self.dmu) ** 2))
        g1loc = -2.0 * (mu - self.mu_c) / (self.dmu ** 2)
        f_mu[0] = f[0] * g1loc
        if g1:
            g1[0] = g1loc
        if g2:
            g2[0] = 4.0 * (((mu - self.mu_c) / (self.dmu ** 2)) ** 2) - 2.0 / (self.dmu ** 2)

    def g1short(self, mu):
        return -2.0 * (mu - self.mu_c) / (self.dmu ** 2)


specSgaInt = [
    ('mu_c', nb.float64),
    ('dmu', nb.float64),
    ('a4', nb.float64)
]


@jitclass(specSgaInt)
class SGAIntegrand:
    """Calculation Kernel for Super Gaussian distribution function (SGAdf)"""

    def __init__(self):
        self.mu_c = 0.0
        self.dmu = 0.0
        self.a4 = 0.0

    def F(self, mu):
        d2 = (mu - self.mu_c) ** 2
        return math.exp(-(d2 + self.a4 * (d2 ** 2)) / (self.dmu ** 2))


specSGA = [
    ('B', nb.float64),
    ('alpha_c', nb.float64),
    ('mu_c', nb.float64),
    ('dmu', nb.float64),

    ('EPS_mu0', nb.float64),
    ('a4', nb.float64)
]


@jitclass(specSGA)
class SGAdf:
    """Describes a distribution similar to Gaussian distribution near maximum mu0, but decreases more rapidly at some
       angular distance away from mu0."""

    def __init__(self, Parms=None, OK=None):
        self.B = 0.0
        self.alpha_c = 0.0
        self.mu_c = 0.0
        self.dmu = 0.0
        self.a4 = 0.0

        self.alpha_c = Parms[i_alphaC] * math.pi / 180
        self.dmu = Parms[i_dmu]
        self.a4 = Parms[i_a4]

        self.mu_c = math.cos(self.alpha_c)
        sgi = np.array([SGAIntegrand()])
        sgi[0].mu_c = self.mu_c
        sgi[0].dmu = self.dmu
        sgi[0].a4 = self.a4
        err = 0
        self.B = 1.0 / qromb(sgi, -1, 1, 1e-10, err)

        self.EPS_mu0 = 1e-3
        self.EPS_mu0 = self.EPS_mu0 if (self.EPS_mu0 < ((self.dmu ** 2) / 30)) else ((self.dmu ** 2) / 30)
        OK[0] = np.isfinite(self.B) != 0 and self.B > 0.0

    def Falpha(self, mu, sa, f, f_alpha):
        self.Fmu(mu, f, f_alpha, 0, 0)
        f_alpha[0] *= (-sa)

    def Fmu(self, mu, f, f_mu, g1, g2):
        d2 = (mu - self.mu_c) ** 2
        f[0] = self.B * math.exp(-(d2 + self.a4 * (d2 ** 2)) / (self.dmu ** 2))
        g1loc = -2.0 * (mu - self.mu_c) * (1.0 + 2.0 * self.a4 * d2) / (self.dmu ** 2)
        f_mu[0] = f[0] * g1loc
        if g1:
            g1[0] = g1loc
        if g2:
            g2[0] = 2.0 / (self.dmu ** 4) * \
                    (2.0 * d2 * (1.0 - 3.0 * self.a4 * (self.dmu ** 2) + 4.0 * self.a4 * d2 + 4.0 * ((
                        self.a4 * d2) ** 2)) - (self.dmu ** 2))

    def g1short(self, mu):
        d2 = (mu - self.mu_c) ** 2
        return -2.0 * (mu - self.mu_c) * (1.0 + 2.0 * self.a4 * d2) / (self.dmu ** 2)


# specStd = [
#     ('N_intervals', nb.int64),
#     ('E_x', nb.float64[:]),
#     ('logscale', nb.float64[:]),
#     ('nb', nb.float64),
#
#     ('EPS_mu0', nb.float64),
#     ('PK_on', nb.int32),
#     ('F1', nb.pyobject),
#     ('F2', nb.pyobject)
# ]
#
#
# @jitclass(specStd)
class Std_DF:
    """An object to handle the switching and execution between the analytical Standard Distribution Functions."""

    def __init__(self, Parms=None, k=None, OK=None, empty=None, kap_on=None, Done=None):

        self.N_intervals = 0
        self.E_x = np.full(10, bigNeg)
        self.logscale = np.full(9, bigNeg)
        self.nb = 0.0
        self.EPS_mu0 = 0.0
        self.PK_on = 0

        self.F1 = np.array([None])
        self.F2 = np.array([None])

        E_id = Parms[i_EId]
        mu_id = Parms[i_muId]

        kap_on[0] = 0

        if E_id == THM:
            self.F1[0] = THMdf(Parms, OK, empty)
            Done[0] = 1
        elif E_id == PLW:
            self.F1[0] = PLWdf(Parms, OK, empty)
            Done[0] = 1
        elif E_id == DPL:
            self.F1[0] = DPLdf(Parms, OK, empty)
            Done[0] = 1
        elif E_id == TNT:
            pass
        elif E_id == TNP:
            pass
        elif E_id == TNG:
            pcr = mc * math.sqrt((((kB * Parms[i_T0] / mc2 + 1.0) ** 2) - 1.0) / Parms[i_epskappa])
            Gcr = math.sqrt(1.0 + ((pcr / mc) ** 2))
            Ecr = mc2 * (Gcr - 1.0)

            if not k:
                self.F1[0] = THMdf(Parms, OK, empty, Ecr)
            else:
                E_max = Parms[i_Emax] * eV * 1e6
                G_max = E_max / mc2 + 1.0
                p_max = mc * math.sqrt((G_max ** 2) - 1.0)
                delta = Parms[i_delta1]

                if E_max > Ecr:
                    thm = THMdf(Parms, OK, empty)

                    if OK[0] and not empty[0]:
                        fcr = np.array([bigNeg])
                        tmp = np.array([bigNeg])
                        Acr = np.array([bigNeg])
                        nb = np.array([bigNeg])

                        if E_id == TNT:
                            thm.FE(Ecr, fcr, tmp)
                            Acr[0] = fcr[0] * pow(Ecr, delta)
                            nb[0] = Acr[0] * 2.0 * math.pi / (delta - 1.0) * (
                                    pow(Ecr, 1.0 - delta) - pow(E_max, 1.0 - delta))
                        elif E_id == TNP:
                            thm.Fp(pcr, fcr, tmp)
                            Acr[0] = fcr[0] * pow(pcr, delta)
                            nb[0] = Acr[0] * 2.0 * math.pi / (delta - 3.0) * (
                                    pow(pcr, 3.0 - delta) - pow(p_max, 3.0 - delta))
                        elif E_id == TNG:
                            thm.FE(Ecr, fcr, tmp)
                            Acr[0] = fcr[0] * mc2 * pow(Gcr, delta)
                            nb[0] = Acr[0] * 2.0 * math.pi / (delta - 1.0) * (
                                    pow(Gcr, 1.0 - delta) - pow(G_max, 1.0 - delta))

                        ParmsLoc = np.full(InSize, bigNeg)

                        # np.copyto(ParmsLoc, Parms)
                        ParmsLoc[:] = Parms

                        ParmsLoc[i_Emin] = Ecr / eV / 1e6
                        ParmsLoc[i_nb] = nb

                        if E_id == TNT:
                            self.F1[0] = PLWdf(ParmsLoc, OK, empty)
                        elif E_id == TNP:
                            self.F1[0] = PLPdf(ParmsLoc, OK, empty)
                        elif E_id == TNG:
                            self.F1[0] = PLGdf(ParmsLoc, OK, empty)

                else:
                    OK[0] = 1
                    empty[0] = 1

                Done[0] = 1

        elif E_id == KAP:
            self.F1[0] = KAPdf(Parms, OK, empty)
            Done[0] = 1
            kap_on[0] = 1

        elif E_id == PLP:
            self.F1[0] = PLPdf(Parms, OK, empty)
            Done[0] = 1

        elif E_id == PLG:
            self.F1[0] = PLGdf(Parms, OK, empty)
            Done[0] = 1

        elif E_id == TPL:
            pass

        elif E_id == TDP:
            if not k:
                self.F1[0] = THMdf(Parms, OK, empty)
                mu_id = ISO
            else:
                if E_id == TPL:
                    self.F1[0] = PLWdf(Parms, OK, empty)
                elif E_id == TDP:
                    self.F1[0] = DPLdf(Parms, OK, empty)

                Done[0] = 1

        else:
            OK[0] = 1
            empty[0] = 1
            Done[0] = 1

        if OK[0] and not (empty[0]):
            self.nb = self.F1[0].nb
            self.N_intervals = self.F1[0].N_intervals

            self.logscale = self.F1[0].logscale
            self.E_x = self.F1[0].E_x

            if mu_id == ISO:
                pass
            elif mu_id == ISO1:
                self.F2[0] = ISOdf(OK)
            elif mu_id == ELC:
                self.F2[0] = ELCdf(Parms, OK)
            elif mu_id == GAU:
                self.F2[0] = GAUdf(Parms, OK)
            elif mu_id == GAB:
                self.F2[0] = GABdf(Parms, OK)
            elif mu_id == SGA:
                self.F2[0] = SGAdf(Parms, OK)
            else:
                self.F2[0] = ISOdf(OK)

            if OK[0]:
                self.EPS_mu0 = self.F2[0].EPS_mu0

    def Fp(self, p, p_z, p_n, f, df_dp, df_dalpha):
        f1 = np.array([bigNeg])
        f2 = np.array([bigNeg])
        df1_dp = np.array([bigNeg])
        df2_dalpha = np.array([bigNeg])

        self.F1[0].Fp(p, f1, df1_dp)

        mu = (p_z / p) if (p > 0.0) else 0.0
        sa = p_n / p if (p > 0.0) else 0.0

        if mu > 1.0:
            mu = 1.0
        if mu < (-1.0):
            mu = -1.0
        if sa > 1.0:
            sa = 1.0
        if sa < (-1.0):
            sa = -1.0

        self.F2[0].Falpha(mu, sa, f2, df2_dalpha)

        f[0] = f1[0] * f2[0]
        df_dp[0] = df1_dp[0] * f2[0]
        df_dalpha[0] = f1[0] * df2_dalpha[0]

    def FE(self, E, mu, f, df_dE, df_dmu, g1, g2):
        if not f:
            g1[0] = self.F2[0].g1short(mu)
        else:
            f1 = np.array([bigNeg])
            f2 = np.array([bigNeg])
            df1_dE = np.array([bigNeg])
            df2_dmu = np.array([bigNeg])

            self.F1[0].FE(E, f1, df1_dE)
            self.F2[0].Fmu(mu, f2, df2_dmu, g1, g2)
            f[0] = f1[0] * f2[0]
            df_dE[0] = df1_dE[0] * f2[0]
            df_dmu[0] = f1[0] * df2_dmu[0]
