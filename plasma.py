from plasma_vars import *

import math
import numpy as np
from numba.experimental import jitclass
from numba import njit
import numba as nb


specEmWave = [
    ('Valid', nb.int32),
    ('sigma', nb.int32),

    ('nu', nb.float64),
    ('nu_p', nb.float64),
    ('nu_B', nb.float64),
    ('theta', nb.float64),

    ('ct', nb.float64),
    ('st', nb.float64),
    ('y', nb.float64),
    ('N', nb.float64),
    ('N_z', nb.float64),
    ('T', nb.float64),
    ('L', nb.float64),
    ('Zfactor', nb.float64),
]


@jitclass(specEmWave)
class EmWave:
    """An object meant to describe the various elements and properties of an electromagnetic wave, such as those emitted
       by gyrosynchotron activity."""

    def __init__(self, _nu=None, _theta=None, _sigma=None, _nu_p=None, _nu_B=None, LT_on=None, Zh_on=None):
        self.Valid = 0
        self.sigma = 0

        self.nu = 0.0
        self.nu_p = 0.0
        self.nu_B = 0.0
        self.theta = 0.0

        self.ct = 0.0
        self.st = 0.0
        self.y = 0.0
        self.N = 0.0
        self.N_z = 0.0
        self.T = 0.0
        self.L = 0.0
        self.Zfactor = 0.0

        self.nu = _nu
        self.theta = _theta
        self.nu_p = _nu_p
        self.nu_B = _nu_B
        self.sigma = _sigma

        # cutoff frequency
        nu_c = (self.nu_B / 2 + math.sqrt((self.nu_p ** 2) + (self.nu_B ** 2) / 4)) if (self.sigma == -1) else self.nu_p

        if self.nu <= nu_c:
            self.Valid = 0
        else:
            self.Valid = 1

            self.ct = math.cos(self.theta)
            self.st = math.sin(self.theta)

            if (math.fabs(self.ct)) < cst_min:
                self.ct = cst_min * (1.0 if (self.ct > 0) else (-1.0 if (self.ct < 0) else 0))
                self.st = math.sqrt(1.0 - (self.ct ** 2)) * (1.0 if (self.st > 0) else (-1.0 if (self.st < 0) else 0))
            if (math.fabs(self.st)) < cst_min:
                self.st = cst_min * (1.0 if (self.st > 0) else (-1.0 if (self.st < 0) else 0))
                self.ct = math.sqrt(1.0 - (self.st ** 2)) * (1.0 if (self.ct > 0) else (-1.0 if (self.ct < 0) else 0))

            self.y = self.nu / self.nu_B

            u = (self.nu_B / self.nu) ** 2
            v = (self.nu_p / self.nu) ** 2

            Delta = math.sqrt(((u * (self.st ** 2)) ** 2) + 4.0 * u * (((1.0 - v) * self.ct) ** 2))

            # refraction index
            self.N = math.sqrt(1.0 - 2.0 * v * (1.0 - v) / (2.0 * (1.0 - v) - u * (self.st ** 2) + self.sigma * Delta))
            # longitudinal component of the refraction index
            self.N_z = self.N * self.ct

            if LT_on:
                # axial polarization coefficient
                self.T = 2.0 * math.sqrt(u) * (1.0 - v) * self.ct / (u * (self.st ** 2) - self.sigma * Delta)
                # longitudinal polarization coefficient
                self.L = (v * math.sqrt(u) * self.st + self.T * u * v * self.st * self.ct) / (1.0 - u - v + u * v *
                         (self.ct ** 2))

            # Zheleznyakov's correction to free-free (free electron emitting Bremsstrahlung radiation and
            # remaining free)
            if Zh_on:
                self.Zfactor = 2.0 * (u * (self.st ** 2) + 2.0 * ((1.0 - v) ** 2) - self.sigma * ((u * (self.st ** 2))
                               ** 2) / Delta) / ((2.0 * (1.0 - v) - u * (self.st ** 2) + self.sigma * Delta) ** 2) if \
                               u else 1.0

            self.Valid = np.isfinite(self.N)


@njit
def SahaH(n0, T0):
    """
    An expression to relate the ionization state of a Helium Gas to its high temperature, under thermal
    equilibrium.

    :param n0: Density
    :param T0: Temperature
    :return: hydrogen ionization fraction
    """

    x = 0.0

    if T0 > 0.0 and n0 > 0.0:
        xi = math.pow(2.0 * math.pi * me * kB * T0 / (hPl ** 2), 1.5) / n0 * math.exp(-ieH / kB / T0)
        x = 2.0 / (math.sqrt(1.0 + 4.0 / xi) + 1.0) if xi else 0.0

    return x


@njit
def SahaHe(n_p, T0, a12, a2):
    """
    An expression to relate the ionization state of a Helium Gas to its high temperature, under thermal
    equilibrium.

    :param n_p: Density
    :param T0: Temperature
    :param a12: pointer for helium I+II ionization fraction
    :param a2: pointer for helium II ionization fraction
    """

    a12[0] = 0
    a2[0] = 0

    if T0 > 0.0 and n_p > 0.0:
        A = 4.0 * math.pow(2.0 * math.pi * me * kB * T0 / (hPl ** 2), 1.5) / n_p

        xi12 = A * math.exp(-ieHe12 / kB / T0)
        a12[0] = xi12 / (1.0 + xi12)  # helium I+II ionization fraction

        xi2 = A * math.exp(-ieHe2 / kB / T0)
        a2[0] = xi2 / (1.0 + xi2)  # helium II ionization fraction


@njit
def FindIonizationsSolar(n0, T0, n_e, n_H, n_He):
    """
    Function to calculate the degree of solar ionization through the use of two Saha ionization equation functions,
    one for Hydrogen and one for Helium.

    :param n0: Density
    :param T0: Temperature
    :param n_e: pointer for number of electrons
    :param n_H: pointer for number of Hydrogen
    :param n_He: pointer for number of Helium
    """

    bigNeg = -9.2559631349317831e+61

    n_Htotal = n0 * 0.922
    n_Hetotal = n0 * 0.078

    a = SahaH(n_Htotal, T0)  # Hydrogen ionization fraction
    n_p = n_Htotal * a
    n_H[0] = n_Htotal * (1.0 - a)

    a12 = np.array([bigNeg])
    a2 = np.array([bigNeg])

    SahaHe(n_p, T0, a12, a2)
    n_He[0] = n_Hetotal * (1.0 - a12[0])
    n_e[0] = n_p + n_Hetotal * (a12[0] + a2[0]) + n_Htotal * 1e-3
