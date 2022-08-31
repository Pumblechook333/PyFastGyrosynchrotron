from plasma_header import *

from plasma import EmWave
from coulomb import lnC1
from zeta import Zeta_Solar
from extmath import Gamma

import math
import numpy as np
from numba import njit


@njit
def CoulombOld(T0, nu):
    """Calculation kernel for holding information about an EMWave object's current coulomb logarithm."""

    return 18.2 + 1.5 * math.log(T0) - math.log(nu) if (T0 < 2e5) else 24.573 + math.log(T0 / nu)


@njit
def ZmeanOld(T0):
    """Calculation kernel for holding information about the current average zeta function."""

    return 1.146 if (T0 > 3.5e4) else 1.0


# @njit
def FF_jk_Maxwell(w, ne, T0, ab, j, k):
    """
    For when kappa is infinite, this method approximates the gyrosynchotron regime for isotropic power-law energy
    electron distributions. Swaps between the following two methods:

    |

    The simple analytical Dulk & Marsh (1982) approximation is employed for a limited range of harmonic numbers
    (20-100), and moderate viewing angles (30-80 degrees) relative to the magnetic field. No thermal plasma effect
    is included. This method can be used for rough estimates and parametric dependencies, but falls apart for
    quantitative treatment or detailed modelling.

    |

    The modern, fast numerical Petrosian-Klein (Petrosian 1981, Klein 1987) approximation calculates GS radiation
    including the plasma effect, making it valid for a broader range of parameters, at high harmonic frequencies.
    This method is only valid for isotropic and weakly anisotropic angular distributions of emitting electrons.
    """

    if ab < 0:  # classical Dulk's formula
        lnC = CoulombOld(T0, w[0].nu)
        kFF = 9.786e-3 * (ne ** 2) * lnC / (w[0].N * (w[0].nu ** 2) * T0 * math.sqrt(T0))

        k[0] = kFF * w[0].Zfactor * ZmeanOld(T0)
        j[0] = (k[0]) * ((w[0].N * w[0].nu / c) ** 2) * kB * T0

    else:  # modern formulae with correct Coulomb logarithm and zeta-function
        lnC = lnC1(T0, w[0].nu)
        zeta = Zeta_Solar(T0, w[0].nu, ab)

        jff = 8 * e * e * e * e * e * e * w[0].N / (3.0 * math.sqrt(2.0 * math.pi) * math.sqrt(me * me * me) * c * c *
              c) * (ne ** 2) * lnC / math.sqrt(kB * T0) * (1.0 + zeta)
        kff = 8 * e * e * e * e * e * e / (3.0 * math.sqrt(2.0 * math.pi) * w[0].N * c * (w[0].nu ** 2) * math.sqrt(me
              * me * me)) * (ne ** 2) * lnC / (math.sqrt(kB * T0) * kB * T0) * (1.0 + zeta)

        j[0] = jff * w[0].Zfactor
        k[0] = kff * w[0].Zfactor


@njit
def FF_jk_kappa(w, ne, T0, kappa, ab, j, k):  # ///
    """
    For when kappa is finite, this method approximates the gyrosynchotron regime for anisotropic power-law energy
    electron distributions. Swaps between the following two methods:

    |

    The fast exact classical Gyrosynchotron calculations of Fleishman and Kuznetov (2014).

    |

    The modern, fast numerical Petrosian-Klein (Petrosian 1981, Klein 1987) approximation calculates GS radiation
    including the plasma effect, making it valid for a broader range of parameters, at high harmonic frequencies.
    This method is only valid for isotropic and weakly anisotropic angular distributions of emitting electrons.
    """

    Ak = Gamma(kappa + 1.0) / Gamma(kappa - 0.5) * pow(kappa - 1.5, -1.5)

    if ab < 0:  # classical formulae from Fleishman & Kuznetsov (2014)
        lnC = CoulombOld(T0, w[0].nu)

        kFF = Ak * 8.0 * ((e * e * e) ** 2) * (ne ** 2) * lnC / (3.0 * math.sqrt(2.0 * math.pi) * w[0].N * c * (w[0].nu
              ** 2) * me * kB * T0 * math.sqrt(me * kB * T0)) * (1.0 - 0.575 * pow(6.0 / kappa, 1.1) / lnC)

        jFF = Ak * (kappa - 1.5) / kappa * 8.0 * ((e * e * e) ** 2) * w[0].N * (ne ** 2) * lnC / (3.0 * math.sqrt(2.0 *
              math.pi) * mc2 * math.sqrt(mc2) * math.sqrt(kB * T0)) * (1.0 - 0.525 * pow(4.0 / kappa, 1.25) / lnC)

        k[0] = kFF * w[0].Zfactor * ZmeanOld(T0)
        j[0] = jFF * w[0].Zfactor * ZmeanOld(T0)

    else:  # more accurate formulae
        lnC = lnC1(T0, w[0].nu)
        zeta = Zeta_Solar(T0, w[0].nu, ab)

        kFF = Ak * 8 * e * e * e * e * e * e * (ne ** 2) * lnC * (1.0 + zeta) / (3.0 * math.sqrt(2.0 * math.pi) *
              w[0].N * c * (w[0].nu ** 2) * me * kB * T0 * math.sqrt(me * kB * T0))

        jFF = Ak * (kappa - 1.5) / kappa * 8 * e * e * e * e * e * e * w[0].N * (ne ** 2) * lnC * (1.0 + zeta) / (3.0 *
              math.sqrt(2.0 * math.pi) * mc2 * math.sqrt(mc2) * math.sqrt(kB * T0))

        k[0] = kFF * w[0].Zfactor
        j[0] = jFF * w[0].Zfactor


# @njit Note does not need
def Find_jk_FFei(ne, T0, nu_p, nu_B, theta, kappa, abcode, sigma, nu, j, k):
    """
    The correct operations to perform on GS j and k values when the parameter e-ions flag is turned on.
    Switches between use of the FF_jk_kappa and FF_jk_Maxwell equations depending on whether or not kappa is finite.
    """

    w = np.array([EmWave(nu, theta, sigma, nu_p, nu_B, 0, 1)])

    if not w[0].Valid:
        j[0] = 0
        k[0] = 1e100
    elif ne == 0.0:
        j[0] = 0.0
        k[0] = 0.0
    else:
        ab = 0
        if abcode == 1:
            ab = 1
        if abcode == -1:
            ab = -1

        if np.isfinite(kappa):
            FF_jk_kappa(w, ne, T0, kappa, ab, j, k)
        else:
            FF_jk_Maxwell(w, ne, T0, ab, j, k)


@njit
def Find_jk_FFen(ne, nH, nHe, T0, nu_p, nu_B, theta, sigma, nu, j, k):
    """The correct operations to perform on GS j and k values when the parameter e-neutral flag is turned on."""

    w = EmWave(nu, theta, sigma, nu_p, nu_B, 0, 1)

    if not w.Valid:
        j[0] = 0
        k[0] = 1e100
    elif ne == 0.0:
        j[0] = 0.0
        k[0] = 0.0
    else:
        jH = 0.0
        kH = 0.0

        if ne > 0 and nH > 0 and 2500 < T0 < 50000:
            kT = math.sqrt(kB * T0 / ieH)
            xi = 4.862 * kT * (1.0 - 0.2096 * kT + 0.0170 * kT * kT - 0.00968 * kT * kT * kT)
            kH = 1.2737207e-11 * ne * nH * math.sqrt(T0) / ((nu ** 2) * w.N) * math.exp(-xi)

            jH = kH * ((w.N * nu / c) ** 2) * kB * T0

        jHe = 0.0
        kHe = 0.0

        if ne > 0 and nHe > 0 and 2500 < T0 < 25000:
            kT = math.sqrt(kB * T0 / ieH)
            kHe = 5.9375453e-13 * ne * nHe * math.sqrt(T0) / ((nu ** 2) * w.N) * (
                    1.868 + 7.415 * kT - 22.56 * kT * kT + 15.59 * kT * kT * kT)

            jHe = kHe * ((w.N * nu / c) ** 2) * kB * T0

        j[0] = (jH + jHe) * w.Zfactor
        k[0] = (kH + kHe) * w.Zfactor
