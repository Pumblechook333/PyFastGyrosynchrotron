from interface_header import *
from plasma_header import *

from std_df import Std_DF
from arr_df import Arr_DF
from gs import Find_jk_GS
from ff import Find_jk_FFei, Find_jk_FFen

# For assessing performance of FindLocalJK
import time

import math
import numpy as np
from numba import njit, jit


# @njit
def FindLocalJK(nu, Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, jX, jO, kX, kO, ne_total):
    """Begins the process of finding a localized set of emissivity (j) and absorption (k) coefficients at a particular
       eigen-mode."""

    bigNeg = -9.2559631349317831e+61
    dNaN = np.inf

    res = 0

    EM_flag = int(Parms[i_EMflag])
    DFtypeGlobalFlag = Lparms[i_arrKeyG]
    DFtypeLocalFlag = int(Parms[i_arrKeyL])

    kappa = dNaN

    Ndf = 0
    df = np.full(10, None)

    # initializing the analytical distribution function:
    if (not (DFtypeGlobalFlag & 2)) and (not (DFtypeLocalFlag & 2)):
        k = 0
        Done = np.array([0])
        OK = np.array([bigNeg])
        empty = np.array([bigNeg])
        kap_on = np.array([bigNeg])

        while not Done:
            df[Ndf] = Std_DF(Parms, k, OK, empty, kap_on, Done)

            if not OK[0]:
                res = 1

            if OK[0] and not empty[0]:
                Ndf += 1
                if kap_on[0]:
                    kappa = Parms[i_epskappa]
            else:
                df[Ndf] = None
            k += 1

    # initializing the array distribution function
    if (not (DFtypeGlobalFlag & 1)) and (not (DFtypeLocalFlag & 1)):
        OK = np.array([bigNeg])
        empty = np.array([bigNeg])

        df[Ndf] = Arr_DF(Lparms, E_arr, mu_arr, f_arr, OK, empty)

        if not OK[0]:
            res = 2

        if OK[0] and not empty[0]:
            Ndf += 1
        else:
            df[Ndf] = None

    df[Ndf] = 0

    if not res:
        nb = 0.0  # additional energetic electron density
        for i in range(0, Ndf):
            nb += df[i].nb

        ne_total[0] = Parms[i_n0] + nb

        # nu_p = plasma frequency, nu_B = electrons around magnetic field
        nu_p = e * math.sqrt(ne_total[0] / me / math.pi)
        nu_B = e * Parms[i_B] / me / c / (2.0 * math.pi)
        theta = Parms[i_theta] * math.pi / 180

        Nnu = Lparms[i_Nnu]

        jX.fill(0.0)
        kX.fill(0.0)
        jO.fill(0.0)
        kO.fill(0.0)

        if not (EM_flag & 1) and Ndf:  # GS is on and nonthermal electrons are present
            nu_cr = nu_B * Rparms[i_nuCr]
            nu_cr_WH = nu_B * Rparms[i_nuWH]

            Npoints = Lparms[i_Nnodes]
            if Npoints >= 0:
                Npoints = Npoints if (Npoints > 16) else 16
            if Npoints < 0:
                Npoints = 0

            Q_on = Lparms[i_QoptKey] == 0
            m_on = Lparms[i_matchKey] == 0

            print("Performing Find_jk_GS (-1)...")
            Find_jk_GS(df, nu, Nnu, -1, theta, nu_p, nu_B, nu_cr, nu_cr_WH, Npoints, Q_on, m_on, jX, kX)
            print("Performing Find_jk_GS (1)...")
            Find_jk_GS(df, nu, Nnu, 1, theta, nu_p, nu_B, nu_cr, nu_cr_WH, Npoints, Q_on, m_on, jO, kO)
            print("Both Find_jk_GS performed. ")

        if not (EM_flag & 2):  # e-ions is on
            j_loc = np.array([bigNeg])
            k_loc = np.array([bigNeg])

            for i in range(0, Nnu):
                Find_jk_FFei(Parms[i_n0], Parms[i_T0], nu_p, nu_B, theta, kappa, Parms[i_abcode], -1, nu[i], j_loc,
                             k_loc)
                jX[i] += j_loc[0]
                kX[i] += k_loc[0]
                Find_jk_FFei(Parms[i_n0], Parms[i_T0], nu_p, nu_B, theta, kappa, Parms[i_abcode], 1, nu[i], j_loc,
                             k_loc)
                jO[i] += j_loc[0]
                kO[i] += k_loc[0]

        if not (EM_flag & 4):  # e-neutrals is on
            j_loc = np.array([bigNeg])
            k_loc = np.array([bigNeg])

            for i in range(0, Nnu):
                Find_jk_FFen(Parms[i_n0], Parms[i_nH], Parms[i_nHe], Parms[i_T0], nu_p, nu_B, theta, -1, nu[i], j_loc,
                             k_loc)
                jX[i] += j_loc[0]
                kX[i] += k_loc[0]
                Find_jk_FFen(Parms[i_n0], Parms[i_nH], Parms[i_nHe], Parms[i_T0], nu_p, nu_B, theta, 1, nu[i], j_loc,
                             k_loc)
                jO[i] += j_loc[0]
                kO[i] += k_loc[0]

    return res


@njit
def RadiationTransfer(nu, Nz, dz, ne, B, theta, jX, jO, kX, kO, Lw, Rw, Ls, Rs, Le, Re):
    """Performs the calculations needed to determine the resulting intensities and damping factors of the gyrosynchotron
       emissions."""

    for i in range(0, Nz):
        tau = -kO[i] * dz[i]
        eO = math.exp(tau) if (tau < 700) else 0.0
        dIO = 0.0 if (kO[i] == 0.0 or tau > 700) else (jO[i] / kO[i] * ((1.0 - eO) if (1.0 - eO) else (-tau)))
        tau = -kX[i] * dz[i]
        eX = math.exp(tau) if (tau < 700) else 0.0
        dIX = 0.0 if (kX[i] == 0.0 or tau > 700) else (jX[i] / kX[i] * ((1.0 - eX) if (1.0 - eX) else (-tau)))

        # Theta represents the angle at which the sun is being viewed; never greater than 90 degrees (pi/2 radians)
        if (i > 0) and (
                ((theta[i] > (math.pi / 2)) ^ (theta[i - 1] > (math.pi / 2))) and (ne[i] > 0) and (ne[i - 1] > 0)):
            a = Lw[0]
            Lw[0] = Rw[0]
            Rw[0] = a

            B_avg = (B[i] + B[i - 1]) / 2
            ne_avg = (ne[i] + ne[i - 1]) / 2
            da_dz = (math.fabs(theta[i] - theta[i - 1])) / (dz[i] + dz[i - 1]) * 2

            QT = e * e * e * e * e / (32 * math.pi * math.pi * me * me * me * me * c * c * c * c) * ne_avg * (B_avg **
                 2) * B_avg / (nu ** 4) / da_dz
            QT = math.exp(-QT)
            a = Le[0] * QT + Re[0] * (1.0 - QT)
            Re[0] = Re[0] * QT + Le[0] * (1.0 - QT)
            Le[0] = a

        if theta[i] > (math.pi / 2):
            Lw[0] = dIX + Lw[0] * eX
            Ls[0] = dIX + Ls[0] * eX
            Le[0] = dIX + Le[0] * eX
            Rw[0] = dIO + Rw[0] * eO
            Rs[0] = dIO + Rs[0] * eO
            Re[0] = dIO + Re[0] * eO
        else:
            Lw[0] = dIO + Lw[0] * eO
            Ls[0] = dIO + Ls[0] * eO
            Le[0] = dIO + Le[0] * eO
            Rw[0] = dIX + Rw[0] * eX
            Rs[0] = dIX + Rs[0] * eX
            Re[0] = dIX + Re[0] * eX


# @njit
def MW_Transfer(Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, RL):
    """Handles the looping and loading of functions FindLocalJK and RadiationTransfer in order to return the final GS
       calculations into the RL results array."""

    bigNeg = -9.2559631349317831e+61

    res = 0

    Nnu = Lparms[i_Nnu]
    nu = np.full(Nnu, bigNeg)
    if Rparms[i_nu0] > 0:
        nu[0] = Rparms[i_nu0]
        dnu = math.pow(10.0, Rparms[i_dnu])
        for i in range(1, Nnu):
            nu[i] = nu[i - 1] * dnu

    else:
        for i in range(0, Nnu):
            nu[i] = RL[i * OutSize + iRL_nu] * 1e9

    Nz = Lparms[i_Nz]
    dz = np.full(Nz, bigNeg)
    ne_total = np.full(Nz, bigNeg)
    B = np.full(Nz, bigNeg)
    theta = np.full(Nz, bigNeg)

    jX = np.full((Nz, Nnu), bigNeg)
    jO = np.full((Nz, Nnu), bigNeg)
    kX = np.full((Nz, Nnu), bigNeg)
    kO = np.full((Nz, Nnu), bigNeg)

    err = 0

    # Debugging timer
    t1 = t2 = tTot = 0

    for i in range(0, Nz):
        if not err:

            dz[i] = Parms[i * InSize + i_dz]
            B[i] = Parms[i * InSize + i_B]
            theta[i] = Parms[i * InSize + i_theta] * math.pi / 180

            Parms1 = np.roll(Parms, -(i*InSize))
            ne_total1 = np.roll(ne_total, -i)
            f_arr1 = np.roll(f_arr, -(i * Lparms[i_NE] * Lparms[i_Nmu]))

            # # Time elapsed between findLocalJK run
            td = t2 - t1
            tTot += td

            # Timing
            print("Performing FindLocalJK ", i, " / ", Nz)
            print("Time Elapsed: ", td, " s \n")
            t1 = time.perf_counter()

            err = FindLocalJK(nu, Lparms, Rparms, Parms1, E_arr, mu_arr,
                              f_arr1, jX[i], jO[i], kX[i], kO[i], ne_total1)

            t2 = time.perf_counter()

            f_arr = np.roll(f_arr1, i * Lparms[i_NE] * Lparms[i_Nmu])
            ne_total = np.roll(ne_total1, i)
            Parms = np.roll(Parms1, i*InSize)

    print("Total findLocalJK section elapsed time", tTot, " s \n")

    if err:
        res = err
    else:
        Sang = Rparms[i_S] / ((AU ** 2) * sfu)

        Lw = np.array([bigNeg])
        Rw = np.array([bigNeg])
        Ls = np.array([bigNeg])
        Rs = np.array([bigNeg])
        Le = np.array([bigNeg])
        Re = np.array([bigNeg])

        kO_loc = np.full(Nz, bigNeg)
        kX_loc = np.full(Nz, bigNeg)
        jO_loc = np.full(Nz, bigNeg)
        jX_loc = np.full(Nz, bigNeg)

        for i in range(0, Nnu):
            Lw[0] = RL[i * OutSize + iRL_Lw] / Sang
            Rw[0] = RL[i * OutSize + iRL_Rw] / Sang
            Ls[0] = RL[i * OutSize + iRL_Ls] / Sang
            Rs[0] = RL[i * OutSize + iRL_Rs] / Sang
            Le[0] = RL[i * OutSize + iRL_Le] / Sang
            Re[0] = RL[i * OutSize + iRL_Re] / Sang

            for j in range(0, Nz):
                jX_loc[j] = jX[j][i]
                jO_loc[j] = jO[j][i]
                kX_loc[j] = kX[j][i]
                kO_loc[j] = kO[j][i]

            RadiationTransfer(nu[i], Nz, dz, ne_total, B, theta, jX_loc, jO_loc, kX_loc, kO_loc, Lw, Rw, Ls, Rs, Le, Re)

            RL[i * OutSize + iRL_nu] = nu[i] / 1e9
            RL[i * OutSize + iRL_Lw] = Lw[0] * Sang
            RL[i * OutSize + iRL_Rw] = Rw[0] * Sang
            RL[i * OutSize + iRL_Ls] = Ls[0] * Sang
            RL[i * OutSize + iRL_Rs] = Rs[0] * Sang
            RL[i * OutSize + iRL_Le] = Le[0] * Sang
            RL[i * OutSize + iRL_Re] = Re[0] * Sang

    return res
