from mw_main import MW_Transfer
import numpy as np
#from numba import njit
from plasma import FindIonizationsSolar
from interface_header import *
import multiprocessing as mp


def getMW(Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, RL):
    """This is the function which can be called directly from a python module to perform the total Gyrosynchotron
       Calculation.

       It prepares the provided parameters for handling by the MW_Transfer module of the mw_main.py module."""

    # USED TO INSTANCE ALL NUMERICAL POINTERS
    bigNeg = -9.2559631349317831e+61

    Nz = Lparms[i_Nz]

    tempParms = np.full(np.shape(Parms), bigNeg)

    tempParms[:] = Parms

    for i in range(0, Nz):
        p = np.roll(tempParms, -(i*InSize))

        if p[i_T0] < 1e5 and p[i_np] == 0 and p[i_nH] == 0:
            ne = np.array([bigNeg])
            nH = np.array([bigNeg])
            nHe = np.array([bigNeg])

            FindIonizationsSolar(p[i_n0], p[i_T0], ne, nH, nHe)

            p[i_n0] = ne[0]
            p[i_nH] = nH[0]
            p[i_nHe] = nHe[0]

        tempParms = np.roll(p, i*InSize)

    # np.copyto(Parms, tempParms)
    Parms[:] = tempParms

    res = MW_Transfer(Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, RL)

    return res


def getMWMain(Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, RL):
    """A modified version of getMW which does not modify the Parms array before passing it into MW_Transfer"""

    res = MW_Transfer(Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, RL)

    return res


def parmARGV(pix, Lparms_M, Rparms_M, Parms_M, E_arr, mu_arr, f_arr_M, Nz, NE, Nnu, RL_M, Nmu, res_M):
    """Calls the get_MW function of mw_main.py on a slice of data from the pool provided."""

    ARGV = np.empty(7)
    ARGV[0] = (Lparms_M + 1)
    ARGV[1] = (Rparms_M + pix * RpSize)
    ARGV[2] = (Parms_M + pix * Nz * InSize)
    ARGV[3] = E_arr
    ARGV[4] = mu_arr
    ARGV[5] = (f_arr_M + pix * Nz * NE * Nmu)
    ARGV[6] = (RL_M + pix * Nnu * OutSize)

    res_M[pix] = getMW(ARGV[0], ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5], ARGV[6])


def getMWSlice(Lparms_M, Rparms_M, Parms_M, E_arr, mu_arr, f_arr_M, RL_M):
    """A modified version of getMW which splits the provided data into various pools to be calculated simultaneously
       via multithreading."""

    # USED TO INSTANCE ALL NUMERICAL POINTERS
    bigNeg = -9.2559631349317831e+61

    res = 0

    Npix = Lparms_M[i_Npix]
    Nz = Lparms_M[i_Nz + 1]
    Nnu = Lparms_M[i_Nnu + 1]
    NE = Lparms_M[i_NE + 1]
    Nmu = Lparms_M[i_Nmu + 1]

    res_M = np.full(Npix, bigNeg)

    # # omp parallel for substitution
    # if __name__ == 'interface':
    pool = mp.Pool(mp.cpu_count())
    for pix in range(0, Npix):
        pool.apply(parmARGV, args=(Npix, pix, Lparms_M, Rparms_M, Parms_M, E_arr, mu_arr, f_arr_M, Nz, NE, Nnu,
                                   RL_M, Nmu, res_M))

    pool.close()

    for i in range(0, Npix):
        res = (res != 0) or (res_M[i] != 0)
    #del res_M

    return res
