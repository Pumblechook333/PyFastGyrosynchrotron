import matplotlib.pyplot as plt
from interface import *

Nf = 100  # number of frequencies
NSteps = 30  # number of nodes along the line-of-sight

Lparms = np.zeros(11, dtype='int')  # array of dimensions etc.
Lparms[0] = NSteps
Lparms[1] = Nf

Rparms = np.zeros(5, dtype='float')  # array of global floating-point parameters
Rparms[0] = 1e20  # area, cm^2
Rparms[1] = 1e9  # starting frequency to calculate spectrum, Hz
Rparms[2] = 0.02  # logarithmic step in frequency
Rparms[3] = 12  # f^C
Rparms[4] = 12  # f^WH

L = 1e10  # total source depth, cm

ParmLocal = np.zeros(24, dtype='float')  # array of voxel parameters - for a single voxel
ParmLocal[0] = L / NSteps  # voxel depth, cm
ParmLocal[1] = 3e7  # T_0, K
ParmLocal[2] = 3e9  # n_0 - thermal electron density, cm^{-3}
ParmLocal[3] = 180  # B - magnetic field, G
ParmLocal[6] = 3  # distribution over energy (PLW is chosen)
ParmLocal[7] = 1e6  # n_b - nonthermal electron density, cm^{-3}
ParmLocal[9] = 0.1  # E_min, MeV
ParmLocal[10] = 10.0  # E_max, MeV
ParmLocal[12] = 4.0  # \delta_1
ParmLocal[14] = 3  # distribution over pitch-angle (GLC is chosen)
ParmLocal[15] = 70  # loss-cone boundary, degrees
ParmLocal[16] = 0.2  # \Delta\mu

Parms = np.zeros((24, NSteps), dtype='float', order='F')  # 2D array of input parameters - for multiple voxels

# NOTE - Filling the Parms array follows Fortran row-major (row, column) format.
for i in range(NSteps):
    Parms[:, i] = ParmLocal  # most of the parameters are the same in all voxels
    Parms[4, i] = 50.0 + 30.0 * i / (NSteps - 1)  # the viewing angle varies from 50 to 80 degrees along the LOS
Parms = Parms.flatten('F')

RL = np.zeros((7, Nf), dtype='float', order='F')  # input/output array
RL = RL.flatten('F')

dummy = np.zeros(1, dtype='float')

res = 0
profiling = 1

if not profiling:
    # calculating the emission for analytical distribution (array -> off),
    # the unused parameters can be set to any value
    res = getMW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)

    RL = RL.reshape((7, Nf), order='F')

    # retrieving the results (each index is 100 columns)
    f = RL[0]
    I_L = RL[5]
    I_R = RL[6]

    print("Entirety of RL: \n", RL, '\n')
    print("f = RL[0] = \n", f, "\n\n I_L = RL[5] = \n", I_L, "\n\n I_R = RL[6] = \n", I_R)

    # plotting the results
    plt.figure(1)
    plt.plot(f, I_L + I_R)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Total intensity (analytical)')
    plt.xlabel('Frequency, GHz')
    plt.ylabel('Intensity, sfu')

    plt.figure(2)
    plt.plot(f, (I_L - I_R) / (I_L + I_R))
    plt.xscale('log')
    plt.title('Circular polarization degree (analytical)')
    plt.xlabel('Frequency, GHz')
    plt.ylabel('Polarization degree')

    plt.show()
else:
    import cProfile
    import pstats
    from pstats import SortKey

    cProfile.run("getMW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)", "thirtyRunStats")

    p = pstats.Stats('thirtyRunStats')
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(30)


