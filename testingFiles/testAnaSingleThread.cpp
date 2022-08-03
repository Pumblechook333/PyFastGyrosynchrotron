#include "IDLinterface.h"
#include <iostream>

using namespace std;

void printArray(double a[100]) {
    cout << "{ ";

    for (int i = 0; i < 100; i++) {
        cout << a[i] << ", ";
    }

    cout << " }";

}

//int testAnaSingleThread(){
int main() {

    const int Nf = 100;       // number of frequencies
    const int NSteps = 30;    // number of nodes along the line-of-sight

    int Lparms[11] = { 0 };   // array of dimensions etc.
    Lparms[0] = NSteps;
    Lparms[1] = Nf;

    double Rparms[11] = { 0 };  // array of global floating-point parameters
    Rparms[0] = 1e20;   // area, cm^2
    Rparms[1] = 1e9;    // starting frequency to calculate spectrum, Hz
    Rparms[2] = 0.02;   // logarithmic step in frequency
    Rparms[3] = 12;     // f^C
    Rparms[4] = 12;     // f^WH

    double L = 1e10;    // total source depth, cm 

    double ParmLocal[24] = { 0 };       // array of voxel parameters - for a single voxel
    ParmLocal[0] = L / NSteps;  // voxel depth, cm
    ParmLocal[1] = 3e7;         // T_0, K
    ParmLocal[2] = 3e9;         // n_0, K
    ParmLocal[3] = 180;         // B - magnetic field, G
    ParmLocal[6] = 3;           // distribution over energy (PLW is chosen)
    ParmLocal[7] = 1e6;         // n_b - nonthermal electron density, cm^{-3}
    ParmLocal[9] = 0.1;         // E_min, MeV
    ParmLocal[10] = 10.0;       // E_max, MeV
    ParmLocal[12] = 4.0;        // \delta_1
    ParmLocal[14] = 3;          // distribution over pitch - angle(GLC is chosen)
    ParmLocal[15] = 70;         // loss - cone boundary, degrees
    ParmLocal[16] = 0.2;        // \Delta\mu

    double Parms[NSteps][24] = { 0 };   // 2D array of input parameters - for multiple voxels

    for (int i = 0; i < NSteps; i++) {
        for (int j = 0; j < 24; j++) {
            Parms[i][j] = ParmLocal[j];                        // most of the parameters are the same in all voxels
        }
        Parms[i][4] = 50.0 + 30.0 * i / (NSteps - 1);   // the viewing angle varies from 50 to 80 degrees along the LOS
    }

    double ParmsFlat[NSteps * 24] = { 0 };

    for (int i = 0; i < NSteps; i++){
        for (int j = 0; j < 24; j++) {
            ParmsFlat[i*24 + j] = Parms[i][j];
        }
    }

    double RL[7][Nf] = { 0 };       // input / output array

    double RLFlat[7 * Nf] = { 0 };

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < Nf; j++) {
            RLFlat[i*7 + j] = RL[i][j];
        }
    }

    double dummy[1] = { 0 };

    // calculating the emission for analytical distribution(array->off),
    // the unused parameters can be set to any value

    //     const int argc = 7;
    //     void **argv[argc] = { Lparms, Rparms, Parms, dummy, dummy, dummy, RL };

    int* LparmsP = Lparms;
    double* RparmsP = Rparms;
    double* ParmsP = ParmsFlat;
    double* dummyP = dummy;
    double* RLP = RLFlat;

    // All Parameters must be SINGLE POINTERS before going beyond this point 
    double res = GET_MW_CPP(LparmsP, RparmsP, ParmsP, dummyP, dummyP, dummyP, RLP);

    double RLReformed[7][Nf] = { 0 };

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < Nf; j++) {
            RLReformed[i][j] = RLFlat[(j * 7) + i];
        }
    }

    // retrieving the results (each index is 100 columns)
    double f[Nf] = { 0 }; 
    double I_L[Nf] = { 0 };
    double I_R[Nf] = { 0 };

    for (int i = 0; i < Nf; i++) {
        f[i] = RLReformed[0][i];
        I_L[i] = RLReformed[5][i];
        I_R[i] = RLReformed[6][i];
    }


    cout << "f = RL[0] = \n";
    printArray(f);
    cout << "\n\nI_L = RL[5] = \n";
    printArray(I_L);
    cout << "\n\nI_R = RL[6] = \n";
    printArray(I_R);


    //// plotting the results // revisit this if necessary
    //plt.figure(1)
    //plt.plot(f, I_L + I_R)
    //plt.xscale('log')
    //plt.yscale('log')
    //plt.title('Total intensity (analytical)')
    //plt.xlabel('Frequency, GHz')
    //plt.ylabel('Intensity, sfu')

    //plt.figure(2)
    //plt.plot(f, (I_L - I_R) / (I_L + I_R))
    //plt.xscale('log')
    //plt.title('Circular polarization degree (analytical)')
    //plt.xlabel('Frequency, GHz')
    //plt.ylabel('Polarization degree')

    //plt.show()

    return 0;
}