# PyFastGyrosynchotron
  Sabastian Fernandes | NJIT Applied Physics Undergraduate | Summer 2021 - Summer 2022

- Fast codes for computing gyrosynchotron radio emission, written in python. Adapted from here: https://github.com/kuznetsov-radio/gyrosynchrotron
- All mathematical calculations and descriptions are taken from the papers and documentation found here: https://sites.google.com/site/fgscodes/gs

  Originally written in CPP, these codes have been translated to be read and executed from a native python environment. This project was undertaken as an experiment in order to see how python's automatic memory assingment and dynamic typing would affect the execution time of the CPP codes within the kuznetsov-radio repository linked above.
  
  Additionally, translation of the fast GS codes into python will serve to promote a more accessible, open-sourced nature to the codes that is more easily readable and modifiable. Many of the calculations within could be easily adaptable to another project requiring similar computation, for scientists hoping to move away from lower-level languages such as IDL and CPP; much of the previous CPP code structure has been accordingly shortened and simplified via translation.

* The main underlying technical feature of the code itself is its use of numpy arrays to perform massive amounts of calculations on large parameter-fed numpy arrays, and return the values up the callstack for further processing, in order to closely mimic CPP pointer objects.
