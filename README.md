# PeriodicFMM
The C++/C/Fortran/Python wrapper for flexibly periodic FMM and operator data generator to be used with PVFMM.

The flexibly periodic KIFMM algorithm is described in:
* Yan, W. & Shelley, M. Flexibly imposing periodicity in kernel independent FMM: A multipole-to-local operator approach. Journal of Computational Physics 355, 214–232 (2018).

The flexibly periodic Stokeslet Image algorithm is described in:
* Yan, W. & Shelley, M. Universal image systems for non-periodic and periodic Stokes flows above a no-slip wall. Journal of Computational Physics 375, 263–270 (2018).

Please cite them if you find this package useful.  

Dependency:
The C++ template library Eigen is necessary to compile the test driver main.cpp. The wrapper class itself (FMMWrapper.h/.cpp) does not rely on Eigen.

## Capability:
### FMMWrapper.h/.cpp
* Singular Stokeslet kernel force -> velocity
* Regularized Stokeslet kernel force + torque + spreading -> velocity + angular velocity 

With NONE/PX/PXY/PXYZ periodicity
### FMMWrapperWall2D.h/.cpp
* Singular Stokeslet kernel force -> velocity above a no slip wall

With NONE/PX/PXY periodicity


## How-To:

### 0. Follow `InstallationInstructions.md` to setup the toolchain.

### 1. Generate the M2L matrix 

Go to folders in M2LLaplace or M2LStokes and modify the Makefile to use the correct c++ compiler and the correct path to Eigen library header files.

After compilation, type:
```bash
XXX.X p > ./M2LYYYaD3Dpb
```
where XXX.X is the name of the executable, YYY is the name of the kernel (Stokes, LaplaceCharge, LaplaceDipole, etc), p is the discretization number for KIFMM, 'a' is periodic dimension, and 'b' is the point number for the equiv surface. 'b' controls the tradeoff between accuracy and cost. 'b=10' gives single precision and 'p=16' gives double precision, roughly. For exmaple, M2L2D3Dp8 means doubly periodic in 3D space with p=8. See FMMWrapper.cpp for details.

For example
```bash
localhost:periodicfmm wyan$ cd ./M2LStokes/2D3D/
localhost:2D3D wyan$ ./Stokes2D3D.X 8 > ./M2LStokes2D3Dp8
```

The executable also solves a tiny test problem to check the accuracy after the M2L operator is calculated. At the end of the above execution process, you should see something like:
```bash
localhost:2D3D wyan$ tail ./M2LStokes2D3Dp8
-4.527578312263358384e-11
Usample M2L total: 1.567028182748916532e+00
 1.769944468765771717e-10
-4.527578312263358384e-11
Usample Ewald: 1.567028149573803653e+00
-4.150067072042485956e-19
 0.000000000000000000e+00
error 3.317511287903585071e-08
 1.769944472915838697e-10
-4.527578312263358384e-11
```


### 2. Perform test and calculations

Copy those M2LYYYaD3Dpb files for Stokes kernel to the subfolder /pdata/ in $PVFMM_DIR, remove any extra lines to make sure the M2L files start with "i j value" lines like:
```bash
localhost:2D3D wyan$ head ./M2LStokes2D3Dp8
0 0 -8.963371808378797212e+00
0 1 -1.614059953274547965e+01
0 2 2.183755414811182050e+01
0 3 6.150120972933304841e+00
```

The correct folder structure should look like:
```bash
localhost:2D3D wyan$ cd $PVFMM_DIR
localhost:pvfmm wyan$ ls ./pdata/M2LStokes*
./pdata/M2LStokes1D3Dp10 ./pdata/M2LStokes2D3Dp10 ./pdata/M2LStokes3D3Dp10
./pdata/M2LStokes1D3Dp12 ./pdata/M2LStokes2D3Dp12 ./pdata/M2LStokes3D3Dp12
./pdata/M2LStokes1D3Dp14 ./pdata/M2LStokes2D3Dp14 ./pdata/M2LStokes3D3Dp14
./pdata/M2LStokes1D3Dp16 ./pdata/M2LStokes2D3Dp16 ./pdata/M2LStokes3D3Dp16
./pdata/M2LStokes1D3Dp6  ./pdata/M2LStokes2D3Dp6  ./pdata/M2LStokes3D3Dp6
./pdata/M2LStokes1D3Dp8  ./pdata/M2LStokes2D3Dp8  ./pdata/M2LStokes3D3Dp8
```

The wrapper files are located in the folder 'FMM':
```bash
PeriodicFMM/ $ ls ./FMM
FMMWrapper.cpp  FMMWrapper.h  FMMWrapper.o  FMMWrapperWall2D.cpp  FMMWrapperWall2D.h  LaplaceCustomKernel.hpp
PeriodicFMM/ $ 
```

The test-driving routines are located in the folder 'ModuleTest'
```bash
PeriodicFMM/ $ ls ./ModuleTest/
StokesFMM3D  StokesFMMWall2D  MakefileInc.mk
PeriodicFMM/ $ 
```
Modify the file MakefileInc.mk properly.

To run the demo and tests for Stokes 3D periodic FMM:
```bash
PeriodicFMM/ $ cd ModuleTest/StokesFMM3D/
StokesFMM3D/ $ make
make: 'TestStokes3D.X' is up to date.
StokesFMM3D/ $ TestStokes3D.X --help
```
to get command line options. 

To run the demo and tests for Stokes 3D periodic FMM above a wall:
```bash
PeriodicFMM/ $ cd ModuleTest/StokesFMMWall2D/
StokesFMMWall2D/ $ make
make: 'TestStokesWall2D.X' is up to date.
StokesFMMWall2D/ $ ./TestStokesWall2D.X --help
```
to get command line options. 

This test driver demonstrates how to use the wrapper FMM class and can perform tests for FreeSpace, SP, DP, TP in 3D space, for random & chebyshev points, with OpenMP & MPI.

# Warning
This wrapper only works with the new_BC branch of my fork of PVFMM.


