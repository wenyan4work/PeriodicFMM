# PeriodicFMM
The periodic wrapper and operator data generator to be used with PVFMM
The algorithm is described in:

Wen Yan & Michael Shelley, Flexibly imposing periodicity in kernel independent FMM: A Multipole-To-Local operator approach, Volume 355, 15 February 2018, Pages 214-232

Now published: https://doi.org/10.1016/j.jcp.2017.11.012 in Journal of Computional Physics. Please cite if you find it useful.  


Dependency:
The C++ template library Eigen is necessary to compile the test driver main.cpp. The wrapper class itself (FMMWrapper.h/.cpp) does not rely on Eigen.

## How-To:


### 1. Generate the M2L matrix 

Go to folders in M2LLaplace or M2LStokes and modify the Makefile to use the correct c++ compiler and the correct path to Eigen library header files.

After compilation, type:
```bash
XXX.X p > ./M2LYYYaD3Dpb
```
where XXX.X is the name of the executable, YYY is the name of the kernel (Stokes, LaplaceCharge, LaplaceDipole, etc), p is the discretization number for KIFMM, a is periodic dimension, and b is the point number for the equiv surface. For exmaple, M2L2D3Dp8 means doubly periodic in 3D space with p=8. See FMMWrapper.cpp for details.

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

Modify the Makefile properly, and then type:
```bash
make
./StokesTest3D3D.X --help
```
to get command line options. This test driver demonstrates how to use the wrapper FMM class and can perform tests for FreeSpace, SP, DP, TP in 3D space, for random & chebyshev points, with OpenMP & MPI.

# Warning
This wrapper only works with the new_BC branch of my fork of PVFMM.


