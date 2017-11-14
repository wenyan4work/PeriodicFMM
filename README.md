# PeriodicFMM
The periodic wrapper and operator data generator to be used with PVFMM
The algorithm is described in:

Flexibly imposing periodicity in kernel independent FMM: A Multipole-To-Local operator approach
Wen Yan & Michael Shelley
Now on arXiv: https://arxiv.org/abs/1705.02043, to be published in Journal of Computional Physics soon. Please cite if you find it useful.  

Dependency:
The C++ template library Eigen is necessary to compile the files.

--------------------------------
How-To:

1. Generate the M2L matrix 
Go to folders in M2LLaplace or M2LStokes, modify the variables pCheck and pEquiv in the main function to generate the M2L matrix with desired number of p for check and equiv surfaces. pCheck = pEquiv is recommended.

Modify the Makefile to use the correct c++ compiler and the correct path to Eigen library header files.

After compilation, type:

XXX.X > ./M2LaD3Dpb

where XXX.X is the name of the executable, a is periodic dimension, and b is the point number for the equiv surface. For exmaple, M2L2D3Dp8 means doubly periodic in 3D space with p=8. See FMMWrapper.cpp for details.

2. Perform test and calculations
Copy those M2LaD3Dpb files for Stokes kernel to the subfolder /pdata/ in $PVFMM_DIR, modify the Makefile properly, and then execute the executable.

./StokesTest3D3D.X --help

to get command line options.

$$$$$$$$$$$$$$$$$$$$$$$$
Important:
The corresponding data for the desired dimension and point number p must be available in the foler pdata/ otherwise the executable will crash due to the failure of loading M2L matrix data.
$$$$$$$$$$$$$$$$$$$$$$$$



