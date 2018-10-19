# Introduction
Follow these three steps to use the software:
* Setup the compiler and math library toolchain
* Install PVFMM library
* Use this software

# Toolchain
Install the following libraries and compiler:
* A mpi c++ compiler with c++14 support
* BLAS/LAPACK libraries with standard interfaces
* FFTW-3 library with interfaces for float and double routines
  
On Linux, the following combination is recommended:
* intel c++ + intel mpi + intel MKL
* g++ + openmpi/mpich + intel MKL
* g++ + openmpi/mpich + openblas + fftw-3

When using intel MKL, make sure that the proper OpenMP runtime library is linked, and never link one executable to multiple OpenMP runtime libraries. Read the `linking in detail` section of the intel MKL documentation carefully if you have any questions. Note that the intel linking advisor sometimes gives wrong combinations of linking line.

On Macs, the situation is very complicated because macOS's default toolchain is completely broken in terms of OpenMP. 
Macports is recommended (instead of homebrew) because it allows you to compile the toolchain from scratch and completely independent of macOS's default broken toolchain. 
Further, if you want to use MKL make sure you use clang++ (from Macports) as the mpi compiler and link all executables and libraries to intel OpenMP runtime library instead of libomp carried by clang++, using the `-fopenmp=libiomp5` flag. 
Otherwise, you have only one choice:
* clang++/g++ + openmpi/mpich + openblas + fftw-3

The system default blas and lapack on macOS are way too slow, and the thread number control is broken.

# PVFMM
Download the `new_BC` branch of PVFMM from https://github.com/wenyan4work/pvfmm
. Then compile it with the toolchain introduced in the first step. 

Here is an example configure script on macOS with Macports:
```bash
./configure --prefix=/..../ \
CXXFLAGS="-std=c++14 -O3 -march=native" \
LDFLAGS="-L/.../MacPorts/lib/ -lopenblas" \
--with-openmp-flag="fopenmp" \
--with-fftw-include="/.../MacPorts/include/" \
--with-fftw-lib="-L/.../MacPorts/lib/ -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f"
```
The linking flags to BLAS and LAPACK are specified through `LDFLAGS`.

Here is an example configure script on Linux with Intel MKL + Intel C++ + Intel MPI
```bash
./configure MPICXX=mpicxx --prefix=/..../ \
CXXFLAGS="-qno-offload -mtune=broadwell -xcore-avx2 -O3 -std=c++14 -DFFTW3_MKL" \
--with-fftw-include="$MKLROOT/include/fftw" --with-fftw-lib="-lmkl_rt" \
--with-openmp-flag='qopenmp' \
```
The flag `-mtune=broadwell -xcore-avx2` can be changed to match the hardware.


After successful configuration, type:
```bash
make
make all-examples
```

Then, run example files to make sure PVFMM works as expected:
```bash
cd ./examples/bin
fmm_pts -N 8192 -omp 4 -m 10 -ker 1 
fmm_pts -N 8192 -omp 4 -m 10 -ker 2
fmm_pts -N 8192 -omp 4 -m 10 -ker 3
```

After checking the results of examples, type
```bash
make install
```

# Use this software
This software is not supposed to be installed as a library. Instead, take the cpp/hpp files you need and copy them to your project's source code directory. 
The MakefileInc.mk file in `ModuleTest` folder is an example of how to write your own Makefile. 