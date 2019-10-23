cmake_minimum_required(VERSION 3.9)
project(PeriodicFMM VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 14)

set(MPI_CXX_SKIP_MPICXX true
  CACHE BOOL "If true, the MPI-2 C++ bindings are disabled using definitions.")
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)
find_package(PVFMM REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(Boost REQUIRED)
# find_package(PythonLibs)

include_directories("FMM/include"
#  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${PVFMM_INCLUDE_DIR}
  ${MKL_INCLUDE_DIRS}
  )
if (DEFINED ENV{CONDA_PREFIX})
  message("Including conda paths")
  set(CONDA_PREFIX ENV{CONDA_PREFIX})
  include_directories("$ENV{CONDA_PREFIX}/include")
endif()

add_subdirectory(FMM)
add_subdirectory(M2LStokes)
add_subdirectory(M2LLaplace)
add_subdirectory(examples)
