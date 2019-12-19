cmake_minimum_required(VERSION 3.10)
project(PeriodicFMM VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 14)

set(MPI_CXX_SKIP_MPICXX true
  CACHE BOOL "If true, the MPI-2 C++ bindings are disabled using definitions.")
# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# library
find_package(MKL REQUIRED)
find_package(PVFMM REQUIRED)
find_package(Eigen3 REQUIRED)
# requried by python interface
find_package(Boost REQUIRED COMPONENTS python numpy)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

include_directories("FMM/include"
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${MKL_INCLUDE_DIRS}
  )

message("Python Includes: " ${PYTHON_INCLUDE_DIRS})
message("Python Libs: " ${PYTHON_LIBRARIES})

# Avoid compiler warnings from PVFMM headers
include_directories(SYSTEM ${PVFMM_INCLUDE_DIR})

if (DEFINED ENV{CONDA_PREFIX})
  message("Including conda paths")
  set(CONDA_PREFIX ENV{CONDA_PREFIX})
  include_directories(SYSTEM "$ENV{CONDA_PREFIX}/include")
endif()

add_subdirectory(FMM)
add_subdirectory(M2LStokes)
add_subdirectory(M2LLaplace)
add_subdirectory(examples)
