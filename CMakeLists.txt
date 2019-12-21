cmake_minimum_required(VERSION 3.10)
project(
  PeriodicFMM
  VERSION 0.1
  LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)

# compiler flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using intel
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qno-offload")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qno-offload")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

set(MPI_CXX_SKIP_MPICXX
    true
    CACHE BOOL
          "If true, the MPI-2 C++ bindings are disabled using definitions.")
# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# library
find_package(PVFMM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(MKL REQUIRED)

enable_testing()

add_subdirectory(Lib)
add_subdirectory(Examples)
add_subdirectory(M2LStokes)
add_subdirectory(M2LLaplace)
