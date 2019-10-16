cmake_minimum_required(VERSION 3.0)
project(PeriodicFMM)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)

find_package(Boost)
find_package(PythonLibs)

include_directories("include"
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  )
if (DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX ENV{CONDA_PREFIX})
  include_directories("$ENV{CONDA_PREFIX}/include")
else ()
  message("Conda not found")
endif()

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()


add_definitions("-std=c++14")

set(PeriodicFMM_SOURCES
  src/FMMWrapper-c.cpp
  #  src/FMMWrapper-py.cpp
  src/FMMWrapper.cpp
  src/FMMWrapperWall2D-c.cpp
  src/FMMWrapperWall2D.cpp
  )

add_library(PeriodicFMM SHARED ${PeriodicFMM_SOURCES})
add_library(PeriodicFMM_static STATIC ${PeriodicFMM_SOURCES})
target_link_libraries(PeriodicFMM ${MPI_LIBRARIES})
target_link_libraries(PeriodicFMM_static ${MPI_LIBRARIES})

if (MPI_COMPILE_FLAGS)
  message("Setting MPI compile flags")
  set_target_properties(PeriodicFMM PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(PeriodicFMM_static PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  message("Setting MPI link flags")
  set_target_properties(PeriodicFMM PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(PeriodicFMM_static PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

if (OpenMP_CXX_FLAGS)
  set_target_properties(PeriodicFMM PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
  set_target_properties(PeriodicFMM_static PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

set_target_properties(PeriodicFMM PROPERTIES PUBLIC_HEADER "\
include/FMM/EigenDef.hpp;\
include/FMM/FMMWrapper-c.h;\
include/FMM/FMMWrapper.hpp;\
include/FMM/FMMWrapperWall2D-c.h;\
include/FMM/FMMWrapperWall2D.hpp;\
include/FMM/LaplaceLayerKernel.hpp;\
include/FMM/SVD_pvfmm.hpp;\
include/FMM/StokesRegSingleLayerKernel.hpp;\
include/FMM/Timer.hpp;\
include/FMM/cmdparser.hpp")

install(TARGETS PeriodicFMM PeriodicFMM_static
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/FMM
  )
