project(FMM VERSION 0.1 LANGUAGES CXX)

set(PeriodicFMM_SOURCES
  src/FMMWrapper-c.cpp
  src/FMMWrapper.cpp
  src/FMMWrapperWall2D-c.cpp
  src/FMMWrapperWall2D.cpp
  )

set(PeriodicFMM_PYTHON_SOURCE src/FMMWrapper-py.cpp)

add_library(objFMM OBJECT ${PeriodicFMM_SOURCES})
set_property(TARGET objFMM PROPERTY POSITION_INDEPENDENT_CODE 1)
target_link_libraries(objFMM pvfmm OpenMP::OpenMP_CXX MPI::MPI_CXX)

add_library(PeriodicFMM SHARED $<TARGET_OBJECTS:objFMM>)
target_link_libraries(PeriodicFMM
  pvfmm
  mkl_rt
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  )

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

add_library(PeriodicFMM_static STATIC $<TARGET_OBJECTS:objFMM>)
target_link_libraries(PeriodicFMM_static
  pvfmm
  mkl_rt
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  )

add_library(PyPeriodicFMM SHARED ${PeriodicFMM_PYTHON_SOURCE})
set_target_properties(PyPeriodicFMM PROPERTIES PREFIX "")
target_link_libraries(PyPeriodicFMM
  PeriodicFMM
  pvfmm
  mkl_rt
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  ${PYTHON_LIBRARIES}
  Boost::python
  Boost::numpy
  )

install(TARGETS PeriodicFMM PeriodicFMM_static
#  EXPORT PeriodicFMMConfig
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/FMM
  )

#install(EXPORT PeriodicFMMConfig DESTINATION share/PeriodicFMM/cmake)

find_program(PYTHON "python")

if (PYTHON)
  set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in")
  set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  set(DEPS        PyPeriodicFMM)
  set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/dummy")

  configure_file(${SETUP_PY_IN} ${SETUP_PY})

  add_custom_command(OUTPUT ${OUTPUT}
    COMMAND ${PYTHON} ${SETUP_PY} build
    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
    DEPENDS ${DEPS}
    )

  add_custom_target(PyPeriodicFMM-install ALL DEPENDS ${OUTPUT})

  install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()
