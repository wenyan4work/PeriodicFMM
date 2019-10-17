project(FMM VERSION 0.1 LANGUAGES CXX)

set(PeriodicFMM_SOURCES
  src/FMMWrapper-c.cpp
  #  src/FMMWrapper-py.cpp
  src/FMMWrapper.cpp
  src/FMMWrapperWall2D-c.cpp
  src/FMMWrapperWall2D.cpp
  )

add_library(objFMM OBJECT ${PeriodicFMM_SOURCES})
set_property(TARGET objFMM PROPERTY POSITION_INDEPENDENT_CODE 1)

add_library(PeriodicFMM SHARED $<TARGET_OBJECTS:objFMM>)
add_library(PeriodicFMM_static STATIC $<TARGET_OBJECTS:objFMM>)

target_link_libraries(objFMM ${PVFMM_LIBRARIES} OpenMP::OpenMP_CXX MPI::MPI_CXX)

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
  EXPORT PeriodicFMMConfig
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/FMM
  )

install(EXPORT PeriodicFMMConfig DESTINATION share/PeriodicFMM/cmake)
