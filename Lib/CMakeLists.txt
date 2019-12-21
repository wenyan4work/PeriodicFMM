# part 1, core library
add_library(
  PeriodicFMM SHARED src/FMMWrapper.cpp src/FMMWrapper-c.cpp
                     src/FMMWrapperWall2D.cpp src/FMMWrapperWall2D-c.cpp)
target_include_directories(
  PeriodicFMM
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${PVFMM_INCLUDE_DIR} ${MKL_INCLUDE_DIRS})
target_compile_definitions(PeriodicFMM PUBLIC "PVFMM_FFTW3_MKL")
target_compile_options(PeriodicFMM PUBLIC ${OpenMP_CXX_FLAGS}
                                          ${MPI_CXX_INCLUDE_DIRS})

# install core library and headers
include(GNUInstallDirs)
install(
  TARGETS PeriodicFMM
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
