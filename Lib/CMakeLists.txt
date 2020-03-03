# part 1, core library
add_library(
  PeriodicFMM SHARED src/FMMWrapper.cpp src/FMMWrapper-c.cpp
                     src/FMMWrapperWall2D.cpp src/FMMWrapperWall2D-c.cpp)
target_include_directories(
  PeriodicFMM
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR})
target_link_libraries(
  PeriodicFMM PUBLIC ${PVFMM_LIB_DIR}/${PVFMM_SHARED_LIB} ${PVFMM_DEP_LIB}
                     OpenMP::OpenMP_CXX MPI::MPI_CXX)

# install core library and headers
include(GNUInstallDirs)
install(
  TARGETS PeriodicFMM
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
