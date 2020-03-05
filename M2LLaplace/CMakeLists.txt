add_executable(
  M2LLaplace
  src/main.cpp
  src/Laplace1D3D.cpp
  src/Laplace2D3D.cpp
  src/Laplace3D3D.cpp
  src/Laplace1D3DDipole.cpp
  src/Laplace2D3DDipole.cpp
  src/Laplace3D3DDipole.cpp)

target_link_libraries(
  M2LLaplace
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  ${PVFMM_LIB_DIR}/${PVFMM_SHARED_LIB}
  ${PVFMM_DEP_LIB}
  PeriodicFMM)
target_include_directories(
  M2LLaplace PRIVATE ${CMAKE_SOURCE_DIR}/Util ${PVFMM_INCLUDE_DIR}/pvfmm
                     ${PVFMM_DEP_INCLUDE_DIR})
