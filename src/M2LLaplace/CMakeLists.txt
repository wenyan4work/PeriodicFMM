
add_executable(M2LLaplace
  src/main.cpp
  src/Laplace1D3D.cpp
  src/Laplace2D3D.cpp
  src/Laplace1D3DDipole.cpp
  src/Laplace2D3DDipole.cpp
  src/Laplace3D3DDipole.cpp)

target_link_libraries(M2LLaplace
  ${PVFMM_LIBRARIES}
  PeriodicFMM
  mkl_rt
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX)

install(TARGETS M2LLaplace
  RUNTIME DESTINATION bin
  )
