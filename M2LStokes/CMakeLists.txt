
add_executable(M2LStokes
  src/main.cpp
  src/Stokes1D3D.cpp
  src/Stokes2D3D.cpp
  src/Stokes3D3D.cpp)


target_link_libraries(M2LStokes
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX)

install(TARGETS M2LStokes
  RUNTIME DESTINATION bin
  )
