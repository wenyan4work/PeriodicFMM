
add_executable(FMMPeriodic_C
  C/main.c
  )

target_link_libraries(FMMPeriodic_C
  pvfmm
  PeriodicFMM
  mkl_rt
  m
  stdc++
  OpenMP::OpenMP_C
  MPI::MPI_CXX
  )

install(TARGETS FMMPeriodic_C
  RUNTIME DESTINATION bin
  )

target_compile_options(FMMPeriodic_C PUBLIC "-DPVFMM_FFTW3_MKL")

add_executable(StokesFMM3D
  StokesFMM3D/main.cpp
  StokesFMM3D/ChebNodal.cpp
  StokesFMM3D/regularized_stokeslet.cpp
  )

target_link_libraries(StokesFMM3D
  pvfmm
  PeriodicFMM
  mkl_rt
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  )

target_compile_options(StokesFMM3D PUBLIC "-DPVFMM_FFTW3_MKL")

install(TARGETS StokesFMM3D
  RUNTIME DESTINATION bin
  )


add_executable(StokesFMMWall2D
  StokesFMMWall2D/main.cpp
  StokesFMMWall2D/ChebNodal.cpp
  )

target_link_libraries(StokesFMMWall2D
  pvfmm
  PeriodicFMM
  mkl_rt
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX)

target_compile_options(StokesFMMWall2D PUBLIC "-DPVFMM_FFTW3_MKL")

install(TARGETS StokesFMMWall2D
  RUNTIME DESTINATION bin
  )


