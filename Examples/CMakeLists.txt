add_executable(Example_C C/main.c)
target_compile_definitions(Example_C PRIVATE "PVFMM_FFTW3_MKL")
target_include_directories(Example_C PRIVATE ${CMAKE_SOURCE_DIR}/Lib/include)
target_link_libraries(
  Example_C
  PeriodicFMM
  ${PVFMM_LIBRARIES}
  ${MKL_LIBRARIES}
  stdc++
  m
  OpenMP::OpenMP_C
  MPI::MPI_C)
add_test(NAME Example_C COMMAND mpirun -n 4 Example_C)
# set_tests_properties(TpetraUtil PROPERTIES PASS_REGULAR_EXPRESSION
# "TestPassed;All ok" )

# install(TARGETS FMMPeriodic_C RUNTIME DESTINATION bin )

# target_compile_options(FMMPeriodic_C PUBLIC "-DPVFMM_FFTW3_MKL")

# add_executable(StokesFMM3D StokesFMM3D/main.cpp StokesFMM3D/ChebNodal.cpp
# StokesFMM3D/regularized_stokeslet.cpp )

# target_link_libraries(StokesFMM3D pvfmm PeriodicFMM mkl_rt Eigen3::Eigen
# OpenMP::OpenMP_CXX MPI::MPI_CXX )

# target_compile_options(StokesFMM3D PUBLIC "-DPVFMM_FFTW3_MKL")
# target_compile_options(StokesFMM3D PUBLIC ${OpenMP_CXX_FLAGS})

# install(TARGETS StokesFMM3D RUNTIME DESTINATION bin )

# add_executable(StokesFMMWall2D StokesFMMWall2D/main.cpp
# StokesFMMWall2D/ChebNodal.cpp )

# target_link_libraries(StokesFMMWall2D pvfmm PeriodicFMM mkl_rt Eigen3::Eigen
# OpenMP::OpenMP_CXX MPI::MPI_CXX)

# target_compile_options(StokesFMMWall2D PUBLIC "-DPVFMM_FFTW3_MKL")
# target_compile_options(StokesFMMWall2D PUBLIC ${OpenMP_CXX_FLAGS})

# install(TARGETS StokesFMMWall2D RUNTIME DESTINATION bin )
