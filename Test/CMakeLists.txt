add_executable(StkFMM3D.X StokesFMM3D/main.cpp
                          StokesFMM3D/regularized_stokeslet.cpp)
target_include_directories(StkFMM3D.X PRIVATE ${CMAKE_SOURCE_DIR}/Util)
target_link_libraries(StkFMM3D.X PRIVATE PeriodicFMM Eigen3::Eigen
                                         OpenMP::OpenMP_CXX MPI::MPI_CXX)

add_executable(StkFMMWall2D.X StokesFMMWall2D/main.cpp)
target_include_directories(StkFMMWall2D.X PRIVATE ${CMAKE_SOURCE_DIR}/Util)
target_link_libraries(StkFMMWall2D.X PRIVATE PeriodicFMM Eigen3::Eigen
                                             OpenMP::OpenMP_CXX MPI::MPI_CXX)
