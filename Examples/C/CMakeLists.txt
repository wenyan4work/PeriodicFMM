add_executable(Example_C main.c)
target_include_directories(Example_C PRIVATE ${CMAKE_SOURCE_DIR}/Lib/include)
target_link_libraries(
  Example_C
  PRIVATE PeriodicFMM
          OpenMP::OpenMP_C
          MPI::MPI_C)
add_test(NAME Example_C COMMAND mpirun -n 4 Example_C)
