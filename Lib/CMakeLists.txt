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

# part 2, python interface
# PVFMM should be compiled with -fPIC 

if(PyInterface)
  add_library(PyPeriodicFMM SHARED src/Interface-py.cpp)
  set_target_properties(PyPeriodicFMM PROPERTIES PREFIX "" SUFFIX ".so")
  target_include_directories(
    PyPeriodicFMM
    PUBLIC $<INSTALL_INTERFACE:include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
           ${PVFMM_INCLUDE_DIR} ${MKL_INCLUDE_DIRS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})
  target_compile_definitions(PyPeriodicFMM PUBLIC "PVFMM_FFTW3_MKL")
  target_compile_options(PyPeriodicFMM PUBLIC ${OpenMP_CXX_FLAGS}
                                              ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(
    PyPeriodicFMM PUBLIC PeriodicFMM ${pybind11_LIBRARIES} ${PVFMM_LIBRARIES}
                         OpenMP::OpenMP_CXX ${MKL_LIBRARIES})
endif()
# set(PeriodicFMM_PYTHON_SOURCE src/FMMWrapper-py.cpp)

# add_library(objFMM OBJECT src/FMMWrapper.cpp src/FMMWrapperWall2D.cpp)
# set_property(TARGET objFMM PROPERTY POSITION_INDEPENDENT_CODE 1)
# target_compile_options(objFMM PUBLIC ${OpenMP_CXX_FLAGS})

# add_library(PeriodicFMM SHARED $<TARGET_OBJECTS:objFMM>)
# target_link_libraries(PeriodicFMM pvfmm)

# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") target_link_options(PeriodicFMM
# PUBLIC -flat_namespace -undefined suppress) endif()

# set_target_properties(PeriodicFMM PROPERTIES PUBLIC_HEADER "\
# include/FMM/FMMWrapper-c.h;\ include/FMM/FMMWrapper.hpp;\
# include/FMM/FMMWrapperWall2D-c.h;\ include/FMM/FMMWrapperWall2D.hpp;\
# include/FMM/LaplaceLayerKernel.hpp;\ include/FMM/SVD_pvfmm.hpp;\
# include/FMM/StokesRegSingleLayerKernel.hpp;\ include/FMM/Timer.hpp;\
# include/FMM/cmdparser.hpp")

# add_library(PeriodicFMM_static STATIC $<TARGET_OBJECTS:objFMM>) #
# target_link_libraries(PeriodicFMM_static #   pvfmm #   mkl_rt #
# OpenMP::OpenMP_CXX #   MPI::MPI_CXX #   )
# target_compile_options(PeriodicFMM_static PUBLIC ${OpenMP_CXX_FLAGS})

# add_library(PyPeriodicFMM SHARED ${PeriodicFMM_PYTHON_SOURCE})
# target_link_libraries(PyPeriodicFMM PeriodicFMM_static pvfmm Boost::numpy )
# target_compile_options(PyPeriodicFMM PUBLIC ${OpenMP_CXX_FLAGS})

# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") target_link_options(PyPeriodicFMM
# PUBLIC -flat_namespace -undefined suppress) endif ()

# install(TARGETS PeriodicFMM PeriodicFMM_static #  EXPORT PeriodicFMMConfig
# ARCHIVE DESTINATION lib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION
# include/FMM )

# #install(EXPORT PeriodicFMMConfig DESTINATION share/PeriodicFMM/cmake)

# find_program(PYTHON "python")

# if (PYTHON) set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in")
# set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py") set(DEPS
# PyPeriodicFMM) set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/dummy")

# configure_file(${SETUP_PY_IN} ${SETUP_PY})

# add_custom_command(OUTPUT ${OUTPUT} COMMAND ${PYTHON} ${SETUP_PY} build
# COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT} DEPENDS ${DEPS} )

# add_custom_target(PyPeriodicFMM-install ALL DEPENDS ${OUTPUT})

# install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)") endif()
