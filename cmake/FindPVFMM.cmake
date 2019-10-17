#  Copyright Olivier Parcollet 2010.
#  Copyright Simons Foundation 2019
#    Author: Nils Wentzell
#    Customized for PVFMM by: Robert Blackwell

#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

#
# This module looks for PVFMM.
# It sets up : PVFMM_INCLUDE_DIR, PVFMM_LIBRARIES

find_path(PVFMM_INCLUDE_DIR
  NAMES pvfmm.hpp
  PATHS
    $ENV{CONDA_PREFIX}/include
    ENV CPATH
    ENV C_INCLUDE_PATH
    ENV CPLUS_INCLUDE_PATH
    ENV OBJC_INCLUDE_PATH
    ENV OBJCPLUS_INCLUDE_PATH
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
  DOC "Include Directory for PVFMM"
)

find_library(PVFMM_LIBRARIES
  NAMES pvfmm
  PATHS
    $ENV{CONDA_PREFIX}/lib
    $ENV{CONDA_PREFIX}/lib
    ENV LIBRARY_PATH
    ENV LD_LIBRARY_PATH
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
  DOC "PVFMM library"
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PVFMM DEFAULT_MSG PVFMM_LIBRARIES PVFMM_INCLUDE_DIR)

mark_as_advanced(PVFMM_INCLUDE_DIR PVFMM_LIBRARIES)

# Interface target
# We refrain from creating an imported target since those cannot be exported
add_library(pvfmm INTERFACE)
target_link_libraries(pvfmm INTERFACE ${PVFMM_LIBRARIES})
target_include_directories(pvfmm SYSTEM INTERFACE ${PVFMM_INCLUDE_DIR})
