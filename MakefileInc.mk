CXX= mpicxx
LINK= $(CXX)

# optimized
CFLAGS=-std=c99 -O3 -fopenmp
# TODO: automate detection of eigen directory (using cmake probably)
CXXFLAGS=-I$(CONDA_PREFIX)/include/eigen3 -fopenmp -O3
LINKFLAGS=-lpvfmm -lstdc++ -fopenmp -lfftw3 -lblas -lmpi_cxx -lm

# debug
DEBUGMODE:=no

# debug flags
# CXXFLAGS += -DFMMTIMING
# CXXFLAGS += -DFMMDEBUG

ifeq ($(DEBUGMODE), yes)
	CXXFLAGS:=$(subst -O3, ,$(CXXFLAGS))
	LINKFLAGS:=$(subst -O3, ,$(LINKFLAGS))
	CXXFLAGS:=$(CXXFLAGS) -O0 -g
	LINKFLAGS:=$(LINKFLAGS) -O0 -g
else
	CXXFLAGS:=$(CXXFLAGS) -DNDEBUG
	LINKFLAGS:=$(LINKFLAGS) -DNDEBUG
endif
