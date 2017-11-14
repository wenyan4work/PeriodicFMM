# change this to match your system
SFTPATH=/Users/wyan/local
EIGEN= $(SFTPATH)/include/eigen3
CXX= mpicxx -Wa,-q
LINK = $(CXX)

CXXFLAGS= -O3 -fopenmp -std=c++11 -march=native # -ftree-vectorizer-verbose=1
LINKFLAGS= $(CXXFLAGS)

# set of libraries to correctly link to the target
USERLIB =  
USERINCLUDE = -I$(EIGEN) -I./

INCLUDE_DIRS = $(USERINCLUDE)
LIBRARY_DIRS =  
LIBRARIES = $(USERLIB)

# System-specific settings
SHELL = /bin/bash
SYSLIB =	
SIZE =	size

# Files

SRC =   Stokes3D3D.cpp
INC =   ../../common/SVD_pvfmm.hpp 
         
# Definitions

EXE =   $(SRC:.cpp=.X)
OBJ =   $(SRC:.cpp=.o)

# Link rule
$(EXE):	$(OBJ)
	$(LINK) $(OBJ)  -o $(EXE) $(LINKFLAGS) $(SYSLIB) $(LIBRARY_DIRS) $(LIBRARIES)
	$(SIZE) $(EXE)

# Compilation rules
.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o

# Individual dependencies
$(OBJ): $(INC)

all: $(EXE)

clean: 
	rm ./*.o
	rm ./$(EXE)
