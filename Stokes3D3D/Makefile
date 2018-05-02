# change this to match your system
SFTPATH=/mnt/home/wyan/local
EIGEN= $(SFTPATH)/include/eigen3
CXX= mpicxx
LINK = $(CXX)

EIGEN= $(SFTPATH)/include/eigen3
include $(PVFMM_DIR)/MakeVariables

CXXFLAGS= $(CXXFLAGS_PVFMM) -DFMMTIMING
LINKFLAGS= $(CXXFLAGS_PVFMM) $(LDLIBS_PVFMM)

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

SRC =   main.cpp ChebNodal.cpp FMMWrapper.cpp
INC =   ChebNodal.h FMMWrapper.h Ewald.hpp
         
# Definitions

EXE =   StokesTest3D3D.X
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
	rm ./$(OBJ)
	rm ./$(EXE)
