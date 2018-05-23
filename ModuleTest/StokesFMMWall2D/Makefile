include ../../MakefileInc.mk

INCLUDE_DIRS = $(USERINCLUDE) -I../../
LIBRARY_DIRS =
LIBRARIES = $(USERLIB)
# Files

SRC =   main.cpp ChebNodal.cpp ../../FMM/FMMWrapperWall2D.cpp
INC =   ChebNodal.h ../../FMM/FMMWrapperWall2D.h ../../FMM/LaplaceCustomKernel.hpp
         
# Definitions
CXXFLAGS += -DFMMTIMING

EXE =   TestStokesWall2D.X
OBJ =   $(SRC:.cpp=.o)

# Link rule
$(EXE):	$(OBJ)
	$(LINK) $(OBJ)  -o $(EXE) $(LINKFLAGS) $(SYSLIB) $(LIBRARY_DIRS) $(LIBRARIES)

# Compilation rules
.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o

# Individual dependencies
$(OBJ): $(INC)

all: $(EXE)

clean: 
	rm ./$(OBJ)
	rm ./$(EXE)
