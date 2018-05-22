include ../MakefileInc.mk

INCLUDE_DIRS = $(USERINCLUDE) -I../../
LIBRARY_DIRS =
LIBRARIES =
# Files

SRC =   main.f90 ../../Interface/FMMWrapper-f.f90 ../../Interface/FMMWrapperWall2D-f.f90 ../../FMM/FMMWrapper.cpp ../../FMM/FMMWrapperWall2D.cpp ../../Interface/FMMWrapper-c.cpp ../../Interface/FMMWrapperWall2D-c.cpp
INC =   ../../FMM/FMMWrapper.h ../../Interface/FMMWrapper-c.h ../../Interface/FMMWrapperWall2D-c.h
         
# Definitions
CXXFLAGS += -DFMMTIMING

CC = mpicc
CFLAGS = $(CXXFLAGS)

FC = mpifc
FFLAGS = $(CXXFLAGS) -std03 # fortran 2003 in intel compiler

LINK := mpifc
LINKFLAGS := $(FFLAGS) $(LINKFLAGS) -cxxlib # link with cxx runtime in intel compiler

EXE = TestF03API.X
OBJ = ../../Interface/FMMWrapper-f.o ../../Interface/FMMWrapperWall2D-f.o main.o ../../FMM/FMMWrapper.o ../../FMM/FMMWrapperWall2D.o ../../Interface/FMMWrapper-c.o ../../Interface/FMMWrapperWall2D-c.o

# Link rule
$(EXE):	$(OBJ) $(SRC)
	$(LINK) $(OBJ)  -o $(EXE) $(LINKFLAGS) $(SYSLIB) $(LIBRARY_DIRS) $(LIBRARIES)

# Compilation rules
../../Interface/FMMWrapper-f.o:
	$(FC) $(FFLAGS) $(INCLUDE_DIRS) -c ../../Interface/FMMWrapper-f.f90 -o ../../Interface/FMMWrapper-f.o
../../Interface/FMMWrapperWall2D-f.o:
	$(FC) $(FFLAGS) $(INCLUDE_DIRS) -c ../../Interface/FMMWrapperWall2D-f.f90 -o ../../Interface/FMMWrapperWall2D-f.o
main.o: ../../Interface/FMMWrapper-f.o ../../Interface/FMMWrapperWall2D-f.o
	$(FC) $(FFLAGS) $(INCLUDE_DIRS) -c main.f90 -o main.o
.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o
.c.o:
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $*.c -o $*.o

# Individual dependencies
$(OBJ): $(INC)

all: $(EXE)

clean: 
	rm ./$(OBJ)
	rm ./*.mod
	rm ./$(EXE)
