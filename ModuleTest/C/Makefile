include ../../MakefileInc.mk

INCLUDE_DIRS = $(USERINCLUDE) -I../../
LIBRARY_DIRS =
LIBRARIES =
# Files

SRC =   main.c ../../FMM/FMMWrapper.cpp ../../FMM/FMMWrapperWall2D.cpp ../../Interface/FMMWrapper-c.cpp ../../Interface/FMMWrapperWall2D-c.cpp
INC =   ../../FMM/FMMWrapper.h ../../Interface/FMMWrapper-c.h ../../Interface/FMMWrapperWall2D-c.h
         
# Definitions
CXXFLAGS += -DFMMTIMING

CC = mpicc
CFLAGS = $(CXXFLAGS) -std=c99

LINK := mpicc
LINKFLAGS := $(CFLAGS) $(LINKFLAGS)

EXE = TestCAPI.X
OBJ = main.o ../../FMM/FMMWrapper.o ../../FMM/FMMWrapperWall2D.o ../../Interface/FMMWrapper-c.o ../../Interface/FMMWrapperWall2D-c.o

# Link rule
$(EXE):	$(OBJ)
	$(LINK) $(OBJ)  -o $(EXE) $(LINKFLAGS) $(SYSLIB) $(LIBRARY_DIRS) $(LIBRARIES)

# Compilation rules
.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o
.c.o:
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $*.c -o $*.o

# Individual dependencies
$(OBJ): $(INC)

all: $(EXE)

clean: 
	rm ./$(OBJ)
	rm ./$(EXE)
