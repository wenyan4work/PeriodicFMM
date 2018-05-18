include ../MakefileInc.mk

INCLUDE_DIRS = $(Trilinos_INCLUDE_DIRS) $(Trilinos_TPL_INCLUDE_DIRS) $(USERINCLUDE) -I../../
LIBRARY_DIRS = $(Trilinos_LIBRARY_DIRS) $(Trilinos_TPL_LIBRARY_DIRS) 
LIBRARIES = $(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES) $(USERLIB)
# Files

SRC =   main.c ../../FMM/FMMWrapper.cpp ../../FMM/FMMWrapperWall2D.cpp ../../Interface/FMMWrapper-c.cpp ../../Interface/FMMWrapperWall2D-c.cpp
INC =   ../../FMM/FMMWrapper.h ../../Interface/FMMWrapper-c.h ../../Interface/FMMWrapperWall2D-c.h
         
# Definitions
CXXFLAGS += -DFMMTIMING

CC = mpicc
CFLAGS = $(CXXFLAGS) -std=c99

EXE =   TestCAPI.X
OBJ =   $(SRC:.cpp=.o) main.o

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
