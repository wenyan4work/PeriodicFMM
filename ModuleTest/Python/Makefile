include ../../MakefileInc.mk

# location of the Python header files
PYTHON_VERSION = 3.6m
PYTHON_DIR = /Users/wyan/miniconda3/envs/py3
PYTHON_INCLUDE = $(PYTHON_DIR)/include/python$(PYTHON_VERSION)/
PYTHON_LIB = $(PYTHON_DIR)/lib/

# location of the Boost Python include files and library
# for boost installed through conda 
BOOST_INC = $(PYTHON_DIR)/include
BOOST_LIB = $(PYTHON_DIR)/lib

# Define includes
INCLUDE_DIRS = -I../../ -I$(PYTHON_INCLUDE) -I$(BOOST_INC)

# Definitions -Wl,-rpath,$(BOOST_LIB) \
CXXFLAGS += -DFMMTIMING -fPIC
LINKFLAGS := $(LINKFLAGS) -shared -Wl,-undefined,error \
        -L$(BOOST_LIB)  -lboost_python -lboost_numpy -lboost_system \
		-L$(PYTHON_LIB) -lpython$(PYTHON_VERSION) 

# Target and sources
TARGET = periodic_fmm
SRC = ../../Interface/FMMWrapper-py.cpp ../../FMM/FMMWrapper.cpp ../../FMM/FMMWrapperWall2D.cpp 
INC = ../../FMM/FMMWrapper.h ../../FMM/FMMWrapperWall2D.h ../../Util/Timer.hpp
OBJ = ../../Interface/FMMWrapper-py.o ../../FMM/FMMWrapper.o ../../FMM/FMMWrapperWall2D.o

# Link rule
$(TARGET).so: $(OBJ)
	$(LINK) $(OBJ) $(LINKFLAGS) -o $(TARGET).so

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o

# Individual dependencies
$(OBJ): $(INC)


clean: 
	rm -f $(OBJ)
	rm -f $(TARGET).so

