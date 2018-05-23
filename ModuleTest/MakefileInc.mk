# CCB014
SFTPATH=/mnt/home/wyan/local

# Mac 
# SFTPATH=/Users/wyan/local

include $(PVFMM_DIR)/MakeVariables
EIGEN= $(SFTPATH)/include/eigen3
PVFMM= $(SFTPATH)/include/pvfmm

USERINCLUDE = -I$(EIGEN) -I./
USERLIB_DIRS = -L$(SFTPATH)/lib

CXX= mpicxx
LINK= $(CXX)

# optimized
CXXFLAGS= $(CXXFLAGS_PVFMM) 
LINKFLAGS= $(CXXFLAGS) $(LDLIBS_PVFMM) 

# debug
DEBUGMODE:= no

# debug flags
# CXXFLAGS += -DFMMTIMING 
# CXXFLAGS += -DFMMDEBUG

ifeq ($(DEBUGMODE), yes)
CXXFLAGS:= $(subst -O3, ,$(CXXFLAGS))
LINKFLAGS:= $(subst -O3, ,$(LINKFLAGS))
CXXFLAGS := $(CXXFLAGS) -O0 -g
LINKFLAGS := $(LINKFLAGS) -O0 -g
else
CXXFLAGS:= $(CXXFLAGS) -DNDEBUG
LINKFLAGS:= $(LINKFLAGS) -DNDEBUG
endif
