#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>

#include "FMM/FMMWrapper.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

void FMM_UpdateTree(FMM_Wrapper* fmm, np::ndarray trg_coord, np::ndarray src_coord){
  std::cout << "Hello from FMM_UpdateTree" << std::endl;

  // Transform ndarray to std::vectors
  int ntrg = trg_coord.shape(0);
  char* ctrg = trg_coord.get_data();
  std::vector<double> vtrg(ctrg, ctrg + ntrg * 3);
 
  int nsrc = src_coord.shape(0);
  char* csrc = src_coord.get_data();
  std::vector<double> vsrc(csrc, csrc + nsrc * 3);

  // Call method
  fmm->FMM_UpdateTree(vsrc, vtrg);

  return;
}


BOOST_PYTHON_MODULE(periodic_fmm)
{
  using namespace boost::python;
  
  // Initialize numpy
  Py_Initialize();
  np::initialize();

  // Class FMM_Wrapper
  enum_<FMM_Wrapper::PAXIS>("PAXIS")
    .value("NONE", FMM_Wrapper::NONE)
    .value("PX", FMM_Wrapper::PX)
    .value("PY", FMM_Wrapper::PY)
    .value("PZ", FMM_Wrapper::PZ)
    .value("PXY", FMM_Wrapper::PXY)
    .value("PXZ", FMM_Wrapper::PXZ)
    .value("PYZ", FMM_Wrapper::PYZ)
    .value("PXYZ", FMM_Wrapper::PXYZ);
  class_<FMM_Wrapper>("FMM_Wrapper", init<int, int, int, FMM_Wrapper::PAXIS>())
    .def("FMM_TreeClear", &FMM_Wrapper::FMM_TreeClear)
    .def("FMM_DataClear", &FMM_Wrapper::FMM_DataClear)
    .def("FMM_Evaluate", &FMM_Wrapper::FMM_Evaluate)
    // .def("FMM_UpdateTree", &FMM_Wrapper::FMM_UpdateTree)
    .def("FMM_SetBox", &FMM_Wrapper::FMM_SetBox)
    ;
  
  // Define functions
  def("FMM_UpdateTree", FMM_UpdateTree);
  
}


