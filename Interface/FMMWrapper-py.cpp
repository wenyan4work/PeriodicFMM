#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>

#include "FMM/FMMWrapper.h"
#include "FMM/MiniFMM.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

// typedef struct FMM_Wrapper FMM_Wrapper;


void FMM(np::ndarray trg_coord){

  std::cout << "Hello from C++" << std::endl;
  std::cout << "Goodbye from C++" << std::endl;
  return;
}








BOOST_PYTHON_MODULE(periodic_fmm)
{
  using namespace boost::python;
  // using namespace boost::python::numpy;

  // Initialize numpy
  Py_Initialize();
  np::initialize();

  // Definitions
  def("FMM", FMM);

  // Class
  // 
  // enum_<Mini_FMM::PAXIS>("PAXIS")
  // .value("NONE", Mini_FMM::NONE);

  // Define Mock class
  enum_<Mini_FMM::PAXIS>("Mini_PAXIS")
    .value("NONE", Mini_FMM::NONE)
    .value("PX", Mini_FMM::PX)
    .value("PY", Mini_FMM::PY)
    .value("PZ", Mini_FMM::PZ)
    .value("PXY", Mini_FMM::PXY)
    .value("PXZ", Mini_FMM::PXZ)
    .value("PYZ", Mini_FMM::PYZ)
    .value("PXYZ", Mini_FMM::PXYZ);
  class_<Mini_FMM>("Mini_FMM", init<int, Mini_FMM::PAXIS>())
    .def("saludo", &Mini_FMM::saludo);

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
  // FMM_Wrapper(int mult_order = 10, int max_pts = 2000, int init_depth = 0, PAXIS pbc_ = PAXIS::NONE);
  class_<FMM_Wrapper>("FMM_Wrapper", init<int, int, int, FMM_Wrapper::PAXIS>());
  // class_<FMM_Wrapper>("FMM_Wrapper");

  
}


