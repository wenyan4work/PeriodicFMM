#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>

#include "FMM/FMMWrapper.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

void FMM(np::ndarray trg_coord){

  std::cout << "Hello from C++" << std::endl;
  std::cout << "Goodbye from C++" << std::endl;
  return;
}




BOOST_PYTHON_MODULE(periodic_fmm)
{
  // using namespace boost::python;
  // using namespace boost::python::numpy;

  // Initialize numpy
  Py_Initialize();
  np::initialize();

  // Definitions
  def("FMM", FMM);
}
