#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>


namespace bp = boost::python;
namespace np = boost::python::numpy;

void FMM(){

  //std::cout << "Hello from C++" << std::endl;
  //std::cout << "Goodbye from C++" << std::endl;
  return;
}




BOOST_PYTHON_MODULE(periodic_fmm)
{
  using namespace boost::python;

  // Initialize numpy
  // Py_Initialize();
  // np::initialize();

  // Definitions
  def("FMM", FMM);
}
