#include <boost/python.hpp>
#include <stdio.h>
#include <boost/python.hpp>
#include <iostream>
#include <math.h>
#include <vector>


namespace bp = boost::python;

void FMM(){
  std::cout << "Hello from C++" << std::endl;
  
  std::cout << "Goodbye from C++" << std::endl;
  return;
}




BOOST_PYTHON_MODULE(periodic_fmm)
{
  using namespace boost::python;
  // boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  def("FMM", FMM);
}
