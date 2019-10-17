#include <iostream>

namespace Stokes1D3D {
int main(int argc, char *argv[]);
}

namespace Stokes2D3D {
int main(int argc, char *argv[]);
}

namespace Stokes3D3D {
int main(int argc, char *argv[]);
}

int main(int argc, char *argv[]) {
  int dim = atoi(argv[1]);

  switch (dim){
  case 1:
    Stokes1D3D::main(argc - 1, argv + 1);
    break;
  case 2:
    Stokes2D3D::main(argc - 1, argv + 1);
    break;
  case 3:
    Stokes3D3D::main(argc - 1, argv + 1);
    break;
  default:
      std::cerr << "Invalid dimension " << dim << std::endl;
      break;
  }
  
  return 0;
}
