#include <iostream>

namespace Laplace1D3D {
int main(int argc, char *argv[]);
}

namespace Laplace2D3D {
int main(int argc, char *argv[]);
}

namespace Laplace1D3DDipole {
int main(int argc, char *argv[]);
}

namespace Laplace2D3DDipole {
int main(int argc, char *argv[]);
}

namespace Laplace3D3DDipole {
int main(int argc, char *argv[]);
}

int main(int argc, char *argv[]) {
  int dim = atoi(argv[1]);

  switch (dim) {
  case 1:
    Laplace1D3DDipole::main(argc - 1, argv + 1);
    break;
  case 2:
    Laplace2D3DDipole::main(argc - 1, argv + 1);
    break;
  case 3:
    Laplace1D3DDipole::main(argc - 1, argv + 1);
    break;
  case 4:
    Laplace1D3D::main(argc - 1, argv + 1);
    break;
  case 5:
    Laplace2D3D::main(argc - 1, argv + 1);
    break;
  default:
      std::cerr << "Invalid dimension " << dim << std::endl;
      break;
  }
  
  return 0;
}
