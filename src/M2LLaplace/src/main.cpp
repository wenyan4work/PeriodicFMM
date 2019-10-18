#include <iostream>
#include <string>

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
    if (argc != 4) {
        std::cerr << "Input -[cd] {dim} {N}\n";
        return 1;
    }

    bool dipoleFlag = false;
    if (std::string(argv[1]) == "-d") {
        dipoleFlag = true;
    } else if (std::string(argv[1]) == "-c") {
    } else {
        std::cerr << "Invalid flag " << argv[1] << ".\n";
        return 1;
    }

    int dim = atoi(argv[2]);
    argc -= 2;
    argv += 2;

    if (dipoleFlag) {
        switch (dim) {
        case 1:
            Laplace1D3DDipole::main(argc, argv);
            break;
        case 2:
            Laplace2D3DDipole::main(argc, argv);
            break;
        case 3:
            Laplace1D3DDipole::main(argc, argv);
            break;
        default:
            std::cerr << "Dimension {" << dim
                      << "} not supported for dipole.\n";
            return 1;
        }
    } else {
        switch (dim) {
        case 1:
            Laplace1D3D::main(argc, argv);
            break;
        case 2:
            Laplace2D3D::main(argc, argv);
            break;
        default:
            std::cerr << "Dimension {" << dim
                      << "} not supported for charge.\n";
            return 1;
        }
    }
    return 0;
}
