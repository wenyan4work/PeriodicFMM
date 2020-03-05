/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

#include <pvfmm.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <mpi.h>

#define DIRECTLAYER 2
#define PI314 (static_cast<double>(3.1415926535897932384626433))

namespace Laplace3D3D {

// chebfmm Input function
void fn_input(const double *coord, int n, double *out) {
    for (int i = 0; i < n; i++) {
        out[i] = 1;
    }
}

double integrate(const double t[3], double lb = 0, double ub = 1) {
    double scale = ub - lb;
    double ts[3] = {(t[0] - lb) / scale, (t[1] - lb) / scale,
                    (t[2] - lb) / scale};
    // integrate numerically
    // Integrate[1/Sqrt[(x-tx)^2+(y-ty)^2+(z-tz)^2],{x,lb,ub},{y,lb,ub},{z,lb,ub}]

    // Set kernel.
    const pvfmm::Kernel<double> &kernel_fn =
        pvfmm::LaplaceKernel<double>::potential();
    // Construct tree.
    size_t max_pts = 100;
    std::vector<double> trg_coord(3);
    trg_coord[0] = ts[0];
    trg_coord[1] = ts[1];
    trg_coord[2] = ts[2];
    int cheb_deg = 6;
    int mult_order = 6;
    MPI_Comm comm = MPI_COMM_WORLD;
    auto *tree =
        ChebFMM_CreateTree(cheb_deg, kernel_fn.ker_dim[0], fn_input, trg_coord,
                           comm, 1e-4, max_pts, pvfmm::FreeSpace);

    // Load matrices.
    pvfmm::ChebFMM<double> matrices;
    matrices.Initialize(mult_order, cheb_deg, MPI_COMM_WORLD, &kernel_fn);

    // FMM Setup
    tree->SetupFMM(&matrices);

    // Run FMM
    size_t n_trg = trg_coord.size() / PVFMM_COORD_DIM;
    std::vector<double> trg_value(n_trg);
    pvfmm::ChebFMM_Evaluate(tree, trg_value, n_trg);
    delete tree;
    return trg_value[0] * 4 * PI314 * scale*scale;
}

using EVec3 = Eigen::Vector3d;

inline double ERFC(double x) { return std::erfc(x); }
inline double ERF(double x) { return std::erf(x); }

// real and wave sum of 2D Laplace kernel Ewald

// xm: target, xn: source
inline double realSum(const double xi, const EVec3 &xn, const EVec3 &xm) {
    EVec3 rmn = xm - xn;
    double rnorm = rmn.norm();
    if (rnorm < 1e-10) {
        return 0;
    }
    return ERFC(rnorm * xi) / rnorm;
}

inline double gKernelEwald(const EVec3 &xm, const EVec3 &xn) {
    const double xi = 2; // recommend for box=1 to get machine precision
    EVec3 target = xm;
    EVec3 source = xn;
    target[0] = target[0] - floor(target[0]); // periodic BC
    target[1] = target[1] - floor(target[1]);
    target[2] = target[2] - floor(target[2]);
    source[0] = source[0] - floor(source[0]);
    source[1] = source[1] - floor(source[1]);
    source[2] = source[2] - floor(source[2]);

    // real sum
    int rLim = 4;
    double Kreal = 0;
    for (int i = -rLim; i <= rLim; i++) {
        for (int j = -rLim; j <= rLim; j++) {
            for (int k = -rLim; k <= rLim; k++) {
                Kreal += realSum(xi, target, source - EVec3(i, j, k));
            }
        }
    }

    // wave sum
    int wLim = 4;
    double Kwave = 0;
    EVec3 rmn = target - source;
    const double xi2 = xi * xi;
    const double rmnnorm = rmn.norm();
    for (int i = -wLim; i <= wLim; i++) {
        for (int j = -wLim; j <= wLim; j++) {
            for (int k = -wLim; k <= wLim; k++) {
                if (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                EVec3 kvec = EVec3(i, j, k) * (2 * PI314);
                double k2 = kvec.dot(kvec);
                Kwave +=
                    4 * PI314 * cos(kvec.dot(rmn)) * exp(-k2 / (4 * xi2)) / k2;
            }
        }
    }

    double Kself = rmnnorm < 1e-10 ? -2 * xi / sqrt(PI314) : 0;

    return Kreal + Kwave + Kself - PI314 / xi2;
}

inline double gKernel(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    return rnorm < 1e-10 ? 0 : 1 / rnorm;
}

// Out of Direct Sum Layer, far field part
inline double gKernelFF(const EVec3 &target, const EVec3 &source) {
    double fEwald = gKernelEwald(target, source);
    const int N = DIRECTLAYER;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                double gFree = gKernel(target, source + EVec3(i, j, k));
                fEwald -= gFree;
            }
        }
    }

    // add neutralizing background within DIRECTLAYER
    fEwald -= integrate(target.data(),-N,N+1);

    //   {
    //     std::cout << "source:" << source << std::endl
    //               << "target:" << target << std::endl
    //               << "gKernalFF" << fEwald << std::endl;
    //   }
    return fEwald;
}

/**
 * \brief Returns the coordinates of points on the surface of a cube.
 * \param[in] p Number of points on an edge of the cube is (n+1)
 * \param[in] c Coordinates to the centre of the cube (3D array).
 * \param[in] alpha Scaling factor for the size of the cube.
 * \param[in] depth Depth of the cube in the octree.
 * \return Vector with coordinates of points on the surface of the cube in the
 * format [x0 y0 z0 x1 y1 z1 .... ].
 */

template <class Real_t>
std::vector<Real_t> surface(int p, Real_t *c, Real_t alpha, int depth) {
    size_t n_ = (6 * (p - 1) * (p - 1) + 2); // Total number of points.

    std::vector<Real_t> coord(n_ * 3);
    coord[0] = coord[1] = coord[2] = -1.0;
    size_t cnt = 1;
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = -1.0;
            coord[cnt * 3 + 1] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = (2.0 * j - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * i - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = -1.0;
            coord[cnt * 3 + 2] = (2.0 * (j + 1) - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = (2.0 * j - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = -1.0;
            cnt++;
        }
    for (size_t i = 0; i < (n_ / 2) * 3; i++)
        coord[cnt * 3 + i] = -coord[i];

    Real_t r = 0.5 * pow(0.5, depth);
    Real_t b = alpha * r;
    for (size_t i = 0; i < n_; i++) {
        coord[i * 3 + 0] = (coord[i * 3 + 0] + 1.0) * b + c[0];
        coord[i * 3 + 1] = (coord[i * 3 + 1] + 1.0) * b + c[1];
        coord[i * 3 + 2] = (coord[i * 3 + 2] + 1.0) * b + c[2];
    }
    return coord;
}

int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    MPI_Init(&argc, &argv);

    {
        double t[3] = {0.2, 0.3, 0.4};
        std::cout << integrate(t, -3, 4) << std::endl;
    }

    {
        EVec3 samplePoint(0.72, 0.7, 0.8);
        EVec3 chargePoint(0.11, 0.2, 0.3);
        std::cout << gKernelEwald(chargePoint, chargePoint) << std::endl;
        std::cout << gKernelEwald(chargePoint + EVec3(0.001, 0, 0),
                                  chargePoint) -
                         gKernel(chargePoint + EVec3(0.001, 0, 0), chargePoint)
                  << std::endl;
        double pot = gKernelEwald(samplePoint, chargePoint);
        std::cout << pot << std::endl;
        double pot1 = pot;
        const int N = 60;
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    EVec3 negPoint(i, j, k);
                    negPoint *= (1.0 / N);
                    double p = gKernelEwald(samplePoint, negPoint) *
                               (-1.0 / pow(N, 3));
#pragma omp atomic
                    pot += p;
                }
            }
        }
        std::cout << pot << std::endl;
        std::cout << pot - pot1 << std::endl;
        // std::exit(0);
    }

    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    const int pEquiv = atoi(argv[1]); // (8-1)^2*6 + 2 points
    const int pCheck = atoi(argv[1]);
    const double scaleEquiv = 1.05;
    const double scaleCheck = 2.95;
    const double pCenterEquiv[3] = {
        -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {
        -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};

    const double scaleLEquiv = 1.05;
    const double scaleLCheck = 2.95;
    const double pCenterLEquiv[3] = {
        -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2};
    const double pCenterLCheck[3] = {
        -(scaleLCheck - 1) / 2, -(scaleLCheck - 1) / 2, -(scaleLCheck - 1) / 2};

    auto pointMEquiv = surface(
        pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointMCheck = surface(
        pCheck, (double *)&(pCenterCheck[0]), scaleCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    auto pointLEquiv = surface(
        pEquiv, (double *)&(pCenterLCheck[0]), scaleLCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointLCheck = surface(
        pCheck, (double *)&(pCenterLEquiv[0]), scaleLEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    // calculate the operator M2L with least square
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointLCheck.size() / 3;
    Eigen::MatrixXd M2L(equivN, equivN); // Laplace, 1->1

    Eigen::MatrixXd A(1 * checkN, 1 * equivN);
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                               pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l],
                                         pointLEquiv[3 * l + 1],
                                         pointLEquiv[3 * l + 2]);
            A(k, l) = gKernel(Cpoint, Lpoint);
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);

// #pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        //		std::cout << "debug:" << Mpoint << std::endl;

        // assemble linear system
        Eigen::VectorXd f(checkN);
        for (int k = 0; k < checkN; k++) {
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            //			std::cout<<"debug:"<<k<<std::endl;
            // sum the images
            f[k] = gKernelFF(Cpoint, Mpoint);
        }
        std::cout << "debug:" << f << std::endl;

        M2L.col(i) = (ApinvU.transpose() * (ApinvVT.transpose() * f));
    }
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Precomputing time:" << duration / 1e6 << std::endl;

    // dump M2L
    // for (int i = 0; i < equivN; i++) {
    //     for (int j = 0; j < equivN; j++) {
    //         std::cout << i << " " << j << " " << std::scientific
    //                   << std::setprecision(18) << M2L(i, j) << std::endl;
    //     }
    // }

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        chargePoint(2);
    std::vector<double> chargeValue(2);
    chargePoint[0] = Eigen::Vector3d(0.6, 0.6, 0.6);
    chargeValue[0] = -1;
    chargePoint[1] = Eigen::Vector3d(0.1, 0.1, 0.1);
    chargeValue[1] = 1.0;

    // solve M
    A.resize(checkN, equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    Eigen::VectorXd f(checkN);
    for (int k = 0; k < checkN; k++) {
        double temp = 0;
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
                               pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < chargePoint.size(); p++) {
            temp = temp + gKernel(Cpoint, chargePoint[p]) * (chargeValue[p]);
        }
        f(k) = temp;
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1],
                                   pointMEquiv[3 * l + 2]);
            A(k, l) = gKernel(Mpoint, Cpoint);
        }
    }
    pinv(A, ApinvU, ApinvVT);
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));

    std::cout << "Msource sum: " << Msource.sum() << std::endl;

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    Eigen::Vector3d samplePoint = chargePoint[0];
    double Usample = 0;
    double UsampleSP = 0;
    double EwaldFF = 0;
    for (int i = -DIRECTLAYER; i < 1 + DIRECTLAYER; i++) {
        for (int j = -DIRECTLAYER; j < 1 + DIRECTLAYER; j++) {
            for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
                for (size_t p = 0; p < chargePoint.size(); p++) {
                    Usample +=
                        gKernel(samplePoint, chargePoint[p] + EVec3(i, j, k)) *
                        chargeValue[p];
                }
            }
        }
    }

    for (size_t p = 0; p < chargePoint.size(); p++) {
        EwaldFF += gKernelFF(samplePoint, chargePoint[p]) * chargeValue[p];
    }
    for (int p = 0; p < equivN; p++) {
        Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1],
                               pointLEquiv[3 * p + 2]);
        UsampleSP += gKernel(samplePoint, Lpoint) * M2Lsource[p];
    }

    std::cout << "samplePoint:" << samplePoint.transpose() << std::endl;
    std::cout << "Usample NF:" << Usample << std::endl;
    std::cout << "Usample FF:" << UsampleSP << std::endl;
    std::cout << "Ewald FF:" << EwaldFF << std::endl;
    std::cout << "Usample FF+NF total:" << UsampleSP + Usample << std::endl;
    const double p0 =
        gKernelEwald(samplePoint, chargePoint[0]) * chargeValue[0] +
        gKernelEwald(samplePoint, chargePoint[1]) * chargeValue[1];
    std::cout << "Ewald " << p0 << std::endl;

    samplePoint = chargePoint[0] - EVec3(0.001, 0.001, 0.001);
    Usample = 0;
    UsampleSP = 0;
    EwaldFF = 0;

    for (int i = -DIRECTLAYER; i < 1 + DIRECTLAYER; i++) {
        for (int j = -DIRECTLAYER; j < 1 + DIRECTLAYER; j++) {
            for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
                for (size_t p = 0; p < chargePoint.size(); p++) {
                    Usample +=
                        gKernel(samplePoint, chargePoint[p] + EVec3(i, j, k)) *
                        chargeValue[p];
                }
            }
        }
    }

    for (size_t p = 0; p < chargePoint.size(); p++) {
        EwaldFF += gKernelFF(samplePoint, chargePoint[p]) * chargeValue[p];
    }
    for (int p = 0; p < equivN; p++) {
        Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1],
                               pointLEquiv[3 * p + 2]);
        UsampleSP += gKernel(samplePoint, Lpoint) * M2Lsource[p];
    }

    std::cout << "samplePoint:" << samplePoint.transpose() << std::endl;
    std::cout << "Usample NF:" << Usample << std::endl;
    std::cout << "Usample FF:" << UsampleSP << std::endl;
    std::cout << "Ewald FF:" << EwaldFF << std::endl;
    std::cout << "Usample FF+NF total:" << UsampleSP + Usample << std::endl;
    const double p1 =
        gKernelEwald(samplePoint, chargePoint[0]) * chargeValue[0] +
        gKernelEwald(samplePoint, chargePoint[1]) * chargeValue[1] -
        gKernel(samplePoint, chargePoint[0]) * chargeValue[0];
    std::cout << "Ewald " << p1 << std::endl;
    std::cout << "Error : " << std::setprecision(16) << p0 + p1 << std::endl;
    std::cout << "Error : " << std::setprecision(16) << p0 - p1 << std::endl;
    MPI_Finalize();
    return 0;
}

} // namespace Laplace3D3D

#undef DIRECTLAYER
#undef PI314
