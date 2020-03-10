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

enum DistribType { UnifGrid, RandUnif, RandGaus, RandElps, RandSphr };

template <class Real_t>
std::vector<Real_t> point_distrib(DistribType dist_type, size_t N,
                                  MPI_Comm comm) {
    int np, myrank;
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &myrank);
    static size_t seed = myrank + 1;
    seed += np;
    srand48(seed);

    std::vector<Real_t> coord;
    switch (dist_type) {
    case UnifGrid: {
        size_t NN = (size_t)round(pow((double)N, 1.0 / 3.0));
        size_t N_total = NN * NN * NN;
        size_t start = myrank * N_total / np;
        size_t end = (myrank + 1) * N_total / np;
        for (size_t i = start; i < end; i++) {
            coord.push_back(((Real_t)((i / 1) % NN) + 0.5) / NN);
            coord.push_back(((Real_t)((i / NN) % NN) + 0.5) / NN);
            coord.push_back(((Real_t)((i / (NN * NN)) % NN) + 0.5) / NN);
        }
    } break;
    case RandUnif: {
        size_t N_total = N;
        size_t start = myrank * N_total / np;
        size_t end = (myrank + 1) * N_total / np;
        size_t N_local = end - start;
        coord.resize(N_local * 3);

        for (size_t i = 0; i < N_local * 3; i++)
            coord[i] = ((Real_t)drand48());
    } break;
    case RandGaus: {
        Real_t e = 2.7182818284590452;
        Real_t log_e = log(e);
        size_t N_total = N;
        size_t start = myrank * N_total / np;
        size_t end = (myrank + 1) * N_total / np;

        for (size_t i = start * 3; i < end * 3; i++) {
            Real_t y = -1;
            while (y <= 0 || y >= 1) {
                Real_t r1 = sqrt(-2 * log(drand48()) / log_e) *
                            cos(2 * M_PI * drand48());
                Real_t r2 = pow(0.5, i * 10 / N_total);
                y = 0.5 + r1 * r2;
            }
            coord.push_back(y);
        }
    } break;
    case RandElps: {
        size_t N_total = N;
        size_t start = myrank * N_total / np;
        size_t end = (myrank + 1) * N_total / np;
        size_t N_local = end - start;
        coord.resize(N_local * 3);

        const Real_t r = 0.45;
        const Real_t center[3] = {0.5, 0.5, 0.5};
        for (size_t i = 0; i < N_local; i++) {
            Real_t *y = &coord[i * 3];
            Real_t phi = 2 * M_PI * drand48();
            Real_t theta = M_PI * drand48();
            y[0] = center[0] + 0.25 * r * sin(theta) * cos(phi);
            y[1] = center[1] + 0.25 * r * sin(theta) * sin(phi);
            y[2] = center[2] + r * cos(theta);
        }
    } break;
    case RandSphr: {
        size_t N_total = N;
        size_t start = myrank * N_total / np;
        size_t end = (myrank + 1) * N_total / np;
        size_t N_local = end - start;
        coord.resize(N_local * 3);

        const Real_t center[3] = {0.5, 0.5, 0.5};
        for (size_t i = 0; i < N_local; i++) {
            Real_t *y = &coord[i * 3];
            Real_t r = 1;
            while (r > 0.5 || r == 0) {
                y[0] = drand48();
                y[1] = drand48();
                y[2] = drand48();
                r = sqrt((y[0] - center[0]) * (y[0] - center[0]) +
                         (y[1] - center[1]) * (y[1] - center[1]) +
                         (y[2] - center[2]) * (y[2] - center[2]));
                y[0] = center[0] + 0.45 * (y[0] - center[0]) / r;
                y[1] = center[1] + 0.45 * (y[1] - center[1]) / r;
                y[2] = center[2] + 0.45 * (y[2] - center[2]) / r;
            }
        }
    } break;
    default:
        break;
    }
    return coord;
}

std::vector<double> integrate(const double *t, const int nt, double lb = 0,
                              double ub = 1) {
    // integrate numerically
    // Integrate[1/Sqrt[(x-tx)^2+(y-ty)^2+(z-tz)^2],{x,lb,ub},{y,lb,ub},{z,lb,ub}]
    double scale = ub - lb;
    std::vector<double> trg_coord(nt * 3);
    for (int i = 0; i < nt; i++) {
        trg_coord[3 * i + 0] = (t[3 * i + 0] - lb) / scale;
        trg_coord[3 * i + 1] = (t[3 * i + 1] - lb) / scale;
        trg_coord[3 * i + 2] = (t[3 * i + 2] - lb) / scale;
    }

    // Set kernel.
    const pvfmm::Kernel<double> &kernel_fn =
        pvfmm::LaplaceKernel<double>::potential();

    // Construct tree.
    size_t max_pts = 100;
    int cheb_deg = 14;
    int mult_order = 14;
    MPI_Comm comm = MPI_COMM_WORLD;

    pvfmm::ChebFMM_Tree<double> tree(comm);
    { // Initialize tree
        pvfmm::ChebFMM_Data<double> tree_data;
        tree_data.cheb_deg = cheb_deg;
        tree_data.data_dof = kernel_fn.ker_dim[0];
        tree_data.input_fn = fn_input;
        tree_data.tol = 1e-10;
        bool adap = false;

        tree_data.dim = PVFMM_COORD_DIM;
        tree_data.max_depth = PVFMM_MAX_DEPTH;
        tree_data.max_pts = max_pts;

        // Set refinement point coordinates.
        tree_data.pt_coord = point_distrib<double>(UnifGrid, 12800, comm);

        // Set target points.
        tree_data.trg_coord = trg_coord;

        // Initialize with input data.
        tree.Initialize(&tree_data);
        tree.InitFMM_Tree(adap, pvfmm::FreeSpace);
    }

    // Load matrices.
    pvfmm::ChebFMM<double> matrices;
    matrices.Initialize(mult_order, cheb_deg, comm, &kernel_fn);

    // FMM Setup
    tree.SetupFMM(&matrices);

    // Run FMM
    size_t n_trg = trg_coord.size() / PVFMM_COORD_DIM;
    std::vector<double> trg_value(n_trg);
    pvfmm::ChebFMM_Evaluate(&tree, trg_value, n_trg);
    // std::cout << pvfmm::Vector<double>(trg_value) << '\n'; // print output
    for (auto &v : trg_value) {
        v *= 4 * PI314 * scale * scale;
    }
    return trg_value;
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
    // double fEwald=0;
    // int N = 2*DIRECTLAYER;
    // for (int i = -N; i < N + 1; i++) {
    //     for (int j = -N; j < N + 1; j++) {
    //         for (int k = -N; k < N + 1; k++) {
    //             double gFree = gKernel(target, source + EVec3(i, j, k));
    //             fEwald += gFree;
    //         }
    //     }
    // }

    int N = DIRECTLAYER;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                double gFree = gKernel(target, source + EVec3(i, j, k));
                fEwald -= gFree;
            }
        }
    }

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

    std::cout << std::scientific << std::setprecision(15);
    // {
    //     std::vector<double> t = {1.2, 0.3, 0.4, 2.5, 1.2, 0.8};
    //     const auto &v = integrate(t.data(), 2, -2, 3);
    //     std::cout << std::scientific << std::setprecision(15) //
    //               << v[0] << std::endl
    //               << v[1] << std::endl;
    //     std::exit(0);
    // }

    //     {
    //         EVec3 samplePoint(0.72, 0.7, 0.8);
    //         EVec3 chargePoint(0.11, 0.2, 0.3);
    //         std::cout << gKernelEwald(chargePoint, chargePoint) << std::endl;
    //         std::cout << gKernelEwald(chargePoint + EVec3(0.001, 0, 0),
    //                                   chargePoint) -
    //                          gKernel(chargePoint + EVec3(0.001, 0, 0),
    //                          chargePoint)
    //                   << std::endl;
    //         double pot = gKernelEwald(samplePoint, chargePoint);
    //         std::cout << pot << std::endl;
    //         double pot1 = pot;
    //         const int N = 60;
    // #pragma omp parallel for
    //         for (int i = 0; i < N; i++) {
    //             for (int j = 0; j < N; j++) {
    //                 for (int k = 0; k < N; k++) {
    //                     EVec3 negPoint(i, j, k);
    //                     negPoint *= (1.0 / N);
    //                     double p = gKernelEwald(samplePoint, negPoint) *
    //                                (-1.0 / pow(N, 3));
    // #pragma omp atomic
    //                     pot += p;
    //                 }
    //             }
    //         }
    //         std::cout << pot << std::endl;
    //         std::cout << pot - pot1 << std::endl;
    //         // std::exit(0);
    //     }

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

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        chargePoint(2);
    std::vector<double> chargeValue(2);
    chargePoint[0] = Eigen::Vector3d(0.61, 0.345, 0.767);
    chargeValue[0] = -1.0;
    chargePoint[1] = Eigen::Vector3d(0.1, 0.1, 0.1);
    chargeValue[1] = 1.0;

    chargeValue.resize(2);
    chargePoint.resize(2);

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

    // neutralizing background
    const auto &neuPotCpoint =
        integrate(pointLCheck.data(), checkN, -DIRECTLAYER, DIRECTLAYER + 1);
    std::cout << "background potential" << std::endl;
    for (auto &v : neuPotCpoint) {
        std::cout << v << std::endl;
    }

    // #pragma omp parallel for
    for (int i = 0; i < 1; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        // assemble linear system
        Eigen::VectorXd f(checkN);
        for (int k = 0; k < checkN; k++) {
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            //			std::cout<<"debug:"<<k<<std::endl;
            // sum the images
            for (int p = 0; p < chargeValue.size(); p++)
                f[k] += (gKernelFF(Cpoint, chargePoint[p]) + neuPotCpoint[k]) *
                        chargeValue[p];
            // f[k] = gKernelFF(Cpoint, Mpoint) - gKernelFF(Cpoint, neupoint);
        }

        M2L.col(i) = (ApinvU.transpose() * (ApinvVT.transpose() * f));

        std::cout << "debug: \n" << f << std::endl;
        std::cout << "debug M: \n" << M2L.col(i) << std::endl;
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

    // // solve M
    // A.resize(checkN, equivN);
    // ApinvU.resize(A.cols(), A.rows());
    // ApinvVT.resize(A.cols(), A.rows());
    // Eigen::VectorXd f(checkN);
    // for (int k = 0; k < checkN; k++) {
    //     double temp = 0;
    //     Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
    //                            pointMCheck[3 * k + 2]);
    //     for (size_t p = 0; p < chargePoint.size(); p++) {
    //         temp = temp + gKernel(Cpoint, chargePoint[p]) * (chargeValue[p]);
    //     }
    //     f(k) = temp;
    //     for (int l = 0; l < equivN; l++) {
    //         Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l +
    //         1],
    //                                pointMEquiv[3 * l + 2]);
    //         A(k, l) = gKernel(Mpoint, Cpoint);
    //     }
    // }
    // pinv(A, ApinvU, ApinvVT);
    // Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() *
    // f));

    Eigen::VectorXd Msource = M2L.col(0);
    std::cout << "Msource sum: " << Msource.sum() << std::endl;

    Eigen::VectorXd M2Lsource = (Msource);

    {
        // verify. Three components:
        // 1. direct near field
        // 2. background near field
        // 3. far field

        Eigen::Vector3d samplePoint = chargePoint[0];
        double UsampleNF = 0;
        double UsampleBGNF = 0;
        double UsampleFF = 0;
        double EwaldFF = 0;
        double Ewald = 0;

        for (int i = -DIRECTLAYER; i < 1 + DIRECTLAYER; i++) {
            for (int j = -DIRECTLAYER; j < 1 + DIRECTLAYER; j++) {
                for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
                    for (size_t p = 0; p < chargePoint.size(); p++) {
                        UsampleNF += gKernel(samplePoint,
                                             chargePoint[p] + EVec3(i, j, k)) *
                                     chargeValue[p];
                    }
                }
            }
        }

        for (int p = 0; p < equivN; p++) {
            Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1],
                                   pointLEquiv[3 * p + 2]);
            UsampleFF += gKernel(samplePoint, Lpoint) * M2Lsource[p];
        }

        const auto &val =
            integrate(samplePoint.data(), 1, -DIRECTLAYER, DIRECTLAYER + 1);

        for (size_t p = 0; p < chargePoint.size(); p++) {
            EwaldFF += gKernelFF(samplePoint, chargePoint[p]) * chargeValue[p];
            Ewald += gKernelEwald(samplePoint, chargePoint[p]) * chargeValue[p];
            UsampleBGNF += val[0] * chargeValue[p];
        }

        std::cout << "samplePoint:" << samplePoint.transpose() << std::endl;
        std::cout << "Usample NF:" << UsampleNF << std::endl;
        std::cout << "Usample BGNF:" << UsampleBGNF << std::endl;
        std::cout << "Usample FF:" << UsampleFF << std::endl;
        std::cout << "EwaldFF:" << EwaldFF << std::endl;
        std::cout << "FF Error: " << EwaldFF - (UsampleFF + UsampleBGNF)
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}

} // namespace Laplace3D3D

#undef DIRECTLAYER
#undef PI314
