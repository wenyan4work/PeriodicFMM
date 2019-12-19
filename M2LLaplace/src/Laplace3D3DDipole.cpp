/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>

#define DIRECTLAYER 2
#define PI314 (static_cast<double>(3.1415926535897932384626433))
#define E271 (static_cast<double>(2.7182818284590452354))

namespace Laplace3D3DDipole {

using EVec3 = Eigen::Vector3d;
using EMat3 = Eigen::Matrix3d;

inline double ERFC(double x) { return std::erfc(x); }
inline double ERF(double x) { return std::erf(x); }

inline double f(double r, double eta) {
    return ERFC(sqrt(PI314 / eta) * r) / r;
}

inline double fp(double r, double eta) {
    return -ERFC(sqrt(PI314 / eta) * r) / (r * r) -
           2 * exp(-PI314 * r * r / eta) / (r * sqrt(eta));
}

// real and wave sum of 2D Laplace kernel Ewald

// xm: target, xn: source
inline EVec3 realSum(const double eta, const EVec3 &xn, const EVec3 &xm) {
    EVec3 rmn = xm - xn;
    double rnorm = rmn.norm();
    if (rnorm < 1e-14) {
        return EVec3(0, 0, 0);
    }
    return -fp(rnorm, eta) / rnorm * rmn;
}

inline EVec3 gKernelEwald(const EVec3 &xm, const EVec3 &xn) {
    const double eta = 1.0; // recommend for box=1 to get machine precision
    EVec3 target = xm;
    EVec3 source = xn;
    target[0] = target[0] - floor(target[0]); // periodic BC
    target[1] = target[1] - floor(target[1]);
    target[2] = target[2] - floor(target[2]);
    source[0] = source[0] - floor(source[0]);
    source[1] = source[1] - floor(source[1]);
    source[2] = source[2] - floor(source[2]);

    // real sum
    int rLim = 6;
    EVec3 Kreal(0, 0, 0);
    for (int i = -rLim; i <= rLim; i++) {
        for (int j = -rLim; j <= rLim; j++) {
            for (int k = -rLim; k <= rLim; k++) {
                EVec3 rmn = target - source + EVec3(i, j, k);
                if (rmn.norm() < 1e-13) {
                    continue;
                }
                Kreal += realSum(eta, EVec3(0, 0, 0), rmn);
            }
        }
    }

    // wave sum
    int wLim = 6;
    EVec3 Kwave(0, 0, 0);
    EVec3 rmn = target - source;

    for (int i = -wLim; i <= wLim; i++) {
        for (int j = -wLim; j <= wLim; j++) {
            for (int k = -wLim; k <= wLim; k++) {
                if (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                EVec3 kvec = EVec3(i, j, k);
                double knorm = kvec.norm();
                Kwave += 2 * PI314 * sin(2 * PI314 * kvec.dot(rmn)) *
                         exp(-eta * PI314 * knorm * knorm) /
                         (PI314 * knorm * knorm) * kvec;
            }
        }
    }

    return Kreal + Kwave;
}

inline EVec3 gKernel(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < 1e-14) {
        return EVec3(0, 0, 0);
    } else {
        return rst / pow(rnorm, 3);
    }
}

inline Eigen::Matrix3d gKernelGrad(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < 1e-14) {
        return Eigen::Matrix3d::Zero();
    } else {
        return Eigen::Matrix3d::Identity() / pow(rnorm, 3) -
               3 * rst * rst.transpose() / pow(rnorm, 5);
    }
}

// Out of Direct Sum Layer, far field part
inline EVec3 gKernelFF(const EVec3 &target, const EVec3 &source) {
    EVec3 fEwald = gKernelEwald(target, source);
    const int N = DIRECTLAYER;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                EVec3 gFree = gKernel(target, source - EVec3(i, j, k));
                fEwald -= gFree;
            }
        }
    }

    // {
    //   std::cout << "source:" << source << std::endl
    //             << "target:" << target << std::endl
    //             << "gKernalFF" << fEwald << std::endl;
    // }
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

    // testing Ewald routine
    double zeroTest = gKernelEwald(EVec3(0.5, 0.5, 0.5), EVec3(0.5, 0.5, 0.5))
                          .dot(EVec3(1, 1, 1));
    std::cout << std::setprecision(16) << "zeroTest: " << zeroTest << std::endl;

    double centerTest = gKernelEwald(EVec3(0, 0, 0), EVec3(0.5, 0.5, 0.5))
                            .dot(EVec3(0.5, 0.5, 0.5));
    std::cout << std::setprecision(16) << "centerTest: " << centerTest
              << " error: " << centerTest - 0 << std::endl;

    centerTest = gKernelEwald(EVec3(0.2, 0.3, 0.4), EVec3(0.3, 0.6, 0.5))
                     .dot(EVec3(3, 2, 1));
    std::cout << std::setprecision(16) << "centerTest2: " << centerTest
              << " error: " << centerTest + 23.48660380315382667504
              << std::endl;

    centerTest = gKernelEwald(EVec3(0.7, 0.9, 0.7), EVec3(0.2, 0.3, 0.4))
                     .dot(EVec3(0.1, 2, 0.3));
    std::cout << std::setprecision(16) << "centerTest2: " << centerTest
              << " error: " << centerTest + 0.83918927151112920892 << std::endl;

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

    auto pointMEquiv =
        surface(pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv, 0);
    // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointMCheck = surface(
        pCheck, (double *)&(pCenterCheck[0]), scaleCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth =0

    auto pointLEquiv = surface(
        pEquiv, (double *)&(pCenterLCheck[0]), scaleLCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth =  0
    auto pointLCheck = surface(
        pCheck, (double *)&(pCenterLEquiv[0]), scaleLEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    // calculate the operator M2L with least square
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointLCheck.size() / 3;
    Eigen::MatrixXd M2L(3 * equivN, 3 * equivN); // Laplace, 1->1

    Eigen::MatrixXd A(3 * checkN, 3 * equivN);
    A.setZero();
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                               pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l],
                                         pointLEquiv[3 * l + 1],
                                         pointLEquiv[3 * l + 2]);
            EVec3 temp = gKernel(Cpoint, Lpoint);
            A(3 * k, 3 * l) = temp[0];
            A(3 * k + 1, 3 * l + 1) = temp[1];
            A(3 * k + 2, 3 * l + 2) = temp[2];
            // A.block(k, l, 1, 3) = gKernel(Cpoint, Lpoint).transpose();
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        //		std::cout << "debug:" << Mpoint << std::endl;

        // assemble linear system
        Eigen::VectorXd f0(3 * checkN);
        Eigen::VectorXd f1(3 * checkN);
        Eigen::VectorXd f2(3 * checkN);
        f0.setZero();
        f1.setZero();
        f2.setZero();
        for (int k = 0; k < checkN; k++) {
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            //			std::cout<<"debug:"<<k<<std::endl;
            // sum the images
            EVec3 temp = gKernelFF(Cpoint, Mpoint);
            f0[3 * k] = temp[0];
            f1[3 * k + 1] = temp[1];
            f2[3 * k + 2] = temp[2];
        }
        // std::cout << "debug:" << f0 << std::endl;

        M2L.block(0, 3 * i, 3 * equivN, 1) =
            (ApinvU.transpose() * (ApinvVT.transpose() * f0));
        M2L.block(0, 3 * i + 1, 3 * equivN, 1) =
            (ApinvU.transpose() * (ApinvVT.transpose() * f1));
        M2L.block(0, 3 * i + 2, 3 * equivN, 1) =
            (ApinvU.transpose() * (ApinvVT.transpose() * f2));
    }

    // dump M2L
    for (int i = 0; i < 3 * equivN; i++) {
        for (int j = 0; j < 3 * equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific
                      << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        dipolePoint(1);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        dipoleValue(1);
    dipolePoint[0] = Eigen::Vector3d(0.2, 0.3, 0.4);
    dipoleValue[0] = Eigen::Vector3d(0.1, 0.2, 0.3);

    // solve M
    A.resize(3 * checkN, 3 * equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    Eigen::VectorXd f(3 * checkN);
    for (int k = 0; k < checkN; k++) {
        EVec3 sum(0, 0, 0);
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
                               pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < dipolePoint.size(); p++) {
            EVec3 temp = gKernel(Cpoint, dipolePoint[p]);
            sum[0] += temp[0] * dipoleValue[p][0];
            sum[1] += temp[1] * dipoleValue[p][1];
            sum[2] += temp[2] * dipoleValue[p][2];
        }
        f[3 * k] = sum[0];
        f[3 * k + 1] = sum[1];
        f[3 * k + 2] = sum[2];
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1],
                                   pointMEquiv[3 * l + 2]);
            // A(k, l) = gKernel(Mpoint, Cpoint);
            EVec3 temp = gKernel(Cpoint, Mpoint);
            A(3 * k, 3 * l) = temp[0];
            A(3 * k + 1, 3 * l + 1) = temp[1];
            A(3 * k + 2, 3 * l + 2) = temp[2];
        }
    }
    pinv(A, ApinvU, ApinvVT);
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));

    std::cout << "Msource: " << Msource << std::endl;

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    Eigen::Vector3d samplePoint(0.7, 0.9, 0.7);
    //    Eigen::Vector3d samplePoint = dipolePoint[0];
    double Usample = 0;
    double UsampleSP = 0;

    for (int i = -DIRECTLAYER; i < 1 + DIRECTLAYER; i++) {
        for (int j = -DIRECTLAYER; j < 1 + DIRECTLAYER; j++) {
            for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
                for (size_t p = 0; p < dipolePoint.size(); p++) {
                    Usample +=
                        gKernel(samplePoint, dipolePoint[p] + EVec3(i, j, k))
                            .dot(dipoleValue[p]);
                }
            }
        }
    }

    for (int p = 0; p < equivN; p++) {
        Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1],
                               pointLEquiv[3 * p + 2]);
        EVec3 M2Lsp(M2Lsource[3 * p], M2Lsource[3 * p + 1],
                    M2Lsource[3 * p + 2]);
        UsampleSP += gKernel(samplePoint, Lpoint).dot(M2Lsp);
    }

    std::cout << "samplePoint:" << samplePoint << std::endl;
    std::cout << "Usample NF:" << Usample << std::endl;
    std::cout << "Usample FF:" << UsampleSP << std::endl;
    std::cout << "Usample FF+NF total:" << UsampleSP + Usample << std::endl;

    std::cout
        << "Error : "
        << UsampleSP + Usample -
               gKernelEwald(samplePoint, dipolePoint[0]).dot(dipoleValue[0])
        << std::endl;

    return 0;
}

} // namespace Laplace3D3DDipole

#undef DIRECTLAYER
#undef PI314
#undef E271
