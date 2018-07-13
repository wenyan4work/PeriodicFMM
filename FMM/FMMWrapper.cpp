/*
 * FMMWrapper.cpp
 *
 *  Created on: Oct 20, 2016
 *      Author: wyan
 */

#include "FMMWrapper.h"

#include <cassert>
#include <limits>

#include <mpi.h>
#include <omp.h>

extern pvfmm::PeriodicType pvfmm::periodicType;

// return fraction part between [0,1)
/*
 * This function is only applied in the PERIODIC DIRECTION
 *
 * The user of the library must ensure that all points are located within [0,1)
 *
 * */
inline double fracwrap(double x) { return x - floor(x); }

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

FMM_Wrapper::FMM_Wrapper(int mult_order, int max_pts, int init_depth, PAXIS pbc_)
    : mult_order(mult_order), max_pts(max_pts), init_depth(init_depth), pbc(pbc_), xlow(0), xhigh(1), ylow(0), yhigh(1),
      zlow(0), zhigh(1), scaleFactor(1), xshift(0), yshift(0), zshift(0)
#ifndef FMMTIMING
      ,
      myTimer(false)
#endif
{

    // set periodic boundary condition
    switch (pbc) {
    case PAXIS::NONE:
        pvfmm::periodicType = pvfmm::PeriodicType::NONE;
        break;
    case PAXIS::PXYZ:
        pvfmm::periodicType = pvfmm::PeriodicType::PXYZ;
        break;
    case PAXIS::PX:
        pvfmm::periodicType = pvfmm::PeriodicType::PX;
        break;
    case PAXIS::PY:
        pvfmm::periodicType = pvfmm::PeriodicType::PX; // use axis rotation
        break;
    case PAXIS::PZ:
        pvfmm::periodicType = pvfmm::PeriodicType::PX; // use axis rotation
        break;
    case PAXIS::PXY:
        pvfmm::periodicType = pvfmm::PeriodicType::PXY;
        break;
    case PAXIS::PXZ:
        pvfmm::periodicType = pvfmm::PeriodicType::PXY; // use axis rotation
        break;
    case PAXIS::PYZ:
        pvfmm::periodicType = pvfmm::PeriodicType::PXY; // use axis rotation
        break;
    }
    pm2l = nullptr;
    if (pbc != NONE) {
        if (mult_order != (mult_order / 2) * 2 || mult_order < 6 || mult_order > 16) {
            printf("periodic M2L data available only for p=6,8,10,12,14,16\n");
        } else if (pbc == PAXIS::PXYZ) {
            switch (mult_order) {
            case 6:
                pm2l = readM2LMat("M2LStokes3D3Dp6", 6);
                break;
            case 8:
                pm2l = readM2LMat("M2LStokes3D3Dp8", 8);
                break;
            case 10:
                pm2l = readM2LMat("M2LStokes3D3Dp10", 10);
                break;
            case 12:
                pm2l = readM2LMat("M2LStokes3D3Dp12", 12);
                break;
            case 14:
                pm2l = readM2LMat("M2LStokes3D3Dp14", 14);
                break;
            case 16:
                pm2l = readM2LMat("M2LStokes3D3Dp16", 16);
                break;
            default:
                std::cout << "no m2l data at corresponding p, exit now" << std::endl;
                exit(1);
                break;
            }
        } else if (pbc == PAXIS::PX || pbc == PAXIS::PY || pbc == PAXIS::PZ) {
            switch (mult_order) {
            case 6:
                pm2l = readM2LMat("M2LStokes1D3Dp6", 6);
                break;
            case 8:
                pm2l = readM2LMat("M2LStokes1D3Dp8", 8);
                break;
            case 10:
                pm2l = readM2LMat("M2LStokes1D3Dp10", 10);
                break;
            case 12:
                pm2l = readM2LMat("M2LStokes1D3Dp12", 12);
                break;
            case 14:
                pm2l = readM2LMat("M2LStokes1D3Dp14", 14);
                break;
            case 16:
                pm2l = readM2LMat("M2LStokes1D3Dp16", 16);
                break;
            default:
                std::cout << "no m2l data at corresponding p, exit now" << std::endl;
                exit(1);
                break;
            }
        } else if (pbc == PAXIS::PXY || pbc == PAXIS::PXZ || pbc == PAXIS::PYZ) {
            switch (mult_order) {
            case 6:
                pm2l = readM2LMat("M2LStokes2D3Dp6", 6);
                break;
            case 8:
                pm2l = readM2LMat("M2LStokes2D3Dp8", 8);
                break;
            case 10:
                pm2l = readM2LMat("M2LStokes2D3Dp10", 10);
                break;
            case 12:
                pm2l = readM2LMat("M2LStokes2D3Dp12", 12);
                break;
            case 14:
                pm2l = readM2LMat("M2LStokes2D3Dp14", 14);
                break;
            case 16:
                pm2l = readM2LMat("M2LStokes2D3Dp16", 16);
                break;
            default:
                std::cout << "no m2l data at corresponding p, exit now" << std::endl;
                exit(1);
                break;
            }
        }

        this->pEquiv = mult_order; // (8-1)^2*6 + 2 points

        this->scaleLEquiv = RAD1; // RAD1 = 2.95 defined in pvfmm_common.h
        this->pCenterLEquiv[0] = -(scaleLEquiv - 1) / 2;
        this->pCenterLEquiv[1] = -(scaleLEquiv - 1) / 2;
        this->pCenterLEquiv[2] = -(scaleLEquiv - 1) / 2;

        pointLEquiv = surface(pEquiv, (double *)&(pCenterLEquiv[0]), scaleLEquiv,
                              0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

        equivN = 6 * (pEquiv - 1) * (pEquiv - 1) + 2;
    }
    MPI_Comm comm = MPI_COMM_WORLD;

    const pvfmm::Kernel<double> &kernel_fn = pvfmm::StokesKernel<double>::velocity();

    matrix.Initialize(mult_order, comm, &kernel_fn);
    treePtr = nullptr;

#ifdef FMMDEBUG
    pvfmm::Profile::Enable(true);
#endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        printf("Stokes3D FMM Initialized\n");
}

void FMM_Wrapper::FMM_SetBox(double xlow_, double xhigh_, double ylow_, double yhigh_, double zlow_, double zhigh_) {
    xlow = xlow_;
    xhigh = xhigh_;
    ylow = ylow_;
    yhigh = yhigh_;
    zlow = zlow_;
    zhigh = zhigh_;
    // find and calculate scale & shift factor to map the box to [0,1)
    xshift = -xlow;
    yshift = -ylow;
    zshift = -zlow;
    double xlen = xhigh - xlow;
    double ylen = yhigh - ylow;
    double zlen = zhigh - zlow;
    scaleFactor = 1 / std::max(zlen, std::max(xlen, ylen));
    // new coordinate = (x+xshift)*scaleFactor, in (0,1)
    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        std::cout << "box x" << xlen << "box y" << ylen << "box z" << zlen << std::endl;
        std::cout << "scale factor" << scaleFactor << std::endl;
    }

    // validate box setting, ensure fitting in a cubic box [0,1)^3
    const double eps = pow(10, -10) / scaleFactor;
    switch (pbc) {
    case PAXIS::NONE:
        // for PNONE, scale max length to (0,1), all choices are valid
        break;
    case PAXIS::PX:
        // for PX,PY,PZ, max must be the periodic direction
        if (xlen < ylen || xlen < zlen) {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    case PAXIS::PY:
        if (ylen < xlen || ylen < zlen) {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    case PAXIS::PZ:
        if (zlen < xlen || zlen < ylen) {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    case PAXIS::PXY:
        // for PXY,PXZ,PYZ, periodic direcitons must have equal size, and larger than the third direction
        if (fabs(xlen - ylen) < eps && xlen >= zlen) {
        } else {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    case PAXIS::PXZ:
        if (fabs(xlen - zlen) < eps && xlen >= ylen) {
        } else {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    case PAXIS::PYZ:
        if (fabs(zlen - ylen) < eps && zlen >= xlen) {
        } else {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    case PAXIS::PXYZ:
        // for PXYZ, must be cubic
        if (fabs(xlen - ylen) < eps && fabs(xlen - ylen) < eps && fabs(xlen - zlen) < eps) {
        } else {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("box x: %f, y: %f, z: %f\n", xlen, ylen, zlen);
        printf("shift x: %f, y: %f, z: %f\n", xshift, yshift, zshift);
        printf("scale factor: %f\n", scaleFactor);
    }
}

void FMM_Wrapper::FMM_UpdateTree(const std::vector<double> &src_coord, const std::vector<double> &trg_coord) {

    myTimer.start();

    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0)
        printf("FMM_UpdateTree\n");

    FMM_TreeClear();

    if (myRank == 0)
        printf("tree deleted\n");

    treePtr = new pvfmm::PtFMM_Tree(MPI_COMM_WORLD);

    treeData.dim = 3;
    treeData.max_depth = 15;
    treeData.max_pts = max_pts;

    // Set source points, with scale
    //	treeData.src_coord = src_coord;
    const int nsrc = src_coord.size() / 3;
    treeData.src_coord.Resize(nsrc * 3);
    if (pbc == PAXIS::PY) {
// rotate y axis to x axis to use the x 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i] = fracwrap((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i + 1] = ((src_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.src_coord[3 * i] << treeData.src_coord[3 * i + 1] << treeData.src_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PXZ) {
// rotate y axis to z axis to use the xy 2d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i + 1] = fracwrap((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i] = fracwrap((src_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.src_coord[3 * i] << treeData.src_coord[3 * i + 1] << treeData.src_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PX) {
// no rotation
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i] = fracwrap((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i + 1] = ((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.src_coord[3 * i] << treeData.src_coord[3 * i + 1] << treeData.src_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PYZ) {
// rotate x axis to z axis
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i] = fracwrap((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i + 1] = fracwrap((src_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.src_coord[3 * i] << treeData.src_coord[3 * i + 1] << treeData.src_coord[3 * i + 2]
                      << std::endl;
#endif
        }

    } else if (pbc == PAXIS::PXYZ) {
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i] = fracwrap((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i + 1] = fracwrap((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i + 2] = fracwrap((src_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.src_coord[3 * i] << treeData.src_coord[3 * i + 1] << treeData.src_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PZ) {
// rotate z to x
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i + 1] = ((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i] = fracwrap((src_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.src_coord[3 * i] << treeData.src_coord[3 * i + 1] << treeData.src_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PXY) {
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i] = fracwrap((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i + 1] = fracwrap((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.src_coord[3 * i] << treeData.src_coord[3 * i + 1] << treeData.src_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else {
        assert(pbc == PAXIS::NONE);
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i] = ((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i + 1] = ((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.src_coord[3 * i] << treeData.src_coord[3 * i + 1] << treeData.src_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    }
    {
        // set to NULL. currently no support for surf source
        const int nsurf = 0;
        treeData.surf_coord.Resize(nsurf * 3);
    }

    // Set target points.
    // use the same rotation and periodic wrap as source

    const int ntrg = trg_coord.size() / 3;
    treeData.trg_coord.Resize(ntrg * 3);
    if (pbc == PAXIS::PY) {
// rotate y axis to x axis to use the x 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i] = fracwrap((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i + 1] = ((trg_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.trg_coord[3 * i] << treeData.trg_coord[3 * i + 1] << treeData.trg_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PXZ) {
// rotate y axis to z axis to use the xy 2d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i + 1] = fracwrap((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i] = fracwrap((trg_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.trg_coord[3 * i] << treeData.trg_coord[3 * i + 1] << treeData.trg_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PX) {
// no rotation
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i] = fracwrap((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i + 1] = ((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.trg_coord[3 * i] << treeData.trg_coord[3 * i + 1] << treeData.trg_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PYZ) {
// rotate x axis to z axis
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i] = fracwrap((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i + 1] = fracwrap((trg_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.trg_coord[3 * i] << treeData.trg_coord[3 * i + 1] << treeData.trg_coord[3 * i + 2]
                      << std::endl;
#endif
        }

    } else if (pbc == PAXIS::PXYZ) {
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i] = fracwrap((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i + 1] = fracwrap((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i + 2] = fracwrap((trg_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.trg_coord[3 * i] << treeData.trg_coord[3 * i + 1] << treeData.trg_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PZ) {
// rotate z to x
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i + 1] = ((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i] = fracwrap((trg_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.trg_coord[3 * i] << treeData.trg_coord[3 * i + 1] << treeData.trg_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else if (pbc == PAXIS::PXY) {
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i] = fracwrap((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i + 1] = fracwrap((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.trg_coord[3 * i] << treeData.trg_coord[3 * i + 1] << treeData.trg_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    } else {
        assert(pbc == PAXIS::NONE);
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i] = ((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i + 1] = ((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i + 2] + zshift) * scaleFactor);
#ifdef FMMDEBUG
            std::cout << treeData.trg_coord[3 * i] << treeData.trg_coord[3 * i + 1] << treeData.trg_coord[3 * i + 2]
                      << std::endl;
#endif
        }
    }

    // prevent PVFMM from breaking down with coord=1
    const int NS = treeData.src_coord.Dim();
    const int NT = treeData.trg_coord.Dim();
    const double eps = std::numeric_limits<double>::epsilon() * 100;
#pragma omp parallel for
    for (int i = 0; i < NS; i++) {
        if (treeData.src_coord[i] > 1 - eps)
            treeData.src_coord[i] = 1 - eps;
    }
#pragma omp parallel for
    for (int i = 0; i < NT; i++) {
        if (treeData.trg_coord[i] > 1 - eps)
            treeData.trg_coord[i] = 1 - eps;
    }

    treeData.pt_coord = treeData.trg_coord;

    treePtr->Initialize(&treeData);
    bool adap = true;

    treePtr->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtr->SetupFMM(&matrix);
    myTimer.stop("Stokes3D FMM tree setup");

#ifdef FMMDEBUG
    std::cout << "SetupFMM Complete" << std::endl;
#endif
}

void FMM_Wrapper::FMM_Evaluate(std::vector<double> &trg_val, const int n_trg, std::vector<double> *src_val) {
    FMM_DataClear();
    if (src_val == nullptr) {
        printf("Error, no source value\n");
        return;
    }

    // in place rotate of src_val;
    const int nsrc = src_val->size() / 3;
    if (pbc == PAXIS::PXZ) {
// rotate y axis to z axis to use the z 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            double x = (*src_val)[3 * i];
            double y = (*src_val)[3 * i + 1];
            double z = (*src_val)[3 * i + 2];
            (*src_val)[3 * i] = z;
            (*src_val)[3 * i + 1] = x;
            (*src_val)[3 * i + 2] = y;
        }
    } else if (pbc == PAXIS::PYZ) {
// rotate x axis to z axis
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            double x = (*src_val)[3 * i];
            double y = (*src_val)[3 * i + 1];
            double z = (*src_val)[3 * i + 2];
            (*src_val)[3 * i] = y;
            (*src_val)[3 * i + 1] = z;
            (*src_val)[3 * i + 2] = x;
        }
    } else if (pbc == PAXIS::PY) {
        // rotate y axis to x axis to use the x 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            double x = (*src_val)[3 * i];
            double y = (*src_val)[3 * i + 1];
            double z = (*src_val)[3 * i + 2];
            (*src_val)[3 * i] = y;
            (*src_val)[3 * i + 1] = z;
            (*src_val)[3 * i + 2] = x;
        }
    } else if (pbc == PAXIS::PZ) {
        // rotate z axis to x axis to use the x 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            double x = (*src_val)[3 * i];
            double y = (*src_val)[3 * i + 1];
            double z = (*src_val)[3 * i + 2];
            (*src_val)[3 * i] = z;
            (*src_val)[3 * i + 1] = x;
            (*src_val)[3 * i + 2] = y;
        }

    } else {
        // no rotate
    }

    myTimer.start();
    PtFMM_Evaluate(treePtr, trg_val, n_trg, src_val, nullptr);
    myTimer.stop("Stokes Near Field");

#ifdef FMMDEBUG
    std::cout << "before pxyz" << trg_val[0] << std::endl;
    std::cout << trg_val[1] << std::endl;
    std::cout << trg_val[2] << std::endl;
#endif
    if (pbc != NONE) {
        myTimer.start();
        // calcM(treeData.trg_coord, trg_val, *src_val);
        calcMStokes(trg_val);
        myTimer.stop("Stokes Far Field");
    }

    // scale and rotate back
    if (pbc == PAXIS::PXZ) {
// rotate y axis to z axis to use the z 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            double x = (*src_val)[3 * i];
            double y = (*src_val)[3 * i + 1];
            double z = (*src_val)[3 * i + 2];
            (*src_val)[3 * i] = y;
            (*src_val)[3 * i + 1] = z;
            (*src_val)[3 * i + 2] = x;
        }
#pragma omp parallel for
        for (size_t i = 0; i < n_trg; i++) {
            double x = trg_val[3 * i];
            double y = trg_val[3 * i + 1];
            double z = trg_val[3 * i + 2];
            trg_val[3 * i] = y * scaleFactor;
            trg_val[3 * i + 1] = z * scaleFactor;
            trg_val[3 * i + 2] = x * scaleFactor;
        }
    } else if (pbc == PAXIS::PYZ) {
// rotate x axis to z axis
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            double x = (*src_val)[3 * i];
            double y = (*src_val)[3 * i + 1];
            double z = (*src_val)[3 * i + 2];
            (*src_val)[3 * i] = z;
            (*src_val)[3 * i + 1] = x;
            (*src_val)[3 * i + 2] = y;
        }
#pragma omp parallel for
        for (size_t i = 0; i < n_trg; i++) {
            double x = trg_val[3 * i];
            double y = trg_val[3 * i + 1];
            double z = trg_val[3 * i + 2];
            trg_val[3 * i] = z * scaleFactor;
            trg_val[3 * i + 1] = x * scaleFactor;
            trg_val[3 * i + 2] = y * scaleFactor;
        }
    } else if (pbc == PAXIS::PY) {
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            double x = (*src_val)[3 * i];
            double y = (*src_val)[3 * i + 1];
            double z = (*src_val)[3 * i + 2];
            (*src_val)[3 * i] = z;
            (*src_val)[3 * i + 1] = x;
            (*src_val)[3 * i + 2] = y;
        }
#pragma omp parallel for
        for (size_t i = 0; i < n_trg; i++) {
            double x = trg_val[3 * i];
            double y = trg_val[3 * i + 1];
            double z = trg_val[3 * i + 2];
            trg_val[3 * i] = z * scaleFactor;
            trg_val[3 * i + 1] = x * scaleFactor;
            trg_val[3 * i + 2] = y * scaleFactor;
        }
    } else if (pbc == PAXIS::PZ) {
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            double x = (*src_val)[3 * i];
            double y = (*src_val)[3 * i + 1];
            double z = (*src_val)[3 * i + 2];
            (*src_val)[3 * i] = y;
            (*src_val)[3 * i + 1] = z;
            (*src_val)[3 * i + 2] = x;
        }
#pragma omp parallel for
        for (size_t i = 0; i < n_trg; i++) {
            double x = trg_val[3 * i];
            double y = trg_val[3 * i + 1];
            double z = trg_val[3 * i + 2];
            trg_val[3 * i] = y * scaleFactor;
            trg_val[3 * i + 1] = z * scaleFactor;
            trg_val[3 * i + 2] = x * scaleFactor;
        }
    }

    else {
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < n_trg; i++) {
            trg_val[3 * i] *= scaleFactor;
            trg_val[3 * i + 1] *= scaleFactor;
            trg_val[3 * i + 2] *= scaleFactor;
        }
    }

#ifdef FMMTIMING
    myTimer.dump();
#endif
}

double *FMM_Wrapper::readM2LMat(const char *fname, const int p) {
    const int size = 3 * (6 * (p - 1) * (p - 1) + 2);
    double *fdata = new double[size * size];

    char *pvfmm_dir = getenv("PVFMM_DIR");
    std::stringstream st;
    st << pvfmm_dir;
    st << "/pdata/";
    st << fname;
    FILE *fin = fopen(st.str().c_str(), "r");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int iread, jread;
            double fread;
            fscanf(fin, "%d %d %lf\n", &iread, &jread, &fread);
            if (i != iread || j != jread) {
                printf("read ij error \n");
            }
#ifdef FMMDEBUG
//			printf("%d %d %g\n", iread, jread, fread);
#endif
            fdata[i * size + j] = fread;
        }
    }

    fclose(fin);
    return fdata;
}

void FMM_Wrapper::calcMStokes(std::vector<double> &trgValue) {
    // add periodization for stokes G

    // make a copy to do correction
    pvfmm::Vector<double> v = treePtr->RootNode()->FMMData()->upward_equiv; // the value calculated by pvfmm
    auto &trgCoord = treeData.trg_coord;

    assert(v.Dim() == 3 * this->equivN);

    // add to trg_value
    const int n_trg = trgCoord.Dim() / 3;
    assert(trgCoord.Dim() == trg_value.size());
    const double pi = 3.1415926535897932384626433;

    int M = 3 * equivN;
    int N = 3 * equivN; // checkN = equivN in this code.
    M2Lsource.resize(M);
    assert(M2Lsource.size() == v.Dim());

#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double temp = 0;
        //#pragma unroll 4
        for (int j = 0; j < N; j++) {
            temp += pm2l[i * N + j] * v[j];
        }
        M2Lsource[i] = temp;
    }

    const pvfmm::Kernel<double> &kernelG = pvfmm::StokesKernel<double>::velocity();
    const size_t chunkSize = 2000; // each chunk has 2000 target points.
    const size_t chunkNumber = floor(n_trg / chunkSize) + 1;
    // printf("chunkSize, chunkNumber: %zu, %zu\n", chunkSize, chunkNumber);
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < chunkNumber; i++) {
        const size_t idTrgLow = i * chunkSize;
        const size_t idTrgHigh = (i + 1 < chunkNumber) ? idTrgLow + chunkSize : n_trg; // not inclusive
        //        printf("i, idTrgLow, idTrgHigh: %d, %d, %d\n", i, idTrgLow, idTrgHigh);
        kernelG.k_l2t->ker_poten(pointLEquiv.data(), equivN, M2Lsource.data(), 1, &(trgCoord[3 * idTrgLow]),
                                 idTrgHigh - idTrgLow, &(trgValue[3 * idTrgLow]), NULL);
    }
}

void FMM_Wrapper::FMM_DataClear() {
    // clear data, keep tree
    if (treePtr != nullptr)
        treePtr->ClearFMMData();
}

void FMM_Wrapper::FMM_TreeClear() {
    // clear Tree, delete tree
    FMM_DataClear();

    if (treePtr != nullptr) {
        delete treePtr;
        treePtr = nullptr;
    }
}

FMM_Wrapper::~FMM_Wrapper() {
    FMM_TreeClear();
    if (pm2l != nullptr) {
        delete[] pm2l;
    }
}
