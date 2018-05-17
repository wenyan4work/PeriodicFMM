/*
 * FMMWrapper.cpp
 *
 *  Created on: Oct 20, 2016
 *      Author: wyan
 */

#include "FMMWrapperWall2D.h"
#include "LaplaceCustomKernel.hpp"

#include <cassert>
#include <limits>

#include <mpi.h>
#include <omp.h>

extern pvfmm::PeriodicType pvfmm::periodicType;

/*
 * return fraction part between [0,1)
 * This function is only applied in the PERIODIC DIRECTION
 * The user of the library must ensure that all points are located within [0,1)
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

FMM_WrapperWall2D::FMM_WrapperWall2D(int mult_order, int max_pts, int init_depth, PAXIS pbc_)
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
    case PAXIS::PXY:
        pvfmm::periodicType = pvfmm::PeriodicType::PXY;
        break;
    }

    pm2lStokes = nullptr;
    pm2lLapCharge = nullptr;
    pm2lLapDipole = nullptr;

    if (pbc != NONE) {
        if (mult_order != (mult_order / 2) * 2 || mult_order < 6 || mult_order > 16) {
            printf("periodic M2L data available only for p=6,8,10,12,14,16\n");
            exit(1);
        } else if (pbc == PAXIS::PXY) {
            switch (mult_order) {
            case 6:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp6", 6, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp6", 6, 1);
                pm2lLapDipole = readM2LMat("M2LLapDipole2D3Dp6", 6, 3);
                break;
            case 8:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp8", 8, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp8", 8, 1);
                pm2lLapDipole = readM2LMat("M2LLapDipole2D3Dp8", 8, 3);
                break;
            case 10:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp10", 10, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp10", 10, 1);
                pm2lLapDipole = readM2LMat("M2LLapDipole2D3Dp10", 10, 3);
                break;
            case 12:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp12", 12, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp12", 12, 1);
                pm2lLapDipole = readM2LMat("M2LLapDipole2D3Dp12", 12, 3);
                break;
            case 14:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp14", 14, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp14", 14, 1);
                pm2lLapDipole = readM2LMat("M2LLapDipole2D3Dp14", 14, 3);
                break;
            case 16:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp16", 16, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp16", 16, 1);
                pm2lLapDipole = readM2LMat("M2LLapDipole2D3Dp16", 16, 3);
                break;
            default:
                printf("no m2l data at corresponding p, exit now\n");
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

    const pvfmm::Kernel<double> &kernelStokes = pvfmm::StokesKernel<double>::velocity();
    matrixStokes.Initialize(mult_order, comm, &kernelStokes);

    const pvfmm::Kernel<double> &kernelLapCharge = pvfmm::LaplaceCustomKernel<double>::potentialGradient();
    matrixLapChargeF.Initialize(mult_order, comm, &kernelLapCharge);
    matrixLapChargeFZ.Initialize(mult_order, comm, &kernelLapCharge);

    const pvfmm::Kernel<double> &kernelLapDipole = pvfmm::LaplaceCustomKernel<double>::dipoleGradient();
    matrixLapDipole.Initialize(mult_order, comm, &kernelLapDipole);

    treePtrStokes = nullptr;
    treePtrLapDipole = nullptr;
    treePtrLapChargeF = nullptr;
    treePtrLapChargeFZ = nullptr;

#ifdef FMMDEBUG
    pvfmm::Profile::Enable(true);
#endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        printf("StokesWall2D FMM Initialized\n");
}

void FMM_WrapperWall2D::FMM_SetBox(double xlow_, double xhigh_, double ylow_, double yhigh_, double zlow_,
                                   double zhigh_) {
    xlow = xlow_;
    xhigh = xhigh_;
    ylow = ylow_;
    yhigh = yhigh_;
    zlow = zlow_;
    zhigh = zhigh_;

    const double xlen = xhigh - xlow;
    const double ylen = yhigh - ylow;
    const double zlen = zhigh - zlow;

    // Wall plane at z=0.5, fit everything into the box x=[0,1), y=[0,1), z=[0.5,1)

    // step 1, validity check
    if (this->pbc == PAXIS::NONE) {
        if (zlen >= 0.5 * std::max(xlen, ylen)) {
            printf("z axis height too large\n");
            exit(1);
        }
    } else if (this->pbc == PAXIS::PXY) {
        if (xhigh - xlow != yhigh - ylow) {
            printf("x and y length must be identical for doubly periodic systems\n");
            exit(1);
        }
        if (zlen >= 0.5 * xlen || zlen >= 0.5 * ylen) {
            printf("z axis height too large\n");
            exit(1);
        }
    }

    // step 2, setting xyz shift and scale factor
    // find and calculate scale & shift factor to map the box to [0,1)
    scaleFactor = 1 / std::max(xlen, ylen); // new coordinate = (x+xshift)*scaleFactor, in (0,1)

    xshift = -xlow;
    yshift = -ylow;
    zshift = 0.5 / scaleFactor - zlow;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("box x: %f, y: %f, z: %f\n", xlen, ylen, zlen);
        printf("shift x: %f, y: %f, z: %f\n", xshift, yshift, zshift);
        printf("scale factor: %f\n", scaleFactor);
    }
}

void FMM_WrapperWall2D::treeSetup(pvfmm::PtFMM_Data &treeData, const std::vector<double> &src_coord,
                                  const std::vector<double> &trg_coord, bool withOriginal) {
    // no rotate, image wall at z = 0.5
    //    printf("start setup for treeData\n");

    const int nsrc = src_coord.size() / 3;

    if (withOriginal) {
        treeData.src_coord.Resize(nsrc * 3 * 2); // including the original
    } else {
        treeData.src_coord.Resize(nsrc * 3); // image only
    }

    if (pbc == PAXIS::PXY) {
// original
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i] = fracwrap((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i + 1] = fracwrap((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i + 2] + zshift) * scaleFactor);
        }
        if (withOriginal) {
// image
#pragma omp parallel for
            for (size_t i = nsrc; i < nsrc * 2; i++) {
                treeData.src_coord[3 * i] = treeData.src_coord[3 * (i - nsrc)];
                treeData.src_coord[3 * i + 1] = treeData.src_coord[3 * (i - nsrc) + 1];
                treeData.src_coord[3 * i + 2] = 1.0 - treeData.src_coord[3 * (i - nsrc) + 2];
            }
        }
    } else {
// original
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i] = ((src_coord[3 * i] + xshift) * scaleFactor);
            treeData.src_coord[3 * i + 1] = ((src_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.src_coord[3 * i + 2] = ((src_coord[3 * i + 2] + zshift) * scaleFactor);
        }
        if (withOriginal) {
// image
#pragma omp parallel for
            for (size_t i = nsrc; i < nsrc * 2; i++) {
                treeData.src_coord[3 * i] = treeData.src_coord[3 * (i - nsrc)];
                treeData.src_coord[3 * i + 1] = treeData.src_coord[3 * (i - nsrc) + 1];
                treeData.src_coord[3 * i + 2] = 1.0 - treeData.src_coord[3 * (i - nsrc) + 2];
            }
        }
    }

    if (!withOriginal) {
#pragma omp parallel for
        for (size_t i = 0; i < nsrc; i++) {
            treeData.src_coord[3 * i + 2] = 1.0 - treeData.src_coord[3 * i + 2];
        }
    }

    treeData.surf_coord.Resize(0);
#ifdef FMMDEBUG
    printf("src_coord setup for treeData\n");
#endif

    // Set target points.
    // use the same rotation and periodic wrap as source

    const int ntrg = trg_coord.size() / 3;
    treeData.trg_coord.Resize(ntrg * 3);
    // image target values not needed

    if (pbc == PAXIS::PXY) {
// original
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i] = fracwrap((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i + 1] = fracwrap((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i + 2] + zshift) * scaleFactor);
        }

    } else {
        assert(pbc == PAXIS::NONE);
#pragma omp parallel for
        for (size_t i = 0; i < ntrg; i++) {
            treeData.trg_coord[3 * i] = ((trg_coord[3 * i] + xshift) * scaleFactor);
            treeData.trg_coord[3 * i + 1] = ((trg_coord[3 * i + 1] + yshift) * scaleFactor);
            treeData.trg_coord[3 * i + 2] = ((trg_coord[3 * i + 2] + zshift) * scaleFactor);
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

#ifdef FMMDEBUG
    printf("trg_coord setup for treeData\n");
    treePointDump(treeData);
#endif
}

void FMM_WrapperWall2D::treePointDump(const pvfmm::PtFMM_Data &treeData) {
    const int nsrc = treeData.src_coord.Dim() / 3;
    const int ntrg = treeData.trg_coord.Dim() / 3;

    printf("tree src\n");
    for (int i = 0; i < nsrc; i++) {
        printf("%f,%f,%f\n", treeData.src_coord[3 * i], treeData.src_coord[3 * i + 1], treeData.src_coord[3 * i + 2]);
    }

    printf("tree trg\n");
    for (int i = 0; i < ntrg; i++) {
        printf("%f,%f,%f\n", treeData.trg_coord[3 * i], treeData.trg_coord[3 * i + 1], treeData.trg_coord[3 * i + 2]);
    }
}

void FMM_WrapperWall2D::FMM_UpdateTree(const std::vector<double> &src_coord, const std::vector<double> &trg_coord,
                                       const std::vector<double> *surf_coordPtr) {
    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0)
        printf("FMM_UpdateTree\n");

    FMM_TreeClear();

    if (myRank == 0)
        printf("tree deleted\n");

    myTimer.start();
    treeStokes(src_coord, trg_coord);
    myTimer.stop("Stokes tree construction");

    myTimer.start();
    treeLapDipole(src_coord, trg_coord);
    myTimer.stop("Lap Dipole tree construction");

    myTimer.start();
    treeLapChargeF(src_coord, trg_coord);
    myTimer.stop("Lap Charge F tree construction");

    myTimer.start();
    treeLapChargeFZ(src_coord, trg_coord);
    myTimer.stop("Lap Charge FZ tree construction");

    // printf("SetupFMM Complete\n");
}

void FMM_WrapperWall2D::FMM_Evaluate(std::vector<double> &trg_val, const int n_trg, std::vector<double> *src_val,
                                     std::vector<double> *surf_valPtr) {
    FMM_DataClear();
    if (n_trg != trg_val.size() / 3) {
        printf("n_trg error \n");
        exit(1);
    }
    // in place rotate of src_val;
    if (src_val == nullptr) {
        printf("Error, no source value\n");
        return;
    }
    if (surf_valPtr != nullptr) {
        printf("Error, surfval not implemented\n");
        return;
    }
    const int nsrc = src_val->size() / 3;
    const int ntrg = trg_val.size() / 3;

    // evaluate 1, Stokes FMM
    evalStokes(trg_val, n_trg, src_val);

    // evaluate 2, LaplaceDipole FMM
    evalLapDipole(trg_val, n_trg, src_val);

    // evaluate 3, LaplaceCharge FMM
    evalLapChargeF(trg_val, n_trg, src_val);

    // evaluate 4, LaplaceCharge FMM
    evalLapChargeFZ(trg_val, n_trg, src_val);

#ifdef FMMDEBUG
    printf("before pxyz: %f,%f,%f", trg_val[0], trg_val[1], trg_val[2]);
#endif

    myTimer.start();
    sumImageSystem(trg_val);
    myTimer.stop("FMM system summation");

#ifdef FMMTIMING
    myTimer.dump();
#endif
}

double *FMM_WrapperWall2D::readM2LMat(const char *fname, const int p, int kDim) {
    const int size = kDim * (6 * (p - 1) * (p - 1) + 2);
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
                exit(1);
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

void FMM_WrapperWall2D::calcMStokes() {
    // add periodization for stokes G

    // make a copy to do correction
    pvfmm::Vector<double> v = treePtrStokes->RootNode()->FMMData()->upward_equiv; // the value calculated by pvfmm
    auto &trgCoord = treeDataStokes.trg_coord;

    assert(v.Dim() == 3 * this->equivN);

    // add to trg_value
    const int n_trg = trgCoord.Dim() / 3;
    assert(trg_coord.Dim() == trg_value.size());
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
            temp += pm2lStokes[i * N + j] * v[j];
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
                                 idTrgHigh - idTrgLow, &(trgValueStokes[3 * idTrgLow]), NULL);
    }
}

void FMM_WrapperWall2D::calcMLapCharge(int treeSelect) {

    pvfmm::Vector<double> *vPtr = nullptr;
    pvfmm::Vector<double> *trgCoordPtr = nullptr;
    std::vector<double> *trgValuePtr = nullptr;

    // no copy
    if (treeSelect == 0) {
        vPtr = &(treePtrLapChargeF->RootNode()->FMMData()->upward_equiv); // the value calculated by pvfmm
        trgCoordPtr = &(treeDataLapChargeF.trg_coord);
        trgValuePtr = &trgValueLapChargeF;
    } else if (treeSelect == 1) {
        vPtr = &(treePtrLapChargeFZ->RootNode()->FMMData()->upward_equiv); // the value calculated by pvfmm
        trgCoordPtr = &(treeDataLapChargeFZ.trg_coord);
        trgValuePtr = &trgValueLapChargeFZ;
    }

    // calculate M2L, Laplace kernel 1x1
    int M = equivN;
    int N = equivN; // checkN = equivN in this code.
    M2Lsource.resize(M);

#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double temp = 0;
#pragma omp simd
        for (int j = 0; j < N; j++) {
            temp += pm2lLapCharge[i * N + j] * (*vPtr)[j];
        }
        M2Lsource[i] = temp;
    }

    // printf("pm2l finished for lap tree %d", treeSelect);
    // calculate M2L, Laplace kernel 1x1, with gradient 3*1
    auto &trgCoord = *trgCoordPtr;
    auto &trgValue = *trgValuePtr;
    const int nTrg = trgCoord.Dim() / 3;
    if (nTrg * 4 != trgValue.size()) {
        printf("trg coord and value size error for lap %d", treeSelect);
        exit(1);
    }

    const pvfmm::Kernel<double> &kernelLap = pvfmm::LaplaceCustomKernel<double>::potentialGradient();
    const size_t chunkSize = 2000; // each chunk has 2000 target points.
    const size_t chunkNumber = floor(nTrg / chunkSize) + 1;
    // printf("chunkSize, chunkNumber: %zu, %zu\n", chunkSize, chunkNumber);
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < chunkNumber; i++) {
        const size_t idTrgLow = i * chunkSize;
        const size_t idTrgHigh = (i + 1 < chunkNumber) ? idTrgLow + chunkSize : nTrg; // not inclusive
        //        printf("i, idTrgLow, idTrgHigh: %d, %d, %d\n", i, idTrgLow, idTrgHigh);
        kernelLap.k_l2t->ker_poten(pointLEquiv.data(), equivN, M2Lsource.data(), 1, &(trgCoord[3 * idTrgLow]),
                                   idTrgHigh - idTrgLow, &(trgValue[4 * idTrgLow]), NULL);
    }

    return;
}

void FMM_WrapperWall2D::calcMLapDipole() {

    const auto &v = treePtrLapDipole->RootNode()->FMMData()->upward_equiv;

    // calculate M2L, Laplace kernel 1x1
    int M = 3 * equivN;
    int N = 3 * equivN; // checkN = equivN in this code.
    M2Lsource.resize(M);

#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double temp = 0;
#pragma omp simd
        for (int j = 0; j < N; j++) {
            temp += pm2lLapDipole[i * N + j] * v[j];
        }
        M2Lsource[i] = temp;
    }

    // calculate M2L, Laplace kernel 1x1, with gradient 3*1
    const double factor4pi = 1 / (4 * (double)3.1415926535897932384626433);
    auto &trgCoord = treeDataLapDipole.trg_coord;
    auto &trgValue = trgValueLapDipole;

    const int nTrg = trgCoord.Dim() / 3;
    if (nTrg * 4 != trgValue.size()) {
        printf("trg coord and value size error for lap dipole ");
        exit(1);
    }

    const pvfmm::Kernel<double> &kernelD = pvfmm::LaplaceCustomKernel<double>::dipoleGradient();
    const size_t chunkSize = 2000; // each chunk has 2000 target points.
    const size_t chunkNumber = floor(nTrg / chunkSize) + 1;
    // printf("chunkSize, chunkNumber: %zu, %zu\n", chunkSize, chunkNumber);
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < chunkNumber; i++) {
        const size_t idTrgLow = i * chunkSize;
        const size_t idTrgHigh = (i + 1 < chunkNumber) ? idTrgLow + chunkSize : nTrg; // not inclusive
        //        printf("i, idTrgLow, idTrgHigh: %d, %d, %d\n", i, idTrgLow, idTrgHigh);
        kernelD.k_l2t->ker_poten(pointLEquiv.data(), equivN, M2Lsource.data(), 1, &(trgCoord[3 * idTrgLow]),
                                 idTrgHigh - idTrgLow, &(trgValueLapDipole[4 * idTrgLow]), NULL);
    }
    // printf("finish dipole m\n");

    return;
}

void FMM_WrapperWall2D::sumImageSystem(std::vector<double> &trgValue) {
    // sum the four FMM results into the final result
    const int ntrg = trgValue.size() / 3;

// part 1, the Stokes FMM
#pragma omp parallel for
    for (int i = 0; i < ntrg * 3; i++) {
        trgValue[i] = trgValueStokes[i];
    }

// part 2, Laplace Dipole
#pragma omp parallel for
    for (int i = 0; i < ntrg; i++) {
        const double x3 = treeDataLapDipole.trg_coord[3 * i + 2] - 0.5;
        double ud[3];
        ud[0] = x3 * trgValueLapDipole[4 * i + 1];
        ud[1] = x3 * trgValueLapDipole[4 * i + 2];
        ud[2] = x3 * trgValueLapDipole[4 * i + 3] - trgValueLapDipole[4 * i];
        trgValue[3 * i] += ud[0];
        trgValue[3 * i + 1] += ud[1];
        trgValue[3 * i + 2] += ud[2];
    }

// part 3, Laplace charge F
#pragma omp parallel for
    for (int i = 0; i < ntrg; i++) {
        const double x3 = treeDataLapChargeF.trg_coord[3 * i + 2] - 0.5;
        double uL1[3];
        uL1[0] = -0.5 * x3 * trgValueLapChargeF[4 * i + 1];
        uL1[1] = -0.5 * x3 * trgValueLapChargeF[4 * i + 2];
        uL1[2] = -0.5 * x3 * trgValueLapChargeF[4 * i + 3] + 0.5 * trgValueLapChargeF[4 * i];
        trgValue[3 * i] += uL1[0];
        trgValue[3 * i + 1] += uL1[1];
        trgValue[3 * i + 2] += uL1[2];
    }

// part 4, Laplace charge FZ
#pragma omp parallel for
    for (int i = 0; i < ntrg; i++) {
        trgValue[3 * i] += 0.5 * trgValueLapChargeFZ[4 * i + 1];
        trgValue[3 * i + 1] += 0.5 * trgValueLapChargeFZ[4 * i + 2];
        trgValue[3 * i + 2] += 0.5 * trgValueLapChargeFZ[4 * i + 3];
    }

// scale Back
#pragma omp parallel for
    for (int i = 0; i < ntrg * 3; i++) {
        trgValue[i] *= scaleFactor;
    }
}

void FMM_WrapperWall2D::treeStokes(const std::vector<double> &src_coord, const std::vector<double> &trg_coord) {
    treePtrStokes = new pvfmm::PtFMM_Tree(MPI_COMM_WORLD);
    treeDataStokes.dim = 3;
    treeDataStokes.max_depth = 15;
    treeDataStokes.max_pts = max_pts;
    treeSetup(treeDataStokes, src_coord, trg_coord, true);
    treeDataStokes.pt_coord = treeDataStokes.src_coord.Dim() > treeDataStokes.trg_coord.Dim()
                                  ? treeDataStokes.src_coord
                                  : treeDataStokes.trg_coord; // use src_coord to construct FMM tree
    treePtrStokes->Initialize(&treeDataStokes);
    bool adap = true;
    treePtrStokes->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtrStokes->SetupFMM(&matrixStokes);
}

void FMM_WrapperWall2D::treeLapDipole(const std::vector<double> &src_coord, const std::vector<double> &trg_coord) {
    treePtrLapDipole = new pvfmm::PtFMM_Tree(MPI_COMM_WORLD);
    treeDataLapDipole.dim = 3;
    treeDataLapDipole.max_depth = 15;
    treeDataLapDipole.max_pts = max_pts;
    treeSetup(treeDataLapDipole, src_coord, trg_coord, false);
    treeDataLapDipole.pt_coord = treeDataStokes.pt_coord; // use src_coord to construct FMM tree
    treePtrLapDipole->Initialize(&treeDataLapDipole);
    bool adap = true;
    treePtrLapDipole->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtrLapDipole->SetupFMM(&matrixLapDipole);
}

void FMM_WrapperWall2D::treeLapChargeF(const std::vector<double> &src_coord, const std::vector<double> &trg_coord) {
    treePtrLapChargeF = new pvfmm::PtFMM_Tree(MPI_COMM_WORLD);
    treeDataLapChargeF.dim = 3;
    treeDataLapChargeF.max_depth = 15;
    treeDataLapChargeF.max_pts = max_pts;
    treeSetup(treeDataLapChargeF, src_coord, trg_coord, true);
    treeDataLapChargeF.pt_coord = treeDataStokes.pt_coord; // use src_coord to construct FMM tree
    treePtrLapChargeF->Initialize(&treeDataLapChargeF);
    bool adap = true;
    treePtrLapChargeF->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtrLapChargeF->SetupFMM(&matrixLapChargeF);
}

void FMM_WrapperWall2D::treeLapChargeFZ(const std::vector<double> &src_coord, const std::vector<double> &trg_coord) {
    treePtrLapChargeFZ = new pvfmm::PtFMM_Tree(MPI_COMM_WORLD);
    treeDataLapChargeFZ.dim = 3;
    treeDataLapChargeFZ.max_depth = 15;
    treeDataLapChargeFZ.max_pts = max_pts;
    treeSetup(treeDataLapChargeFZ, src_coord, trg_coord, true);
    treeDataLapChargeFZ.pt_coord = treeDataStokes.pt_coord; // use src_coord to construct FMM tree
    treePtrLapChargeFZ->Initialize(&treeDataLapChargeFZ);
    bool adap = true;
    treePtrLapChargeFZ->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtrLapChargeFZ->SetupFMM(&matrixLapChargeFZ);
}

void FMM_WrapperWall2D::evalStokes(std::vector<double> &trg_val, const int n_trg, std::vector<double> *src_val) {
    const int nsrc = src_val->size() / 3;
    const int ntrg = trg_val.size() / 3;
    // evaluate 1, Stokes FMM
    srcValueStokes.resize(3 * nsrc * 2);
    myTimer.start();
// original
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        srcValueStokes[3 * i] = (*src_val)[3 * i];         // fx
        srcValueStokes[3 * i + 1] = (*src_val)[3 * i + 1]; // fy
        srcValueStokes[3 * i + 2] = 0;                     // fz =0
    }
// image
#pragma omp parallel for
    for (int i = nsrc; i < 2 * nsrc; i++) {
        srcValueStokes[3 * i] = -srcValueStokes[3 * (i - nsrc)];         // -fx
        srcValueStokes[3 * i + 1] = -srcValueStokes[3 * (i - nsrc) + 1]; // -fy
        srcValueStokes[3 * i + 2] = 0;                                   // fz =0
    }
    trgValueStokes.resize(ntrg * 3);
    PtFMM_Evaluate(treePtrStokes, trgValueStokes, ntrg, &srcValueStokes, nullptr);
    myTimer.stop("Stokes Near Field");
    myTimer.start();
    if (pbc != NONE) {
        calcMStokes();
    }
    myTimer.stop("Stokes Far Field");
}

void FMM_WrapperWall2D::evalLapDipole(std::vector<double> &trg_val, const int n_trg, std::vector<double> *src_val) {
    const int nsrc = src_val->size() / 3;
    const int ntrg = trg_val.size() / 3;
    // evaluate 2, LaplaceDipole FMM
    srcValueLapDipole.resize(3 * nsrc);
    myTimer.start();
// orginal, no image needed for dipole term
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        const double y3 = -(treeDataLapDipole.src_coord[3 * i + 2] - 0.5);
        srcValueLapDipole[3 * i] = -y3 * (*src_val)[3 * i];
        srcValueLapDipole[3 * i + 1] = -y3 * (*src_val)[3 * i + 1];
        srcValueLapDipole[3 * i + 2] = y3 * (*src_val)[3 * i + 2];
    }
    trgValueLapDipole.resize(ntrg * 4); // value + gradient
    PtFMM_Evaluate(treePtrLapDipole, trgValueLapDipole, ntrg, &srcValueLapDipole, nullptr);
    myTimer.stop("Lap Dipole Near Field");
    myTimer.start();
    if (pbc != NONE) {
        calcMLapDipole();
    }
    myTimer.stop("Lap Dipole Far Field");
    // treePtrLapDipole->ClearFMMData();
}

void FMM_WrapperWall2D::evalLapChargeF(std::vector<double> &trg_val, const int n_trg, std::vector<double> *src_val) {
    const int nsrc = src_val->size() / 3;
    const int ntrg = trg_val.size() / 3;
    // evaluate 3, LaplaceCharge FMM
    srcValueLapChargeF.resize(nsrc * 2);
    myTimer.start();
// original
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        srcValueLapChargeF[i] = (*src_val)[3 * i + 2]; // fz
    }
// image
#pragma omp parallel for
    for (int i = nsrc; i < 2 * nsrc; i++) {
        srcValueLapChargeF[i] = -srcValueLapChargeF[i - nsrc]; // -fz
    }
    trgValueLapChargeF.resize(ntrg * 4); // value + gradient
    PtFMM_Evaluate(treePtrLapChargeF, trgValueLapChargeF, ntrg, &srcValueLapChargeF, nullptr);
    myTimer.stop("Lap Charge F Near Field");
    myTimer.start();
    if (pbc != NONE) {
        calcMLapCharge(0);
    }
    myTimer.stop("Lap Charge F Far Field");
    // treePtrLapChargeF->ClearFMMData();
}

void FMM_WrapperWall2D::evalLapChargeFZ(std::vector<double> &trg_val, const int n_trg, std::vector<double> *src_val) {
    const int nsrc = src_val->size() / 3;
    const int ntrg = trg_val.size() / 3;
    // evaluate 4, LaplaceCharge FMM
    srcValueLapChargeFZ.resize(nsrc * 2);
    myTimer.start();
// original
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        const double y3 = treeDataLapChargeFZ.src_coord[3 * i + 2] - 0.5;
        srcValueLapChargeFZ[i] = y3 * (*src_val)[3 * i + 2]; // fz
    }
// image
#pragma omp parallel for
    for (int i = nsrc; i < 2 * nsrc; i++) {
        srcValueLapChargeFZ[i] = -srcValueLapChargeFZ[i - nsrc]; // -fz
    }
    trgValueLapChargeFZ.resize(ntrg * 4); // value + gradient
    PtFMM_Evaluate(treePtrLapChargeFZ, trgValueLapChargeFZ, ntrg, &srcValueLapChargeFZ, nullptr);
    myTimer.stop("Lap Charge FZ Near Field");
    myTimer.start();
    if (pbc != NONE) {
        calcMLapCharge(1);
    }
    myTimer.stop("Lap Charge FZ Far Field");
    // treePtrLapChargeFZ->ClearFMMData();
}

void FMM_WrapperWall2D::FMM_DataClear() {
    // clear data, keep tree

    if (treePtrStokes != nullptr) {
        treePtrStokes->ClearFMMData();
    }
    if (treePtrLapChargeF != nullptr) {
        treePtrLapChargeF->ClearFMMData();
    }
    if (treePtrLapChargeFZ != nullptr) {
        treePtrLapChargeFZ->ClearFMMData();
    }
    if (treePtrLapDipole != nullptr) {
        treePtrLapDipole->ClearFMMData();
    }

    // clear old Data
    srcValueStokes.clear();
    srcValueLapChargeF.clear();
    srcValueLapChargeFZ.clear();
    srcValueLapDipole.clear();

    trgValueStokes.clear();
    trgValueLapChargeF.clear();
    trgValueLapChargeFZ.clear();
    trgValueLapDipole.clear();

    return;
}

void FMM_WrapperWall2D::FMM_TreeClear() {
    // clear Tree, delete tree
    FMM_DataClear();

    if (treePtrStokes != nullptr) {
        delete treePtrStokes;
        treePtrStokes = nullptr; // after delete the pointer is not equal to nullptr
    }
    if (treePtrLapChargeF != nullptr) {
        delete treePtrLapChargeF;
        treePtrLapChargeF = nullptr;
    }
    if (treePtrLapChargeFZ != nullptr) {
        delete treePtrLapChargeFZ;
        treePtrLapChargeFZ = nullptr;
    }
    if (treePtrLapDipole != nullptr) {
        delete treePtrLapDipole;
        treePtrLapDipole = nullptr;
    }
}

FMM_WrapperWall2D::~FMM_WrapperWall2D() {
    FMM_TreeClear();
    if (pm2lStokes != nullptr) {
        delete[] pm2lStokes;
        pm2lStokes = nullptr;
    }
    if (pm2lLapCharge != nullptr) {
        delete[] pm2lLapCharge;
        pm2lLapCharge = nullptr;
    }
    if (pm2lLapDipole != nullptr) {
        delete[] pm2lLapDipole;
        pm2lLapDipole = nullptr;
    }
}