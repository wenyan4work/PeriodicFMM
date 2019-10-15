/*
 * FMMWrapper.cpp
 *
 *  Created on: Oct 20, 2016
 *      Author: wyan
 */

#include <FMMWrapperWall2D.hpp>
#include <LaplaceLayerKernel.hpp>

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
    case PAXIS::PX:
        pvfmm::periodicType = pvfmm::PeriodicType::PX;
        break;
    case PAXIS::PXY:
        pvfmm::periodicType = pvfmm::PeriodicType::PXY;
        break;
    }

    pm2lStokes = nullptr;
    pm2lLapCharge = nullptr;

    if (pbc != NONE) {
        if (mult_order != (mult_order / 2) * 2 || mult_order < 6 || mult_order > 16) {
            printf("periodic M2L data available only for p=6,8,10,12,14,16\n");
            exit(1);
        } else if (pbc == PAXIS::PX) {
            switch (mult_order) {
            case 6:
                pm2lStokes = readM2LMat("M2LStokes1D3Dp6", 6, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge1D3Dp6", 6, 1);
                break;
            case 8:
                pm2lStokes = readM2LMat("M2LStokes1D3Dp8", 8, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge1D3Dp8", 8, 1);
                break;
            case 10:
                pm2lStokes = readM2LMat("M2LStokes1D3Dp10", 10, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge1D3Dp10", 10, 1);
                break;
            case 12:
                pm2lStokes = readM2LMat("M2LStokes1D3Dp12", 12, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge1D3Dp12", 12, 1);
                break;
            case 14:
                pm2lStokes = readM2LMat("M2LStokes1D3Dp14", 14, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge1D3Dp14", 14, 1);
                break;
            case 16:
                pm2lStokes = readM2LMat("M2LStokes1D3Dp16", 16, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge1D3Dp16", 16, 1);
                break;
            default:
                printf("no m2l data at corresponding p, exit now\n");
                exit(1);
                break;
            }
        } else if (pbc == PAXIS::PXY) {
            switch (mult_order) {
            case 6:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp6", 6, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp6", 6, 1);
                break;
            case 8:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp8", 8, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp8", 8, 1);
                break;
            case 10:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp10", 10, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp10", 10, 1);
                break;
            case 12:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp12", 12, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp12", 12, 1);
                break;
            case 14:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp14", 14, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp14", 14, 1);
                break;
            case 16:
                pm2lStokes = readM2LMat("M2LStokes2D3Dp16", 16, 3);
                pm2lLapCharge = readM2LMat("M2LLapCharge2D3Dp16", 16, 1);
                break;
            default:
                printf("no m2l data at corresponding p, exit now\n");
                exit(1);
                break;
            }
        }

        this->pEquiv = mult_order; // (8-1)^2*6 + 2 points

        this->scaleLEquiv = PVFMM_RAD1; // RAD1 = 2.95 defined in pvfmm_common.h
        this->pCenterLEquiv[0] = -(scaleLEquiv - 1) / 2;
        this->pCenterLEquiv[1] = -(scaleLEquiv - 1) / 2;
        this->pCenterLEquiv[2] = -(scaleLEquiv - 1) / 2;

        pointLEquiv = surface(pEquiv, (double *)&(pCenterLEquiv[0]), scaleLEquiv, 0);
        // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

        equivN = 6 * (pEquiv - 1) * (pEquiv - 1) + 2;
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    const pvfmm::Kernel<double> &kernelStokes = pvfmm::StokesKernel<double>::velocity();
    matrixStokes.Initialize(mult_order, comm, &kernelStokes);

    const pvfmm::Kernel<double> &kernelLapCharge = pvfmm::LaplaceLayerKernel<double>::PGrad();
    matrixLapCharge.Initialize(mult_order, comm, &kernelLapCharge);

    const pvfmm::Kernel<double> &kernelLapDipole = pvfmm::LaplaceLayerKernel<double>::PGrad();
    matrixLapDipole.Initialize(mult_order, comm, &kernelLapDipole);

    treePtrStokes = nullptr;
    treePtrLapDipole = nullptr;
    treePtrLapCharge = nullptr;

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
    } else if (this->pbc == PAXIS::PX) {
        if (yhigh - ylow > xhigh - xlow) {
            printf("periodic in x direction. box size in y must be smaller than box size in x");
            exit(1);
        }
        if (zlen >= 0.5 * xlen || zlen >= 0.5 * ylen) {
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

void FMM_WrapperWall2D::treePointDump(const pvfmm::PtFMM_Data<double> &treeData) {
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

void FMM_WrapperWall2D::FMM_UpdateTree(const std::vector<double> &src_coord, const std::vector<double> &trg_coord) {
    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0)
        printf("FMM_UpdateTree\n");

    FMM_TreeClear();

    if (myRank == 0)
        printf("tree deleted\n");

    scalePoints(src_coord, trg_coord);

    myTimer.start();
    treeStokes();
    myTimer.stop("Stokes tree construction");

    myTimer.start();
    treeLapDipole();
    myTimer.stop("Lap Dipole tree construction");

    myTimer.start();
    treeLapCharge();
    myTimer.stop("Lap Charge tree construction");
    // printf("SetupFMM Complete\n");
}

void FMM_WrapperWall2D::FMM_Evaluate(std::vector<double> &trg_val, const int n_trg, std::vector<double> *src_val) {
    if (n_trg != trg_val.size() / 3) {
        printf("n_trg error \n");
        exit(1);
    }
    // in place rotate of src_val;
    if (src_val == nullptr) {
        printf("Error, no source value\n");
        exit(1);
    }

    const int nsrc = src_val->size() / 3;

    // evaluate 1, Stokes FMM
    evalStokes(n_trg, src_val);

    // evaluate 2, LaplaceDipole FMM
    evalLapDipole(n_trg, src_val);

    // evaluate 3, LaplaceCharge FMM
    evalLapCharge(n_trg, src_val);

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
    assert(trgCoord.Dim() == trgValueStokes.size());
    const double pi = 3.1415926535897932384626433;

    int M = 3 * equivN;
    int N = 3 * equivN; // checkN = equivN in this code.
    M2Lsource.resize(M);
    assert(M2Lsource.size() == v.Dim());

#pragma omp parallel for default(none) shared(M, N, v, pm2lStokes, M2Lsource)
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
        vPtr = &(treePtrLapCharge->RootNode()->FMMData()->upward_equiv); // the value calculated by pvfmm
        trgCoordPtr = &(treeDataLapCharge.trg_coord);
        trgValuePtr = &trgValueLapCharge;
    } else if (treeSelect == 1) {
        vPtr = &(treePtrLapDipole->RootNode()->FMMData()->upward_equiv); // the value calculated by pvfmm
        trgCoordPtr = &(treeDataLapDipole.trg_coord);
        trgValuePtr = &trgValueLapDipole;
    }

    // for (int j = 0; j < equivN; j++) {
    //     printf("%lf\n", (*vPtr)[j]);
    // }

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

    const pvfmm::Kernel<double> &kernelLap = pvfmm::LaplaceLayerKernel<double>::PGrad();
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
        const double x3 = trgCoordScaled[3 * i + 2] - 0.5;
        double ud[3];
        ud[0] = x3 * trgValueLapDipole[4 * i + 1];
        ud[1] = x3 * trgValueLapDipole[4 * i + 2];
        ud[2] = x3 * trgValueLapDipole[4 * i + 3] - trgValueLapDipole[4 * i];
        trgValue[3 * i + 0] += ud[0];
        trgValue[3 * i + 1] += ud[1];
        trgValue[3 * i + 2] += ud[2];
    }

// part 3, Laplace charge FZ
#pragma omp parallel for
    for (int i = 0; i < ntrg; i++) {
        double uL2[3];
        uL2[0] = 0.5 * trgValueLapCharge[4 * i + 1];
        uL2[1] = 0.5 * trgValueLapCharge[4 * i + 2];
        uL2[2] = 0.5 * trgValueLapCharge[4 * i + 3];
        trgValue[3 * i + 0] += uL2[0];
        trgValue[3 * i + 1] += uL2[1];
        trgValue[3 * i + 2] += uL2[2];
    }

// scale Back
#pragma omp parallel for
    for (int i = 0; i < ntrg * 3; i++) {
        trgValue[i] *= scaleFactor;
    }
}

void FMM_WrapperWall2D::treeStokes() {
    treePtrStokes = new pvfmm::PtFMM_Tree<double>(MPI_COMM_WORLD);
    treeDataStokes.dim = 3;
    treeDataStokes.max_depth = 15;
    treeDataStokes.max_pts = max_pts;
    treeSetupStokes();
    treeDataStokes.pt_coord = treeDataStokes.src_coord.Dim() > treeDataStokes.trg_coord.Dim()
                                  ? treeDataStokes.src_coord
                                  : treeDataStokes.trg_coord; // use src_coord to construct FMM tree
    treePtrStokes->Initialize(&treeDataStokes);
    bool adap = true;
    treePtrStokes->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtrStokes->SetupFMM(&matrixStokes);
}

void FMM_WrapperWall2D::treeLapDipole() {
    // both SL and DL
    treePtrLapDipole = new pvfmm::PtFMM_Tree<double>(MPI_COMM_WORLD);
    treeDataLapDipole.dim = 3;
    treeDataLapDipole.max_depth = 15;
    treeDataLapDipole.max_pts = max_pts;
    treeSetupDipole();
    treeDataLapDipole.pt_coord = treeDataStokes.pt_coord; // use src_coord to construct FMM tree
    treePtrLapDipole->Initialize(&treeDataLapDipole);
    bool adap = true;
    treePtrLapDipole->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtrLapDipole->SetupFMM(&matrixLapDipole);
}

void FMM_WrapperWall2D::treeLapCharge() {
    treePtrLapCharge = new pvfmm::PtFMM_Tree<double>(MPI_COMM_WORLD);
    treeDataLapCharge.dim = 3;
    treeDataLapCharge.max_depth = 15;
    treeDataLapCharge.max_pts = max_pts;
    treeSetupCharge();
    treeDataLapCharge.pt_coord = treeDataStokes.pt_coord; // use src_coord to construct FMM tree
    treePtrLapCharge->Initialize(&treeDataLapCharge);
    bool adap = true;
    treePtrLapCharge->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtrLapCharge->SetupFMM(&matrixLapCharge);
}

void FMM_WrapperWall2D::evalStokes(const int ntrg, std::vector<double> *src_val) {
    const int nsrc = src_val->size() / 3;
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
    for (int i = 0; i < nsrc; i++) {
        srcValueStokes[3 * (i + nsrc)] = -(*src_val)[3 * i];         // -fx
        srcValueStokes[3 * (i + nsrc) + 1] = -(*src_val)[3 * i + 1]; // -fy
        srcValueStokes[3 * (i + nsrc) + 2] = 0;                      // fz =0
    }
    trgValueStokes.resize(ntrg * 3);
    PtFMM_Evaluate(treePtrStokes, trgValueStokes, ntrg, &srcValueStokes);
    myTimer.stop("Stokes Near Field");
    myTimer.start();
    if (pbc != NONE) {
        calcMStokes();
    }
    myTimer.stop("Stokes Far Field");
}

void FMM_WrapperWall2D::evalLapDipole(const int ntrg, std::vector<double> *src_val) {
    const int nsrc = src_val->size() / 3;
    // evaluate 2, LaplaceDipole FMM
    // SL: src + image
    // DL: image only
    srcValueLapDipoleSL.resize(2 * nsrc);
    srcValueLapDipoleDL.resize(3 * nsrc);
    myTimer.start();

// SL : src + image charge values
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        srcValueLapDipoleSL[i] = -0.5 * (*src_val)[3 * i + 2];
    }

#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        srcValueLapDipoleSL[i + nsrc] = 0.5 * (*src_val)[3 * i + 2];
    }

    // DL : dipole values
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        const double y3 = (srcCoordScaled[3 * i + 2] - 0.5);
        srcValueLapDipoleDL[3 * i + 0] = -y3 * (*src_val)[3 * i + 0];
        srcValueLapDipoleDL[3 * i + 1] = -y3 * (*src_val)[3 * i + 1];
        srcValueLapDipoleDL[3 * i + 2] = y3 * (*src_val)[3 * i + 2];
    }

    trgValueLapDipole.resize(ntrg * 4); // PGrad Kernel
    PtFMM_Evaluate(treePtrLapDipole, trgValueLapDipole, ntrg, &srcValueLapDipoleSL, &srcValueLapDipoleDL);
    myTimer.stop("Lap Dipole Near Field");
    myTimer.start();
    if (pbc != NONE) {
        calcMLapCharge(1);
    }
    myTimer.stop("Lap Dipole Far Field");
}

void FMM_WrapperWall2D::evalLapCharge(const int ntrg, std::vector<double> *src_val) {
    const int nsrc = src_val->size() / 3;
    // evaluate 3, LaplaceCharge FMM
    srcValueLapCharge.resize(nsrc * 2);
    myTimer.start();
// original
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        const double y3 = (srcCoordScaled[3 * i + 2] - 0.5);
        srcValueLapCharge[i] = y3 * (*src_val)[3 * i + 2]; // fz
    }
// image
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        const double y3 = (srcCoordScaled[3 * i + 2] - 0.5);
        srcValueLapCharge[i + nsrc] = -y3 * (*src_val)[3 * i + 2]; // -fz
    }

    trgValueLapCharge.resize(ntrg * 4); // value + gradient
    PtFMM_Evaluate(treePtrLapCharge, trgValueLapCharge, ntrg, &srcValueLapCharge);
    myTimer.stop("Lap Charge F Near Field");
    myTimer.start();
    if (pbc != NONE) {
        calcMLapCharge(0);
    }
    myTimer.stop("Lap Charge F Far Field");
    // treePtrLapChargeF->ClearFMMData();
}

void FMM_WrapperWall2D::FMM_DataClear() {
    // clear data, keep tree

    if (treePtrStokes != nullptr) {
        treePtrStokes->ClearFMMData();
    }
    if (treePtrLapCharge != nullptr) {
        treePtrLapCharge->ClearFMMData();
    }
    if (treePtrLapDipole != nullptr) {
        treePtrLapDipole->ClearFMMData();
    }

    // clear old Data
    srcValueStokes.clear();
    srcValueLapCharge.clear();
    srcValueLapDipoleSL.clear();
    srcValueLapDipoleDL.clear();

    trgValueStokes.clear();
    trgValueLapCharge.clear();
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
    if (treePtrLapCharge != nullptr) {
        delete treePtrLapCharge;
        treePtrLapCharge = nullptr;
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
}

void FMM_WrapperWall2D::scalePoints(const std::vector<double> &srcCoord, const std::vector<double> &trgCoord) {
    // filled srcCoordScaled, srcImageCoordScaled, trgCoordScaled

    // srcCoordScaled:
    const int nsrc = srcCoord.size() / 3;
    const int ntrg = trgCoord.size() / 3;

    srcCoordScaled.resize(nsrc * 3);
    trgCoordScaled.resize(ntrg * 3);

    const double xshift = this->xshift;
    const double yshift = this->yshift;
    const double zshift = this->zshift;
    const double scaleFactor = this->scaleFactor;

    if (pbc == PAXIS::PX) {
#pragma omp parallel for firstprivate(xshift, yshift, zshift, scaleFactor) shared(srcCoordScaled, srcCoord)
        for (size_t i = 0; i < nsrc; i++) {
            srcCoordScaled[3 * i] = fracwrap((srcCoord[3 * i] + xshift) * scaleFactor);
            srcCoordScaled[3 * i + 1] = ((srcCoord[3 * i + 1] + yshift) * scaleFactor);
            srcCoordScaled[3 * i + 2] = ((srcCoord[3 * i + 2] + zshift) * scaleFactor);
        }
#pragma omp parallel for firstprivate(xshift, yshift, zshift, scaleFactor) shared(trgCoordScaled, trgCoord)
        for (size_t i = 0; i < ntrg; i++) {
            trgCoordScaled[3 * i] = fracwrap((trgCoord[3 * i] + xshift) * scaleFactor);
            trgCoordScaled[3 * i + 1] = ((trgCoord[3 * i + 1] + yshift) * scaleFactor);
            trgCoordScaled[3 * i + 2] = ((trgCoord[3 * i + 2] + zshift) * scaleFactor);
        }
    } else if (pbc == PAXIS::PXY) {
#pragma omp parallel for firstprivate(xshift, yshift, zshift, scaleFactor) shared(srcCoordScaled, srcCoord)
        for (size_t i = 0; i < nsrc; i++) {
            srcCoordScaled[3 * i] = fracwrap((srcCoord[3 * i] + xshift) * scaleFactor);
            srcCoordScaled[3 * i + 1] = fracwrap((srcCoord[3 * i + 1] + yshift) * scaleFactor);
            srcCoordScaled[3 * i + 2] = ((srcCoord[3 * i + 2] + zshift) * scaleFactor);
        }
#pragma omp parallel for firstprivate(xshift, yshift, zshift, scaleFactor) shared(trgCoordScaled, trgCoord)
        for (size_t i = 0; i < ntrg; i++) {
            trgCoordScaled[3 * i] = fracwrap((trgCoord[3 * i] + xshift) * scaleFactor);
            trgCoordScaled[3 * i + 1] = fracwrap((trgCoord[3 * i + 1] + yshift) * scaleFactor);
            trgCoordScaled[3 * i + 2] = ((trgCoord[3 * i + 2] + zshift) * scaleFactor);
        }
    } else {
#pragma omp parallel for firstprivate(xshift, yshift, zshift, scaleFactor) shared(srcCoordScaled, srcCoord)
        for (size_t i = 0; i < nsrc; i++) {
            srcCoordScaled[3 * i] = ((srcCoord[3 * i] + xshift) * scaleFactor);
            srcCoordScaled[3 * i + 1] = ((srcCoord[3 * i + 1] + yshift) * scaleFactor);
            srcCoordScaled[3 * i + 2] = ((srcCoord[3 * i + 2] + zshift) * scaleFactor);
        }
#pragma omp parallel for firstprivate(xshift, yshift, zshift, scaleFactor) shared(trgCoordScaled, trgCoord)
        for (size_t i = 0; i < ntrg; i++) {
            trgCoordScaled[3 * i] = ((trgCoord[3 * i] + xshift) * scaleFactor);
            trgCoordScaled[3 * i + 1] = ((trgCoord[3 * i + 1] + yshift) * scaleFactor);
            trgCoordScaled[3 * i + 2] = ((trgCoord[3 * i + 2] + zshift) * scaleFactor);
        }
    }

    // prevent PVFMM from breaking down with coord=1
    const double eps = std::numeric_limits<double>::epsilon() * 100;
    const int NS = nsrc * 3;
    const int NT = ntrg * 3;
#pragma omp parallel for shared(srcCoordScaled)
    for (int i = 0; i < NS; i++) {
        // printf("src scaled %lf\n", srcCoordScaled[i]);
        if (srcCoordScaled[i] > 1 - eps)
            srcCoordScaled[i] = 1 - eps;
    }
#pragma omp parallel for shared(trgCoordScaled)
    for (int i = 0; i < NT; i++) {
        // printf("trg scaled %lf\n", trgCoordScaled[i]);
        if (trgCoordScaled[i] > 1 - eps)
            trgCoordScaled[i] = 1 - eps;
    }

    // image of src
    srcImageCoordScaled.resize(nsrc * 3);
#pragma omp parallel for
    for (size_t i = 0; i < nsrc; i++) {
        srcImageCoordScaled[3 * i] = srcCoordScaled[3 * i];
        srcImageCoordScaled[3 * i + 1] = srcCoordScaled[3 * i + 1];
        srcImageCoordScaled[3 * i + 2] = 1 - srcCoordScaled[3 * i + 2];
    }
}

void FMM_WrapperWall2D::treeSetupStokes() {
    // SL: src + image
    // DL: empty
    const int nsrc = srcCoordScaled.size() / 3;
    const int ntrg = trgCoordScaled.size() / 3;

    treeDataStokes.src_coord.Resize(2 * 3 * nsrc);
#pragma omp parallel for
    for (int i = 0; i < 3 * nsrc; i++) {
        treeDataStokes.src_coord[i] = srcCoordScaled[i];
    }
#pragma omp parallel for
    for (int i = 0; i < 3 * nsrc; i++) {
        treeDataStokes.src_coord[i + 3 * nsrc] = srcImageCoordScaled[i];
    }

    treeDataStokes.trg_coord.Resize(ntrg * 3);
#pragma omp parallel for
    for (int i = 0; i < 3 * ntrg; i++) {
        treeDataStokes.trg_coord[i] = trgCoordScaled[i];
    }

    treeDataStokes.surf_coord.Resize(0);

    treeDataStokes.src_value.Resize(3 * nsrc * 2);
    treeDataStokes.trg_value.Resize(3 * ntrg);
    treeDataStokes.surf_value.Resize(0);

    // printf("tree data Stokes\n");
    // treePointDump(treeDataStokes);
}
void FMM_WrapperWall2D::treeSetupCharge() {
    // SL: src + image
    // DL: empty
    const int nsrc = srcCoordScaled.size() / 3;
    const int ntrg = trgCoordScaled.size() / 3;

    treeDataLapCharge.src_coord.Resize(2 * 3 * nsrc);
#pragma omp parallel for
    for (int i = 0; i < 3 * nsrc; i++) {
        treeDataLapCharge.src_coord[i] = srcCoordScaled[i];
    }
#pragma omp parallel for
    for (int i = 0; i < 3 * nsrc; i++) {
        treeDataLapCharge.src_coord[i + 3 * nsrc] = srcImageCoordScaled[i];
    }

    treeDataLapCharge.trg_coord.Resize(ntrg * 3);
#pragma omp parallel for
    for (int i = 0; i < 3 * ntrg; i++) {
        treeDataLapCharge.trg_coord[i] = trgCoordScaled[i];
    }
    treeDataLapCharge.surf_coord.Resize(0);

    // treeDataLapCharge.src_value.Resize(nsrc * 2);
    // treeDataLapCharge.trg_value.Resize(ntrg * 4);
    // treeDataLapCharge.surf_value.Resize(0);

    // printf("tree data lap charge\n");
    // treePointDump(treeDataLapCharge);
}

void FMM_WrapperWall2D::treeSetupDipole() {
    // SL: src + image
    // DL: image
    const int nsrc = srcCoordScaled.size() / 3;
    const int ntrg = trgCoordScaled.size() / 3;

    treeDataLapDipole.src_coord.Resize(2 * 3 * nsrc);
#pragma omp parallel for
    for (int i = 0; i < 3 * nsrc; i++) {
        treeDataLapDipole.src_coord[i] = srcCoordScaled[i];
    }
#pragma omp parallel for
    for (int i = 0; i < 3 * nsrc; i++) {
        treeDataLapDipole.src_coord[i + 3 * nsrc] = srcImageCoordScaled[i];
    }

    treeDataLapDipole.trg_coord.Resize(ntrg * 3);
#pragma omp parallel for
    for (int i = 0; i < 3 * ntrg; i++) {
        treeDataLapDipole.trg_coord[i] = trgCoordScaled[i];
    }

    treeDataLapDipole.surf_coord.Resize(nsrc * 3);
#pragma omp parallel for
    for (int i = 0; i < 3 * nsrc; i++) {
        treeDataLapDipole.surf_coord[i] = srcImageCoordScaled[i];
    }

    // treeDataLapDipole.src_value.Resize(nsrc * 2);
    // treeDataLapDipole.trg_value.Resize(ntrg * 4);
    // treeDataLapDipole.surf_value.Resize(nsrc * 3);

    // printf("tree data lap dipole\n");
    // treePointDump(treeDataLapDipole);
}
