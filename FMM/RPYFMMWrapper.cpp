/*
 * FMMWrapper.cpp
 *
 *  Created on: Oct 20, 2016
 *      Author: wyan
 */

#include "RPYFMMWrapper.h"

#include <cassert>
#include <limits>

#include <mpi.h>
#include <omp.h>

extern pvfmm::PeriodicType pvfmm::periodicType;

// TreeClear(), optional, should be called BEFORE TreeUpdate.

// return fraction part between [0,1)
/*
 * This function is only applied in the PERIODIC DIRECTION
 *
 * The user of the library must ensure that all points are located within [0,1)
 *
 * */
inline double fracwrap(double x) { return x - floor(x); }

RPYFMM_Wrapper::RPYFMM_Wrapper(int mult_order, int max_pts, int init_depth, PAXIS pbc_, KERNEL kernel_)
    : mult_order(mult_order), max_pts(max_pts), init_depth(init_depth), pbc(pbc_), kernel(kernel_), xlow(0), xhigh(1),
      ylow(0), yhigh(1), zlow(0), zhigh(1), scaleFactor(1), xshift(0), yshift(0), zshift(0) {

    // set periodic boundary condition
    switch (pbc) {
    case PAXIS::NONE:
        periodicType = PeriodicType::NONE;
        break;
    case PAXIS::PXYZ:
        periodicType = PeriodicType::PXYZ;
        break;
    case PAXIS::PX:
        periodicType = PeriodicType::PZ; // use axis rotation
        break;
    case PAXIS::PY:
        periodicType = PeriodicType::PZ; // use axis rotation
        break;
    case PAXIS::PZ:
        periodicType = PeriodicType::PZ;
        break;
    case PAXIS::PXY:
        periodicType = PeriodicType::PXY;
        break;
    case PAXIS::PXZ:
        periodicType = PeriodicType::PXY; // use axis rotation
        break;
    case PAXIS::PYZ:
        periodicType = PeriodicType::PXY; // use axis rotation
        break;
    }

    pm2lG = nullptr;
    pm2lRPY = nullptr;

    if (pbc != NONE) {
        if (mult_order != (mult_order / 2) * 2 || mult_order < 6 || mult_order > 16) {
            printf("periodic M2L data available only for p=6,8,10,12,14,16\n");
        } else if (pbc == PAXIS::PXYZ) {
            std::string dataName = "M2LStokes3D3DpX";
            dataName.replace(14, 1, std::to_string(mult_order));
            std::cout << "reading M2L data: " << dataName << std::endl;
            pm2lG = readM2LMat(dataName.c_str(), mult_order);

        } else if (pbc == PAXIS::PX || pbc == PAXIS::PY || pbc == PAXIS::PZ) {
            std::string dataName = "M2LStokes1D3DpX";
            dataName.replace(14, 1, std::to_string(mult_order));
            std::cout << "reading M2L data: " << dataName << std::endl;
            pm2lG = readM2LMat(dataName.c_str(), mult_order);

        } else if (pbc == PAXIS::PXY || pbc == PAXIS::PXZ || pbc == PAXIS::PYZ) {
            std::string dataName = "M2LStokes2D3DpX";
            dataName.replace(14, 1, std::to_string(mult_order));
            std::cout << "reading M2L data: " << dataName << std::endl;
            pm2lG = readM2LMat(dataName.c_str(), mult_order);
        }

        if (kernel == KERNEL::GRPY) {
            pm2lRPY = pm2lG;
            // read data for RPY, use data for Laplace dipole
            // if (mult_order != (mult_order / 2) * 2 || mult_order < 6 || mult_order > 16) {
            //     printf("periodic M2L data available only for p=6,8,10,12,14,16\n");
            // } else if (pbc == PAXIS::PXYZ) {
            //     std::string dataName = "M2LLapDipole3D3DpX";
            //     dataName.replace(17, 1, std::to_string(mult_order));
            //     std::cout << "reading M2L data: " << dataName << std::endl;
            //     pm2lRPY = readM2LMat(dataName.c_str(), mult_order);

            // } else if (pbc == PAXIS::PX || pbc == PAXIS::PY || pbc == PAXIS::PZ) {
            //     std::cout << "1D data not ready yet" << std::endl;
            //     exit(1);

            // } else if (pbc == PAXIS::PXY || pbc == PAXIS::PXZ || pbc == PAXIS::PYZ) {
            //     std::string dataName = "M2LLapDipole2D3DpX";
            //     dataName.replace(17, 1, std::to_string(mult_order));
            //     std::cout << "reading M2L data: " << dataName << std::endl;
            //     pm2lRPY = readM2LMat(dataName.c_str(), mult_order);
            // }
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
    const pvfmm::Kernel<double> &kernelG = pvfmm::StokesKernel<double>::velocity();      // 1/8pi (I/r+rr/r^3)
    const pvfmm::Kernel<double> &kernelRPY = pvfmm::RPYCustomKernel<double>::velocity(); // 1/4pi (I/r^3-3rr/r^5)

    MPI_Comm comm = MPI_COMM_WORLD;

    treePtrG = nullptr;
    matrixG.Initialize(mult_order, comm, &kernelG);

    treePtrRPY = nullptr;
    if (kernel == GRPY) {
        matrixRPY.Initialize(mult_order, comm, &kernelRPY);
    }
#ifdef FMMDEBUG
    pvfmm::Profile::Enable(true);
#endif

    printf("FMM Initialized\n");
}

RPYFMM_Wrapper::~RPYFMM_Wrapper() {
    FMM_TreeClear();
    safeDeletePtr(pm2lG);
    // safeDeletePtr(pm2lRPY);
}

void RPYFMM_Wrapper::FMM_TreeClear() {
    FMM_DataClear();
    safeDeletePtr(treePtrG);
    safeDeletePtr(treePtrRPY);
}

void RPYFMM_Wrapper::FMM_DataClear() {
    if (treePtrG != nullptr) {
        treePtrG->ClearFMMData();
    }
    if (treePtrRPY != nullptr) {
        treePtrRPY->ClearFMMData();
    }
}

void RPYFMM_Wrapper::FMM_SetBox(double xlow_, double xhigh_, double ylow_, double yhigh_, double zlow_, double zhigh_) {
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

    std::cout << "box x" << xlen << "box y" << ylen << "box z" << zlen << std::endl;
    std::cout << "scale factor" << scaleFactor << std::endl;

    // validate box setting, ensure fitting in a cubic box [0,1]^3
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
}

void RPYFMM_Wrapper::setupCoord(const std::vector<double> &coordIn, std::vector<double> &coord) const {
    // apply scale and rotation to internal
    // Set source points, with scale

    const int npts = coordIn.size() / 3;
    coord.resize(npts * 3);

    if (pbc == PAXIS::PY) {
// rotate y axis to z axis to use the z 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            coord[3 * i + 1] = ((coordIn[3 * i] + xshift) * scaleFactor);
            coord[3 * i + 2] = fracwrap((coordIn[3 * i + 1] + yshift) * scaleFactor);
            coord[3 * i] = ((coordIn[3 * i + 2] + zshift) * scaleFactor);
        }
    } else if (pbc == PAXIS::PXZ) {
// rotate y axis to z axis to use the xy 2d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            coord[3 * i + 1] = fracwrap((coordIn[3 * i] + xshift) * scaleFactor);
            coord[3 * i + 2] = ((coordIn[3 * i + 1] + yshift) * scaleFactor);
            coord[3 * i] = fracwrap((coordIn[3 * i + 2] + zshift) * scaleFactor);
        }
    } else if (pbc == PAXIS::PX) {
// rotate x axis to z axis
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            coord[3 * i + 2] = fracwrap((coordIn[3 * i] + xshift) * scaleFactor);
            coord[3 * i] = ((coordIn[3 * i + 1] + yshift) * scaleFactor);
            coord[3 * i + 1] = ((coordIn[3 * i + 2] + zshift) * scaleFactor);
        }
    } else if (pbc == PAXIS::PYZ) {
// rotate x axis to z axis
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            coord[3 * i + 2] = ((coordIn[3 * i] + xshift) * scaleFactor);
            coord[3 * i] = fracwrap((coordIn[3 * i + 1] + yshift) * scaleFactor);
            coord[3 * i + 1] = fracwrap((coordIn[3 * i + 2] + zshift) * scaleFactor);
        }

    } else if (pbc == PAXIS::PXYZ) {
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            coord[3 * i] = fracwrap((coordIn[3 * i] + xshift) * scaleFactor);
            coord[3 * i + 1] = fracwrap((coordIn[3 * i + 1] + yshift) * scaleFactor);
            coord[3 * i + 2] = fracwrap((coordIn[3 * i + 2] + zshift) * scaleFactor);
        }
    } else if (pbc == PAXIS::PZ) {
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            coord[3 * i] = ((coordIn[3 * i] + xshift) * scaleFactor);
            coord[3 * i + 1] = ((coordIn[3 * i + 1] + yshift) * scaleFactor);
            coord[3 * i + 2] = fracwrap((coordIn[3 * i + 2] + zshift) * scaleFactor);
        }
    } else if (pbc == PAXIS::PXY) {
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            coord[3 * i] = fracwrap((coordIn[3 * i] + xshift) * scaleFactor);
            coord[3 * i + 1] = fracwrap((coordIn[3 * i + 1] + yshift) * scaleFactor);
            coord[3 * i + 2] = ((coordIn[3 * i + 2] + zshift) * scaleFactor);
        }
    } else {
        assert(pbc == PAXIS::NONE);
// no rotate
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            coord[3 * i] = ((coordIn[3 * i] + xshift) * scaleFactor);
            coord[3 * i + 1] = ((coordIn[3 * i + 1] + yshift) * scaleFactor);
            coord[3 * i + 2] = ((coordIn[3 * i + 2] + zshift) * scaleFactor);
        }
    }
    return;
}

void RPYFMM_Wrapper::FMM_UpdateTree(const std::vector<double> &src_coord, const std::vector<double> &trg_coord) {
    int np, myrank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &myrank);

    FMM_TreeClear();
    std::cout << "FMM Tree Cleared" << std::endl;

    treePtrG = new pvfmm::PtFMM_Tree(comm);
    treeDataG.dim = 3;
    treeDataG.max_depth = 15;
    treeDataG.max_pts = max_pts;

    // setup point coordinates
    setupCoord(src_coord, this->srcCoord);
    setupCoord(trg_coord, this->trgCoord);

    std::cout << "Coord setup" << std::endl;

    const bool adap = true;

    // setup Tree
    treeDataG.surf_coord.Resize(0);
    treeDataG.src_coord = srcCoord;
    treeDataG.trg_coord = trgCoord;
    treeDataG.pt_coord = treeDataG.trg_coord;
    treePtrG->Initialize(&treeDataG);
    treePtrG->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
    treePtrG->SetupFMM(&matrixG);
    std::cout << "treeG setup" << std::endl;

    if (kernel == GRPY) {
        // setup RPY, same as treeDataG
        treePtrRPY = new pvfmm::PtFMM_Tree(comm);

        treeDataRPY.dim = treeDataG.dim;
        treeDataRPY.max_depth = treeDataG.max_depth;
        treeDataRPY.max_pts = treeDataG.max_pts;

        treeDataRPY.surf_coord.Resize(0);
        treeDataRPY.src_coord = srcCoord;
        treeDataRPY.trg_coord = trgCoord;
        treeDataRPY.pt_coord = treeDataRPY.trg_coord;

        treePtrRPY->Initialize(&treeDataRPY);
        treePtrRPY->InitFMM_Tree(adap, pbc == NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
        treePtrRPY->SetupFMM(&matrixRPY);

        std::cout << "treeRPY setup" << std::endl;
    }

    std::cout << "SetupFMM Complete" << std::endl;
}

void RPYFMM_Wrapper::rotateValue(const std::vector<double> &value, std::vector<double> &valueRotate) const {

    const int npts = value.size() / 3;
    valueRotate.resize(3 * npts);
    if (pbc == PAXIS::PY || pbc == PAXIS::PXZ) {
// rotate y axis to z axis to use the z 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            double x = value[3 * i];
            double y = value[3 * i + 1];
            double z = value[3 * i + 2];
            valueRotate[3 * i] = z;
            valueRotate[3 * i + 1] = x;
            valueRotate[3 * i + 2] = y;
        }
    } else if (pbc == PAXIS::PX || pbc == PAXIS::PYZ) {
// rotate x axis to z axis
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            double x = value[3 * i];
            double y = value[3 * i + 1];
            double z = value[3 * i + 2];
            valueRotate[3 * i] = y;
            valueRotate[3 * i + 1] = z;
            valueRotate[3 * i + 2] = x;
        }
    } else {
// no rotate
// simple copy
#pragma omp parallel for
        for (size_t i = 0; i < 3 * npts; i++) {
            valueRotate[i] = value[i];
        }
    }
}

void RPYFMM_Wrapper::rotateBackValue(const std::vector<double> &valueRotate, std::vector<double> &value) const {
    // rotate back, no scale
    const int npts = valueRotate.size() / 3;
    if (value.size() != 3 * npts) {
        printf("size error in rorate back\n");
        exit(1);
    }

    if (pbc == PAXIS::PY || pbc == PAXIS::PXZ) {
// rotate y axis to z axis to use the z 1d periodic data
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            double x = valueRotate[3 * i];
            double y = valueRotate[3 * i + 1];
            double z = valueRotate[3 * i + 2];
            value[3 * i] = y;
            value[3 * i + 1] = z;
            value[3 * i + 2] = x;
        }

    } else if (pbc == PAXIS::PX || pbc == PAXIS::PYZ) {
// rotate x axis to z axis
#pragma omp parallel for
        for (size_t i = 0; i < npts; i++) {
            double x = valueRotate[3 * i];
            double y = valueRotate[3 * i + 1];
            double z = valueRotate[3 * i + 2];
            value[3 * i] = z;
            value[3 * i + 1] = x;
            value[3 * i + 2] = y;
        }

    } else {
// no rotate
// simple copy
#pragma omp parallel for
        for (size_t i = 0; i < 3 * npts; i++) {
            value[i] = valueRotate[i];
        }
    }
}

void RPYFMM_Wrapper::FMM_EvaluateG(std::vector<double> &trg_val, const int nTrg, const std::vector<double> &src_val) {

    if (src_val.size() != srcCoord.size()) {
        printf("src val & src coord size error \n");
        exit(1);
    }
    srcValueG.resize(src_val.size());
    rotateValue(src_val, srcValueG);

    trgValueG.resize(nTrg * 3);

    std::vector<double> *surf_valPtr = nullptr;
    PtFMM_Evaluate(treePtrG, trgValueG, nTrg, &srcValueG, surf_valPtr);

#ifdef FMMDEBUG
    printf("before pxyz: %lf,%lf,%lf\n", trgValueG[0], trgValueG[1], trgValueG[2]);
#endif

    if (pbc != NONE) {
        calcMG();
    }

// scale back
#pragma omp parallel for
    for (int i = 0; i < nTrg * 3; i++) {
        trgValueG[i] *= scaleFactor;
    }

    // rotate back
    trg_val.resize(nTrg * 3);
    rotateBackValue(trgValueG, trg_val);

    return;
}

void RPYFMM_Wrapper::FMM_EvaluateRPY(std::vector<double> &trg_val, const int nTrg, const std::vector<double> &src_val) {

    if (kernel != GRPY) {
        printf("RPY kernel not initialized error\n");
        exit(1);
    }

    if (src_val.size() != srcCoord.size()) {
        printf("src val & src coord size error \n");
        exit(1);
    }
    srcValueRPY.resize(src_val.size());
    rotateValue(src_val, srcValueRPY);

    trgValueRPY.resize(nTrg * 3);

    std::vector<double> *surf_valPtr = nullptr;
    PtFMM_Evaluate(treePtrRPY, trgValueRPY, nTrg, &srcValueRPY, surf_valPtr);

    //#ifdef FMMDEBUG
    printf("before pxyz: %lf,%lf,%lf\n", trgValueRPY[0], trgValueRPY[1], trgValueRPY[2]);
    //#endif

    if (pbc != NONE) {
        calcMRPY();
    }

    // scale back
    const double sf = scaleFactor * scaleFactor * scaleFactor;
#pragma omp parallel for
    for (int i = 0; i < nTrg * 3; i++) {
        trgValueRPY[i] *= sf;
    }

    // rotate back
    trg_val.resize(nTrg * 3);
    rotateBackValue(trgValueRPY, trg_val);

    return;
}

double *RPYFMM_Wrapper::readM2LMat(const char *fname, const int p) {
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
            fdata[i * size + j] = fread;
        }
    }

    fclose(fin);
    return fdata;
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

void RPYFMM_Wrapper::calcMG() {
    pvfmm::Vector<double> v = treePtrG->RootNode()->FMMData()->upward_equiv; // the value calculated by pvfmm
    assert(v.Dim() == 3 * this->equivN);

    // add to trg_value
    auto &trgCoord = treeDataG.trg_coord;
    const size_t nTrg = trgCoord.Dim() / 3;

    int M = 3 * equivN;
    int N = 3 * equivN; // checkN = equivN in this code.
    M2Lsource.resize(3 * equivN);
    assert(M2Lsource.size() == v.Dim());
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double temp = 0;
#pragma omp simd
        for (int j = 0; j < N; j++) {
            temp += pm2lG[i * N + j] * v[j];
        }
        M2Lsource[i] = temp;
    }

    const pvfmm::Kernel<double> &kernelG = pvfmm::StokesKernel<double>::velocity();

    //  typedef void (*Ker_t)(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out,
    // mem::MemoryManager* mem_mgr);
    // dof should be set to 1

    const size_t chunkSize = 2000; // each chunk has 2000 target points.
    const size_t chunkNumber = floor(nTrg / chunkSize) + 1;
    printf("chunkSize, chunkNumber: %d, %d\n", chunkSize, chunkNumber);
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < chunkNumber; i++) {
        const size_t idTrgLow = i * chunkSize;
        const size_t idTrgHigh = (i + 1 < chunkNumber) ? idTrgLow + chunkSize : nTrg; // not inclusive
        //        printf("i, idTrgLow, idTrgHigh: %d, %d, %d\n", i, idTrgLow, idTrgHigh);
        kernelG.k_l2t->ker_poten(pointLEquiv.data(), equivN, M2Lsource.data(), 1, &(trgCoord[3 * idTrgLow]),
                                 idTrgHigh - idTrgLow, &(trgValueG[3 * idTrgLow]), NULL);
    }
}

void RPYFMM_Wrapper::calcMRPY() {
    pvfmm::Vector<double> v = treePtrRPY->RootNode()->FMMData()->upward_equiv; // the value calculated by pvfmm
    assert(v.Dim() == 3 * this->equivN);

    // add to trg_value
    auto &trgCoord = treeDataRPY.trg_coord;
    const size_t nTrg = trgCoord.Dim() / 3;

    int M = 3 * equivN;
    int N = 3 * equivN; // checkN = equivN in this code.
    M2Lsource.resize(3 * equivN);
    assert(M2Lsource.size() == v.Dim());

#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double temp = 0;
#pragma omp simd
        for (int j = 0; j < N; j++) {
            temp += pm2lRPY[i * N + j] * v[j];
        }
        M2Lsource[i] = temp;
    }

    // cancel the net flow
    double ux = 0, uy = 0, uz = 0;
    for (int i = 0; i < equivN; i++) {
        ux += v[3 * i];
        uy += v[3 * i + 1];
        uz += v[3 * i + 2];
    }
    //    printf("%f,%f,%f\n", ux, uy, uz);
    const pvfmm::Kernel<double> &kernelRPY = pvfmm::RPYCustomKernel<double>::velocity();

    const size_t chunkSize = 2000; // each chunk has 2000 target points.
    const size_t chunkNumber = floor(nTrg / chunkSize) + 1;
    printf("chunkSize, chunkNumber: %d, %d\n", chunkSize, chunkNumber);
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < chunkNumber; i++) {
        const size_t idTrgLow = i * chunkSize;
        const size_t idTrgHigh = (i + 1 < chunkNumber) ? idTrgLow + chunkSize : nTrg; // not inclusive
        //        printf("i, idTrgLow, idTrgHigh: %d, %d, %d\n", i, idTrgLow, idTrgHigh);
        kernelRPY.k_l2t->ker_poten(pointLEquiv.data(), equivN, M2Lsource.data(), 1, &(trgCoord[3 * idTrgLow]),
                                   idTrgHigh - idTrgLow, &(trgValueRPY[3 * idTrgLow]), NULL);
    }

    // // cancel net flow
    // #pragma omp parallel for
    //     for (size_t i = 0; i < nTrg; i++) {
    //         trgValueRPY[3 * i] += ux;
    //         trgValueRPY[3 * i + 1] += uy;
    //         trgValueRPY[3 * i + 2] += uz;
    //     }
}
