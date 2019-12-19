/*
 * FMMWrapper.h
 *
 *  Created on: Oct 6, 2016
 *      Author: wyan
 */

#ifndef INCLUDE_FMMWRAPPERWALL2D_H_
#define INCLUDE_FMMWRAPPERWALL2D_H_

// a wrapper for pvfmm
// choose kernel at compile time
#include <mpi.h>
#include <pvfmm.hpp>

#include "Timer.hpp"

class FMM_WrapperWall2D {
  public:
    enum PAXIS { NONE = 0, PX = 1, PXY = 4 };

    const PAXIS pbc;

    FMM_WrapperWall2D(int mult_order = 10, int max_pts = 2000,
                      int init_depth = 0, PAXIS pbc_ = PAXIS::NONE);

    ~FMM_WrapperWall2D();

    void FMM_TreeClear();

    void FMM_DataClear();

    void FMM_Evaluate(std::vector<double> &trg_val, const size_t n_trg,
                      std::vector<double> *src_val);

    void FMM_UpdateTree(const std::vector<double> &src_coord,
                        const std::vector<double> &trg_coord);

    void FMM_SetBox(double, double, double, double, double, double);

  private:
    double xlow, xhigh; // box
    double ylow, yhigh;
    double zlow, zhigh;

    // for Stokes FMM
    pvfmm::PtFMM<double> matrixStokes;
    pvfmm::PtFMM_Tree<double> *treePtrStokes;
    pvfmm::PtFMM_Data<double> treeDataStokes;

    // for Laplace Charge FMM
    pvfmm::PtFMM<double> matrixLapCharge;
    pvfmm::PtFMM_Tree<double> *treePtrLapCharge;
    pvfmm::PtFMM_Data<double> treeDataLapCharge;

    // for Laplace Charge + Dipole FMM
    pvfmm::PtFMM<double> matrixLapDipole;
    pvfmm::PtFMM_Tree<double> *treePtrLapDipole;
    pvfmm::PtFMM_Data<double> treeDataLapDipole;

    const int mult_order;
    const int max_pts;
    const int init_depth;

    double scaleFactor;
    double xshift, yshift, zshift;

    double *readM2LMat(const char *, const int, int);

    double *pm2lStokes;    // the periodizing operator for Stokes kernel
    double *pm2lLapCharge; // the periodizing operator for Laplace Charge kernel

    void calcMStokes();

    void calcMLapCharge(int); // 0 for charge, 1 for dipole

    std::vector<double> srcValueStokes;
    std::vector<double> trgValueStokes; // stokes value only

    std::vector<double> srcValueLapCharge; // charge F value and gradient
    std::vector<double> trgValueLapCharge; // charge F value and gradient

    std::vector<double> srcValueLapDipoleSL; // dipole value and gradient
    std::vector<double> srcValueLapDipoleDL; // dipole value and gradient
    std::vector<double> trgValueLapDipole;   // dipole value and gradient

    std::vector<double> srcCoordScaled;
    std::vector<double> srcImageCoordScaled;
    std::vector<double> trgCoordScaled;

    int pEquiv;
    size_t equivN;
    double scaleLEquiv;      // = 1.05;
    double pCenterLEquiv[3]; // = { -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) /
                             // 2, -(scaleLEquiv - 1) / 2 };
    std::vector<double> M2Lsource; // the equivalent sources after the operation

    std::vector<double> pointLEquiv;
    Timer myTimer;

    // = surface(pEquiv, (double *) &(pCenterLCheck[0]), scaleLCheck, 0);
    // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    void scalePoints(const std::vector<double> &srcCoord,
                     const std::vector<double> &trgCoord);

    void treeSetupStokes(); // SL only
    void treeSetupDipole(); // SL+DL
    void treeSetupCharge(); // SL only

    void treePointDump(const pvfmm::PtFMM_Data<double> &);

    void sumImageSystem(std::vector<double> &);

    // small private functions for timing
    void treeStokes();
    void treeLapDipole();
    void treeLapCharge();

    void evalStokes(const int, std::vector<double> *);
    void evalLapDipole(const int, std::vector<double> *);
    void evalLapCharge(const int, std::vector<double> *);
};

#endif /* INCLUDE_FMMWRAPPER_H_ */
