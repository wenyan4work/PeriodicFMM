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

#include "Util/Timer.hpp"

class FMM_WrapperWall2D {
  public:
    enum PAXIS { NONE, PXY };

    const PAXIS pbc;

    FMM_WrapperWall2D(int mult_order = 10, int max_pts = 2000, int init_depth = 0, PAXIS pbc_ = PAXIS::NONE);

    ~FMM_WrapperWall2D();

    void FMM_TreeClear();

    void FMM_DataClear();

    void FMM_Evaluate(std::vector<double> &, const int, std::vector<double> *,
                      std::vector<double> *surf_coordPtr = NULL);
    void FMM_UpdateTree(const std::vector<double> &, const std::vector<double> &,
                        const std::vector<double> *surf_valPtr = NULL);

    void FMM_SetBox(double, double, double, double, double, double);

  private:
    Timer myTimer;
    double xlow, xhigh; // box
    double ylow, yhigh;
    double zlow, zhigh;

    // for Stokes FMM
    pvfmm::PtFMM matrixStokes;
    pvfmm::PtFMM_Tree *treePtrStokes;
    pvfmm::PtFMM_Data treeDataStokes;

    // for Laplace Charge FMM
    pvfmm::PtFMM matrixLapChargeF;
    pvfmm::PtFMM_Tree *treePtrLapChargeF;
    pvfmm::PtFMM_Data treeDataLapChargeF;

    // for Laplace Charge FMM
    pvfmm::PtFMM matrixLapChargeFZ;
    pvfmm::PtFMM_Tree *treePtrLapChargeFZ;
    pvfmm::PtFMM_Data treeDataLapChargeFZ;

    // for Laplace Dipole FMM
    pvfmm::PtFMM matrixLapDipole;
    pvfmm::PtFMM_Tree *treePtrLapDipole;
    pvfmm::PtFMM_Data treeDataLapDipole;

    const int mult_order;
    const int max_pts;
    const int init_depth;

    double scaleFactor;
    double xshift, yshift, zshift;

    double *readM2LMat(const char *, const int, int);

    double *pm2lStokes;    // the periodizing operator for Stokes kernel
    double *pm2lLapCharge; // the periodizing operator for Laplace Charge kernel
    double *pm2lLapDipole; // the periodizing operator for Laplace Dipole kernel

    void calcMStokes();

    void calcMLapCharge(int); // 0 for F, 1 for FZ

    void calcMLapDipole();

    std::vector<double> srcValueStokes;
    std::vector<double> trgValueStokes; // stokes value only

    std::vector<double> srcValueLapDipole; // dipole value and gradient
    std::vector<double> trgValueLapDipole; // dipole value and gradient

    std::vector<double> srcValueLapChargeF; // charge F value and gradient
    std::vector<double> trgValueLapChargeF; // charge F value and gradient

    std::vector<double> srcValueLapChargeFZ; // charge FZ value and gradient
    std::vector<double> trgValueLapChargeFZ; // charge FZ value and gradient

    int pEquiv;
    int equivN;
    double scaleLEquiv;            // = 1.05;
    double pCenterLEquiv[3];       // = { -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2 };
    std::vector<double> M2Lsource; // the equivalent sources after the operation

    std::vector<double> pointLEquiv;
    // = surface(pEquiv, (double *) &(pCenterLCheck[0]), scaleLCheck, 0);
    // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    void treeSetup(pvfmm::PtFMM_Data &, const std::vector<double> &, const std::vector<double> &, bool);

    void treePointDump(const pvfmm::PtFMM_Data &);

    void sumImageSystem(std::vector<double> &);

    // small private functions for timing
    void treeStokes(const std::vector<double> &, const std::vector<double> &);
    void treeLapDipole(const std::vector<double> &, const std::vector<double> &);
    void treeLapChargeF(const std::vector<double> &, const std::vector<double> &);
    void treeLapChargeFZ(const std::vector<double> &, const std::vector<double> &);

    void evalStokes(std::vector<double> &, const int, std::vector<double> *);
    void evalLapDipole(std::vector<double> &, const int, std::vector<double> *);
    void evalLapChargeF(std::vector<double> &, const int, std::vector<double> *);
    void evalLapChargeFZ(std::vector<double> &, const int, std::vector<double> *);
};

#endif /* INCLUDE_FMMWRAPPER_H_ */
