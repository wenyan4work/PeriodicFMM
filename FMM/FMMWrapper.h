/*
 * FMMWrapper.h
 *
 *  Created on: Oct 6, 2016
 *      Author: wyan
 */

#ifndef INCLUDE_FMMWRAPPER_H_
#define INCLUDE_FMMWRAPPER_H_

// a wrapper for pvfmm
// choose kernel at compile time
#include <mpi.h>
#include <pvfmm.hpp>

#include "Util/Timer.hpp"

class FMM_Wrapper {
  public:
    enum PAXIS { NONE = 0, PX = 1, PY = 2, PZ = 3, PXY = 4, PXZ = 5, PYZ = 6, PXYZ = 7 };

    const PAXIS pbc;

    FMM_Wrapper(int mult_order = 10, int max_pts = 2000, int init_depth = 0, PAXIS pbc_ = PAXIS::NONE);

    ~FMM_Wrapper();

    void FMM_TreeClear();

    void FMM_DataClear();

    void FMM_Evaluate(std::vector<double> &trg_val, const int n_trg, std::vector<double> *src_val);

    void FMM_UpdateTree(const std::vector<double> &src_coord, const std::vector<double> &trg_coord);

    void FMM_SetBox(double, double, double, double, double, double);

  private:
    Timer myTimer;
    double xlow, xhigh; // box
    double ylow, yhigh;
    double zlow, zhigh;

    // for Stokes FMM
    pvfmm::PtFMM matrix;
    pvfmm::PtFMM_Tree *treePtr;
    pvfmm::PtFMM_Data treeData;

    double scaleFactor;
    double xshift, yshift, zshift;

    const int mult_order;
    const int max_pts;
    const int init_depth;

    double *readM2LMat(const char *, const int);

    double *pm2l; // the periodizing operator

    void calcMStokes(std::vector<double> &trgValue);

    int pEquiv;
    int equivN;
    double scaleLEquiv;            // = 1.05;
    double pCenterLEquiv[3];       // = { -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2 };
    std::vector<double> M2Lsource; // the equivalent sources after the operation
    std::vector<double> pointLEquiv;
    // = surface(pEquiv, (double *) &(pCenterLCheck[0]), scaleLCheck, 0);
    // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
};

#endif /* INCLUDE_FMMWRAPPER_H_ */
