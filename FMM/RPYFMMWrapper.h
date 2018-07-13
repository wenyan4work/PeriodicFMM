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
#include "mpi.h"
#include "pvfmm.hpp"

#include "Eigen/Dense"

class RPYFMM_Wrapper {
  public:
    enum PAXIS { NONE, PXYZ, PX, PY, PZ, PXY, PXZ, PYZ };

    enum KERNEL { G, GRPY };

    RPYFMM_Wrapper(int mult_order = 10, int max_pts = 500, int init_depth = 0, PAXIS pbc_ = PAXIS::NONE,
                   KERNEL kernel_ = KERNEL::G);

    ~RPYFMM_Wrapper();

    void FMM_TreeClear();

    void FMM_DataClear();

    void FMM_EvaluateG(std::vector<double> &, const int, const std::vector<double> &);

    void FMM_EvaluateRPY(std::vector<double> &, const int, const std::vector<double> &);

    void FMM_UpdateTree(const std::vector<double> &, const std::vector<double> &);

    void FMM_SetBox(double, double, double, double, double, double);

  private:
    PAXIS pbc;
    KERNEL kernel;

    pvfmm::PtFMM matrixG;
    pvfmm::PtFMM_Tree *treePtrG;
    pvfmm::PtFMM_Data treeDataG;

    pvfmm::PtFMM matrixRPY;
    pvfmm::PtFMM_Tree *treePtrRPY;
    pvfmm::PtFMM_Data treeDataRPY;

    const int mult_order;
    const int max_pts;
    const int init_depth;

    double xlow, xhigh; // box
    double ylow, yhigh;
    double zlow, zhigh;
    double scaleFactor;
    double xshift, yshift, zshift;

    std::vector<double> srcCoord; // scaled coordinate
    std::vector<double> trgCoord;
    std::vector<double> srcValueG; // scaled value
    std::vector<double> trgValueG;
    std::vector<double> srcValueRPY; // scaled value
    std::vector<double> trgValueRPY;

    void setupCoord(const std::vector<double> &, std::vector<double> &) const; // setup the internal srcCoord and
                                                                               // trgCoord, with proper rotation and BC

    double *readM2LMat(const char *, const int); // choose to read G or RPY data

    double *pm2lG; // the periodizing operator for G

    double *pm2lRPY; // operator data for RPY

    void calcMG();

    void calcMRPY();

    void rotateValue(const std::vector<double> &, std::vector<double> &) const;

    void rotateBackValue(const std::vector<double> &, std::vector<double> &) const;

    template <class T>
    void safeDeletePtr(T *ptr) {
        if (ptr != nullptr) {
            delete ptr;
            ptr = nullptr;
        }
    }

    int pEquiv;
    int equivN;
    double scaleLEquiv;            // = 1.05;
    double pCenterLEquiv[3];       // = { -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2 };
    std::vector<double> M2Lsource; // the equivalent sources after the operation

    std::vector<double> pointLEquiv; // = surface(pEquiv, (double *) &(pCenterLCheck[0]), scaleLCheck, 0); // center at
                                     // 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
};

template <class Real_t>
std::vector<Real_t> surface(int, Real_t *, Real_t, int);

#endif /* INCLUDE_FMMWRAPPER_H_ */
