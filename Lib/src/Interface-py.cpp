/**
 * @file Interface-py.cpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-12-20
 *  Migrated from Floren's Boost.python version to pybind11 version
 * @copyright Copyright (c) 2019
 *
 */
#include "PeriodicFMM/FMMWrapper.hpp"
#include "PeriodicFMM/FMMWrapperWall2D.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(PyPeriodicFMM, m) {
    py::class_<FMM_Wrapper>(m, "FMM_Wrapper")
        .def(py::init<int, int, int, FMM_Wrapper::PAXIS, bool>())
        .def("FMM_SetBox", &FMM_Wrapper::FMM_SetBox)
        .def("FMM_UpdateTree", &FMM_Wrapper::FMM_UpdateTree)
        .def("FMM_Evaluate", &FMM_Wrapper::FMM_Evaluate)
        .def("FMM_TreeClear", &FMM_Wrapper::FMM_TreeClear)
        .def("FMM_DataClear", &FMM_Wrapper::FMM_DataClear);

    py::class_<FMM_WrapperWall2D>(m, "FMM_WrapperWall2D")
        .def(py::init<int, int, int, FMM_WrapperWall2D::PAXIS>())
        .def("FMM_SetBox", &FMM_WrapperWall2D::FMM_SetBox)
        .def("FMM_UpdateTree", &FMM_WrapperWall2D::FMM_UpdateTree)
        .def("FMM_Evaluate", &FMM_WrapperWall2D::FMM_Evaluate)
        .def("FMM_TreeClear", &FMM_WrapperWall2D::FMM_TreeClear)
        .def("FMM_DataClear", &FMM_WrapperWall2D::FMM_DataClear);
}

// void FMM_SetBox(FMM_Wrapper *fmm, double xlow, double xhigh, double ylow,
//                 double yhigh, double zlow, double zhigh) {
//     fmm->FMM_SetBox(xlow, xhigh, ylow, yhigh, zlow, zhigh);
// }

// void FMM_UpdateTree(FMM_Wrapper *fmm, np::ndarray trg_coord,
//                     np::ndarray src_coord) {

//     // Transform ndarray to std::vectors
//     int num_trg = trg_coord.shape(0);
//     double *trg_iter = reinterpret_cast<double *>(trg_coord.get_data());
//     std::vector<double> trg_coord_vec(trg_iter, trg_iter + num_trg * 3);
//     int num_src = src_coord.shape(0);
//     double *src_iter = reinterpret_cast<double *>(src_coord.get_data());
//     std::vector<double> src_coord_vec(src_iter, src_iter + num_src * 3);

//     // Call method
//     fmm->FMM_UpdateTree(src_coord_vec, trg_coord_vec);

//     return;
// }

// void FMM_Evaluate(FMM_Wrapper *fmm, np::ndarray trg_value,
//                   np::ndarray src_value) {

//     // Transform ndarray to std::vectors
//     int num_trg = trg_value.shape(0);
//     std::vector<double> trg_value_vec(num_trg * 3);
//     int num_src = src_value.shape(0);
//     double *src_iter = reinterpret_cast<double *>(src_value.get_data());
//     std::vector<double> src_value_vec(src_iter, src_iter + num_src * 3);

//     // Call method
//     fmm->FMM_Evaluate(trg_value_vec, num_trg, &src_value_vec);

//     // Copy std::vector to ndarray
//     std::copy(trg_value_vec.begin(), trg_value_vec.end(),
//               reinterpret_cast<double *>(trg_value.get_data()));

//     return;
// }

// void FMM_TreeClear(FMM_Wrapper *fmm) { fmm->FMM_TreeClear(); }

// void FMM_DataClear(FMM_Wrapper *fmm) { fmm->FMM_DataClear(); }

// void FMMWall2D_SetBox(FMM_WrapperWall2D *fmm, double xlow, double xhigh,
//                       double ylow, double yhigh, double zlow, double zhigh) {
//     fmm->FMM_SetBox(xlow, xhigh, ylow, yhigh, zlow, zhigh);
// }

// void FMMWall2D_UpdateTree(FMM_WrapperWall2D *fmm, np::ndarray trg_coord,
//                           np::ndarray src_coord) {

//     // Transform ndarray to std::vectors
//     int num_trg = trg_coord.shape(0);
//     double *trg_iter = reinterpret_cast<double *>(trg_coord.get_data());
//     std::vector<double> trg_coord_vec(trg_iter, trg_iter + num_trg * 3);
//     int num_src = src_coord.shape(0);
//     double *src_iter = reinterpret_cast<double *>(src_coord.get_data());
//     std::vector<double> src_coord_vec(src_iter, src_iter + num_src * 3);

//     // Call method
//     fmm->FMM_UpdateTree(src_coord_vec, trg_coord_vec);

//     return;
// }

// void FMMWall2D_Evaluate(FMM_WrapperWall2D *fmm, np::ndarray trg_value,
//                         np::ndarray src_value) {

//     // Transform ndarray to std::vectors
//     int num_trg = trg_value.shape(0);
//     std::vector<double> trg_value_vec(num_trg * 3);
//     int num_src = src_value.shape(0);
//     double *src_iter = reinterpret_cast<double *>(src_value.get_data());
//     std::vector<double> src_value_vec(src_iter, src_iter + num_src * 3);

//     // Call method
//     fmm->FMM_Evaluate(trg_value_vec, num_trg, &src_value_vec);

//     // Copy std::vector to ndarray
//     std::copy(trg_value_vec.begin(), trg_value_vec.end(),
//               reinterpret_cast<double *>(trg_value.get_data()));

//     return;
// }

// void FMMWall2D_TreeClear(FMM_WrapperWall2D *fmm) { fmm->FMM_TreeClear(); }

// void FMMWall2D_DataClear(FMM_WrapperWall2D *fmm) { fmm->FMM_DataClear(); }

// BOOST_PYTHON_MODULE(PyPeriodicFMM) {
//     using namespace boost::python;

//     // Initialize numpy
//     Py_Initialize();
//     np::initialize();

//     // Class FMM_Wrapper
//     enum_<FMM_Wrapper::PAXIS>("FMM_PAXIS")
//         .value("NONE", FMM_Wrapper::NONE)
//         .value("PX", FMM_Wrapper::PX)
//         .value("PXY", FMM_Wrapper::PXY)
//         .value("PXYZ", FMM_Wrapper::PXYZ);
//     class_<FMM_Wrapper>("FMM_Wrapper",
//                         init<int, int, int, FMM_Wrapper::PAXIS>());

//     // Class FMM_WrapperWall2D
//     enum_<FMM_WrapperWall2D::PAXIS>("FMMWall2D_PAXIS")
//         .value("NONE", FMM_WrapperWall2D::NONE)
//         .value("PX", FMM_WrapperWall2D::PX)
//         .value("PXY", FMM_WrapperWall2D::PXY);
//     class_<FMM_WrapperWall2D>("FMM_WrapperWall2D",
//                               init<int, int, int,
//                               FMM_WrapperWall2D::PAXIS>());

//     // Define functions for FMM_Wrapper
//     def("FMM_SetBox", FMM_SetBox);
//     def("FMM_UpdateTree", FMM_UpdateTree);
//     def("FMM_Evaluate", FMM_Evaluate);
//     def("FMM_TreeClear", FMM_TreeClear);
//     def("FMM_DataClear", FMM_DataClear);

//     // Define functions for FMMWall2D_Wrapper
//     def("FMMWall2D_SetBox", FMMWall2D_SetBox);
//     def("FMMWall2D_UpdateTree", FMMWall2D_UpdateTree);
//     def("FMMWall2D_Evaluate", FMMWall2D_Evaluate);
//     def("FMMWall2D_TreeClear", FMMWall2D_TreeClear);
//     def("FMMWall2D_DataClear", FMMWall2D_DataClear);
// }
