/*
 * main.cpp
 *
 *  Created on: Oct 14, 2016
 *      Author: wyan
 */

#include "ChebNodal.h"
#include "Ewald.hpp"
#include "FMM/FMMWrapper.h"
#include "Util/cmdparser.hpp"

#include <chrono>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

#include <Eigen/Dense>

void distributePts(std::vector<double> &pts) {
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    int ptsGlobalSize;
    if (myRank == 0) {
        ptsGlobalSize = pts.size();
        MPI_Bcast(&ptsGlobalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // std::cout << "rank " << myRank << " global size" << ptsGlobalSize << std::endl;
    } else {
        ptsGlobalSize = 0;
        MPI_Bcast(&ptsGlobalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // std::cout << "rank " << myRank << " global size" << ptsGlobalSize << std::endl;
    }

    // bcast to all
    pts.resize(ptsGlobalSize);
    MPI_Bcast(pts.data(), ptsGlobalSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // each take a portion
    const int nPts = ptsGlobalSize / 3;
    // inclusive low
    int indexlow = 3 * floor(myRank * nPts / static_cast<double>(nProcs));
    // non-inclusive high
    int indexhigh = 3 * floor((myRank + 1) * nPts / static_cast<double>(nProcs));
    if (myRank == nProcs - 1) {
        indexhigh = ptsGlobalSize;
    }
    std::vector<double>::const_iterator first = pts.begin() + indexlow;
    std::vector<double>::const_iterator last = pts.begin() + indexhigh;
    std::vector<double> newVec(first, last);
    pts = std::move(newVec);
}

void collectPts(std::vector<double> &pts) {

    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    int ptsLocalSize = pts.size();
    int ptsGlobalSize = 0;

    std::vector<int> recvSize(0);
    std::vector<int> displs(0);
    if (myRank == 0) {
        recvSize.resize(nProcs);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&ptsLocalSize, 1, MPI_INT, recvSize.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    // void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    // MPI_Comm comm)
    for (auto &p : recvSize) {
        ptsGlobalSize += p;
    }
    // std::cout << "rank " << myRank << " globalSize " << ptsGlobalSize << std::endl;
    displs.resize(recvSize.size());
    if (displs.size() > 0) {
        displs[0] = 0;
        for (int i = 1; i < displs.size(); i++) {
            displs[i] = recvSize[i - 1] + displs[i - 1];
        }
    }

    // int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    // void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
    // int root, MPI_Comm comm)
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> ptsRecv(ptsGlobalSize); // size=0 on rank !=0
    // std::cout << "globalSize " << ptsGlobalSize << std::endl;
    if (myRank == 0) {
        MPI_Gatherv(pts.data(), pts.size(), MPI_DOUBLE, ptsRecv.data(), recvSize.data(), displs.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(pts.data(), pts.size(), MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    pts = std::move(ptsRecv);
}

class Trapz { // map from (-1,1)
  public:
    Trapz(int N_) : N(N_) {
        points.resize(N_ + 1);
        weights.resize(N_ + 1);

        for (int i = 0; i < N_ + 1; i++) {
            points[i] = -1 + 2.0 * i / N_;
            weights[i] = 2.0 / N_;
        }
        weights[0] *= 0.5;
        weights.back() *= 0.5;
    }
    int N;
    std::vector<double> points;
    std::vector<double> weights;
};

void calcTrueValueFMM(std::vector<double> &trg_value_true, const std::vector<double> &trg_coord,
                      std::vector<double> &src_value, const std::vector<double> &src_coord, const double box,
                      const double shift, const FMM_Wrapper::PAXIS pset) {
    std::cout << "***************************************************" << std::endl;
    std::cout << "Skip O(N^2) true value calculation for large system" << std::endl;
    std::cout << "Use FMM p=16 as 'true' value                       " << std::endl;
    std::cout << "***************************************************" << std::endl;
    FMM_Wrapper myFMM(16, 1000, 0, pset);
    myFMM.FMM_SetBox(shift, shift + box, shift, shift + box, shift, shift + box);
    myFMM.FMM_UpdateTree(src_coord, trg_coord);
    myFMM.FMM_Evaluate(trg_value_true, trg_coord.size() / 3, &src_value);
    return;
}

void calcTrueValueN2(std::vector<double> &trg_value_true, const std::vector<double> &trg_coord,
                     std::vector<double> &src_value, const std::vector<double> &src_coord, const double box,
                     const double shift, const FMM_Wrapper::PAXIS pset) {

    // calc Ewald accuracy test
    trg_value_true.resize(trg_coord.size());
#pragma omp parallel for
    for (int t = 0; t < trg_coord.size() / 3; t++) {
        Eigen::Vector3d target(trg_coord[3 * t], trg_coord[3 * t + 1], trg_coord[3 * t + 2]);
        // shift and rotate to [0,1)
        target[0] -= shift;
        target[1] -= shift;
        target[2] -= shift;
        target *= (1 / box);
        if (pset == FMM_Wrapper::PAXIS::PX || pset == FMM_Wrapper::PAXIS::PYZ) {
            Eigen::Vector3d temp = target;
            target[0] = temp[1];
            target[1] = temp[2];
            target[2] = temp[0];
        } else if (pset == FMM_Wrapper::PAXIS::PY || pset == FMM_Wrapper::PAXIS::PXZ) {
            Eigen::Vector3d temp = target;
            target[0] = temp[2];
            target[1] = temp[0];
            target[2] = temp[1];
        }

        Eigen::Vector3d targetValue(0, 0, 0);
        for (int s = 0; s < src_coord.size() / 3; s++) {
            Eigen::Vector3d source(src_coord[3 * s], src_coord[3 * s + 1], src_coord[3 * s + 2]);
            Eigen::Vector3d sourceValue(src_value[3 * s], src_value[3 * s + 1], src_value[3 * s + 2]);
            // shift and rotate to [0,1)
            source[0] -= shift;
            source[1] -= shift;
            source[2] -= shift;
            source *= (1 / box);
            if (pset == FMM_Wrapper::PAXIS::PX || pset == FMM_Wrapper::PAXIS::PYZ) {
                Eigen::Vector3d temp = source;
                source[0] = temp[1];
                source[1] = temp[2];
                source[2] = temp[0];
                temp = sourceValue;
                sourceValue[0] = temp[1];
                sourceValue[1] = temp[2];
                sourceValue[2] = temp[0];
            } else if (pset == FMM_Wrapper::PAXIS::PY || pset == FMM_Wrapper::PAXIS::PXZ) {
                Eigen::Vector3d temp = source;
                source[0] = temp[2];
                source[1] = temp[0];
                source[2] = temp[1];
                temp = sourceValue;
                sourceValue[0] = temp[2];
                sourceValue[1] = temp[0];
                sourceValue[2] = temp[1];
            }

            Eigen::Vector3d rst = target - source;
            Eigen::Matrix3d G;
            if (pset == FMM_Wrapper::PAXIS::PXYZ) {
                GkernelEwald3D(rst, G, 1.0);
            } else if (pset == FMM_Wrapper::PAXIS::PXY || pset == FMM_Wrapper::PAXIS::PXZ ||
                       pset == FMM_Wrapper::PAXIS::PYZ) {
                GkernelEwald2D(rst, G); // default box = 1
            } else if (pset == FMM_Wrapper::PAXIS::PX || pset == FMM_Wrapper::PAXIS::PY ||
                       pset == FMM_Wrapper::PAXIS::PZ) {
                Gkernel1D(rst, G); // default box =1
            } else if (pset == FMM_Wrapper::PAXIS::NONE) {
                Gkernel(rst, Eigen::Vector3d(0, 0, 0), G);
            }
            targetValue += (G * sourceValue) / (8 * PI314);
        }
        // rotate and scale back
        if (pset == FMM_Wrapper::PAXIS::PX || pset == FMM_Wrapper::PAXIS::PYZ) {
            trg_value_true[3 * t] = targetValue[2] / box;
            trg_value_true[3 * t + 1] = targetValue[0] / box;
            trg_value_true[3 * t + 2] = targetValue[1] / box;
        } else if (pset == FMM_Wrapper::PAXIS::PY || pset == FMM_Wrapper::PAXIS::PXZ) {
            trg_value_true[3 * t] = targetValue[1] / box;
            trg_value_true[3 * t + 1] = targetValue[2] / box;
            trg_value_true[3 * t + 2] = targetValue[0] / box;
        } else {
            trg_value_true[3 * t] = targetValue[0] / box;
            trg_value_true[3 * t + 1] = targetValue[1] / box;
            trg_value_true[3 * t + 2] = targetValue[2] / box;
        }
    }

    //      std::cout << "-------------true value---------------" << std::endl;
    //  for (auto &t : trg_value_true) {
    //      std::cout << t << std::endl;
    //  }
    //      std::cout << "-------------true value end-----------" << std::endl;
}

void initPts(std::vector<double> &src_coord, std::vector<double> &src_value, std::vector<double> &trg_coord,
             std::vector<double> &trg_value, const cli::Parser &parser) {

    // initialize source and target coord and value
    bool randomS = (parser.get<int>("R") > 0 ? true : false);
    const int ntrgEdge = parser.get<int>("T");
    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");

    FMM_Wrapper::PAXIS pset;
    int pinput = parser.get<int>("P");
    switch (pinput) {
    case 0:
        pset = FMM_Wrapper::PAXIS::NONE;
        break;
    case 1:
        pset = FMM_Wrapper::PAXIS::PX;
        break;
    case 2:
        pset = FMM_Wrapper::PAXIS::PY;
        break;
    case 3:
        pset = FMM_Wrapper::PAXIS::PZ;
        break;
    case 4:
        pset = FMM_Wrapper::PAXIS::PXY;
        break;
    case 5:
        pset = FMM_Wrapper::PAXIS::PXZ;
        break;
    case 6:
        pset = FMM_Wrapper::PAXIS::PYZ;
        break;
    case 7:
        pset = FMM_Wrapper::PAXIS::PXYZ;
        break;
    }

    const int chebN = ntrgEdge;
    ChebNodal chebData(chebN);
    if (pset != FMM_Wrapper::PAXIS::PXYZ) {
        chebData.points[0] += 0;
        chebData.points.back() -= 1e-13; // prevent PVFMM crash with point located at the edge
    }
    //	Trapz chebData(chebN);

    // for (auto &v : chebData.points) {
    //     std::cout << std::setprecision(10) << v << std::endl;
    // }
    const int dimension = chebData.points.size();

    // initialize target points
    if (randomS == true) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::lognormal_distribution<> d(log(0.2), 0.5);

        for (int i = 0; i < ntrgEdge + 1; i++) {
            for (int j = 0; j < ntrgEdge + 1; j++) {
                for (int k = 0; k < ntrgEdge + 1; k++) {
                    double r01 = fmod(d(gen), 1.0);
                    trg_coord.push_back(r01 * box + shift); // x
                    r01 = fmod(d(gen), 1.0);
                    trg_coord.push_back(r01 * box + shift); // y
                    r01 = fmod(d(gen), 1.0);
                    trg_coord.push_back(r01 * box + shift); // z
                    trg_value.push_back(drand48() - 0.5);   // x
                    trg_value.push_back(drand48() - 0.5);   // y
                    trg_value.push_back(drand48() - 0.5);   // z
                }
            }
        }

        FILE *pfile = fopen("randomPoints", "w");
        for (int i = 0; i < trg_coord.size() / 3; i++) {
            fprintf(pfile, "%f\t%f\t%f\n", trg_coord[3 * i], trg_coord[3 * i + 1], trg_coord[3 * i + 2]);
        }
        fclose(pfile);
    } else {

        std::vector<double> &chebMesh = trg_coord;
        std::vector<double> &chebValue = trg_value;
        chebMesh.resize(pow(dimension, 3) * 3);
        chebValue.resize(pow(dimension, 3) * 3);

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                for (int k = 0; k < dimension; k++) {
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k)] =
                        (chebData.points[i] + 1) * box / 2 + shift;
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k) + 1] =
                        (chebData.points[j] + 1) * box / 2 + shift;
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k) + 2] =
                        (chebData.points[k] + 1) * box / 2 + shift;

                    chebValue[3 * (i * dimension * dimension + j * dimension + k)] =
                        (drand48() - 0.5); // RNG on [-0.5,0.5]
                    chebValue[3 * (i * dimension * dimension + j * dimension + k) + 1] =
                        (drand48() - 0.5); // RNG on [-0.5,0.5]
                    chebValue[3 * (i * dimension * dimension + j * dimension + k) + 2] =
                        (drand48() - 0.5); // RNG on [-0.5,0.5]
                }
            }
        }

        FILE *pfile = fopen("chebPoints", "w");
        for (int i = 0; i < chebMesh.size() / 3; i++) {
            fprintf(pfile, "%f\t%f\t%f\n", chebMesh[3 * i], chebMesh[3 * i + 1], chebMesh[3 * i + 2]);
        }
        fclose(pfile);
    }
    // initialize source points
    int nsource = parser.get<int>("S");
    switch (nsource) {
    case 1: {
        src_coord.push_back(0.7 * box + shift);
        src_coord.push_back(0.6 * box + shift);
        src_coord.push_back(0.4 * box + shift);
        src_value.push_back(1.0 / sqrt(14.0));
        src_value.push_back(2.0 / sqrt(14.0));
        src_value.push_back(3.0 / sqrt(14.0));
        for (auto &s : src_coord) {
            std::cout << "src_coord" << s << std::endl;
        }
        for (auto &s : src_value) {
            std::cout << "src_value" << s << std::endl;
        }
    } break;
    case 2: {
        src_coord.push_back(0.7 * box + shift); // 1
        src_coord.push_back(0.6 * box + shift);
        src_coord.push_back(0.5 * box + shift);
        src_coord.push_back(0.2 * box + shift); // 2
        src_coord.push_back(0.8 * box + shift);
        src_coord.push_back(0.7 * box + shift);
        src_value.push_back(1 / sqrt(14.0));
        src_value.push_back(2 / sqrt(14.0));
        src_value.push_back(3 / sqrt(14.0));
        src_value.push_back(-src_value[0]);
        src_value.push_back(-src_value[1]);
        src_value.push_back(-src_value[2]);
        for (auto &s : src_coord) {
            std::cout << "src_coord" << s << std::endl;
        }
        for (auto &s : src_value) {
            std::cout << "src_value" << s << std::endl;
        }
    } break;
    case 4: {                                   // quadrupole, no dipole
        src_coord.push_back(0.1 * box + shift); // 1
        src_coord.push_back(0.1 * box + shift);
        src_coord.push_back(0.1 * box + shift);
        src_coord.push_back(0.2 * box + shift); // 2
        src_coord.push_back(0.2 * box + shift);
        src_coord.push_back(0.2 * box + shift);
        src_coord.push_back(0.3 * box + shift); // 3
        src_coord.push_back(0.3 * box + shift);
        src_coord.push_back(0.3 * box + shift);
        src_coord.push_back(0.4 * box + shift); // 4
        src_coord.push_back(0.4 * box + shift);
        src_coord.push_back(0.4 * box + shift);
        const double f = 1 / sqrt(3);
        src_value.push_back(f); // 1
        src_value.push_back(f);
        src_value.push_back(f);
        src_value.push_back(-f); // 2
        src_value.push_back(-f);
        src_value.push_back(-f);
        src_value.push_back(-f); // 2
        src_value.push_back(-f);
        src_value.push_back(-f);
        src_value.push_back(f); // 4
        src_value.push_back(f);
        src_value.push_back(f);
        for (auto &s : src_coord) {
            std::cout << "src_coord" << s << std::endl;
        }
        for (auto &s : src_value) {
            std::cout << "src_value" << s << std::endl;
        }
    } break;

    default: {
        src_coord = trg_coord;
        src_value = trg_value;
    } break;
    }
    // enforce net charge for 1P and 2P
    if (pset == FMM_Wrapper::PAXIS::PX || pset == FMM_Wrapper::PAXIS::PY || pset == FMM_Wrapper::PAXIS::PZ ||
        pset == FMM_Wrapper::PAXIS::PXY || pset == FMM_Wrapper::PAXIS::PYZ || pset == FMM_Wrapper::PAXIS::PXZ) {
        double fx = 0, fy = 0, fz = 0;
        assert(src_coord.size() == src_value.size());
        for (int i = 0; i < src_coord.size() / 3; i++) {
            fx += src_value[3 * i];
            fy += src_value[3 * i + 1];
            fz += src_value[3 * i + 2];
        }
        fx /= (src_coord.size() / 3);
        fy /= (src_coord.size() / 3);
        fz /= (src_coord.size() / 3);
        for (int i = 0; i < src_coord.size() / 3; i++) {
            src_value[3 * i] -= fx;
            src_value[3 * i + 1] -= fy;
            src_value[3 * i + 2] -= fz;
        }
    }

    return;
}

void testFMM(std::vector<double> &trg_value, std::vector<double> &trg_coord, std::vector<double> &src_value,
             std::vector<double> &src_coord, std::vector<double> &trg_value_true, int p, double box, double shift,
             FMM_Wrapper::PAXIS pset) {
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (myRank == 0) {
        std::cout << "------------------------------" << std::endl;
        std::cout << "FMM Initializing for order p:" << p << " src points: " << src_coord.size() / 3
                  << " trg points: " << trg_coord.size() / 3 << std::endl;
        std::cout << "MPI Procs: " << nProcs << std::endl;
        std::cout << "omp threads: " << omp_get_max_threads() << std::endl;
    }

    distributePts(src_coord);
    distributePts(src_value);
    distributePts(trg_coord);
    distributePts(trg_value);

    FMM_Wrapper myFMM(p, 1000, 0, pset);
    myFMM.FMM_SetBox(shift, shift + box, shift, shift + box, shift, shift + box);
    std::cout << "FMM" << std::endl;

    myFMM.FMM_UpdateTree(src_coord, trg_coord);

    // run a 'noisy fmm' first, to check the old data is properly cleared by DataClear().
    std::vector<double> src_noise = src_value;
    std::fill(src_noise.begin(), src_noise.end(), 1.0);
    myFMM.FMM_Evaluate(trg_value, trg_coord.size() / 3, &src_noise);

    myFMM.FMM_DataClear();
    myFMM.FMM_Evaluate(trg_value, trg_coord.size() / 3, &src_value);
    std::cout << "Tree Evaluated" << std::endl;

    collectPts(src_coord);
    collectPts(src_value);
    collectPts(trg_coord);
    collectPts(trg_value);
    MPI_Barrier(MPI_COMM_WORLD);

    if (myRank == 0) {

        std::cout << "checking error" << std::endl;
        // calc error and max error
        double errorL2 = 0, errorAbs = 0, L2 = 0, errorMaxL2 = 0, maxU = 0;
        double errorMaxRel = 0;
        for (int i = 0; i < trg_value_true.size(); i++) {
            double temp = pow(trg_value_true[i] - trg_value[i], 2);
            // std::cout << trg_value[i]-trg_value_true[i] << std::endl;
            // calc error and max error
            //		if (temp >= pow(1e-5, 2)) {
            //			std::cout << "i" << i << "error L2" << temp <<
            // std::endl;
            //		}
            errorL2 += temp;
            errorAbs += sqrt(temp);
            L2 += pow(trg_value_true[i], 2);
            errorMaxL2 = std::max(sqrt(temp), errorMaxL2);
            maxU = std::max(maxU, fabs(trg_value_true[i]));
            errorMaxRel = std::max(sqrt(temp) / trg_value_true[i], errorMaxRel);
        }

        std::cout << std::setprecision(16) << "Max Abs Error L2: " << (errorMaxL2) << std::endl;
        std::cout << std::setprecision(16) << "Ave Abs Error L2: " << errorAbs / trg_value_true.size() << std::endl;
        std::cout << std::setprecision(16) << "Max Rel Error L2: " << (errorMaxRel) << std::endl;
        std::cout << std::setprecision(16) << "RMS Error L2: " << sqrt(errorL2 / trg_value_true.size()) << std::endl;
        std::cout << std::setprecision(16) << "Relative Error L2: " << sqrt(errorL2 / L2) << std::endl;
    }
}

void configure_parser(cli::Parser &parser) {
    parser.set_optional<int>("P", "periodicity", 0,
                             "0: NONE. 1: PX. 2:PY. 3:PZ. 4:PXY. 5:PXZ. 6:PYZ. 7:PXYZ. Default 0");
    parser.set_optional<int>("T", "ntarget", 2, "target number in each dimension. default 2");
    parser.set_optional<double>("B", "box", 1.0, "box edge length");
    parser.set_optional<double>("M", "move", 0.0, "box origin shift move");
    parser.set_optional<int>("R", "random", 1, "1 for random points, 0 for regular mesh");
    parser.set_optional<int>("S", "source", 1,
                             "1 for point force, 2 for force dipole, 4 for quadrupole, other for same as target.");
}

int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    MPI_Init(&argc, &argv);

    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    std::cout << "Running setting: " << std::endl;
    std::cout << "Periodicity: " << parser.get<int>("P") << std::endl;
    std::cout << "nTarget: " << parser.get<int>("T") << std::endl;
    std::cout << "Box: " << parser.get<double>("B") << std::endl;
    std::cout << "Shift: " << parser.get<double>("M") << std::endl;
    std::cout << "Random: " << parser.get<int>("R") << std::endl;
    std::cout << "Source: " << parser.get<int>("S") << std::endl;

    FMM_Wrapper::PAXIS pset;
    int pinput = parser.get<int>("P");
    switch (pinput) {
    case 0:
        pset = FMM_Wrapper::PAXIS::NONE;
        break;
    case 1:
        pset = FMM_Wrapper::PAXIS::PX;
        break;
    case 2:
        pset = FMM_Wrapper::PAXIS::PY;
        break;
    case 3:
        pset = FMM_Wrapper::PAXIS::PZ;
        break;
    case 4:
        pset = FMM_Wrapper::PAXIS::PXY;
        break;
    case 5:
        pset = FMM_Wrapper::PAXIS::PXZ;
        break;
    case 6:
        pset = FMM_Wrapper::PAXIS::PYZ;
        break;
    case 7:
        pset = FMM_Wrapper::PAXIS::PXYZ;
        break;
    }

    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");

    std::vector<double> src_coord(0);
    std::vector<double> src_value(0);
    std::vector<double> trg_coord(0);
    std::vector<double> trg_value(0);
    std::vector<double> trg_true(0);
    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        // initialize source and target coord and value
        // calc true value
        initPts(src_coord, src_value, trg_coord, trg_value, parser);
        trg_true.resize(trg_value.size());
        // determine which truevalue routine to call
        int trueN2 = 0;
        if (src_coord.size() * trg_coord.size() < 3 * 1e7) {
            trueN2 = 1;
            calcTrueValueN2(trg_true, trg_coord, src_value, src_coord, box, shift, pset);
        }
        MPI_Bcast(&trueN2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (trueN2 == 0) {
            calcTrueValueFMM(trg_true, trg_coord, src_value, src_coord, box, shift, pset);
        }
    } else {
        int trueN2 = 0;
        MPI_Bcast(&trueN2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (trueN2 == 0) {
            calcTrueValueFMM(trg_true, trg_coord, src_value, src_coord, box, shift, pset);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // send to test
    double boxfac = pow(box / 2, 2); // for surface integral, scale the cheb weight from [-1,1] to box length
    for (int p = 6; p <= 16; p += 2) {
        testFMM(trg_value, trg_coord, src_value, src_coord, trg_true, p, box, shift, pset);
        if (parser.get<int>("R") <= 0 && myRank == 0) {
            const int chebN = parser.get<int>("T");
            ChebNodal chebData(chebN);
            const int dimension = chebData.points.size();

            // chebyshev integration
            // calculate cheb Integrate
            double vx = 0, vy = 0, vz = 0;
            int imax, jmax, kmax;
            if (pset == FMM_Wrapper::PAXIS::PX || pset == FMM_Wrapper::PAXIS::PYZ) {
                // surface integration on YZ plane
                imax = 1;
                jmax = dimension;
                kmax = dimension;
            } else if (pset == FMM_Wrapper::PAXIS::PY || pset == FMM_Wrapper::PAXIS::PXZ) {
                // surface integration on XZ plane
                imax = dimension;
                jmax = 1;
                kmax = dimension;
            } else {
                // surface integration on XY plane
                imax = dimension;
                jmax = dimension;
                kmax = 1;
            }

            for (int i = 0; i < imax; i++) {
                for (int j = 0; j < jmax; j++) {
                    for (int k = 0; k < kmax; k++) {
                        double vxp = trg_value[3 * (i * dimension * dimension + j * dimension + k)] *
                                     chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                        double vyp = trg_value[3 * (i * dimension * dimension + j * dimension + k) + 1] *
                                     chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                        double vzp = trg_value[3 * (i * dimension * dimension + j * dimension + k) + 2] *
                                     chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                        vx += vxp;
                        vy += vyp;
                        vz += vzp;
                    }
                }
            }
            std::cout << "net flow: " << vx / chebData.weights[0] << "," << vy / chebData.weights[0] << ","
                      << vz / chebData.weights[0] << std::endl;

            // two other fluxes for 3P
            if (pset == FMM_Wrapper::PAXIS::PXYZ) {
                imax = dimension;
                jmax = 1;
                kmax = dimension;
                vx = 0, vy = 0, vz = 0;
                for (int i = 0; i < imax; i++) {
                    for (int j = 0; j < jmax; j++) {
                        for (int k = 0; k < kmax; k++) {
                            double vxp = trg_value[3 * (i * dimension * dimension + j * dimension + k)] *
                                         chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                            double vyp = trg_value[3 * (i * dimension * dimension + j * dimension + k) + 1] *
                                         chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                            double vzp = trg_value[3 * (i * dimension * dimension + j * dimension + k) + 2] *
                                         chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                            vx += vxp;
                            vy += vyp;
                            vz += vzp;
                        }
                    }
                }
                std::cout << "net flow xz: " << vx / chebData.weights[0] << "," << vy / chebData.weights[0] << ","
                          << vz / chebData.weights[0] << std::endl;

                imax = 1;
                jmax = dimension;
                kmax = dimension;
                vx = 0, vy = 0, vz = 0;
                for (int i = 0; i < imax; i++) {
                    for (int j = 0; j < jmax; j++) {
                        for (int k = 0; k < kmax; k++) {
                            double vxp = trg_value[3 * (i * dimension * dimension + j * dimension + k)] *
                                         chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                            double vyp = trg_value[3 * (i * dimension * dimension + j * dimension + k) + 1] *
                                         chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                            double vzp = trg_value[3 * (i * dimension * dimension + j * dimension + k) + 2] *
                                         chebData.weights[i] * chebData.weights[j] * chebData.weights[k] * boxfac;
                            vx += vxp;
                            vy += vyp;
                            vz += vzp;
                        }
                    }
                }
                std::cout << "net flow yz: " << vx / chebData.weights[0] << "," << vy / chebData.weights[0] << ","
                          << vz / chebData.weights[0] << std::endl;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
