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
#include "regularized_stokeslet.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <mpi.h>

constexpr int MAXP = 16;

void configure_parser(cli::Parser &parser) {
    parser.set_optional<int>("P", "periodicity", 0, "0: NONE. 1: PX. 4:PXY. 7:PXYZ. Default 0");
    parser.set_optional<int>("T", "ntarget", 2, "target number in each dimension. default 2");
    parser.set_optional<double>("B", "box", 1.0, "box edge length");
    parser.set_optional<double>("M", "move", 0.0, "box origin shift move");
    parser.set_optional<int>("R", "random", 1, "1 for random points, 0 for regular mesh");
    parser.set_optional<int>("S", "source", 1,
                             "1 for point force, 2 for force dipole, 4 for quadrupole, other for same as target.");
    parser.set_optional<double>("E", "regularize", 0,
                                "The regularization parameter epsilon. Default to 0, Recommend 1e-4");
}

void distributePts(std::vector<double> &pts, const int dim = 3) {
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
    const int nPts = ptsGlobalSize / dim;
    // inclusive low
    int indexlow = dim * floor(myRank * nPts / static_cast<double>(nProcs));
    // non-inclusive high
    int indexhigh = dim * floor((myRank + 1) * nPts / static_cast<double>(nProcs));
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

void calcTrueValueFMM(std::vector<double> &trg_value_true, const std::vector<double> &trg_coord,
                      std::vector<double> &src_value, const std::vector<double> &src_coord, const double box,
                      const double shift, const FMM_Wrapper::PAXIS pset, const bool reg) {
    std::cout << "***************************************************" << std::endl;
    std::cout << "Skip O(N^2) true value calculation for large system" << std::endl;
    std::cout << "Use FMM p=" << MAXP << " as 'true' value                       " << std::endl;
    std::cout << "***************************************************" << std::endl;
    FMM_Wrapper myFMM(MAXP, 2000, 0, pset, reg);
    myFMM.FMM_SetBox(shift, shift + box, shift, shift + box, shift, shift + box);
    myFMM.FMM_UpdateTree(src_coord, trg_coord);
    myFMM.FMM_Evaluate(trg_value_true, trg_coord.size() / 3, &src_value);
    return;
}

void calcTrueValueN2(std::vector<double> &trg_value_true, const std::vector<double> &trg_coord,
                     const std::vector<double> &src_value, const std::vector<double> &src_coord, const double box,
                     const double shift, const FMM_Wrapper::PAXIS pset, const bool reg) {

    if (reg) {
        const int n_trg = trg_coord.size() / 3;
        trg_value_true.resize(n_trg * 6);
        if (pset == FMM_Wrapper::PAXIS::NONE) {
            combined_flow(src_coord, src_value, trg_coord, trg_value_true);
        }

    } else {
        // calc Ewald accuracy test
        trg_value_true.resize(trg_coord.size());
#pragma omp parallel for
        for (int t = 0; t < trg_coord.size() / 3; t++) {
            Eigen::Vector3d target(trg_coord[3 * t], trg_coord[3 * t + 1], trg_coord[3 * t + 2]);
            // shift and scale to [0,1)
            target[0] -= shift;
            target[1] -= shift;
            target[2] -= shift;
            target *= (1 / box);

            Eigen::Vector3d targetValue(0, 0, 0);
            for (int s = 0; s < src_coord.size() / 3; s++) {
                Eigen::Vector3d source(src_coord[3 * s], src_coord[3 * s + 1], src_coord[3 * s + 2]);
                Eigen::Vector3d sourceValue(src_value[3 * s], src_value[3 * s + 1], src_value[3 * s + 2]);
                // shift and rotate to [0,1)
                source[0] -= shift;
                source[1] -= shift;
                source[2] -= shift;
                source *= (1 / box);

                Eigen::Vector3d rst = target - source;
                Eigen::Matrix3d G;
                if (pset == FMM_Wrapper::PAXIS::PXYZ) {
                    GkernelEwald3D(rst, G, 1.0);
                } else if (pset == FMM_Wrapper::PAXIS::PXY) {
                    GkernelEwald2D(rst, G); // default box = 1
                } else if (pset == FMM_Wrapper::PAXIS::PX) {
                    Gkernel1D(rst, G); // default box =1
                } else if (pset == FMM_Wrapper::PAXIS::NONE) {
                    Gkernel(rst, Eigen::Vector3d(0, 0, 0), G);
                }
                targetValue += (G * sourceValue) / (8 * PI314);
            }
            // scale back
            trg_value_true[3 * t] = targetValue[0] / box;
            trg_value_true[3 * t + 1] = targetValue[1] / box;
            trg_value_true[3 * t + 2] = targetValue[2] / box;
        }
    }
}

void initPts(std::vector<double> &src_coord, std::vector<double> &src_value, std::vector<double> &trg_coord,
             std::vector<double> &trg_value, const cli::Parser &parser) {

    // initialize source and target coord and value
    bool randomS = (parser.get<int>("R") > 0 ? true : false);
    const int ntrgEdge = parser.get<int>("T");
    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");
    const double reg = std::abs(parser.get<double>("E"));
    FMM_Wrapper::PAXIS pset = static_cast<FMM_Wrapper::PAXIS>(parser.get<int>("P"));

    std::random_device rd;
    std::mt19937 gen(rd());

    // initialize target points
    if (randomS == true) {
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
                }
            }
        }
    } else {
        const int chebN = ntrgEdge;
        ChebNodal chebData(chebN);
        if (pset != FMM_Wrapper::PAXIS::PXYZ) {
            chebData.points[0] += 0;
            chebData.points.back() -= 1e-13; // prevent PVFMM crash with point located at the edge
        }
        const int dimension = chebData.points.size();

        std::vector<double> &chebMesh = trg_coord;
        chebMesh.resize(pow(dimension, 3) * 3);
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                for (int k = 0; k < dimension; k++) {
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k)] =
                        (chebData.points[i] + 1) * box / 2 + shift;
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k) + 1] =
                        (chebData.points[j] + 1) * box / 2 + shift;
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k) + 2] =
                        (chebData.points[k] + 1) * box / 2 + shift;
                }
            }
        }
    }
    {
        FILE *pfile = fopen("trgPoints.txt", "w");
        for (int i = 0; i < trg_coord.size() / 3; i++) {
            fprintf(pfile, "%f\t%f\t%f\n", trg_coord[3 * i], trg_coord[3 * i + 1], trg_coord[3 * i + 2]);
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
    } break;
    case 2: {
        src_coord.push_back(0.7 * box + shift); // 1
        src_coord.push_back(0.6 * box + shift);
        src_coord.push_back(0.5 * box + shift);
        src_coord.push_back(0.2 * box + shift); // 2
        src_coord.push_back(0.8 * box + shift);
        src_coord.push_back(0.7 * box + shift);
    } break;
    case 4: {
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
    } break;
    default:
        src_coord = trg_coord;
        break;
    }

    // set src_value
    const int n_src = src_coord.size() / 3;
    std::uniform_real_distribution<> uni(-0.5, 0.5);
    const int SDim = reg > 0 ? 7 : 3;
    src_value.resize(SDim * n_src);
    if (reg > 0) {
        for (int i = 0; i < n_src; i++) {
            for (int j = 0; j < SDim - 1; j++)
                src_value[SDim * i + j] = uni(gen);
            src_value[SDim * i + SDim - 1] = reg;
        }
    } else {
        for (int i = 0; i < n_src; i++) {
            for (int j = 0; j < SDim - 1; j++)
                src_value[SDim * i + j] = uni(gen);
        }
    }

    // enforce net charge for 1P and 2P
    if (pset == FMM_Wrapper::PAXIS::PX || pset == FMM_Wrapper::PAXIS::PXY) {
        double fx = 0, fy = 0, fz = 0;
        for (int i = 0; i < n_src; i++) {
            fx += src_value[SDim * i];
            fy += src_value[SDim * i + 1];
            fz += src_value[SDim * i + 2];
        }
        fx /= n_src;
        fy /= n_src;
        fz /= n_src;
        for (int i = 0; i < n_src; i++) {
            src_value[SDim * i] -= fx;
            src_value[SDim * i + 1] -= fy;
            src_value[SDim * i + 2] -= fz;
        }
    }
    {
        FILE *pfile = fopen("srcPoints.txt", "w");
        for (int i = 0; i < n_src; i++) {
            fprintf(pfile, "%f\t%f\t%f, %f\t%f\t%f\n", src_coord[3 * i], src_coord[3 * i + 1], src_coord[3 * i + 2],
                    src_value[SDim * i], src_value[SDim * i + 1], src_value[SDim * i + 2]);
        }
        fclose(pfile);
    }

    return;
}

void testFMM(std::vector<double> &trg_value, std::vector<double> &trg_coord, std::vector<double> &src_value,
             std::vector<double> &src_coord, std::vector<double> &trg_value_true, const int p,
             const cli::Parser &parser) {

    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");
    const double reg = std::abs(parser.get<double>("E"));
    FMM_Wrapper::PAXIS pset = static_cast<FMM_Wrapper::PAXIS>(parser.get<int>("P"));

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

    FMM_Wrapper myFMM(p, 2000, 0, pset, (reg > 0));

    distributePts(src_coord);
    distributePts(src_value, myFMM.SDim);
    distributePts(trg_coord);
    distributePts(trg_value, myFMM.TDim);

    std::cout << "Set Box" << std::endl;
    myFMM.FMM_SetBox(shift, shift + box, shift, shift + box, shift, shift + box);

    std::cout << "Update Tree" << std::endl;
    myFMM.FMM_UpdateTree(src_coord, trg_coord);

    // run a 'noisy fmm' first, to check the old data is properly cleared by DataClear().
    std::vector<double> src_noise = src_value;
    std::fill(src_noise.begin(), src_noise.end(), 1.0);
    std::cout << "Run FMM 1" << std::endl;
    myFMM.FMM_Evaluate(trg_value, trg_coord.size() / 3, &src_noise);

    myFMM.FMM_DataClear();
    std::cout << "Run FMM 2" << std::endl;
    myFMM.FMM_Evaluate(trg_value, trg_coord.size() / 3, &src_value);
    std::cout << "Tree Evaluated" << std::endl;

    collectPts(src_coord);
    collectPts(src_value);
    collectPts(trg_coord);
    collectPts(trg_value);
    MPI_Barrier(MPI_COMM_WORLD);

    {
        FILE *pfile;
        const int n_trg = trg_coord.size() / 3;
        const int TDim = myFMM.TDim;

        pfile = fopen("trgValues.txt", "w");
        for (int i = 0; i < n_trg; i++) {
            for (int j = 0; j < TDim; j++) {
                fprintf(pfile, "%.10e\t", trg_value[TDim * i + j]);
            }
            fprintf(pfile, "\n");
        }
        fclose(pfile);

        pfile = fopen("trgTrueValues.txt", "w");
        for (int i = 0; i < n_trg; i++) {
            for (int j = 0; j < TDim; j++) {
                fprintf(pfile, "%.10e\t", trg_value_true[TDim * i + j]);
            }
            fprintf(pfile, "\n");
        }
        fclose(pfile);
    }

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
    std::cout << "Regularization: " << parser.get<double>("E") << std::endl;

    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");
    const double reg = parser.get<double>("E");
    const FMM_Wrapper::PAXIS pset = static_cast<FMM_Wrapper::PAXIS>(parser.get<int>("P"));

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
        if (src_coord.size() * trg_coord.size() < 3 * 1e8) {
            trueN2 = 1;
            calcTrueValueN2(trg_true, trg_coord, src_value, src_coord, box, shift, pset, (reg > 0));
        }
        MPI_Bcast(&trueN2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (trueN2 == 0) {
            calcTrueValueFMM(trg_true, trg_coord, src_value, src_coord, box, shift, pset, (reg > 0));
        }
    } else {
        int trueN2 = 0;
        MPI_Bcast(&trueN2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (trueN2 == 0) {
            calcTrueValueFMM(trg_true, trg_coord, src_value, src_coord, box, shift, pset, (reg > 0));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // send to test
    double boxfac = pow(box / 2, 2); // for surface integral, scale the cheb weight from [-1,1] to box length
    for (int p = 6; p <= MAXP; p += 2) {
        testFMM(trg_value, trg_coord, src_value, src_coord, trg_true, p, parser);
        if (parser.get<int>("R") <= 0 && myRank == 0) {
            const int chebN = parser.get<int>("T");
            ChebNodal chebData(chebN);
            const int dimension = chebData.points.size();

            // chebyshev integration
            // calculate cheb Integrate
            double vx = 0, vy = 0, vz = 0;
            int imax, jmax, kmax;

            // X flux through YZ plane
            imax = 1;
            jmax = dimension;
            kmax = dimension;
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
            std::cout << "Flux X through YZ: " << vx / chebData.weights[0] << "," << vy / chebData.weights[0] << ","
                      << vz / chebData.weights[0] << std::endl;

            // Y flux through XZ plane
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
            std::cout << "Flux Y through XZ: " << vx / chebData.weights[0] << "," << vy / chebData.weights[0] << ","
                      << vz / chebData.weights[0] << std::endl;

            // Z flux through YZ plane
            imax = dimension;
            jmax = dimension;
            kmax = 1;
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
            std::cout << "Flux Z through XY: " << vx / chebData.weights[0] << "," << vy / chebData.weights[0] << ","
                      << vz / chebData.weights[0] << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
