/*
 * main.cpp
 *
 *  Created on: Oct 14, 2016
 *      Author: wyan
 */

#include "FMM/FMMWrapperWall2D.h"

#include "ChebNodal.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <mpi.h>

#include "Util/EigenDef.hpp"
#include "Util/cmdparser.hpp"

#define EPS (1e-12) // make sure EPS/10 is still valid
#define MAXP 16

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

void runFMM(std::vector<double> &trgValueWall, std::vector<double> &trgCoordWall, std::vector<double> &trgValuePBC,
            std::vector<double> &trgCoordPBC, std::vector<double> &trgValueSample, std::vector<double> &trgCoordSample,
            std::vector<double> &src_value, std::vector<double> &src_coord, const int p, const double box,
            const double shift, const FMM_WrapperWall2D::PAXIS pset) {
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (myRank == 0) {
        std::cout << "------------------------------" << std::endl;
        std::cout << "FMM Initializing for order p:" << p << " src points: " << src_coord.size() / 3 << std::endl;
        std::cout << "MPI Procs: " << nProcs << std::endl;
        std::cout << "omp threads: " << omp_get_max_threads() << std::endl;
    }

    //    distributePts(src_coord);
    //    distributePts(src_value);

    FMM_WrapperWall2D myFMM(p, 4000, 0, pset);
    myFMM.FMM_SetBox(shift, shift + box, shift, shift + box, shift, shift + box * 0.5 * (1 - EPS / 10));

    // FMM 1, Wall
    std::cout << "FMM Wall ntrg:" << trgCoordWall.size() / 3 << std::endl;
    myFMM.FMM_TreeClear();
    myFMM.FMM_UpdateTree(src_coord, trgCoordWall);
    myFMM.FMM_Evaluate(trgValueWall, trgCoordWall.size() / 3, &src_value);
    std::cout << "Tree Evaluated" << std::endl;

    // FMM 2, PBC
    std::cout << "FMM PBC ntrg:" << trgCoordPBC.size() / 3 << std::endl;
    myFMM.FMM_TreeClear();
    myFMM.FMM_UpdateTree(src_coord, trgCoordPBC);
    myFMM.FMM_Evaluate(trgValuePBC, trgCoordPBC.size() / 3, &src_value);
    std::cout << "Tree Evaluated" << std::endl;

    // FMM 3, Sample
    std::cout << "FMM Sample ntrg:" << trgCoordSample.size() / 3 << std::endl;
    myFMM.FMM_TreeClear();
    myFMM.FMM_UpdateTree(src_coord, trgCoordSample);
    myFMM.FMM_Evaluate(trgValueSample, trgCoordSample.size() / 3, &src_value);
    std::cout << "Tree Evaluated" << std::endl;

    //    collectPts(src_coord);
    //    collectPts(src_value);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

void initPtsTrgWall(std::vector<double> &trg_coord, std::vector<double> &trg_value, const cli::Parser &parser) {
    // initialize source and target coord and value
    bool randomS = (parser.get<int>("C") > 0 ? true : false);
    const int ntrgEdge = parser.get<int>("T");
    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");
    int nsource = parser.get<int>("S");

    FMM_WrapperWall2D::PAXIS pset;
    int pinput = parser.get<int>("P");
    switch (pinput) {
    case 0:
        pset = FMM_WrapperWall2D::PAXIS::NONE;
        break;
    case 4:
        pset = FMM_WrapperWall2D::PAXIS::PXY;
        break;
    }

    const int chebN = ntrgEdge;
    ChebNodal chebData(chebN);
    chebData.points[0] += 0;
    chebData.points.back() -= EPS; // prevent PVFMM crash with point located at the edge

    const int dimension = chebData.points.size();

    if (randomS == true) {
        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(nsource);

        std::uniform_real_distribution<> u(-0.5, 0.5);
        std::lognormal_distribution<> d(log(0.2), 0.5);
        trg_coord.resize(0);

        for (int i = 0; i < ntrgEdge + 1; i++) {
            for (int j = 0; j < ntrgEdge + 1; j++) {
                double r01;
                r01 = fmod(d(gen), 1.0);
                trg_coord.push_back(r01 * box + shift); // x
                r01 = fmod(d(gen), 1.0);
                trg_coord.push_back(r01 * box + shift); // y
                trg_coord.push_back(0 * box + shift);   // z on the wall
            }
        }
        FILE *pfile = fopen("randomPointsWall.txt", "w");
        for (int i = 0; i < trg_coord.size() / 3; i++) {
            fprintf(pfile, "%e\t%e\t%e\n", trg_coord[3 * i], trg_coord[3 * i + 1], trg_coord[3 * i + 2]);
        }
        fclose(pfile);
    } else {

        std::vector<double> &chebMesh = trg_coord;
        std::vector<double> &chebValue = trg_value;
        chebMesh.resize(pow(dimension, 2) * 3);
        chebValue.resize(pow(dimension, 2) * 3);

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                chebMesh[3 * (i * dimension + j)] = (chebData.points[i] + 1) * box / 2 + shift;
                chebMesh[3 * (i * dimension + j) + 1] = (chebData.points[j] + 1) * box / 2 + shift;
                chebMesh[3 * (i * dimension + j) + 2] = shift; // z on the no slip wall
            }
        }

        FILE *pfile = fopen("chebPointsWall.txt", "w");
        for (int i = 0; i < chebMesh.size() / 3; i++) {
            fprintf(pfile, "%e\t%e\t%e\n", chebMesh[3 * i], chebMesh[3 * i + 1], chebMesh[3 * i + 2]);
        }
        fclose(pfile);
    }

    trg_value.resize(trg_coord.size());
}

void initPtsTrgPBC(std::vector<double> &trg_coord, std::vector<double> &trg_value, const cli::Parser &parser) {
    // initialize source and target coord and value
    bool randomS = (parser.get<int>("C") > 0 ? true : false);
    const int ntrgEdge = parser.get<int>("T");
    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");
    const double zratio = parser.get<double>("Z");
    int nsource = parser.get<int>("S");

    FMM_WrapperWall2D::PAXIS pset;
    int pinput = parser.get<int>("P");
    switch (pinput) {
    case 0:
        pset = FMM_WrapperWall2D::PAXIS::NONE;
        break;
    case 4:
        pset = FMM_WrapperWall2D::PAXIS::PXY;
        break;
    }

    const int chebN = ntrgEdge;
    ChebNodal chebData(chebN);

    chebData.points[0] += 0;
    chebData.points.back() -= EPS; // prevent PVFMM crash with point located at the edge

    const int dimension = chebData.points.size();

    if (randomS == true) {
        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(nsource);

        std::uniform_real_distribution<> u(-0.5, 0.5);
        std::lognormal_distribution<> d(log(0.2), 0.5);
        trg_coord.resize(0);
        // wall y = 0
        for (int i = 0; i < ntrgEdge + 1; i++) {
            for (int j = 0; j < ntrgEdge + 1; j++) {
                double r01;
                r01 = fmod(d(gen), 1.0);
                trg_coord.push_back(r01 * box + shift); // x
                r01 = fmod(d(gen), 1.0);
                trg_coord.push_back(0 * box + shift); // y
                r01 = fmod(d(gen), 1.0);
                trg_coord.push_back(r01 * zratio * box + shift); // z on the wall
            }
        }
        const int nptsWall = trg_coord.size() / 3; // assert = ntrgEdge ^2
        trg_coord.resize(12 * nptsWall);
        // wall y=1
        for (int i = 0; i < nptsWall; i++) {
            trg_coord[3 * (i + nptsWall)] = trg_coord[3 * i];              // x
            trg_coord[3 * (i + nptsWall) + 1] = ((1 - EPS) * box + shift); // y
            trg_coord[3 * (i + nptsWall) + 2] = trg_coord[3 * i + 2];      // z
        }

        // wall x = 0
        for (int i = 0; i < nptsWall; i++) {
            trg_coord[3 * (i + 2 * nptsWall)] = (0 * box + shift);        // x
            trg_coord[3 * (i + 2 * nptsWall) + 1] = trg_coord[3 * i];     // y = x
            trg_coord[3 * (i + 2 * nptsWall) + 2] = trg_coord[3 * i + 2]; // z
        }

        // wall x = 1
        for (int i = 0; i < nptsWall; i++) {
            trg_coord[3 * (i + 3 * nptsWall)] = ((1 - EPS) * box + shift); // x
            trg_coord[3 * (i + 3 * nptsWall) + 1] = trg_coord[3 * i];      // y = x
            trg_coord[3 * (i + 3 * nptsWall) + 2] = trg_coord[3 * i + 2];  // z
        }

        FILE *pfile = fopen("randomPointsPBC.txt", "w");
        for (int i = 0; i < trg_coord.size() / 3; i++) {
            fprintf(pfile, "%e\t%e\t%e\n", trg_coord[3 * i], trg_coord[3 * i + 1], trg_coord[3 * i + 2]);
        }
        fclose(pfile);
    } else {

        trg_coord.resize(pow(dimension, 2) * 3);

        // wall, y=0
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                trg_coord[3 * (i * dimension + j)] = (chebData.points[i] + 1) * box / 2 + shift;
                trg_coord[3 * (i * dimension + j) + 1] = 0 * box + shift;
                trg_coord[3 * (i * dimension + j) + 2] = (chebData.points[j] + 1) * zratio * box / 2 + shift;
            }
        }

        const int nptsWall = trg_coord.size() / 3; // assert = ntrgEdge ^2
        trg_coord.resize(12 * nptsWall);
        // wall y=1
        for (int i = 0; i < nptsWall; i++) {
            trg_coord[3 * (i + nptsWall)] = trg_coord[3 * i];                                     // x
            trg_coord[3 * (i + nptsWall) + 1] = ((chebData.points.back() + 1) * box / 2 + shift); // y
            trg_coord[3 * (i + nptsWall) + 2] = trg_coord[3 * i + 2];                             // z
        }

        // wall x = 0
        for (int i = 0; i < nptsWall; i++) {
            trg_coord[3 * (i + 2 * nptsWall)] = (0 * box + shift);        // x
            trg_coord[3 * (i + 2 * nptsWall) + 1] = trg_coord[3 * i];     // y = x
            trg_coord[3 * (i + 2 * nptsWall) + 2] = trg_coord[3 * i + 2]; // z
        }

        // wall x = 1
        for (int i = 0; i < nptsWall; i++) {
            trg_coord[3 * (i + 3 * nptsWall)] = ((chebData.points.back() + 1) * box / 2 + shift); // x
            trg_coord[3 * (i + 3 * nptsWall) + 1] = trg_coord[3 * i];                             // y = x
            trg_coord[3 * (i + 3 * nptsWall) + 2] = trg_coord[3 * i + 2];                         // z
        }

        FILE *pfile = fopen("chebPointsPBC.txt", "w");
        for (int i = 0; i < trg_coord.size() / 3; i++) {
            fprintf(pfile, "%e\t%e\t%e\n", trg_coord[3 * i], trg_coord[3 * i + 1], trg_coord[3 * i + 2]);
        }
        fclose(pfile);
    }

    trg_value.resize(trg_coord.size());
}

void initPtsTrgSample(std::vector<double> &trg_coord, std::vector<double> &trg_value, const cli::Parser &parser,
                      bool randomS) {
    // Sample points, at least 100*EPS from the wall and boundaries
    const double gap = 0.1;

    // initialize source and target coord and value
    // bool randomS = (parser.get<int>("R") > 0 ? true : false);
    const int ntrgEdge = parser.get<int>("T");
    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");
    const double zratio = parser.get<double>("Z");
    int nsource = parser.get<int>("S");

    FMM_WrapperWall2D::PAXIS pset;
    int pinput = parser.get<int>("P");
    switch (pinput) {
    case 0:
        pset = FMM_WrapperWall2D::PAXIS::NONE;
        break;
    case 4:
        pset = FMM_WrapperWall2D::PAXIS::PXY;
        break;
    }

    const int chebN = ntrgEdge;
    ChebNodal chebData(chebN);

    //    chebData.points[0] += 0;
    //    chebData.points.back() -= EPS;

    const int dimension = chebData.points.size();

    if (randomS == true) {
        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(nsource);

        std::uniform_real_distribution<> u(-0.5, 0.5);
        std::lognormal_distribution<> d(log(0.2), 0.5);
        trg_coord.resize(0);
        for (int i = 0; i < ntrgEdge + 1; i++) {
            for (int j = 0; j < ntrgEdge + 1; j++) {
                for (int k = 0; k < ntrgEdge + 1; k++) {
                    double r01;
                    r01 = fmod(d(gen), 1.0) * (1.0 - gap * 2) + gap;
                    trg_coord.push_back(r01 * box + shift); // x
                    r01 = fmod(d(gen), 1.0) * (1.0 - gap * 2) + gap;
                    trg_coord.push_back(r01 * box + shift); // y
                    r01 = fmod(d(gen), 1.0) * (1.0 - gap * 2) + gap;
                    trg_coord.push_back(r01 * zratio * box + shift); //

                    trg_value.push_back(u(gen));
                    trg_value.push_back(u(gen));
                    trg_value.push_back(u(gen));
                }
            }
        }
        FILE *pfile = fopen("randomPointsSample.txt", "w");
        for (int i = 0; i < trg_coord.size() / 3; i++) {
            fprintf(pfile, "%e\t%e\t%e\n", trg_coord[3 * i], trg_coord[3 * i + 1], trg_coord[3 * i + 2]);
        }
        fclose(pfile);
    } else {

        std::vector<double> &chebMesh = trg_coord;
        std::vector<double> &chebValue = trg_value;
        chebMesh.resize(pow(dimension, 3) * 3);
        chebValue.resize(0);

        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(nsource);
        std::uniform_real_distribution<> u(-0.5, 0.5);

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                for (int k = 0; k < dimension; k++) {
                    double r01;
                    r01 = fmod(0.5 * (chebData.points[i] + 1), 1.0) * (1.0 - gap * 2) + gap;
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k)] = r01 * box + shift;
                    r01 = fmod(0.5 * (chebData.points[j] + 1), 1.0) * (1.0 - gap * 2) + gap;
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k) + 1] = r01 * box + shift;
                    r01 = fmod(0.5 * (chebData.points[k] + 1), 1.0) * (1.0 - gap * 2) + gap;
                    chebMesh[3 * (i * dimension * dimension + j * dimension + k) + 2] = r01 * zratio * box + shift;
                    chebValue.push_back(u(gen));
                    chebValue.push_back(u(gen));
                    chebValue.push_back(u(gen));
                }
            }
        }

        FILE *pfile = fopen("chebPointsSample.txt", "w");
        for (int i = 0; i < chebMesh.size() / 3; i++) {
            fprintf(pfile, "%e\t%e\t%e\n", chebMesh[3 * i], chebMesh[3 * i + 1], chebMesh[3 * i + 2]);
        }
        fclose(pfile);
    }
    trg_value.resize(trg_coord.size());
}

void initPtsSRC(std::vector<double> &src_coord, std::vector<double> &src_value, const cli::Parser &parser) {
    bool randomS = (parser.get<int>("R") > 0 ? true : false);
    const int ntrgEdge = parser.get<int>("T");
    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");
    const double zratio = parser.get<double>("Z");

    // initialize source points
    int nsource = parser.get<int>("S");
    switch (nsource) {
    case 1: {
        src_coord.push_back(0.7 * box + shift);
        src_coord.push_back(0.6 * box + shift);
        src_coord.push_back(0.4 * box * zratio + shift); // must < 0.5*box
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
        src_coord.push_back(0.5 * box * zratio + shift); // must < 0.5*box
        src_coord.push_back(0.2 * box + shift);          // 2
        src_coord.push_back(0.8 * box + shift);
        src_coord.push_back(0.7 * box * zratio + shift); // must < 0.5*box
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
        src_coord.push_back(0.1 * box * zratio + shift); // must < 0.5*box;
        src_coord.push_back(0.2 * box + shift);          // 2
        src_coord.push_back(0.2 * box + shift);
        src_coord.push_back(0.2 * box * zratio + shift); // must < 0.5*box;
        src_coord.push_back(0.3 * box + shift);          // 3
        src_coord.push_back(0.3 * box + shift);
        src_coord.push_back(0.3 * box * zratio + shift); // must < 0.5*box;
        src_coord.push_back(0.4 * box + shift);          // 4
        src_coord.push_back(0.4 * box + shift);
        src_coord.push_back(0.4 * box * zratio + shift); // must < 0.5*box;
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
        // same as sample points
        initPtsTrgSample(src_coord, src_value, parser, randomS);
        FILE *pfile = fopen("srcPoints.txt", "w");
        for (int i = 0; i < src_coord.size() / 3; i++) {
            fprintf(pfile, "%e\t%e\t%e\n", src_coord[3 * i], src_coord[3 * i + 1], src_coord[3 * i + 2]);
        }
        fclose(pfile);
        pfile = fopen("srcValues.txt", "w");
        for (int i = 0; i < src_value.size() / 3; i++) {
            fprintf(pfile, "%e\t%e\t%e\n", src_value[3 * i], src_value[3 * i + 1], src_value[3 * i + 2]);
        }
        fclose(pfile);
    } break;
    }
}

void checkErrorWall(std::vector<double> &trgValueWall, std::vector<double> &trgCoordWall) {
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (myRank != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    std::cout << "Wall velocity error" << std::endl;
    double eAbsX = 0;
    double eAbsY = 0;
    double eAbsZ = 0;
    const int ntrg = trgValueWall.size() / 3;
    for (int i = 0; i < ntrg; i++) {
        eAbsX = std::max(eAbsX, std::abs(trgValueWall[3 * i]));
        eAbsY = std::max(eAbsY, std::abs(trgValueWall[3 * i + 1]));
        eAbsZ = std::max(eAbsZ, std::abs(trgValueWall[3 * i + 2]));
    }

    printf("Max ABS No Slip Error: %g,%g,%g\n", eAbsX, eAbsY, eAbsZ);
    MPI_Barrier(MPI_COMM_WORLD);
}

void checkErrorPBC(std::vector<double> &trgValuePBC, std::vector<double> &trgCoordPBC) {
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (myRank != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }
    // calc error and max error
    double errorL2 = 0, errorAbs = 0, L2 = 0, errorMaxL2 = 0, maxU = 0;
    double errorMaxRel = 0;
    const int nptsWall = trgValuePBC.size() / (4 * 3);

    // check y PBC
    for (int i = 0; i < 3 * nptsWall; i++) {
        double temp = pow(trgValuePBC[i] - trgValuePBC[i + 3 * nptsWall], 2);

        errorL2 += temp;
        errorAbs += sqrt(temp);
        L2 += pow(trgValuePBC[i], 2);
        errorMaxL2 = std::max(sqrt(temp), errorMaxL2);
        maxU = std::max(maxU, fabs(trgValuePBC[i]));
        errorMaxRel = std::max(sqrt(temp) / trgValuePBC[i], errorMaxRel);
    }
    printf("------------------------");
    std::cout << "checking PBC error in Y direction" << std::endl;
    printf("Max Abs Error L2: %g\n", errorMaxL2);
    printf("Ave Abs Error L2: %g\n", errorAbs / (3 * nptsWall));
    printf("Max Rel Error L2: %g\n", (errorMaxRel));
    printf("RMS Error L2: %g\n", sqrt(errorL2 / (3 * nptsWall)));
    printf("Relative Error L2: %g\n", sqrt(errorL2 / L2));

    // check x PBC
    for (int i = 0; i < 3 * nptsWall; i++) {
        double temp = pow(trgValuePBC[i + 6 * nptsWall] - trgValuePBC[i + 9 * nptsWall], 2);

        errorL2 += temp;
        errorAbs += sqrt(temp);
        L2 += pow(trgValuePBC[i], 2);
        errorMaxL2 = std::max(sqrt(temp), errorMaxL2);
        maxU = std::max(maxU, fabs(trgValuePBC[i]));
        errorMaxRel = std::max(sqrt(temp) / trgValuePBC[i], errorMaxRel);
    }

    printf("------------------------");
    std::cout << "checking PBC error in X direction" << std::endl;
    printf("Max Abs Error L2: %g\n", errorMaxL2);
    printf("Ave Abs Error L2: %g\n", errorAbs / (3 * nptsWall));
    printf("Max Rel Error L2: %g\n", (errorMaxRel));
    printf("RMS Error L2: %g\n", sqrt(errorL2 / (3 * nptsWall)));
    printf("Relative Error L2: %g\n", sqrt(errorL2 / L2));
}

void checkErrorSample(std::vector<double> &trg_value, std::vector<double> &trg_value_true,
                      std::vector<double> &trgCoordSample) {
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (myRank != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }
    printf("------------------------");
    printf("checking Relative error\n");
    // calc error and max error
    double errorL2 = 0, errorAbs = 0, L2 = 0, errorMaxL2 = 0, maxU = 0;
    double errorMaxRel = 0;

    for (int i = 0; i < trg_value_true.size(); i++) {
        double temp = pow(trg_value_true[i] - trg_value[i], 2);
        // std::cout << trg_value[i]-trg_value_true[i] << std::endl;
        // calc error and max error
        //      if (temp >= pow(1e-5, 2)) {
        //          std::cout << "i" << i << "error L2" << temp <<
        // std::endl;
        //      }
        errorL2 += temp;
        errorAbs += sqrt(temp);
        L2 += pow(trg_value_true[i], 2);
        errorMaxL2 = std::max(sqrt(temp), errorMaxL2);
        maxU = std::max(maxU, fabs(trg_value_true[i]));
        errorMaxRel = std::max(sqrt(temp) / trg_value_true[i], errorMaxRel);
    }

    printf("Max Abs Error L2: %g\n", errorMaxL2);
    printf("Ave Abs Error L2: %g\n", errorAbs / trg_value_true.size());
    printf("Max Rel Error L2: %g\n", (errorMaxRel));
    printf("RMS Error L2: %g\n", sqrt(errorL2 / trg_value_true.size()));
    printf("Relative Error L2: %g\n", sqrt(errorL2 / L2));
    MPI_Barrier(MPI_COMM_WORLD);
}

void configure_parser(cli::Parser &parser) {
    parser.set_optional<int>("P", "periodicity", 0,
                             "0: NONE. 1: PX. 2:PY. 3:PZ. 4:PXY. 5:PXZ. 6:PYZ. 7:PXYZ. Default 0");
    parser.set_optional<int>("T", "ntarget", 2, "target number in each dimension. default 2");
    parser.set_optional<double>("B", "box", 1.0, "box edge length");
    parser.set_optional<double>("M", "move", 0.0, "box origin shift move");
    parser.set_optional<int>("S", "source", 1,
                             "1 for point force, 2 for force dipole, 4 for quadrupole, other for same as target.");
    parser.set_optional<double>("Z", "zratio", 0.5, "ratio of z length to box edge length");

    parser.set_optional<int>("R", "randomsource", 1, "1 for random points, 0 for chebyshev mesh");
    parser.set_optional<int>("C", "randomtarget", 1, "1 for random points, 0 for chebyshev mesh");
}

void dumpGrid(std::vector<double> &src_value, std::vector<double> &src_coord, const int p, const double box,
              const double shift, const FMM_WrapperWall2D::PAXIS pset) {

    // x,y from 0 to 1-eps
    // z from 0 to 0.5-eps
    double boxHigh[3]{shift + box, shift + box, shift + box * 0.5};
    double boxLow[3]{shift, shift, shift};

    const int maxMesh = 51;
    const int NX = maxMesh;
    const int NY = maxMesh;
    const int NZ = maxMesh;
    // max 100 points in each dimension
    const double dx = (boxHigh[0] - boxLow[0] - EPS) / (NX - 1);
    const double dy = (boxHigh[1] - boxLow[1] - EPS) / (NY - 1);
    const double dz = (boxHigh[2] - boxLow[2] - EPS) / (NZ - 1);

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    std::vector<double> trgCoord(0);
    std::vector<double> trgValue(0);

    if (myRank == 0) {
        trgCoord.resize(NX * NY * NZ * 3);
        trgValue.resize(NX * NY * NZ * 3);
#pragma omp parallel for
        for (int k = 0; k < NZ; k++) {
            for (int j = 0; j < NY; j++) {
                for (int i = 0; i < NX; i++) {
                    // const Evec3 pos = Evec3(i * dx, j * dy, k * dz) + boxLow;
                    trgCoord[(i + j * (NX) + k * (NX) * (NY)) * 3 + 0] =
                        (i)*dx + boxLow[0]; // avoid hitting the edge of x,y=1
                    trgCoord[(i + j * (NX) + k * (NX) * (NY)) * 3 + 1] = (j)*dy + boxLow[1];
                    trgCoord[(i + j * (NX) + k * (NX) * (NY)) * 3 + 2] = k * dz + boxLow[2]; // fit z from the bottom
                }
            }
        }
        std::fill(trgValue.begin(), trgValue.end(), 0.0);
    } else {
        trgCoord.clear();
        trgValue.clear();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    FMM_WrapperWall2D myFMM(p, 4000, 0, pset);
    myFMM.FMM_SetBox(shift, shift + box, shift, shift + box, shift, shift + box * 0.5 * (1 - EPS / 10));
    MPI_Barrier(MPI_COMM_WORLD);

    myFMM.FMM_UpdateTree(src_coord, trgCoord);

    myFMM.FMM_DataClear();
    myFMM.FMM_Evaluate(trgValue, trgValue.size() / 3, &src_value);

    if (myRank == 0) {
        // step 4 dump
        std::string vtk("flow.vtk");
        FILE *fdump = fopen(vtk.c_str(), "w");
        // VTK file header
        fprintf(fdump, "# vtk DataFile Version 3.0\n");
        fprintf(fdump, "flow velocity\n");
        fprintf(fdump, "ASCII\n");
        fprintf(fdump, "DATASET STRUCTURED_POINTS\n");
        fprintf(fdump, "DIMENSIONS %d %d %d\n", NX, NY, NZ);
        fprintf(fdump, "ORIGIN %f %f %f\n", boxLow[0], boxLow[1], (boxLow[2]));
        fprintf(fdump, "SPACING %f %f %f\n", dx, dy, dz);

        // VTK point properties
        fprintf(fdump, "POINT_DATA %d\n", NX * NY * NZ);
        fprintf(fdump, "SCALARS Velocity float 3\n"); //    SCALARS dataName dataType numComp
        fprintf(fdump, "LOOKUP_TABLE DEFAULT\n");     // (NOT optional) default look up table

        for (int i = 0; i < NX * NY * NZ; i++) {
            fprintf(fdump, "%.6e %.6e %.6e\n", trgValue[3 * i], trgValue[3 * i + 1], trgValue[3 * i + 2]);
        }

        fclose(fdump);
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

    FMM_WrapperWall2D::PAXIS pset;
    int pinput = parser.get<int>("P");
    switch (pinput) {
    case 0:
        pset = FMM_WrapperWall2D::PAXIS::NONE;
        break;
    case 4:
        pset = FMM_WrapperWall2D::PAXIS::PXY;
        break;
    }

    const double box = parser.get<double>("B");
    const double shift = parser.get<double>("M");

    std::vector<double> src_coord(0);
    std::vector<double> src_value(0);

    // initialize target points on the no slip wall
    std::vector<double> trgCoordWall; // on wall to check no slip BC
    std::vector<double> trgValueWall;

    std::vector<double> trgCoordPBC; // on side boundary to check PBC
    std::vector<double> trgValuePBC;

    std::vector<double> trgCoordSample; // random sampling points
    std::vector<double> trgValueSample;
    std::vector<double> trgValueSampleTrue;

    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        // initialize source and target coord and value
        // calc true value
        initPtsSRC(src_coord, src_value, parser);
        initPtsTrgWall(trgCoordWall, trgValueWall, parser);
        initPtsTrgPBC(trgCoordPBC, trgValuePBC, parser);
        initPtsTrgSample(trgCoordSample, trgValueSample, parser, (parser.get<int>("C") > 0 ? true : false));

        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    trgValueSampleTrue.resize(trgValueSample.size());
    runFMM(trgValueWall, trgCoordWall, trgValuePBC, trgCoordPBC, trgValueSampleTrue, trgCoordSample, src_value,
           src_coord, MAXP, box, shift, pset);

    MPI_Barrier(MPI_COMM_WORLD);

    // send to test
    double boxfac = pow(box / 2, 2); // for surface integral, scale the cheb weight from [-1,1] to box length
    for (int p = 6; p <= MAXP; p += 2) {
        MPI_Barrier(MPI_COMM_WORLD);
        runFMM(trgValueWall, trgCoordWall, trgValuePBC, trgCoordPBC, trgValueSample, trgCoordSample, src_value,
               src_coord, p, box, shift, pset);
        checkErrorWall(trgValueWall, trgCoordWall);
        checkErrorPBC(trgValuePBC, trgCoordPBC);
        checkErrorSample(trgValueSample, trgValueSampleTrue, trgCoordSample);
    }

    // dump flow to grid
    dumpGrid(src_value, src_coord, MAXP, box, shift, pset);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
