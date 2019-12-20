#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <PeriodicFMM/FMMWrapper-c.h>
#include <PeriodicFMM/FMMWrapperWall2D-c.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nsrc = 16384;
    const int ntrg = 16384;

    double *srcCoord = malloc(sizeof(double) * 3 * nsrc);
    double *srcValue = malloc(sizeof(double) * 3 * nsrc);
    double *trgCoord = malloc(sizeof(double) * 3 * ntrg);
    double *trgValue = malloc(sizeof(double) * 3 * ntrg);

// some arbitrary data
#pragma omp parallel for
    for (int i = 0; i < nsrc; i++) {
        int seed = rank * nsrc + i;
        srcCoord[3 * i] = fabs(sin(seed));
        srcCoord[3 * i + 1] = fabs(cos(seed));
        srcCoord[3 * i + 2] = fabs(sin(seed * seed));

        srcValue[3 * i] = sin(seed);
        srcValue[3 * i + 1] = sin(sin(seed));
        srcValue[3 * i + 2] = cos(sin(seed));
    }

#pragma omp parallel for
    for (int i = 0; i < ntrg; i++) {
        int seed = rank * nsrc + i;

        trgCoord[3 * i] = fabs(cos(seed));
        trgCoord[3 * i + 1] = fabs(sin(seed));
        trgCoord[3 * i + 2] = fabs(cos(seed * seed));

        trgValue[3 * i] = 0;
        trgValue[3 * i + 1] = 0;
        trgValue[3 * i + 2] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // FMM_Wrapper
    // Evaluate, clear, and Evaluate again
    {
        FMM_Wrapper *fmm = create_fmm_wrapper(12, 2000, 0, 7, 0);
        FMM_SetBox(fmm, 0, 1, 0, 1, 0, 1);
        FMM_UpdateTree(fmm, trgCoord, srcCoord, ntrg, nsrc);
        FMM_Evaluate(fmm, trgValue, srcValue, ntrg, nsrc);
        FMM_DataClear(fmm);
        FMM_Evaluate(fmm, trgValue, srcValue, ntrg, nsrc);
        delete_fmm_wrapper(fmm);
    }

    // FMM_WrapperWall2D
    // Evaluate, clear, and Evaluate again
    {
#pragma omp parallel for
        for (int i = 0; i < nsrc; i++) {
            srcCoord[3 * i + 2] *= 0.499;
        }
#pragma omp parallel for
        for (int i = 0; i < ntrg; i++) {
            trgCoord[3 * i + 2] *= 0.499;
        }

        FMM_WrapperWall2D *fmm = create_fmm_wrapperwall2d(12, 2000, 0, 4);
        FMMWall2D_SetBox(fmm, 0, 1, 0, 1, 0, 0.4999);
        FMMWall2D_UpdateTree(fmm, trgCoord, srcCoord, ntrg, nsrc);
        FMMWall2D_Evaluate(fmm, trgValue, srcValue, ntrg, nsrc);
        FMMWall2D_DataClear(fmm);
        FMMWall2D_Evaluate(fmm, trgValue, srcValue, ntrg, nsrc);
        delete_fmm_wrapperwall2d(fmm);
    }

    free(srcCoord);
    free(trgCoord);
    free(srcValue);
    free(trgValue);

    MPI_Finalize();
    return 0;
}
