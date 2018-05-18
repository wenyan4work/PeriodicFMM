#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include "Interface/FMMWrapper-c.h"
#include "Interface/FMMWrapperWall2D-c.h"

int main(int argc, char** argv) {
    MPI_Init(argc,argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    const int nsrc = 16384;
    const int ntrg = 16384;

    double *srcCoord = malloc(sizeof(double) * 3 * nsrc);
    double *srcValue = malloc(sizeof(double) * 3 * nsrc);
    double *trgCoord = malloc(sizeof(double) * 3 * ntrg);
    double *trgValue = malloc(sizeof(double) * 3 * ntrg);

    // some arbitrary data
    for (int i = 0; i < nsrc; i++) {
        int seed = rank*nsrc+i;
        srcCoord[3 * i] = sin(seed);
        srcCoord[3 * i + 1] = cos(seed);
        srcCoord[3 * i + 2] = sin(exp(seed));

        trgCoord[3 * i] = cos(seed);
        trgCoord[3 * i + 1] = sin(seed);
        trgCoord[3 * i + 2] = cos(exp(seed));

        srcValue[3 * i] = sin(seed);
        srcValue[3 * i + 1] = sin(sin(seed));
        srcValue[3 * i + 2] = cos(sin(seed));

        trgCoord[3 * i] = 0;
        trgCoord[3 * i + 1] = 0;
        trgCoord[3 * i + 2] = 0;
    }

    // FMM_Wrapper
    // Evaluate, clear, and Evaluate again
    {
        FMM_Wrapper *fmm = create_fmm_wrappper(8, 2000, 0, 7);
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
        FMM_WrapperWall2D *fmm = create_fmm_wrappperwall2d(8, 2000, 0, 4);
        FMMWall2D_SetBox(fmm, 0, 1, 0, 1, 0, 1);
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