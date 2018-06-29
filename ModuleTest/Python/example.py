from __future__ import division, print_function
import numpy as np
import sys
try:
    from mpi4py import MPI
except ImportError:
    print('It didn\'t find mpi4py!')

import periodic_fmm as fmm


if __name__ == '__main__':
    print('# Start')

    # FMM parameters
    mult_order = 10
    max_pts = 1024
    init_depth = 0

    # Get MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create sources and targets
    nsrc = 1024
    src_coord = np.random.rand(nsrc, 3)
    src_coord[:,2] *= 0.1
    src_value = np.random.rand(nsrc, 3)
    ntrg = 1024
    trg_coord = np.random.rand(ntrg, 3)
    trg_coord[:,2] *= 0.1
    trg_value = np.zeros((ntrg, 3))
    
    sys.stdout.flush()
    comm.Barrier()

    # Try FMM_Wrapper
    pbc = fmm.FMM_PAXIS.NONE
    myFMM = fmm.FMM_Wrapper(mult_order, max_pts, init_depth, pbc)
    fmm.FMM_SetBox(myFMM, -1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
    fmm.FMM_UpdateTree(myFMM, trg_coord, src_coord)
    fmm.FMM_Evaluate(myFMM, trg_value, src_value)
    fmm.FMM_DataClear(myFMM)
    fmm.FMM_TreeClear(myFMM)
    fmm.FMM_UpdateTree(myFMM, trg_coord, src_coord)
    fmm.FMM_Evaluate(myFMM, trg_value, src_value)

    # Try FMM_WrapperWall2D
    pbc = fmm.FMMWall2D_PAXIS.NONE
    myFMM = fmm.FMM_WrapperWall2D(mult_order, max_pts, init_depth, pbc)
    fmm.FMMWall2D_SetBox(myFMM, -2.0, 2.0, -2.0, 2.0, 0.0, 1.0)
    fmm.FMMWall2D_UpdateTree(myFMM, trg_coord, src_coord)
    fmm.FMMWall2D_Evaluate(myFMM, trg_value, src_value)
    fmm.FMMWall2D_DataClear(myFMM)
    fmm.FMMWall2D_TreeClear(myFMM)
    fmm.FMMWall2D_UpdateTree(myFMM, trg_coord, src_coord)
    fmm.FMMWall2D_Evaluate(myFMM, trg_value, src_value)

    comm.Barrier()
    print('# End')

