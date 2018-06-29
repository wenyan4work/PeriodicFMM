from __future__ import division, print_function
import numpy as np
import sys
try:
    from mpi4py import MPI
    print('MPI imported')
except ImportError:
    print('It didn\'t find mpi4py!')

import periodic_fmm as fmm


if __name__ == '__main__':
    print('# Start')

    # FMM parameters
    mult_order = 10
    max_pts = 1024
    init_depth = 0
    pbc = fmm.FMM_PAXIS.NONE

    # Create sources and targets
    nsrc = 1
    src_coord = np.random.rand(nsrc, 3)
    src_coord[:,2] *= 0.1
    src_value = np.random.rand(nsrc, 3)
    ntrg = 2
    trg_coord = np.random.rand(ntrg, 3)
    trg_coord[:,2] *= 0.1
    trg_value = np.zeros((ntrg, 3))
    
    # Set box dimensions
    box = np.zeros((2,3))
    m = np.minimum(src_coord[:,0].min(), trg_coord[:,0].min()) 
    M = np.maximum(src_coord[:,0].max(), trg_coord[:,0].max()) 
    box[0,0] = m - (M-m) * 0.1
    box[1,0] = M + (M-m) * 0.1
    m = np.minimum(src_coord[:,1].min(), trg_coord[:,1].min()) 
    M = np.maximum(src_coord[:,1].max(), trg_coord[:,1].max()) 
    box[0,1] = m - (M-m) * 0.1
    box[1,1] = M + (M-m) * 0.1
    m = np.minimum(src_coord[:,2].min(), trg_coord[:,2].min()) 
    M = np.maximum(src_coord[:,2].max(), trg_coord[:,2].max()) 
    box[0,2] = m - (M-m) * 0.1
    box[1,2] = M + (M-m) * 0.1

    print(trg_coord)
    print(src_coord)

    # Try FMM_Wrapper
    print('\n\nTry FMM_Wrapper')
    myFMM = fmm.FMM_Wrapper(mult_order, max_pts, init_depth, pbc)
    fmm.FMM_SetBox(myFMM, box[0,0], box[1,0], box[0,1], box[1,1], box[0,2], box[1,2])
    fmm.FMM_UpdateTree(myFMM, trg_coord, src_coord)
    fmm.FMM_Evaluate(myFMM, trg_value, src_value)
    fmm.FMM_DataClear(myFMM)
    fmm.FMM_TreeClear(myFMM)
    fmm.FMM_UpdateTree(myFMM, trg_coord, src_coord)
    fmm.FMM_Evaluate(myFMM, trg_value, src_value)

    # Try FMM_WrapperWall2D
    print('\n\nTry FMM_WrapperWall2D')
    pbc = fmm.FMM_Wall2D_PAXIS.NONE
    myFMM = fmm.FMM_WrapperWall2D(mult_order, max_pts, init_depth, pbc)
    fmm.FMMWall2D_SetBox(myFMM, box[0,0], box[1,0], box[0,1], box[1,1], box[0,2], box[1,2])
    fmm.FMMWall2D_UpdateTree(myFMM, trg_coord, src_coord)
    fmm.FMMWall2D_Evaluate(myFMM, trg_value, src_value)
    fmm.FMMWall2D_DataClear(myFMM)
    fmm.FMMWall2D_TreeClear(myFMM)
    fmm.FMMWall2D_UpdateTree(myFMM, trg_coord, src_coord)
    fmm.FMMWall2D_Evaluate(myFMM, trg_value, src_value)

    print('# End')

