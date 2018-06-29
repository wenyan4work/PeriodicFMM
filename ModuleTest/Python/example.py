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
    pbc = fmm.PAXIS.NONE

    # Create sources and targets
    nsrc = 1
    src_coord = np.random.randn(nsrc, 3)
    ntrg = 2
    trg_coord = np.random.randn(ntrg, 3)
    
    # Set box dimensions
    box = np.zeros((2,3))
    box[0,0] = np.minimum(src_coord[:,0].min(), trg_coord[:,0].min()) * 1.1
    box[1,0] = np.maximum(src_coord[:,0].max(), trg_coord[:,0].max()) * 1.1

    box[0,1] = np.minimum(src_coord[:,1].min(), trg_coord[:,1].min()) * 1.1
    box[1,1] = np.maximum(src_coord[:,1].max(), trg_coord[:,1].max()) * 1.1

    box[0,2] = np.minimum(src_coord[:,2].min(), trg_coord[:,2].min()) * 1.1
    box[1,2] = np.maximum(src_coord[:,2].max(), trg_coord[:,2].max()) * 1.1

    # Try FMM_Wrapper
    myFMM = fmm.FMM_Wrapper(mult_order, max_pts, init_depth, pbc)

    print(trg_coord)
    print(src_coord)
    print(box)
    print('\n\n')
    
    myFMM.FMM_SetBox(box[0,0], box[1,0], box[0,1], box[1,1], box[0,2], box[1,2])
    fmm.FMM_UpdateTree(myFMM, trg_coord, src_coord)
    # myFMM.FMM_UpdateTree(trg_coord, src_coord);
    # FMM_Evaluate(fmm, trgValue, srcValue, ntrg, nsrc);
    # FMM_DataClear(fmm);
    # FMM_Evaluate(fmm, trgValue, srcValue, ntrg, nsrc);

    print('Before end')
    # fmm.FMM_UpdateTree(myFMM, trg_coord, src_coord)
    print('AAA')


    print('# End')

