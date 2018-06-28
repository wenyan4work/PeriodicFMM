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
    ntrg = 10
    trg_coord = np.random.randn(ntrg, 3)


    if True:
        # Try basic function
        print('\n\n')
        fmm.FMM(trg_coord)
        
    if True:
        # Try Mini_FMM
        print('\n\n')
        print('xxx = ', fmm.Mini_PAXIS.PX)
        Mini_FMM = fmm.Mini_FMM(10, fmm.Mini_PAXIS.NONE)
        Mini_FMM.saludo()

    if False:
        # Test MPI
        print('\n\n')
        comm = MPI.COMM_WORLD
        nprocs = comm.Get_size()
        rank   = comm.Get_rank()

        if rank == 0:
            print('rank = ', rank)
            data = 'Hello!'
            comm.send(data, dest=nprocs-1, tag=1)
        elif rank == nprocs-1:
            data = comm.recv(source=0, tag=1)
            print('rank ', rank, ' received ', data)


    if True:
        # Try FMM_Wrapper
        print('\n\n')
        myFMM = fmm.FMM_Wrapper(10, 1024, 0, fmm.PAXIS.NONE)



    print('# End')

