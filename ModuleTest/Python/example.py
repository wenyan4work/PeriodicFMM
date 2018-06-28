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
    ntrg = 10
    trg_coord = np.random.randn(ntrg, 3)

    fmm.FMM(trg_coord)


    print('# End')

