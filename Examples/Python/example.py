import PyPeriodicFMM as fmm
import numpy as np
from mpi4py import MPI
# Get MPI parameters
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print('# Start rank', rank)
# FMM parameters
mult_order = 10
max_pts = 1024
init_depth = 0
# Create sources and targets
nsrc = 1024
src_coord = np.random.rand(nsrc, 3)
src_coord[:, 2] *= 0.1
src_value = np.random.rand(nsrc, 3)
ntrg = 1024
trg_coord = np.random.rand(ntrg, 3)
trg_coord[:, 2] *= 0.1
trg_value = np.zeros((ntrg, 3))

src_coord = src_coord.flatten()
trg_coord = trg_coord.flatten()
src_value = src_value.flatten()
trg_value = trg_value.flatten()
comm.Barrier()
print('# Points created rank', rank)
# Try FMM_Wrapper
pbc = fmm.FMM_Wrapper.PAXIS.NONE
myFMM = fmm.FMM_Wrapper(mult_order, max_pts, init_depth, pbc, False)
myFMM.FMM_SetBox(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
myFMM.FMM_UpdateTree(trg_coord, src_coord)
myFMM.FMM_Evaluate(trg_value, ntrg, src_value)
myFMM.FMM_DataClear()
myFMM.FMM_TreeClear()
myFMM.FMM_UpdateTree(trg_coord, src_coord)
myFMM.FMM_Evaluate(trg_value, ntrg, src_value)
# Try FMM_WrapperWall2D
pbc = fmm.FMM_WrapperWall2D.PAXIS.NONE
myFMM = fmm.FMM_WrapperWall2D(mult_order, max_pts, init_depth, pbc)
myFMM.FMM_SetBox(-2.0, 2.0, -2.0, 2.0, 0.0, 1.0)
myFMM.FMM_UpdateTree(trg_coord, src_coord)
myFMM.FMM_Evaluate(trg_value, ntrg, src_value)
myFMM.FMM_DataClear()
myFMM.FMM_TreeClear()
myFMM.FMM_UpdateTree(trg_coord, src_coord)
myFMM.FMM_Evaluate(trg_value, ntrg, src_value)
comm.Barrier()
print('# End')
