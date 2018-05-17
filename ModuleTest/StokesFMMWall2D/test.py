import os
import multiprocessing


mpimin = 1
mpimax = 2
mpistep = 1
ompmin = 4
ompmax = 12
ompstep = 4


def test(mpi=1, omp=1, intel=1):
    os.environ['OMP_NUM_THREADS'] = str(omp)
    os.environ['MKL_NUM_THREADS'] = str(intel)
    print('export OMP_NUM_THREADS='+str(int(omp)))
    print('export MKL_NUM_THREADS='+str(int(intel)))

    # intel iomp5 must source compilervars.sh intel64 to work properly
    cmd = 'mpiexec -n ' + \
        str(mpi)+' ./TestStokesWall2D.X -S 2 -T 32 -P 0 -R 0'
    print(cmd)
    os.system(cmd)

    cmd = 'mpiexec -n ' + \
        str(mpi)+' ./TestStokesWall2D.X -S 2 -T 32 -P 1 -R 0'
    print(cmd)
    os.system(cmd)

    cmd = 'mpiexec -n ' + \
        str(mpi)+' ./TestStokesWall2D.X -S 2 -T 32 -P 4 -R 0'
    print(cmd)
    os.system(cmd)

    cmd = 'mpiexec -n ' + \
        str(mpi)+' ./TestStokesWall2D.X -S 2 -T 32 -P 7 -R 0'
    print(cmd)
    os.system(cmd)

    return


cpunumber = multiprocessing.cpu_count()

for mpi in range(mpimin, mpimax, mpistep):
    for omp in range(ompmin, ompmax, ompstep):
        if mpi*omp > cpunumber:
            continue
        test(mpi, omp, 1)
