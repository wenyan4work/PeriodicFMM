import os

os.system(
    'mpiexec -n 1 ./TestStokesWall2D.X -S 96 -T 96 -P 0 -R 1 -C 0 | tee ./testS96T96P0R1C0.log')

os.system(
    'mpiexec -n 1 ./TestStokesWall2D.X -S 96 -T 96 -P 4 -R 1 -C 0 | tee ./testS96T96P4R1C0.log')
