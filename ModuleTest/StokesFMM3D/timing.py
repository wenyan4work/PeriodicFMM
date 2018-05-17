import os


os.system(' mpiexec -n 1 ./TestStokes3D.X -P 0 -S 5 -T 96 -R 0 | tee ./TimingP0T96R0')
os.system(' mpiexec -n 1 ./TestStokes3D.X -P 0 -S 5 -T 96 -R 1 | tee ./TimingP0T96R1')

os.system(' mpiexec -n 1 ./TestStokes3D.X -P 3 -S 5 -T 96 -R 0 | tee ./TimingP3T96R0')
os.system(' mpiexec -n 1 ./TestStokes3D.X -P 3 -S 5 -T 96 -R 1 | tee ./TimingP3T96R1')

os.system(' mpiexec -n 1 ./TestStokes3D.X -P 4 -S 5 -T 96 -R 0 | tee ./TimingP4T96R0')
os.system(' mpiexec -n 1 ./TestStokes3D.X -P 4 -S 5 -T 96 -R 1 | tee ./TimingP4T96R1')

os.system(' mpiexec -n 1 ./TestStokes3D.X -P 7 -S 5 -T 96 -R 0 | tee ./TimingP7T96R0')
os.system(' mpiexec -n 1 ./TestStokes3D.X -P 7 -S 5 -T 96 -R 1 | tee ./TimingP7T96R1')
