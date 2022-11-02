#!/bin/bash -l

P=(1 2 4 8 16 32 56)
points=(4096 5792 8192 11585 16384 23170 30651)
iteration=50
num_nodes=1
count=0
max=7

echo "Weak scaling"
for i in ${!P[@]}
do
    echo "Running with ${P[$i]} processors and ${points[$i]} points and ${iteration} iterations"
    srun --nodes 2 -n ${P[$i]} ./poisson ${point[$i]} 50 0 
    srun --nodes 2 -n ${P[$i]} ./poisson ${point[$i]} 50 1
done
echo