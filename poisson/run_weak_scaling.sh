#!/bin/bash
#SBATCH -n 5
module purge
module load intel intel-mpi


for iter in {1..2}; do
    printf "Starting iteration ${iter}\n"
    for n in 1 2 4 8 16 32 56; do
        size=$(echo "scale=0; 4096 * sqrt(${n}.0)" | bc)
        srun  --reservation=Course-math-454-week -A math-454 -n $n poisson $size 50 $s 0 >> output/weak_scaling_sync.txt &
        srun  --reservation=Course-math-454-week -A math-454 -n $n poisson $size 50 $s 1 >> output/weak_scaling_async.txt &
    done
done