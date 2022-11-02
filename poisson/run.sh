#!/bin/bash -l
#SBATCH --reservation=Course-math-454
#SBATCH -A math-454
#SBATCH -N 1


module purge 
module load gcc mvapich2


srun  poisson 512 50