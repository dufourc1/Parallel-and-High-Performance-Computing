#!/bin/bash -l

P=(1 2 4 8 16 32 56)
points=(512 1024 2048 4096)
iteration=50
num_nodes=1
count=0
max=28
module purge
module load intel intel-mpi
echo "Weak scaling"

for iter in {1..2}; do
    for p in ${P[@]}
    do
        echo "Running with $p processes";
        if [ $p -ge 14 ];  then num_nodes=2; fi
        for point in ${points[@]} 
        do
            echo "$count / $max"
            count=$((count+1))
            srun  --reservation=Course-math-454-week -A math-454 --nodes $num_nodes  -n $p ./poisson $point $iteration 0 >> output/strong_scaling_sync.txt &
            srun  --reservation=Course-math-454-week -A math-454 --nodes $num_nodes -n $p ./poisson $point $iteration 1 >> output/strong_scaling_async.txt &
        done
    done
done
echo "Done"
