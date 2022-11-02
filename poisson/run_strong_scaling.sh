#!/bin/bash -l

P=(1 2 4 8 16 32 56)
points=(512 1024 2048 4096)
iteration=50
num_nodes=1
count=0
max=28

echo "Weak scaling"
for p in ${P[@]}
do
    echo "Running with $p processes";
    if [ $p -ge 14 ];  then num_nodes=2; fi
    for point in ${points[@]} 
    do
        echo "$count / $max"
        count=$((count+1))
        srun --nodes $num_nodes  -n $p ./poisson $point $iteration 0 
        srun  --nodes $num_nodes -n $p ./poisson $point $iteration 1
    done
done

echo "Done"