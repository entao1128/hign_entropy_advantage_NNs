#!/bin/bash

process_num=8
gpu_count=1
epoches_per_launch=50000   # this is the number of epoch run in each iteration, change this and the step 
launches=9999

python_script='MNIST_WLMD.py'  # edits as needed for the training code
entropy_ave_script='average_entropy.py'  # edits as needed for the entropy average code
echo 'Process number is: '$process_num
process_max=$((process_num-1))

nvidia-cuda-mps-control -d

for i in $(seq 2239 $launches); do   # loop over steps/iterations, feel free to edit the number as needed
	start_epoch=$(($epoches_per_launch * $i))
	echo '  Starting training step '$start_epoch
	for j in $(seq 0 $process_max); do     # loop over GPU mahcines, feel free to edit the number as needed
		cd './process_'$j
		echo '      Starting GPU '$j
	    CUDA_VISIBLE_DEVICES=$((j%gpu_count)) python3 $python_script $start_epoch $epoches_per_launch $j >> 'output.log' 2>&1 &
	    cd ..
	    sleep 0.1
	done

	echo '    All GPU started'
	wait
	echo '    All training instances completed'

	python3 $entropy_ave_script $process_num
	python3 plot.py &

done

wait
