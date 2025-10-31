#!/bin/bash

process_num=8
gpu_count=4
epoches_per_launch=1000000   # this is the number of epoch run in each iteration, change this and the step 
launches=99

python_script='LM_GPU_Training_Parallelized.py'  # edits as needed for the training code
entropy_ave_script='average_entropy.py'  # edits as needed for the entropy average code
echo 'Process number is: '$process_num
process_max=$((process_num-1))

nvidia-cuda-mps-control -d

for i in $(seq 0 $process_max); do
	if ! test -e './process_'$i; then
		mkdir './process_'$i
		cp $python_script './process_'$i
	fi
done

for i in $(seq 50 $launches); do   # loop over steps/iterations, feel free to edit the number as needed
	start_epoch=$(($epoches_per_launch * $i))
	echo '  Starting training step '$start_epoch
	for j in $(seq 0 $process_max); do     # loop over GPU mahcines, feel free to edit the number as needed
		cd './process_'$j
		echo '      Starting GPU '$j
	    CUDA_VISIBLE_DEVICES=$((j%gpu_count)) python $python_script $start_epoch $epoches_per_launch $j >> 'output.log' 2>&1 &
	    cd ..
	    sleep 1
	done

	echo '    All GPU started'
	wait
	echo '    All training instances completed'

	python $entropy_ave_script $process_num

done
