#!/bin/bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

for i in 3x6 3x8 4x6 4x8
do
	mkdir $i
	cd $i
	for j in {1..100}
	do
		mkdir $j
		cd $j
		echo "$i" > input.txt
		echo "$j" >> input.txt
		python3 ../../train.py < input.txt > output.txt &
		if (( $j % 6 == 0 ))
		then wait
		fi
		cd ..
	done
	wait
	cd ..
done
