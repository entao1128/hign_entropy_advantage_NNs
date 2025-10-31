#!/bin/bash

dataSeeds=`seq 1 24`
labels="3x15"

for dataSeed in $dataSeeds
do
	mkdir -p dataSeed${dataSeed}
	cd dataSeed${dataSeed}
	for label in $labels
	do
		mkdir ${label}
		cd ${label}

		cp ../../inputs/${label}.txt ./input.txt
		echo "${dataSeed} $((dataSeed+100000)) $((dataSeed+200000))" >> input.txt
		cp ../../job.sh .
		sbatch job.sh

		cd ..
	done
	cd ..
	wait
done
