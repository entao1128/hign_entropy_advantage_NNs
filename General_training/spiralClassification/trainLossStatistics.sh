#!/bin/bash

for i in `ls -d */`
do
	cd $i

	rm losses.txt
	for j in `ls -d */`
	do
		cat $j/output.txt | grep 'train loss' | cut -d '=' -f 2 | cut -d ',' -f 1 >> losses.txt
	done

	echo $i
	cat losses.txt | awk '{x+=$0;y+=$0^2}END{print x/NR; print sqrt((y/NR-(x/NR)^2)/NR)}'

	cd ..
done
