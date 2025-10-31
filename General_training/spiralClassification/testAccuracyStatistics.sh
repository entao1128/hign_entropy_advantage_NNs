#!/bin/bash

for i in `ls -d */`
do
	cd $i

	rm accuracies.txt
	for j in `ls -d */`
	do
		cat $j/output.txt | tail -n1 | cut -d ':' -f 2 | sed 's/\%//g' >> accuracies.txt
	done

	echo $i
	cat accuracies.txt | awk '{x+=$0;y+=$0^2}END{print x/NR; print sqrt((y/NR-(x/NR)^2)/NR)}'

	cd ..
done

