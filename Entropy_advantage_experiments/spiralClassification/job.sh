#!/bin/bash
#SBATCH -t 3-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=4G
#SBATCH --partition=low_highcore

source ~/.bashrc
module load gcc/9.2.0
export OMP_NUM_THREADS=16

../../a.out < input.txt > output.txt
rm Conf*.agr
