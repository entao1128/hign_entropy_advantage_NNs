#!/bin/bash
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=entaoyang@alumni.upenn.edu
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1

set -x

#load cuda tool kit from cluster
module load AI/pytorch_23.02-1.13.1-py3

python Kaggle_house_general_training_GPU.py 10 > out

#bash ./CA_house_GPU.sh

#cd /ocean/projects/cis230026p/entaoy/test_0420


#bash ./GPU_paralleize_training.sh
