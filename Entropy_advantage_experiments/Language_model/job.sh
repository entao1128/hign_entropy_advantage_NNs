#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100s:1
#SBATCH --partition=gpu_7d1g

source ~/.bashrc
conda activate torch-gpu

./GPU_paralleize_training.sh
