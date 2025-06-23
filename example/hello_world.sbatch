#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 00:05:00
#SBATCH -p dl_shared
#SBATCH -J hello_world
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -e slurm-%j.out
#SBATCH -o output-%j.out
module load python/miniconda24.4.0
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda deactivate
conda activate atscale
python hello.py

