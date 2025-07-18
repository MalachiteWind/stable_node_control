#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 10:00:00
#SBATCH -p dl_shared
#SBATCH -J sample_size_exp
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -e slurm_output/slurm-%j.out
#SBATCH -o slurm_output/output-%j.out
module load python/miniconda24.4.0
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda deactivate
conda activate atscale

echo "Starting experiment with sample_rate=20"
python train_model.py --seed 1234 --sample_rate 20 --lr 1e-1 --n_epochs 300 --hidden_dim 2 --patience 100 --folder "results/july_17"

echo "All experiments finished."