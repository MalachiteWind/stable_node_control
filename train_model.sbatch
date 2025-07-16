#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 12:00:00
#SBATCH -p slurm
#SBATCH -J sample_size_exp
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH -e slurm_output/slurm-%j.out
#SBATCH -o slurm_output/output-%j.out
module load python/miniconda24.4.0
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda deactivate
conda activate atscale
echo "Starting experiment with sample_rate=1"
python train_model.py --seed 1234 --sample_rate 1 --lr 1e-1 --n_epochs 200 --hidden_dim 4 --patience 50 --folder "results/july_18"

echo "Starting experiment with sample_rate=5"
python train_model.py --seed 1234 --sample_rate 5 --lr 1e-1 --n_epochs 200 --hidden_dim 4 --patience 50 --folder "results/july_18"

echo "Starting experiment with sample_rate=10"
python train_model.py --seed 1234 --sample_rate 10 --lr 1e-1 --n_epochs 200 --hidden_dim 4 --patience 50 --folder "results/july_18"

echo "Starting experiment with sample_rate=20"
python train_model.py --seed 1234 --sample_rate 20 --lr 1e-1 --n_epochs 200 --hidden_dim 4 --patience 50 --folder "results/july_18"

echo "All experiments finished."