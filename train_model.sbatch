#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 20:00:00
#SBATCH -p slurm
#SBATCH -J sample_size_exp
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH -e slurm_output/slurm-%j.out
#SBATCH -o slurm_output/output-%j.out

# Clear all loaded modules before loading your desired one
module purge

# Load your specific Miniconda module
module load python/miniconda24.4.0

# Source the conda profile script
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh

# Activate your conda environment
conda deactivate
conda activate atscale

# Confirm environment and Python version to debug compatibility
echo "Using Python version:"
python --version
echo "Using Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Start your experiment
echo "Starting experiment with noise=0.1"
python train_budworm.py --seed 1234 --noise 0.1 --sample_rate 10 --lr 1e-1 --n_epochs 300 --f_dim 6 --g_dim 2 --patience 100 --folder "results/july_22"
echo "Finished."

echo "Starting experiment with noise=0.05"
python train_budworm.py --seed 1234 --noise 0.05 --sample_rate 10 --lr 1e-1 --n_epochs 300 --f_dim 6 --g_dim 2 --patience 100 --folder "results/july_22"
echo "Finished."

echo "Starting experiment with noise=0.2"
python train_budworm.py --seed 1234 --noise 0.2 --sample_rate 10 --lr 1e-1 --n_epochs 300 --f_dim 6 --g_dim 2 --patience 100 --folder "results/july_22"
echo "Finished."

echo "All experiments finished."
