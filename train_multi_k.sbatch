#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 60:00:00
#SBATCH -p dl_shared
#SBATCH -J multi_k_training
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -e slurm_output/slurm-%j.err
#SBATCH -o slurm_output/slurm-%j.out

# ----------------------------------------------------------------------
# ENVIRONMENT SETUP
# ----------------------------------------------------------------------
module purge
module load python/miniconda24.4.0
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh

conda deactivate
conda activate atscale

echo "Using Python version:"
python --version
echo "Using Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# ----------------------------------------------------------------------
# EXPERIMENTS
# ----------------------------------------------------------------------

SAVE_DIR="results/multi_k_runs"
mkdir -p "$SAVE_DIR"

echo "Starting experiment with decay_val=0.8"
python train_multi_k.py --epochs 1000 --lr 1e-2 --save_folder "$SAVE_DIR/decay_08" --decay_val 0.8
echo "Finished experiment with decay_val=0.8"

# echo "Starting experiment with decay_val=0.9"
# python train_multi_k.py --epochs 1000 --lr 1e-2 --save_folder "$SAVE_DIR/decay_05" --decay_val 0.5
# echo "Finished experiment with decay_val=0.5"


echo "All experiments finished."
