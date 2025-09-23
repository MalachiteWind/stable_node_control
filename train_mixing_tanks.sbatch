#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 100:00:00
#SBATCH -p slurm
#SBATCH -J mixing_tanks_batch_sweep
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

SAVE_DIR="results/mixing_tanks_batch_sweep"
mkdir -p "$SAVE_DIR"

echo "Starting experiment with batch_size=50"
python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 50 --save_folder "$SAVE_DIR/batch_50" --decay_val 0.0 --seed 1234
echo "Finished experiment with batch_size=50"

echo "Starting experiment with batch_size=100"
python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 100 --save_folder "$SAVE_DIR/batch_100" --decay_val 0.0 --seed 1234
echo "Finished experiment with batch_size=100"

echo "Starting experiment with batch_size=150"
python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 150 --save_folder "$SAVE_DIR/batch_150" --decay_val 0.0 --seed 1234
echo "Finished experiment with batch_size=150"

echo "All experiments finished."
