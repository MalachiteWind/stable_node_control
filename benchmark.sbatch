#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 100:00:00
#SBATCH -p slurm
#SBATCH -J benchmark_training
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

SAVE_DIR="results/benchmark"
mkdir -p "$SAVE_DIR"


echo "Starting experiment with decay_val=1.0"
python benchmark.py --epochs 1000 --lr 1e-2 --save_folder "$SAVE_DIR/decay_1" --decay_val 1.0 --seed 1234
echo "Finished experiment with decay_val=1.0"


echo "All experiments finished."
