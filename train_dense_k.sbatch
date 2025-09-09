#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 100:00:00
#SBATCH -p slurm
#SBATCH -J dense_k_training
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

SAVE_DIR="results/dense_k_runs"
mkdir -p "$SAVE_DIR"

# echo "Starting experiment with decay_val=1."
# python train_dense_k.py --epochs 1000 --lr 1e-2 --save_folder "$SAVE_DIR/decay_1" --decay_val 1.0
# echo "Finished experiment with decay_val=1."

# echo "Starting experiment with decay_val=0.8"
# python train_dense_k.py --epochs 1000 --lr 1e-2 --save_folder "$SAVE_DIR/decay_08" --decay_val 0.8 --seed 1234
# echo "Finished experiment with decay_val=0.8"

prev_model="$SAVE_DIR/decay_08/model_opt_states.pt"

echo "Starting experiment with decay_val=0.5"
python train_dense_k.py --epochs 1000 --lr 1e-2 --save_folder "$SAVE_DIR/decay_05" --decay_val 0.5 --model_config "$prev_model" --seed 1234
echo "Finished experiment with decay_val=0.5"


echo "All experiments finished."
