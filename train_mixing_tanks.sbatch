#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 20:00:00
#SBATCH -p slurm
#SBATCH -J g_bounds
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

# SAVE_DIR="results/mixing_tanks_batch_sweep"
# mkdir -p "$SAVE_DIR"

# echo "Starting experiment with batch_size=50"
# python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 50 --save_folder "$SAVE_DIR/batch_50" --decay_val 0.0 --seed 1234
# echo "Finished experiment with batch_size=50"

# echo "Starting experiment with batch_size=100"
# python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 100 --save_folder "$SAVE_DIR/batch_100" --decay_val 0.0 --seed 1234
# echo "Finished experiment with batch_size=100"

# echo "Starting experiment with batch_size=150"
# python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 150 --save_folder "$SAVE_DIR/batch_150" --decay_val 0.0 --seed 1234
# echo "Finished experiment with batch_size=150"


# ----------------------------------------------------------------------
# EXPERIMENTS (t_max observations)
# ----------------------------------------------------------------------
# SAVE_DIR="results/mixing_tanks_batch_sweep_scarce_2"
# mkdir -p "$SAVE_DIR"

# echo "Starting experiment with t_max=25"
# python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 50 --save_folder "$SAVE_DIR/t_max_25" --decay_val 0.0 --seed 1234 --t_max 25.0 --trial_rate 20
# echo "Finished experiment with t_max=25"

# echo "Starting experiment with t_max=40"
# python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 50 --save_folder "$SAVE_DIR/t_max_40" --decay_val 0.0 --seed 1234 --t_max 40.0 --trial_rate 20
# echo "Finished experiment with t_max=40"

# echo "Starting experiment with t_max=75"
# python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 50 --save_folder "$SAVE_DIR/t_max_75" --decay_val 0.0 --seed 1234 --t_max 75.0 --trial_rate 20
# echo "Finished experiment with t_max=75"

# ----------------------------------------------------------------------
# small batch_size + varying g bounds
# ----------------------------------------------------------------------
SAVE_DIR="results/g_bounds"
mkdir -p "$SAVE_DIR"

echo "Starting experiment with g_min=0 g_max=1"
python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 25 --save_folder "$SAVE_DIR/g_0_1" --decay_val 0.0 --seed 1234 --t_max 100.0 --trial_rate 10 --g_min=0.0 --g_max=1.0
echo "Finished experiment with g_min=0 g_max=1"

echo "Starting experiment with g_min=-5 g_max=5"
python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 25 --save_folder "$SAVE_DIR/g_5_5" --decay_val 0.0 --seed 1234 --t_max 100.0 --trial_rate 10 --g_min=-5.0 --g_max=5.0
echo "Finished experiment with g_min=-5 g_max=5"

echo "Starting experiment with g_min=-10 g_max=10"
python train_mixing_tanks.py --epochs 1000 --lr 1e-2 --batch_size 25 --save_folder "$SAVE_DIR/g_10_10" --decay_val 0.0 --seed 1234 --t_max 100.0 --trial_rate 10 --g_min=-10.0 --g_max=10.0
echo "Finished experiment with g_min=-10 g_max=10"




echo "All experiments finished."
