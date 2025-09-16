#!/bin/bash

gamma=0.5
steps=6
alphas=($(python -c "import numpy as np; gamma=$gamma; steps=$steps; arr=[1*(gamma**i) for i in range(steps)]; arr=np.concatenate([arr,[0]]); print(' '.join([str(a) for a in arr]))"))

SAVE_DIR="results/sequential_decay_runs"
mkdir -p "$SAVE_DIR"
mkdir -p slurm_output

summary_file="slurm_output/job_chain_summary.txt"
echo "Job chain submission summary (date: $(date))" > "$summary_file"
echo "Gamma=$gamma, Steps=$steps, Alphas=${alphas[*]}" >> "$summary_file"
echo "" >> "$summary_file"

prev_jobid=""
prev_model=""

for alpha in "${alphas[@]}"; do
    folder="$SAVE_DIR/alpha_${alpha}"
    mkdir -p "$folder"

    # clean string for filenames (avoid decimal points)
    alpha_tag=$(echo $alpha | sed 's/\./p/')

    sbatch_cmd="sbatch"
    if [ -n "$prev_jobid" ]; then
        sbatch_cmd="$sbatch_cmd --dependency=afterok:$prev_jobid"
    fi

    jobid=$($sbatch_cmd <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH -A atscale
#SBATCH -t 20:00:00
#SBATCH -p slurm
#SBATCH -J decay_${alpha}
#SBATCH -N 1
#SBATCH -e slurm_output/slurm-alpha_${alpha_tag}-%j.err
#SBATCH -o slurm_output/slurm-alpha_${alpha_tag}-%j.out

module purge
module load python/miniconda24.4.0
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda deactivate
conda activate atscale

echo "Running with alpha=$alpha"
if [ -z "$prev_model" ]; then
    python sequential_decay_models.py --epochs 1000 --lr 1e-2 --save_folder "$folder" --decay_val "$alpha" --seed 1234
else
    python sequential_decay_models.py --epochs 1000 --lr 1e-2 --save_folder "$folder" --decay_val "$alpha" --model_config "$prev_model" --seed 1234
fi
EOT
)

    prev_jobid="$jobid"
    prev_model="$folder/model_opt_states.pt"

    echo "Submitted job $jobid for alpha=$alpha (logs: slurm_output/slurm-alpha_${alpha_tag}-%j.out)" | tee -a "$summary_file"
done

echo "" >> "$summary_file"
echo "All jobs submitted. Last job ID: $prev_jobid" >> "$summary_file"
