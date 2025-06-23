
README

What the stuff in the sbatch file does...

#!/bin/bash
#SBATCH -p shared               # -p the partition you will run things on
#SBATCH -A ops                    # -A the project that will pay for the run
#SBATCH --reservation=ops  # -R if this is running under a reservation
#SBATCH -J onboarding        # -J the name of the job
#SBATCH -N 1                       # -N the number of nodes (machines) you want
#SBATCH -t 2:00:00               # -t the timelimit for the job  HH:MM:SS format
#SBATCH -n 8                        # -n the number of cpus you want
#SBATCH --gres=gpu:1          # --gres=gpu: the number of gpus you want
#SBATCH -o slurm_output.txt # -o standard output destination
#SBATCH -e slurm_output.txt # -e standard error destination




HOW TO RUN THE SBATCH FILE


> sbatch hello_world.sbatch


you can check that your job submitted by running

>squeue -u USERID


you should an output file created that hello world will be printed to

note that this job will only work if it can find numpy in your python environment.
