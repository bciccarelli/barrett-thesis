#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G a100:1
#SBATCH -t 5:00:00
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0

source activate myEnv

export HF_HOME=/scratch/bkciccar/huggingface
export WANDB_CACHE_DIR=/scratch/bkciccar/wandb

wandb login $WANDB_API_KEY
axolotl train config.yml