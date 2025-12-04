#!/bin/bash
#SBATCH -e slurm_%A_%a.err               # Error file location
#SBATCH -o slurm_%A_%a.out 
#SBATCH -p common, gpu-common, scavenger     # Show who you are and get priority
#SBATCH --array=0-999                # How many jobs do you have (the int variable $SLURM_ARRAY_TASK_ID)
#SBATCH -c 1
#SBATCH --mail-type=END        
#SBATCH --mail-user=am1155@duke.edu       # It will send you an email when the job is finishe$
#SBATCH --mem=10G                # Memory, keep it as 10G
#SBATCH --time=48:00:00            # Time limit hrs:min:sec
#SBATCH --job-name=code_cap_thresholds_batch.out                # How is your output is called, you n$

python3 compass_code_correlated_error.py 
