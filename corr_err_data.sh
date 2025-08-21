#!/bin/bash
#SBATCH -e slurm_%A_%a.err               # Error file location
#SBATCH -o slurm_%A_%a.out 
#SBATCH -p common     # Show who you are and get priority
#SBATCH --array=0-14                # How many jobs do you have (the int variable $SLURM_ARRAY_TASK_ID)
#SBATCH -c 1
#SBATCH --mail-type=END        
#SBATCH --mail-user=am1155@duke.edu       # It will send you an email when the job is finishe$
#SBATCH --mem=10G                # Memory, keep it as 10G
#SBATCH --job-name=code_cap_thresholds_batch.out                # How is your output is called, you n$
#readarray -t PARAMS_ARRAY < params.txt
#PARAMS=${PARAMS_ARRAY[(($SLURM_ARRAY_TASK_ID - 1))]}
python3 compass_code_correlated_error.py 
