#!/bin/bash
#SBATCH -J ler_cutoff_scan              # Job name
#SBATCH -p common,scavenger,gpu-common,scavenger-gpu   # Partition (change if needed)
#SBATCH --array=0-63                  # 4*d × 4*p × 4*eta = 120 jobs
#SBATCH --mem=4G                       # Memory per job
#SBATCH --time=48:00:00                # Time limit
#SBATCH -o logs/ler_%A_%a.out          # STDOUT
#SBATCH -e logs/ler_%A_%a.err          # STDERR
#SBATCH --mail-type=END,FAIL           # Email notifications


# -------------------------
# Create directories
# -------------------------

mkdir -p results

# -------------------------
# Run your script
# -------------------------

python run_ler_cutoff.py
