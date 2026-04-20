#!/bin/bash
#SBATCH --job-name=ler_chunks
#SBATCH --partition=common,gpu-common,scavenger,scavenger-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=24:00:00
#SBATCH --array=0-899
#SBATCH --output=logs/ler_%A_%a.out
#SBATCH --error=logs/ler_%A_%a.err

mkdir -p logs
mkdir -p cluster_ler_chunks

python run_ler_chunk.py \
    --task_id ${SLURM_ARRAY_TASK_ID} \
    --shots_per_chunk 1000 \
    --num_chunks 100 \
    --outdir cluster_ler_chunks \
    --l 3 \
    --basis X \
    --CD_type ZXXZonSqu \
    --d 9