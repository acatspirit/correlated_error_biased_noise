import os
from pathlib import Path

# Parameter sweeps
Ls = [3, 5, 7]
Etas = [0.5, 1, 5]

# Output directory for generated scripts
script_dir = Path("slurm_scripts")
script_dir.mkdir(exist_ok=True)

# Template for Slurm job script
slurm_template = """#!/bin/bash
#SBATCH -J compass_l{l}_eta{eta}
#SBATCH -p common
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -o slurm_l{l}_eta{eta}_%A_%a.out
#SBATCH -e slurm_l{l}_eta{eta}_%A_%a.err
#SBATCH --array=0-99
#SBATCH --mail-type=END
#SBATCH --mail-user=am1155@duke.edu

rep=$SLURM_ARRAY_TASK_ID
OUTDIR=outputs/l_{l}/eta_{eta}
mkdir -p "$OUTDIR"

python3 compass_code_correlated_error.py \\
    --l {l} \\
    --eta {eta} \\
    --rep "$rep" \\
    --outdir "$OUTDIR"
"""

# Generate one script per (l, eta)
for l in Ls:
    for eta in Etas:
        script_content = slurm_template.format(l=l, eta=eta)
        script_path = script_dir / f"l{l}_eta{eta}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)  # make it executable
        print(f"Generated {script_path}")