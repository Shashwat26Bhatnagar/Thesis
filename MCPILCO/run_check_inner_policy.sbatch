#!/bin/bash
#SBATCH --job-name=inner_policy
#SBATCH --partition=Teaching
#SBATCH --account=general-teaching
#SBATCH --qos=teaching
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=results_inner_policy/inner_%A_%a.out
#SBATCH --error=results_inner_policy/inner_%A_%a.err

cd ~/Thesis/MCPILCO

source /opt/conda/etc/profile.d/conda.sh
conda activate rvgp

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "=== K=default, LR=default ==="
    PENSIM_PHASE_BOUNDS="47.5,72.5" python -u check_inner_policy.py
else
    echo "=== K=15, LR=0.00667 ==="
    K=15 LR=0.00667 PENSIM_PHASE_BOUNDS="47.5,72.5" python -u check_inner_policy.py
fi
