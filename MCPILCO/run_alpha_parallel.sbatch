#!/bin/bash
#SBATCH --job-name=reptile_lr
#SBATCH --partition=Teaching
#SBATCH --output=lr_%A_%a.out
#SBATCH --error=lr_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Remaining learning rates / alpha values
LR_VALUES=(0.01 0.02 0.05)

LR=${LR_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Starting LR = $LR"
echo "alpha*k = ${LR}*5"
echo "Job ID = $SLURM_JOB_ID"
echo "Node = $(hostname)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"

mkdir -p results_pensim/lr_sweep

PENSIM_PHASE_BOUNDS="47.5,72.5" python policy_learning/reptile_policy_opt.py \
    -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 \
    -lam 0.01 \
    -iters 20 \
    -inner_k 5 \
    -lr "$LR" \
    -tasks 10 \
    -out "results_pensim/lr_sweep/rep_lr${LR}.pt"

echo "=========================================="
echo "Finished LR = $LR"
echo "=========================================="
