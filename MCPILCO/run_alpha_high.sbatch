#!/bin/bash
#SBATCH --job-name=reptile_hi
#SBATCH --partition=Teaching
#SBATCH --output=alpha_%A_%a.out
#SBATCH --error=alpha_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

ALPHA_VALUES=(0.03 0.04)

ALPHA=${ALPHA_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Starting alpha = $ALPHA"
echo "alpha*k = ${ALPHA}*5"
echo "Job ID = $SLURM_JOB_ID"
echo "Node = $(hostname)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"

mkdir -p results_pensim/alpha_sweep

PENSIM_PHASE_BOUNDS="47.5,72.5" python policy_learning/reptile_policy_opt.py \
    -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 \
    -lam 0.01 \
    -iters 20 \
    -inner_k 5 \
    -lr "$ALPHA" \
    -tasks 10 \
    -out "results_pensim/alpha_sweep/rep_alpha${ALPHA}.pt"

echo "=========================================="
echo "Finished alpha = $ALPHA"
echo "=========================================="
