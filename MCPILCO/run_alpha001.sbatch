#!/bin/bash
#SBATCH --job-name=reptile_a001
#SBATCH --partition=Teaching
#SBATCH --output=alpha001_%j.out
#SBATCH --error=alpha001_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

ALPHA=0.001
K=5

echo "=========================================="
echo "Starting alpha = $ALPHA"
echo "k = $K"
echo "alpha*k = $(python -c "print($ALPHA * $K)")"
echo "Job ID = $SLURM_JOB_ID"
echo "Node = $(hostname)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"

mkdir -p results_pensim/alpha_sweep

PENSIM_PHASE_BOUNDS="47.5,72.5" python policy_learning/reptile_policy_opt.py \
    -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 \
    -lam 0.01 \
    -iters 20 \
    -inner_k "$K" \
    -lr "$ALPHA" \
    -tasks 10 \
    -out "results_pensim/alpha_sweep/rep_alpha${ALPHA}_k${K}.pt"

echo "=========================================="
echo "Finished alpha = $ALPHA, k = $K"
echo "=========================================="
