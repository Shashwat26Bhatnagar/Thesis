#!/bin/bash
#SBATCH --job-name=reptile_k_sweep
#SBATCH --partition=Teaching
#SBATCH --output=ksweep_%A_%a.out
#SBATCH --error=ksweep_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# k-sweep at a FIXED alpha*k, holding the ratio at the paper's own validated
# Table 3 (Omniglot 5-way) setting: lr=0.001, inner_iterations=5 -> alpha*k=0.005.
# Because INNER_LR = LR * K_REF / k inside reptile_policy_opt.py (K_REF=5 fixed
# in that file), setting -lr 0.001 and only varying -inner_k keeps alpha*k pinned
# at exactly 0.005 for every k below -- no per-k alpha needs to be computed by hand.

LR=0.001
K_VALUES=(5 8 10 15 20)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Starting k=$K  (lr=$LR fixed, so alpha*k = ${LR} * 5 = 0.005)"
echo "Job ID = $SLURM_JOB_ID"
echo "Array task = $SLURM_ARRAY_TASK_ID"
echo "Node = $(hostname)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p results_pensim/k_sweep_T0.005

PENSIM_PHASE_BOUNDS="47.5,72.5" python policy_learning/reptile_policy_opt.py \
    -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 \
    -lam 0.01 \
    -iters 20 \
    -inner_k "$K" \
    -lr "$LR" \
    -tasks 10 \
    -out "results_pensim/k_sweep_T0.005/rep_k${K}.pt"

echo "=========================================="
echo "Finished k=$K  (lr=$LR, alpha*k=0.005)"
echo "=========================================="
