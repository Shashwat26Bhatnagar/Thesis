#!/bin/bash
#SBATCH --job-name=lr_sweep
#SBATCH --output=lr_sweep_%j.out
#SBATCH --error=lr_sweep_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=Teaching

# Activate your RVGP environment
source ~/path/to/your/rvgp/bin/activate

cd "$SLURM_SUBMIT_DIR"

for LR in 0.005 0.01 0.02 0.05; do
    echo "=== lr=$LR  (alpha*k = ${LR}*5)"

    PENSIM_PHASE_BOUNDS="47.5,72.5" python policy_learning/reptile_policy_opt.py \
        -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 \
        -lam 0.01 \
        -iters 20 \
        -inner_k 5 \
        -lr "$LR" \
        -tasks 10 \
        -out "/tmp/rep_lr${LR}.pt" \
        2>&1 | tail -12
done
