#!/bin/bash
# Train one policy per lambda (plus the lambda=0 reference) against a FIXED world
# model, then evaluate them with evaluate_lambda_sweep.py.
#
#   ./run_lambda_sweep.sh results_pensim/rbf_model_bnd_rbf_iter0.pt
#   LAMBDAS="0 0.001 0.01 0.1 1" ./run_lambda_sweep.sh <model>
#
# All runs share one world model, so lambda is the ONLY thing that differs.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${1:?usage: $0 <world-model.pt>}
LAMBDAS=${LAMBDAS:-"0 0.001 0.01 0.1 1.0"}
ITERS=${ITERS:-20}
SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; cd $PWD;"
mkdir -p results_pensim

IDS=()
for L in $LAMBDAS; do
  TAG="lam${L//./p}"
  J=$(sbatch --parsable $SB --job-name="po_$TAG" --time=12:00:00 \
      --output="results_pensim/sweep_${TAG}_%j.out" \
      --error="results_pensim/sweep_${TAG}_%j.err" \
      --wrap="$CONDA python -u policy_learning/cdil_policy_optimization.py \
              -model $MODEL -policy_kind rbf -lam $L -iters $ITERS \
              -out results_pensim/cdil_policy_${TAG}.pt")
  echo "lambda=$L -> job $J  (results_pensim/cdil_policy_${TAG}.pt)"
  IDS+=("$J")
done

DEP=$(IFS=:; echo "${IDS[*]}")
JE=$(sbatch --parsable $SB --dependency=afterok:$DEP --job-name=sweep_eval --time=04:00:00 \
     --output="results_pensim/sweep_eval_%j.out" --error="results_pensim/sweep_eval_%j.err" \
     --wrap="$CONDA python -u evaluate_lambda_sweep.py -model $MODEL \
             -ref results_pensim/cdil_policy_lam0.pt \
             -cand 'results_pensim/cdil_policy_lam*.pt'")
echo
echo "evaluation -> job $JE (after all policies finish)"
echo "watch: squeue -u \$USER"
echo "result: results_pensim/sweep_eval_*.out  and  results_pensim/lambda_sweep.json"
