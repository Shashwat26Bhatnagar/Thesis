#!/bin/bash
# =====================================================================================
# Clean lambda sweep: train one policy per lambda against the three phase models,
# select by the per-hour criterion, then explore with the winner.
#
#   ./run_clean_sweep.sh
#   LAMBDAS="0 0.001 0.003 0.01 0.03 0.1 0.3 1" ./run_clean_sweep.sh
#
# Everything lands in results_clean/ and configdata/pensim_clean/. Nothing existing is
# touched.
#
# The loss is W2 + lambda*||a||^2 + chance constraints, and nothing else: no Bernoulli
# gate, no sparsity KL, no violation multiplier, no separate valve policy, no discharge
# override. Each of those was tried and did not survive contact with the data.
#
# EXPLORATION IS DETERMINISTIC (-p_dropout 0). Dropout has repeatedly changed outcomes
# in ways that were not attributable to the policy -- one run completed 230 h at yield
# 3247 deterministic and drained at 121 h with dropout on -- so the comparable numbers
# are the deterministic ones.
#
# LAMBDAS MUST INCLUDE 0: it is the reference every other lambda is scored against.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
RES=$REPO/results_clean
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
DATA=$CFG/pensim_clean
mkdir -p "$RES" "$DATA"

PREFIX=${PREFIX:-$REPO/results_pensim/rbf_model_bnd_rbf_iter0}
LAMBDAS=${LAMBDAS:-"0 0.001 0.01 0.1 1.0"}
ITERS=${ITERS:-20}
TOL=${TOL:-0.05}
N_EXPLORE=${N_EXPLORE:-3}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}

# seed the exploration folder so the CSVs have somewhere to accumulate
if [ ! -f "$DATA/gpei_batch_0.csv" ]; then
  cp "$CFG"/pensim_bnd/random_batch_*.csv "$CFG"/pensim_bnd/gpei_batch_*.csv "$DATA/"
  echo "seeded $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
fi

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

echo "prefix : $PREFIX"
echo "lambdas: $LAMBDAS"
echo "bounds : $PB"

PIDS=(); REF=""
for L in $LAMBDAS; do
  LT="lam${L//./p}"
  POL=$RES/clean_${LT}.pt
  [ "$L" = "0" ] && REF=$POL
  J=$(sbatch --parsable $SB --job-name="c_$LT" --time=14:00:00 \
    --output="$RES/clean_${LT}_%j.out" --error="$RES/clean_${LT}_%j.err" \
    --wrap="$CONDA python -u policy_learning/cdil_policy_clean.py \
            -phase_prefix $PREFIX -lam $L -iters $ITERS -out $POL")
  PIDS+=("$J"); echo "  lambda=$L -> job $J"
done
[ -n "$REF" ] || { echo "ERROR: LAMBDAS must include 0 (the reference)" >&2; exit 1; }

DEP=$(IFS=:; echo "${PIDS[*]}")
J3=$(sbatch --parsable $SB --dependency=afterok:$DEP --job-name=c_eval --time=04:00:00 \
  --output="$RES/clean_eval_%j.out" --error="$RES/clean_eval_%j.err" \
  --wrap="$CONDA python -u evaluate_lambda_sweep.py -phase_prefix $PREFIX \
          -ref $REF -tol $TOL -cand '$RES/clean_lam*.pt' \
          -out $RES/clean_sweep.json -select_out $RES/clean_best.pt")
echo "  eval -> job $J3"

sbatch --dependency=afterok:$J3 $SB --job-name=c_expl --time=14:00:00 \
  --output="$RES/clean_explore_%j.out" --error="$RES/clean_explore_%j.err" \
  --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
          -policy $RES/clean_best.pt -out $DATA -tag clean \
          -n $N_EXPLORE -p_dropout 0.0"
echo "  explore -> queued"
echo
echo "watch  : squeue -u \$USER"
echo "L2 check: grep -h '||a||^2  first' $RES/clean_lam*_*.out"
echo "sweep  : tail -30 $RES/clean_eval_*.out"
echo "yields : grep -h 'total yield' $RES/clean_explore_*.out"
