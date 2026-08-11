#!/bin/bash
# Train policies with the MINIMUM-ACTION penalty against the three phase models.
#
#   ./run_minact.sh                          # lambda 0 and 0.03
#   LAMBDAS="0 0.01 0.03 0.1" ./run_minact.sh
#
# The penalty is  lambda * sum_i ((a_i - lo_i)/span_i)^2  -- distance from each
# channel's PHYSICAL MINIMUM, not from the dataset mean. ||a||^2 in z-units measures
# distance from z=0, which IS the dataset-mean action (sugar 76.5, discharge 254), so
# it can only ever pull actions TO the mean: measured lambda 0.3 -> sugar 77.2,
# lambda 1 -> 76.5, lambda 30 -> 76.5. Physically small actions are the FURTHEST point
# from where that penalty pulls.
#
# Results in results_minact/, exploration in configdata/pensim_minact/.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
RES=$REPO/results_minact
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
DATA=$CFG/pensim_minact
mkdir -p "$RES" "$DATA"
[ -f "$DATA/gpei_batch_0.csv" ] || cp "$CFG"/pensimenv/*.csv "$DATA/"

PREFIX=${PREFIX:-$REPO/results_retrain/rt_model}
LAMBDAS=${LAMBDAS:-"0 0.03"}
ITERS=${ITERS:-20}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}

if [ ! -f "${PREFIX}_phase0.pt" ]; then
    echo "phase models not found at ${PREFIX}_phase{0,1,2}.pt" >&2
    echo "set PREFIX=... to point at them" >&2
    exit 1
fi

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

echo "models : ${PREFIX}_phase{0,1,2}.pt"
echo "lambdas: $LAMBDAS"

PIDS=()
for L in $LAMBDAS; do
  LT="lam${L//./p}"
  J=$(sbatch --parsable $SB --job-name="ma_$LT" --time=14:00:00 \
    --output="$RES/minact_${LT}_%j.out" --error="$RES/minact_${LT}_%j.err" \
    --wrap="$CONDA python -u policy_learning/cdil_policy_minact.py \
            -phase_prefix $PREFIX -lam $L -iters $ITERS -out $RES/minact_${LT}.pt")
  PIDS+=("$J"); echo "  lambda=$L -> job $J"
done

# one deterministic episode per policy, chained after training
DEP=$(IFS=:; echo "${PIDS[*]}")
for L in $LAMBDAS; do
  LT="lam${L//./p}"
  sbatch --dependency=afterok:$DEP $SB --job-name="mx_$LT" --time=06:00:00 \
    --output="$RES/explore_${LT}_%j.out" --error="$RES/explore_${LT}_%j.err" \
    --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
            -policy $RES/minact_${LT}.pt -out $DATA -tag minact_${LT} \
            -n 1 -p_dropout 0.0" > /dev/null
  echo "  explore lambda=$L -> queued"
done

echo
echo "watch : squeue -u \$USER"
echo "trend : grep -h 'dist_from_min  first' $RES/minact_lam*_*.out"
echo "rows  : head -3 $DATA/minact_lam0p03_batch_0.csv"
