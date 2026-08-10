#!/bin/bash
# =====================================================================================
# Full Dyna iteration with hyperparameter selection, submitted as one dependency chain:
#
#   [1] train world model on the accumulated dataset
#   [2] optimise ONE policy per lambda, in parallel, against that model
#   [3] evaluate them and SELECT a lambda      (no simulator involved)
#   [4] explore with the SELECTED policy -> 10 new CSVs back into the same folder
#
#   ./run_full_pipeline.sh bnd 0                     # first iteration
#   ./run_full_pipeline.sh bnd 1                     # after 0 completes
#   LAMBDAS="0 0.01 0.1 1" ./run_full_pipeline.sh bnd 0
#   POLICY_KIND=kan ./run_full_pipeline.sh bnd 0
#   ./run_dyna_loop.sh init                          # ONCE, seeds the data folders
#
# WHY STAGE 3 PUBLISHES TO A FIXED PATH
# All four stages are submitted up front, so stage 4 is created before stage 3 has
# decided anything. The evaluator therefore COPIES the winner to a canonical filename
# (-select_out), and stage 4 reads that path rather than a lambda-specific one.
#
# SELECTION USES NO SIMULATOR. Policies are compared on the frozen world model:
#   good hour  <=>  W2_lam(h) <= W2_ref(h)*(1+tol)  AND  ||a||_lam(h) < ||a||_ref(h)
# counted per hour rather than averaged, so two different action modes cannot average
# into a third that neither policy actually achieves.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_pensim
mkdir -p "$RES"

if [ $# -lt 2 ]; then
    echo "usage: $0 <bnd|unb> <iteration>" >&2
    echo "       LAMBDAS=\"0 0.01 0.1\" POLICY_KIND=rbf $0 bnd 0" >&2
    exit 1
fi
VAR=$1
IT=$2
PREV=$((IT - 1))

POLICY_KIND=${POLICY_KIND:-rbf}
case "$POLICY_KIND" in rbf|mlp|kan) ;; *) echo "POLICY_KIND must be rbf|mlp|kan" >&2; exit 1;; esac

LAMBDAS=${LAMBDAS:-"0 0.001 0.01 0.1 1.0"}
N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
N_POLICY_ITERS=${N_POLICY_ITERS:-20}
N_EXPLORE=${N_EXPLORE:-10}
TOL=${TOL:-0.05}

TAG=${VAR}_${POLICY_KIND}
DATA=$CFG/pensim_$TAG
[ "$VAR" = "bnd" ] && CLIP="-clip10" || CLIP=""

MODEL=$RES/rbf_model_${TAG}_iter${IT}.pt
BEST=$RES/cdil_policy_${TAG}_iter${IT}_best.pt
PREV_BEST=$RES/cdil_policy_${TAG}_iter${PREV}_best.pt

if [ ! -d "$DATA" ]; then
    echo "dataset folder $DATA missing -- run: ./run_dyna_loop.sh init" >&2
    exit 1
fi

# warm start from the previous iteration's SELECTED policy
WARM=""
if [ "$IT" -gt 0 ] && [ -f "$PREV_BEST" ]; then
    WARM="-init_policy $PREV_BEST"
fi

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; cd $REPO;"

echo "=========================================================================="
echo " $VAR / $POLICY_KIND   iteration $IT"
echo "   data    : $DATA  ($(ls "$DATA"/*.csv 2>/dev/null | wc -l) CSVs)"
echo "   model   : $MODEL"
echo "   lambdas : $LAMBDAS"
echo "   warm    : ${WARM:-none (fresh policy)}"
echo "   explore : $N_EXPLORE episodes ${CLIP:+(+/-10% clipped)}"
echo "=========================================================================="

# ---- [1] world model -------------------------------------------------------
J1=$(sbatch --parsable $SB --job-name="tr_${TAG}${IT}" --time=10:00:00 \
  --output="$RES/pipe_${TAG}${IT}_train_%j.out" \
  --error="$RES/pipe_${TAG}${IT}_train_%j.err" \
  --wrap="$CONDA python -u train_rbf_pensim.py -phase -1 -data_dir $DATA \
          -tag ${TAG}_iter${IT} -n_keep $N_KEEP -n_epoch $N_EPOCH -select pivchol")
echo "  [1/4] train           -> job $J1"

# ---- [2] one policy per lambda, all against that model ---------------------
PIDS=()
REF=""
for L in $LAMBDAS; do
  LT="lam${L//./p}"
  POL=$RES/cdil_policy_${TAG}_iter${IT}_${LT}.pt
  [ "$L" = "0" ] && REF=$POL
  J=$(sbatch --parsable $SB --dependency=afterok:$J1 \
      --job-name="po_${TAG}${IT}_${LT}" --time=14:00:00 \
      --output="$RES/pipe_${TAG}${IT}_${LT}_%j.out" \
      --error="$RES/pipe_${TAG}${IT}_${LT}_%j.err" \
      --wrap="$CONDA python -u policy_learning/cdil_policy_optimization.py \
              -model $MODEL -policy_kind $POLICY_KIND -lam $L \
              -iters $N_POLICY_ITERS -out $POL $WARM")
  PIDS+=("$J")
  echo "  [2/4] policy lam=$L    -> job $J"
done
if [ -z "$REF" ]; then
  echo "ERROR: LAMBDAS must include 0 -- it is the reference the others are scored against" >&2
  exit 1
fi

# ---- [3] select lambda (world model only, no simulator) --------------------
DEP=$(IFS=:; echo "${PIDS[*]}")
J3=$(sbatch --parsable $SB --dependency=afterok:$DEP \
  --job-name="ev_${TAG}${IT}" --time=04:00:00 \
  --output="$RES/pipe_${TAG}${IT}_eval_%j.out" \
  --error="$RES/pipe_${TAG}${IT}_eval_%j.err" \
  --wrap="$CONDA python -u evaluate_lambda_sweep.py -model $MODEL \
          -ref $REF -tol $TOL \
          -cand '$RES/cdil_policy_${TAG}_iter${IT}_lam*.pt' \
          -out $RES/lambda_sweep_${TAG}_iter${IT}.json \
          -select_out $BEST")
echo "  [3/4] select lambda   -> job $J3"

# ---- [4] explore with whichever policy stage 3 selected --------------------
J4=$(sbatch --parsable $SB --dependency=afterok:$J3 \
  --job-name="ex_${TAG}${IT}" --time=14:00:00 \
  --output="$RES/pipe_${TAG}${IT}_explore_%j.out" \
  --error="$RES/pipe_${TAG}${IT}_explore_%j.err" \
  --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
          -policy $BEST -out $DATA -tag ${TAG}_iter${IT} \
          -n $N_EXPLORE -p_dropout 0.25 $CLIP")
echo "  [4/4] explore         -> job $J4"

echo
echo "watch : squeue -u \$USER"
echo "logs  : tail -f $RES/pipe_${TAG}${IT}_*_*.out"
echo "choice: cat $RES/cdil_policy_${TAG}_iter${IT}_best_selection.json"
echo "next  : POLICY_KIND=$POLICY_KIND ./run_full_pipeline.sh $VAR $((IT + 1))"
