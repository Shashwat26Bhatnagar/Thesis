#!/bin/bash
# =====================================================================================
# Full pipeline with PHASE-SPECIFIC world models, submitted as one dependency chain:
#
#   [1] train THREE world models, one per fermentation phase        (3 parallel jobs)
#   [2] optimise ONE policy per lambda against all three            (N parallel jobs)
#   [3] evaluate the policies and SELECT a lambda                   (no simulator)
#   [4] explore with the SELECTED policy -> 10 new CSVs
#
#   ./run_phase_pipeline.sh bnd 0
#   PENSIM_PHASE_BOUNDS="47.5,72.5" ./run_phase_pipeline.sh bnd 0
#   LAMBDAS="0 0.01 0.1 1" ./run_phase_pipeline.sh bnd 0
#
# PHASE BOUNDARIES
# Set with PENSIM_PHASE_BOUNDS="b0,b1" -> phases [0,b0), [b0,b1), [b1,inf). The default
# is 35,51. The causal-transition analysis of the penicillin process reports 47.5 h and
# 72.5 h, so pass those if the split is meant to follow that paper. The value used here
# is what the trained models -- and the reported results -- will describe, and it is
# recorded in the log.
#
# WHY THE POLICY STAGE OMITS -model
# cdil_policy_optimization.py loads three phase models when -model is absent, choosing
# one per window by the expert's time. -phase_prefix tells it which files to use, so
# each iteration's models stay separate.
#
# A CAVEAT WORTH WATCHING
# Phase 1 is the narrowest window. Last time it held only 3.5% of the transitions and
# its model trained on 280 points against 800 for the others -- thin enough that its
# kernel matrix became rank-deficient. The per-phase counts are printed by the training
# jobs; check them before trusting the phase-1 model.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_pensim
mkdir -p "$RES"

if [ $# -lt 2 ]; then
    echo "usage: $0 <bnd|unb> <iteration>" >&2
    exit 1
fi
VAR=$1
IT=$2
PREV=$((IT - 1))

POLICY_KIND=${POLICY_KIND:-rbf}
LAMBDAS=${LAMBDAS:-"0 0.001 0.01 0.1 1.0"}
N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
N_POLICY_ITERS=${N_POLICY_ITERS:-20}
N_EXPLORE=${N_EXPLORE:-10}
TOL=${TOL:-0.05}
PB=${PENSIM_PHASE_BOUNDS:-35.0,51.0}

TAG=${VAR}_${POLICY_KIND}
DATA=$CFG/pensim_$TAG
[ "$VAR" = "bnd" ] && CLIP="-clip10" || CLIP=""

PREFIX=$RES/rbf_model_${TAG}_iter${IT}          # -> ${PREFIX}_phase{0,1,2}.pt
BEST=$RES/cdil_policy_${TAG}_iter${IT}_best.pt
PREV_BEST=$RES/cdil_policy_${TAG}_iter${PREV}_best.pt

[ -d "$DATA" ] || { echo "dataset folder $DATA missing -- run ./run_dyna_loop.sh init" >&2; exit 1; }

WARM=""
if [ "$IT" -gt 0 ] && [ -f "$PREV_BEST" ]; then WARM="-init_policy $PREV_BEST"; fi

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

echo "=========================================================================="
echo " $VAR / $POLICY_KIND   iteration $IT   PHASE MODELS"
echo "   phase bounds : $PB  h   -> [0,${PB%%,*}) [${PB%%,*},${PB##*,}) [${PB##*,},inf)"
echo "   data         : $DATA  ($(ls "$DATA"/*.csv 2>/dev/null | wc -l) CSVs)"
echo "   models       : ${PREFIX}_phase{0,1,2}.pt"
echo "   lambdas      : $LAMBDAS"
echo "   warm start   : ${WARM:-none}"
echo "=========================================================================="

# ---- [1] three phase models, in parallel -----------------------------------
TIDS=()
for P in 0 1 2; do
  J=$(sbatch --parsable $SB --job-name="tr_${TAG}${IT}p$P" --time=10:00:00 \
    --output="$RES/phase_${TAG}${IT}_p${P}_%j.out" \
    --error="$RES/phase_${TAG}${IT}_p${P}_%j.err" \
    --wrap="$CONDA python -u train_rbf_pensim.py -phase $P -data_dir $DATA \
            -tag ${TAG}_iter${IT}_phase${P} -n_keep $N_KEEP -n_epoch $N_EPOCH \
            -select pivchol")
  TIDS+=("$J")
  echo "  [1/4] phase $P model   -> job $J"
done
TDEP=$(IFS=:; echo "${TIDS[*]}")

# ---- [2] one policy per lambda, all three models loaded --------------------
PIDS=()
REF=""
for L in $LAMBDAS; do
  LT="lam${L//./p}"
  POL=$RES/cdil_policy_${TAG}_iter${IT}_${LT}.pt
  [ "$L" = "0" ] && REF=$POL
  J=$(sbatch --parsable $SB --dependency=afterok:$TDEP \
      --job-name="po_${TAG}${IT}_${LT}" --time=14:00:00 \
      --output="$RES/phase_${TAG}${IT}_${LT}_%j.out" \
      --error="$RES/phase_${TAG}${IT}_${LT}_%j.err" \
      --wrap="$CONDA python -u policy_learning/cdil_policy_optimization.py \
              -phase_prefix $PREFIX -policy_kind $POLICY_KIND -lam $L \
              -iters $N_POLICY_ITERS -out $POL $WARM")
  PIDS+=("$J")
  echo "  [2/4] policy lam=$L    -> job $J"
done
[ -n "$REF" ] || { echo "ERROR: LAMBDAS must include 0 (the reference)" >&2; exit 1; }

# ---- [3] select lambda -- same three models, no simulator ------------------
PDEP=$(IFS=:; echo "${PIDS[*]}")
J3=$(sbatch --parsable $SB --dependency=afterok:$PDEP \
  --job-name="ev_${TAG}${IT}" --time=04:00:00 \
  --output="$RES/phase_${TAG}${IT}_eval_%j.out" \
  --error="$RES/phase_${TAG}${IT}_eval_%j.err" \
  --wrap="$CONDA python -u evaluate_lambda_sweep.py -phase_prefix $PREFIX \
          -ref $REF -tol $TOL \
          -cand '$RES/cdil_policy_${TAG}_iter${IT}_lam*.pt' \
          -out $RES/lambda_sweep_${TAG}_iter${IT}.json -select_out $BEST")
echo "  [3/4] select lambda   -> job $J3"

# ---- [4] explore with the selected policy ----------------------------------
J4=$(sbatch --parsable $SB --dependency=afterok:$J3 \
  --job-name="ex_${TAG}${IT}" --time=14:00:00 \
  --output="$RES/phase_${TAG}${IT}_explore_%j.out" \
  --error="$RES/phase_${TAG}${IT}_explore_%j.err" \
  --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
          -policy $BEST -out $DATA -tag ${TAG}_iter${IT} \
          -n $N_EXPLORE -p_dropout 0.25 $CLIP")
echo "  [4/4] explore         -> job $J4"

echo
echo "watch : squeue -u \$USER"
echo "phase counts: grep 'phase . \[' $RES/phase_${TAG}${IT}_p*_*.out"
echo "choice: cat $RES/cdil_policy_${TAG}_iter${IT}_best_selection.json"
echo "data  : ls $DATA/*.csv | wc -l"
echo "next  : ./run_phase_pipeline.sh $VAR $((IT + 1))"
