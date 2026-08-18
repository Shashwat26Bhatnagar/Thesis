#!/bin/bash
# =====================================================================================
# Ten iterations of: train policy -> explore ONE episode -> retrain the world models.
#
#   ./run_exp_loop.sh            # iterations 0..9
#   ./run_exp_loop.sh 3 9        # resume at 3
#
# EXACTLY the configuration that produced buggy_batch_0.csv:
#   - three phase world models, selected per window by expert time
#   - objective CHANGED: -LCB_reward + alpha*relu(W2_h - eta) + lambda*||a||^2
#     (everything else identical, including LAMBDA_A now being ACTIVE at 0.01)
#   - static action box alpha=1000, vessel floor alpha=1.0
#   - 5-step (1 h) windows, 150 per iteration, 20 iterations
#   - exploration with -cliprecipe AND BUGGY_RECIPE_CLIP=1, p_dropout 0.25
#
# The clip compares physical actions against smpl-normalised recipe bounds. It binds
# hard early and releases as the profile rises: on buggy_batch_0.csv sugar takes 70
# distinct values (7.0..121.5) and soilbean 35, while discharge and water take only 6
# each. So the early rows sit at the floor and the later ones are the policy's.
#
# Everything is isolated: results_expreps/ and configdata/pensim_expreps/, prefix "exp".
#
# THE STANDARDIZER IS PINNED (-std_from) across all ten iterations. It otherwise
# refits on a folder that grows each iteration -- two models trained two days apart on
# the same folder saw 19953 vs 25290 transitions and pH's std moved 0.0153 -> 0.0329,
# which changes what z means and makes results incomparable between iterations.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_expreps
DATA=$CFG/pensim_expreps
mkdir -p "$RES" "$DATA"

FROM=${1:-0}
TO=${2:-9}
SRC_PREFIX=${SRC_PREFIX:-$REPO/results_pensim/rbf_model_bnd_rbf_iter0}
LAM=${LAM:-0.01}
ETA=${ETA:-0.23}
ALPHA_W2=${ALPHA_W2:-15.0}
KAPPA=${KAPPA:-1.0}
REWARD=${REWARD:-$REPO/results_pensim/reward_model_base.pt}
ITERS=${ITERS:-20}
N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}
DROPOUT=${DROPOUT:-0.25}

[ -f "$REWARD" ] || { echo "reward model not found: $REWARD" >&2; exit 1; }
[ -f "${SRC_PREFIX}_phase0.pt" ] || { echo "models not found: ${SRC_PREFIX}_phase0.pt" >&2; exit 1; }

if [ ! -f "$DATA/gpei_batch_0.csv" ]; then
  cp "$CFG"/pensimenv/*.csv "$DATA/"
  echo "seeded $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
fi
if [ ! -f "$RES/xr_model_iter0_phase0.pt" ]; then
  for p in 0 1 2; do cp "${SRC_PREFIX}_phase${p}.pt" "$RES/xr_model_iter0_phase${p}.pt"; done
  echo "iteration 0 models copied from $SRC_PREFIX"
fi

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

echo "=========================================================================="
echo " exp loop: iterations $FROM..$TO"
echo "   eta=$ETA alpha_W2=$ALPHA_W2 kappa=$KAPPA lambda=$LAM"
echo "   policy iters=$ITERS  dropout=$DROPOUT  clip=BUGGY_RECIPE_CLIP"
echo "   reward: $(basename "$REWARD")"
echo "   data  : $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
echo "=========================================================================="

DEP=""
for IT in $(seq "$FROM" "$TO"); do
  MODEL=$RES/xr_model_iter${IT}
  NEXT=$RES/xr_model_iter$((IT+1))
  POL=$RES/xr_policy_iter${IT}.pt
  PREV=$RES/xr_policy_iter$((IT-1)).pt
  WARM=""; [ -f "$PREV" ] && WARM="-init_policy $PREV"
  D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"

  echo "--- iteration $IT"

  J1=$(sbatch --parsable $SB $D --job-name="xr${IT}" --time=14:00:00 \
    --output="$RES/it${IT}_policy_%j.out" --error="$RES/it${IT}_policy_%j.err" \
    --wrap="$CONDA python -u policy_learning/exp_policy_reps.py \
            -phase_prefix $MODEL -reward_model $REWARD -lam $LAM \
            -eta $ETA -alpha_w2 $ALPHA_W2 -kappa $KAPPA \
            -iters $ITERS -out $POL $WARM")
  echo "  [1/3] policy  -> job $J1"

  J2=$(sbatch --parsable $SB --dependency=afterok:$J1 --job-name="ex${IT}" --time=06:00:00 \
    --output="$RES/it${IT}_explore_%j.out" --error="$RES/it${IT}_explore_%j.err" \
    --wrap="$CONDA BUGGY_RECIPE_CLIP=1 python -u policy_learning/explore_with_policy.py \
            -policy $POL -cliprecipe -out $DATA -tag xr_iter${IT} \
            -n 1 -p_dropout $DROPOUT")
  echo "  [2/3] explore -> job $J2"

  if [ "$IT" -lt "$TO" ]; then
    TIDS=()
    for P in 0 1 2; do
      J=$(sbatch --parsable $SB --dependency=afterok:$J2 --job-name="em${IT}p$P" \
        --time=10:00:00 \
        --output="$RES/it${IT}_m${P}_%j.out" --error="$RES/it${IT}_m${P}_%j.err" \
        --wrap="$CONDA python -u train_rbf_pensim.py -phase $P -data_dir $DATA \
                -tag xr_it$((IT+1))_p${P} -save_dir $RES -n_keep $N_KEEP \
                -n_epoch $N_EPOCH -select pivchol -std_from ${SRC_PREFIX}_phase0.pt; \
                cp $RES/rbf_model_xr_it$((IT+1))_p${P}.pt ${NEXT}_phase${P}.pt")
      TIDS+=("$J")
    done
    DEP=$(IFS=:; echo "${TIDS[*]}")
    echo "  [3/3] models  -> ${TIDS[*]}"
  else
    DEP=$J2
  fi
done

echo
echo "watch : squeue -u \$USER"
echo "W2/eta: grep -h 'violating' $RES/it*_policy_*.out"
echo "reward: grep -h 'reward(LCB)' $RES/it*_policy_*.out"
echo "yields: grep -h 'total yield' $RES/it*_explore_*.out"
echo "rows  : head -3 $DATA/xr_iter0_batch_0.csv"

