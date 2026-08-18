#!/bin/bash
# =====================================================================================
# Iterative loop: REPS-constrained policy -> explore -> retrain world models -> repeat.
#
#   ./run_reps_loop.sh              # 10 iterations
#   ./run_reps_loop.sh 3 9          # resume at iteration 3
#
# Each iteration:
#   [1] train the policy on the CURRENT world models, objective
#           max reward  s.t.  W2_h <= eta
#       as  loss = -LCB_reward + alpha*relu(W2_h - eta) + chance penalties
#   [2] explore ONE episode with it, deterministic
#   [3] retrain the three phase models on everything collected so far
#
# ISOLATED from the original experiment: results_reps/, configdata/pensim_reps/, and
# cdil_policy_clean.py is untouched.
#
# ITERATION 0 STARTS FROM THE EXISTING MODELS. The pretrained repro_bnd_rbf1.pt is a
# POLICY, not a world model, so it cannot seed stage 3; the three phase models it was
# trained against are used instead and are copied into results_reps/ at iteration 0.
#
# ETA = 0.23, measured rather than chosen: imitation-only policies converge to
# 0.219-0.253 per-hour W2 in training units (cdil_policy_selected 0.2288,
# clean_lam0 0.2189), so eta is this architecture's best imitation and the natural
# feasibility boundary. Below it the constraint is inert and the policy pursues yield.
#
# THE STANDARDIZER IS PINNED across all ten iterations (-std_from), because it
# otherwise refits on a folder that grows every iteration: two models trained two days
# apart on the same folder saw 19953 vs 25290 transitions and pH's std moved
# 0.0153 -> 0.0329, which silently changes what z means and makes W2 and eta
# incomparable across iterations. eta is a fixed number, so the space it lives in must
# be fixed too.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_reps
DATA=$CFG/pensim_reps
mkdir -p "$RES" "$DATA"

FROM=${1:-0}
TO=${2:-9}
SRC_PREFIX=${SRC_PREFIX:-$REPO/results_pensim/rbf_model_bnd_rbf_iter0}
REWARD=${REWARD:-$REPO/results_pensim/reward_model_base.pt}
ETA=${ETA:-0.23}
ALPHA_W2=${ALPHA_W2:-100.0}
KAPPA=${KAPPA:-1.0}
ITERS=${ITERS:-20}
N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}

[ -f "$REWARD" ] || { echo "reward model not found: $REWARD" >&2; exit 1; }

# seed: baseline CSVs, and iteration 0's models are the existing ones
if [ ! -f "$DATA/gpei_batch_0.csv" ]; then
  cp "$CFG"/pensimenv/*.csv "$DATA/"
  echo "seeded $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
fi
if [ ! -f "$RES/reps_model_iter0_phase0.pt" ]; then
  for p in 0 1 2; do cp "${SRC_PREFIX}_phase${p}.pt" "$RES/reps_model_iter0_phase${p}.pt"; done
  echo "iteration 0 models copied from $SRC_PREFIX"
fi

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

echo "=========================================================================="
echo " REPS loop: iterations $FROM..$TO"
echo "   eta=$ETA  alpha_W2=$ALPHA_W2  kappa=$KAPPA  policy iters=$ITERS"
echo "   reward model: $(basename "$REWARD")"
echo "   data        : $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
echo "=========================================================================="

DEP=""
for IT in $(seq "$FROM" "$TO"); do
  MODEL=$RES/reps_model_iter${IT}
  NEXT=$RES/reps_model_iter$((IT+1))
  POL=$RES/reps_policy_iter${IT}.pt
  PREV=$RES/reps_policy_iter$((IT-1)).pt
  WARM=""; [ -f "$PREV" ] && WARM="-init_policy $PREV"
  D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"

  echo "--- iteration $IT"

  # [1] policy: maximise reward subject to W2 <= eta
  J1=$(sbatch --parsable $SB $D --job-name="rp${IT}" --time=14:00:00 \
    --output="$RES/it${IT}_policy_%j.out" --error="$RES/it${IT}_policy_%j.err" \
    --wrap="$CONDA python -u policy_learning/cdil_policy_reps.py \
            -phase_prefix $MODEL -reward_model $REWARD \
            -eta $ETA -alpha_w2 $ALPHA_W2 -kappa $KAPPA \
            -iters $ITERS -out $POL $WARM")
  echo "  [1/3] policy  -> job $J1"

  # [2] one deterministic episode
  J2=$(sbatch --parsable $SB --dependency=afterok:$J1 --job-name="rx${IT}" --time=06:00:00 \
    --output="$RES/it${IT}_explore_%j.out" --error="$RES/it${IT}_explore_%j.err" \
    --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
            -policy $POL -out $DATA -tag reps_iter${IT} -n 1 -p_dropout 0.0")
  echo "  [2/3] explore -> job $J2"

  # [3] retrain the three phase models on the accumulated data, z-space PINNED
  if [ "$IT" -lt "$TO" ]; then
    TIDS=()
    for P in 0 1 2; do
      J=$(sbatch --parsable $SB --dependency=afterok:$J2 --job-name="rm${IT}p$P" \
        --time=10:00:00 \
        --output="$RES/it${IT}_m${P}_%j.out" --error="$RES/it${IT}_m${P}_%j.err" \
        --wrap="$CONDA python -u train_rbf_pensim.py -phase $P -data_dir $DATA \
                -tag reps_it$((IT+1))_p${P} -save_dir $RES -n_keep $N_KEEP \
                -n_epoch $N_EPOCH -select pivchol \
                -std_from ${SRC_PREFIX}_phase0.pt; \
                cp $RES/rbf_model_reps_it$((IT+1))_p${P}.pt ${NEXT}_phase${P}.pt")
      TIDS+=("$J")
    done
    DEP=$(IFS=:; echo "${TIDS[*]}")
    echo "  [3/3] models  -> ${TIDS[*]}"
  else
    DEP=$J2
  fi
done

echo
echo "watch  : squeue -u \$USER"
echo "W2/eta : grep -h 'hours violating' $RES/it*_policy_*.out"
echo "reward : grep -h 'reward(LCB)' $RES/it*_policy_*.out"
echo "yields : grep -h 'total yield' $RES/it*_explore_*.out"
echo "models : grep -h 'mean MSE' $RES/it*_m*_*.out"
