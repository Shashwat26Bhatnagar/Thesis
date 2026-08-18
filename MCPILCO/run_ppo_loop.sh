#!/bin/bash
# =====================================================================================
# Ten rounds of: imitation -> PPO fine-tune -> explore -> refit the world models.
#
#   ./run_ppo_loop.sh              # rounds 0..9
#   ./run_ppo_loop.sh 3 9          # resume at 3
#
# Each round:
#   [1] train the policy on the CURRENT world models with the imitation objective
#       (W2 covariance + L2 on actions + chance constraints, NO reward term)
#   [2] fine-tune it with PPO for reward maximisation, rolling out INSIDE the world
#       models -- full 1150-step episodes, so PPO optimises TOTAL batch production
#       rather than the average yield of an isolated hour
#   [3] explore ONE episode in the real simulator with the fine-tuned policy
#   [4] refit the three phase models on everything collected so far
#
# Isolated: results_ppo/ and configdata/pensim_ppo/.
#
# WHY THE TWO STAGES ARE SEPARATE RATHER THAN ONE COMBINED LOSS
# Adding a reward term directly to the imitation loss was tried and the two never
# balanced: at alpha=100 the constraint was 8.1x the reward, at 15 they were level but
# the policy still would not close the discharge valve. Splitting them means the
# imitation stage is a clean optimisation with no competing gradient, and PPO then
# moves from that solution rather than fighting it.
#
# THE STANDARDIZER IS PINNED (-std_from) across all ten rounds. It otherwise refits on
# a folder that grows each round: two models fitted two days apart on the same folder
# saw 19953 vs 25290 transitions and pH's std moved 0.0153 -> 0.0329, which changes
# what z means and makes rounds incomparable.
#
# EXPECTATION, STATED IN ADVANCE. The SMPL paper's own PPO on PenSim scores 2.5231
# mean reward against the recipe baseline's 3.3071 -- PPO from scratch does WORSE than
# the recipe there. The imitation policies here already reach 2.82-3.03 per step, so
# the whole value of this stage rests on the warm start.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_ppo
DATA=$CFG/pensim_ppo
mkdir -p "$RES" "$DATA"

FROM=${1:-0}
TO=${2:-9}
SRC_PREFIX=${SRC_PREFIX:-$REPO/results_pensim/rbf_model_bnd_rbf_iter0}
REWARD=${REWARD:-$REPO/results_pensim/reward_model_base.pt}
LAM=${LAM:-0.01}
ITERS=${ITERS:-20}
PPO_STEPS=${PPO_STEPS:-50000}
KAPPA=${KAPPA:-1.0}
N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}

[ -f "$REWARD" ] || { echo "reward model not found: $REWARD" >&2; exit 1; }
[ -f "${SRC_PREFIX}_phase0.pt" ] || { echo "models not found" >&2; exit 1; }

[ -f "$DATA/gpei_batch_0.csv" ] || cp "$CFG"/pensimenv/*.csv "$DATA/"
if [ ! -f "$RES/ppo_model_iter0_phase0.pt" ]; then
  for p in 0 1 2; do cp "${SRC_PREFIX}_phase${p}.pt" "$RES/ppo_model_iter0_phase${p}.pt"; done
  echo "round 0 models copied from $SRC_PREFIX"
fi

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

echo "=========================================================================="
echo " PPO loop: rounds $FROM..$TO"
echo "   imitation: W2 + ${LAM}*||a||^2 + chance, $ITERS iters, NO reward term"
echo "   PPO      : $PPO_STEPS timesteps in the world models, kappa=$KAPPA"
echo "   data     : $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
echo "=========================================================================="

DEP=""
for IT in $(seq "$FROM" "$TO"); do
  MODEL=$RES/ppo_model_iter${IT}
  NEXT=$RES/ppo_model_iter$((IT+1))
  IMIT=$RES/imit_policy_iter${IT}.pt
  POL=$RES/ppo_policy_iter${IT}.pt
  PREV=$RES/ppo_policy_iter$((IT-1)).pt
  WARM=""; [ -f "$PREV" ] && WARM="-init_policy $PREV"
  D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"

  echo "--- round $IT"

  # [1] imitation only: W2 + L2 + chance
  J1=$(sbatch --parsable $SB $D --job-name="pi${IT}" --time=14:00:00 \
    --output="$RES/it${IT}_imit_%j.out" --error="$RES/it${IT}_imit_%j.err" \
    --wrap="$CONDA python -u policy_learning/cdil_policy_clean.py \
            -phase_prefix $MODEL -lam $LAM -iters $ITERS -out $IMIT $WARM")
  echo "  [1/4] imitation -> job $J1"

  # [2] PPO fine-tuning inside the world models
  J2=$(sbatch --parsable $SB --dependency=afterok:$J1 --job-name="pp${IT}" \
    --time=12:00:00 \
    --output="$RES/it${IT}_ppo_%j.out" --error="$RES/it${IT}_ppo_%j.err" \
    --wrap="$CONDA python -u policy_learning/ppo_finetune.py \
            -phase_prefix $MODEL -reward_model $REWARD -init_policy $IMIT \
            -timesteps $PPO_STEPS -kappa $KAPPA -out $POL")
  echo "  [2/4] PPO       -> job $J2"

  # [3] one deterministic episode in the REAL simulator
  J3=$(sbatch --parsable $SB --dependency=afterok:$J2 --job-name="px${IT}" \
    --time=06:00:00 \
    --output="$RES/it${IT}_explore_%j.out" --error="$RES/it${IT}_explore_%j.err" \
    --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
            -policy $POL -out $DATA -tag ppo_iter${IT} -n 1 -p_dropout 0.0")
  echo "  [3/4] explore   -> job $J3"

  # [4] refit the three phase models, z-space pinned
  if [ "$IT" -lt "$TO" ]; then
    TIDS=()
    for P in 0 1 2; do
      J=$(sbatch --parsable $SB --dependency=afterok:$J3 --job-name="pm${IT}p$P" \
        --time=10:00:00 \
        --output="$RES/it${IT}_m${P}_%j.out" --error="$RES/it${IT}_m${P}_%j.err" \
        --wrap="$CONDA python -u train_rbf_pensim.py -phase $P -data_dir $DATA \
                -tag ppo_it$((IT+1))_p${P} -save_dir $RES -n_keep $N_KEEP \
                -n_epoch $N_EPOCH -select pivchol -std_from ${SRC_PREFIX}_phase0.pt; \
                cp $RES/rbf_model_ppo_it$((IT+1))_p${P}.pt ${NEXT}_phase${P}.pt")
      TIDS+=("$J")
    done
    DEP=$(IFS=:; echo "${TIDS[*]}")
    echo "  [4/4] models    -> ${TIDS[*]}"
  else
    DEP=$J3
  fi
done

echo
echo "watch  : squeue -u \$USER"
echo "imit W2: grep -h '^iter' $RES/it*_imit_*.out | tail -10"
echo "PPO    : grep -h 'ep_rew_mean\|ep_len_mean' $RES/it*_ppo_*.out | tail -20"
echo "yields : grep -h 'total yield' $RES/it*_explore_*.out"
