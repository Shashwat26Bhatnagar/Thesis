#!/bin/bash
# Reward maximisation with imitation as an inductive bias, one policy per phase.
#
#   ./run_rlimit.sh
#
#   loss = -LCB_reward + 15*relu(W2 - eta) + lambda*||(a-lo)/span||^2 + chance
#
# WHY A CONSTRAINT ON W2 RATHER THAN A PENALTY. Replaying gpei's own actions through
# the world model shows the objective ranks the reference BELOW the mean action -- gpei
# wins at only 4 of 15 hours (mean W2 0.253 vs 0.229) despite reaching 3803 yield. So
# minimising W2 cannot produce reference-like behaviour. relu(W2 - eta) does not rank
# them: below eta the term is zero, both are feasible, and the REWARD decides.
#
# eta = 0.35 admits both gpei (0.19-0.34) and the mean action (0.18-0.30) at every
# sampled hour, so W2 acts as a guardrail.
#
# lambda IS PHASE-SPECIFIC. The penalty is distance from the PHYSICAL MINIMUM, and
# gpei is closer to it than the mean only in the growth phase (0.476 vs 1.234 at
# t<20 h; 1.735 vs 1.234 by t=20-47). So 0.2 in phase 0, 0 in phases 1-2.
#
# Discharge is pinned at 0 in phase 0: gpei holds it at exactly 0 for the first ~100 h
# and the vessel should be filling during growth.
set -euo pipefail
cd "$(dirname "$0")"
RES=$PWD/results_rlimit
DATA=/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensim_rlimit
mkdir -p "$RES" "$DATA"

PREFIX=${PREFIX:-$PWD/results_pensim/rbf_model_bnd_rbf_iter0}
REWARD=${REWARD:-$PWD/results_pensim/reward_model_base.pt}
ETA=${ETA:-0.35}
ALPHA_W2=${ALPHA_W2:-15.0}
KAPPA=${KAPPA:-1.0}
ITERS=${ITERS:-20}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}
PHASE0_END=${PB%%,*}

[ -f "$REWARD" ] || { echo "reward model not found: $REWARD" >&2; exit 1; }

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $PWD;"

echo "eta=$ETA  alpha_W2=$ALPHA_W2  kappa=$KAPPA  iters=$ITERS  bounds=$PB"
PIDS=()
for P in 0 1 2; do
  if [ "$P" = "0" ]; then LAM=0.2; FIX="-fix_discharge 0"; else LAM=0.0; FIX=""; fi
  J=$(sbatch --parsable $SB --job-name="rl$P" --time=14:00:00 \
    --output="$RES/rl${P}_%j.out" --error="$RES/rl${P}_%j.err" \
    --wrap="$CONDA python -u policy_learning/rl_imit_phase.py \
            -phase_prefix $PREFIX -reward_model $REWARD -phase $P \
            -eta $ETA -alpha_w2 $ALPHA_W2 -kappa $KAPPA -lam $LAM $FIX \
            -iters $ITERS -out $RES/rl_ph${P}.pt")
  PIDS+=("$J"); echo "  phase $P  lam=$LAM ${FIX:-(discharge free)} -> job $J"
done

DEP=$(IFS=:; echo "${PIDS[*]}")
sbatch --dependency=afterok:$DEP $SB --job-name=rl_expl --time=06:00:00 \
  --output="$RES/explore_%j.out" --error="$RES/explore_%j.err" \
  --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
          -phase_policies $RES/rl -fix_discharge 0 -fix_until $PHASE0_END \
          -out $DATA -tag rlimit -n 1 -p_dropout 0.0"
echo "  explore -> queued"
echo
echo "watch : squeue -u \$USER"
echo "W2/eta: grep -h 'violating' $RES/rl*_*.out"
echo "reward: grep -h 'reward(LCB)' $RES/rl*_*.out"
echo "yield : grep -h 'total yield' $RES/explore_*.out"
