#!/bin/bash
# Reptile over windows, stratified across phases, on the reward + W2-constraint
# objective. ONE policy for the whole batch -- contrast with run_rlimit.sh, which
# trains three separate per-phase policies. Run both in parallel and compare.
#
#   ./run_rlrep.sh                       # k = 1, 5, 15 at fixed alpha*k
#   TASKS=20 ./run_rlrep.sh
#
#   loss = -LCB_reward + 15*relu(W2 - 0.35) + lam(phase)*||(a-lo)/span||^2 + chance
#
# Each window (one expert hour) is a Reptile task; theta resets to theta_0 before every
# task and the outer step averages the endpoints. TASKS are drawn EVENLY FROM EACH
# PHASE: phase 1 holds only 25 of the 150 windows, so a uniform draw of 15 would often
# take none from it and the outer average would follow whichever phases were sampled.
#
# lambda applies to PHASE-0 windows only (0.2). The penalty is distance from the
# physical minimum, and gpei is closer to it than the dataset mean only during growth
# (0.476 vs 1.234 at t<20 h, reversing to 1.735 vs 1.234 by t=20-47).
#
# k = 1 is the CONTROL: the paper's expansion gives an agreement coefficient of
# k(k-1)alpha/2, exactly 0 at k = 1, so that run IS joint training.
set -euo pipefail
cd "$(dirname "$0")"
RES=$PWD/results_rlrep
DATA=/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensim_rlrep
mkdir -p "$RES" "$DATA"

PREFIX=${PREFIX:-$PWD/results_pensim/rbf_model_bnd_rbf_iter0}
REWARD=${REWARD:-$PWD/results_pensim/reward_model_base.pt}
ETA=${ETA:-0.35}
ALPHA_W2=${ALPHA_W2:-15.0}
TASKS=${TASKS:-15}
ITERS=${ITERS:-20}
LR=${LR:-0.02}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}
PHASE0_END=${PB%%,*}

[ -f "$REWARD" ] || { echo "reward model not found: $REWARD" >&2; exit 1; }

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $PWD;"

echo "eta=$ETA  tasks=$TASKS (stratified)  lr=$LR  iters=$ITERS"
PIDS=()
for K in 1 5 15; do
  J=$(sbatch --parsable $SB --job-name="rr_k$K" --time=14:00:00 \
    --output="$RES/rr_k${K}_%j.out" --error="$RES/rr_k${K}_%j.err" \
    --wrap="$CONDA python -u policy_learning/rl_imit_reptile.py \
            -phase_prefix $PREFIX -reward_model $REWARD \
            -eta $ETA -alpha_w2 $ALPHA_W2 -inner_k $K -tasks $TASKS -lr $LR \
            -lam_growth 0.2 -fix_discharge 0 -iters $ITERS \
            -out $RES/rlrep_k${K}.pt")
  PIDS+=("$J"); echo "  k=$K -> job $J"
done

DEP=$(IFS=:; echo "${PIDS[*]}")
for K in 1 5 15; do
  sbatch --dependency=afterok:$DEP $SB --job-name="rx_k$K" --time=06:00:00 \
    --output="$RES/explore_k${K}_%j.out" --error="$RES/explore_k${K}_%j.err" \
    --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
            -policy $RES/rlrep_k${K}.pt -fix_discharge 0 -fix_until $PHASE0_END \
            -out $DATA -tag rlrep_k${K} -n 1 -p_dropout 0.0" > /dev/null
done
echo "  explores -> queued"
echo
echo "watch : squeue -u \$USER"
echo "phases: grep -h -A9 'action by phase' $RES/rr_k*_*.out"
echo "W2/eta: grep -h 'violating' $RES/rr_k*_*.out | tail"
echo "yield : grep -h 'total yield' $RES/explore_*.out"
