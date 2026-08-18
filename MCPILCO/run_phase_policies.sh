#!/bin/bash
# Train one policy per fermentation phase, in parallel, then explore with all three.
#
#   ./run_phase_policies.sh
#   LAM=0 ITERS=20 ./run_phase_policies.sh
#
# Same loss as the single-policy run this follows from: trace-normalised and smoothed
# W2, chance constraints, 5-step windows, lambda = 0. The ONLY change is that each
# policy sees only its own phase's windows.
#
# Per-hour evaluation of the single policy showed it specialises and the early hours
# pay for it: -23% to -32% against the mean action at t = 1-41, +26% to +30% at
# t = 51-71, +7% to +17% at t = 81-141, with the breakpoints on the phase boundaries.
set -euo pipefail
cd "$(dirname "$0")"
RES=$PWD/results_phase
DATA=/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensim_phase
mkdir -p "$RES" "$DATA"

PREFIX=${PREFIX:-$PWD/results_pensim/rbf_model_bnd_rbf_iter0}
LAM=${LAM:-0}
ITERS=${ITERS:-20}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $PWD;"

echo "phase bounds: $PB    lambda=$LAM    iters=$ITERS"
PIDS=()
for P in 0 1 2; do
  J=$(sbatch --parsable $SB --job-name="ph$P" --time=14:00:00 \
    --output="$RES/ph${P}_%j.out" --error="$RES/ph${P}_%j.err" \
    --wrap="$CONDA python -u policy_learning/cdil_policy_phase.py \
            -phase_prefix $PREFIX -phase $P -lam $LAM -iters $ITERS \
            -out $RES/phase_ph${P}.pt")
  PIDS+=("$J"); echo "  phase $P -> job $J"
done

DEP=$(IFS=:; echo "${PIDS[*]}")
sbatch --dependency=afterok:$DEP $SB --job-name=ph_expl --time=06:00:00 \
  --output="$RES/explore_%j.out" --error="$RES/explore_%j.err" \
  --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
          -phase_policies $RES/phase -out $DATA -tag phase -n 1 -p_dropout 0.0"
echo "  explore (all three, selected by time) -> queued"
echo
echo "watch : squeue -u \$USER"
echo "spread: grep -h -A8 'action spread' $RES/ph*_*.out"
echo "yield : grep -h 'total yield' $RES/explore_*.out"
