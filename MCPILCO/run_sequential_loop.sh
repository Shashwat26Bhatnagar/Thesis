#!/bin/bash
# =====================================================================================
# Sequential Dyna loop with gpei's discharge held fixed.
#
#   ./run_sequential_loop.sh                 # 10 iterations from scratch
#   ./run_sequential_loop.sh 3 10            # resume at iteration 3
#
# Each iteration:
#   [1] retrain the three phase world models on everything collected so far
#   [2] retrain the policy (L2 on the FIVE controlled channels; discharge is
#       overridden by the trace, excluded from the loss, and receives no gradient)
#   [3] explore ONE episode, discharge replayed from a gpei batch
#
# WHY DISCHARGE IS FIXED
# Replaying gpei's recorded discharge with the policy on the other five channels gave
# 3486 (90.9% of gpei's 3835), against 3346 (87.3%) with the learned valve. So the
# valve is worth ~3.6 points and that is now spent; the remaining ~9 points sit in
# aeration and back pressure, the two channels gpei ramps (39->73 and 0.66->1.19)
# and ours holds flat. Fixing discharge removes it as a confound while those are
# worked on.
#
# THE TRACE IS CYCLED across gpei_batch_0..9, so the accumulated dataset contains ten
# different pulse timings rather than one repeated -- otherwise the world model would
# see a single discharge pattern and learn nothing about the channel.
#
# HONEST FRAMING: this bootstraps off gpei rather than self-improving. The policy is
# not learning to control discharge; it is learning the other five channels under a
# fixed expert schedule.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_seq
DATA=$CFG/pensim_seq
GPEI=$CFG/pensim_bnd
mkdir -p "$RES" "$DATA"

FROM=${1:-0}
TO=${2:-9}
N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
N_POLICY_ITERS=${N_POLICY_ITERS:-20}
LAM=${LAM:-0.01}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

# seed the dataset once
if [ ! -f "$DATA/gpei_batch_0.csv" ]; then
  cp "$GPEI"/random_batch_*.csv "$GPEI"/gpei_batch_*.csv "$DATA/"
  echo "seeded $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
fi

DEP=""
for IT in $(seq "$FROM" "$TO"); do
  PREFIX=$RES/seq_model_iter${IT}
  POL=$RES/seq_policy_iter${IT}.pt
  PREV=$RES/seq_policy_iter$((IT-1)).pt
  TRACE=$GPEI/gpei_batch_$((IT % 10)).csv          # cycle the pulse timings
  WARM=""; [ -f "$PREV" ] && WARM="-init_policy $PREV"

  echo "=== iteration $IT   data=$(ls "$DATA"/*.csv | wc -l) CSVs   trace=$(basename "$TRACE")"

  # [1] three phase world models
  TIDS=()
  for P in 0 1 2; do
    D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
    J=$(sbatch --parsable $SB $D --job-name="s${IT}m$P" --time=10:00:00 \
      --output="$RES/it${IT}_p${P}_%j.out" --error="$RES/it${IT}_p${P}_%j.err" \
      --wrap="$CONDA python -u train_rbf_pensim.py -phase $P -data_dir $DATA \
              -tag seq_iter${IT}_phase${P} -n_keep $N_KEEP -n_epoch $N_EPOCH \
              -select pivchol")
    TIDS+=("$J")
  done
  TDEP=$(IFS=:; echo "${TIDS[*]}")
  echo "  [1/3] models  -> ${TIDS[*]}"

  # rename to the prefix the policy expects
  JR=$(sbatch --parsable $SB --dependency=afterok:$TDEP --job-name="s${IT}r" --time=00:10:00 \
    --output="$RES/it${IT}_link_%j.out" --error="$RES/it${IT}_link_%j.err" \
    --wrap="cd $RES; for p in 0 1 2; do cp rbf_model_seq_iter${IT}_phase\$p.pt \
            seq_model_iter${IT}_phase\$p.pt; done")

  # [2] policy: five channels, discharge from the trace
  J2=$(sbatch --parsable $SB --dependency=afterok:$JR --job-name="s${IT}p" --time=14:00:00 \
    --output="$RES/it${IT}_policy_%j.out" --error="$RES/it${IT}_policy_%j.err" \
    --wrap="$CONDA python -u policy_learning/cdil_policy_optimization.py \
            -phase_prefix $PREFIX -policy_kind rbf -lam $LAM \
            -discharge_csv $TRACE -iters $N_POLICY_ITERS -out $POL $WARM")
  echo "  [2/3] policy  -> $J2"

  # [3] ONE episode, same trace
  J3=$(sbatch --parsable $SB --dependency=afterok:$J2 --job-name="s${IT}e" --time=04:00:00 \
    --output="$RES/it${IT}_explore_%j.out" --error="$RES/it${IT}_explore_%j.err" \
    --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
            -policy $POL -discharge_csv $TRACE -out $DATA \
            -tag seq_iter${IT} -n 1 -p_dropout 0.0")
  echo "  [3/3] explore -> $J3"
  DEP=$J3
done

echo
echo "watch : squeue -u \$USER"
echo "yields: grep -h 'total yield' $RES/it*_explore_*.out"
echo "models: grep -h 'mean MSE' $RES/it*_p*_*.out"
