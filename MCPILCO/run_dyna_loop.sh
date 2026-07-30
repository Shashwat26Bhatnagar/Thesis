#!/bin/bash
# =====================================================================================
# Dyna loop:  train world model  ->  optimize policy  ->  explore  ->  repeat
#
#   ./run_dyna_loop.sh init                 # one-time: seed both dataset folders
#   ./run_dyna_loop.sh unb 1                # unbounded, iteration 1
#   ./run_dyna_loop.sh bnd 1                # bounded (+/-10%), iteration 1
#   ./run_dyna_loop.sh unb 2                # ... after iteration 1 finishes
#
# The two variants are FULLY SEPARATE: their own dataset folder, models, policies and
# CSV tags. If they shared a data folder each would train on the other's trajectories.
#
#   configdata/pensim_unb/   12 baseline CSVs + 10 per iteration   (accumulating)
#   configdata/pensim_bnd/   same, bounded chain
#
# Within an iteration the three stages are chained with Slurm dependencies, so each
# waits for the previous to finish successfully.
#
# STANDARDIZER: refitted on the UNION every iteration. Frozen iteration-0 stats would
# push the new exploration data (~8x wider pH coverage) to large z-values, i.e. back
# outside the policy's RBF basis -- the divergence failure mode.
# The warm-started policy's CENTRES are therefore remapped into the new z-space
# (see cdil_policy_optimization.py), so its behaviour in PHYSICAL units is preserved.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
BASE=$CFG/pensimenv
RES=$REPO/results_pensim
mkdir -p "$RES"

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; cd $REPO;"

N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
N_POLICY_ITERS=${N_POLICY_ITERS:-20}
N_EXPLORE=${N_EXPLORE:-10}

# ---------------------------------------------------------------- init ----
if [ "${1:-}" = "init" ]; then
  for v in unb bnd; do
    mkdir -p "$CFG/pensim_$v"
    cp "$BASE"/random_batch_*.csv "$BASE"/gpei_batch_*.csv "$CFG/pensim_$v/"
    echo "seeded $CFG/pensim_$v  ($(ls "$CFG/pensim_$v"/*.csv | wc -l) CSVs)"
  done
  exit 0
fi

# NOTE: do not use ${1:?...} with braces in the message -- bash closes the
# parameter expansion at the first '}', and the rest is parsed as a redirection.
if [ $# -lt 2 ]; then
    echo "usage: $0 init            # seed both dataset folders" >&2
    echo "       $0 unb <iter>      # unbounded chain" >&2
    echo "       $0 bnd <iter>      # bounded (+/-10%) chain" >&2
    exit 1
fi
VAR=$1                                             # unb | bnd
IT=$2
PREV=$((IT - 1))
DATA=$CFG/pensim_$VAR
[ "$VAR" = "bnd" ] && CLIP="-clip10" || CLIP=""

MODEL=$RES/rbf_model_${VAR}_iter${IT}.pt
POLICY=$RES/cdil_policy_${VAR}_iter${IT}.pt
PREV_POLICY=$RES/cdil_policy_${VAR}_iter${PREV}.pt

# from iteration 1 on: warm-start the policy (standardizer always refits on the union)
WARM_ARG=""
if [ "$IT" -gt 0 ] && [ -f "$PREV_POLICY" ]; then
  WARM_ARG="-init_policy $PREV_POLICY"
fi

echo "=== $VAR iteration $IT ==="
echo "  data   : $DATA  ($(ls "$DATA"/*.csv 2>/dev/null | wc -l) CSVs)"
echo "  model  : $MODEL"
echo "  policy : $POLICY   ${WARM_ARG:+(warm start)}"
echo "  explore: $N_EXPLORE episodes  ${CLIP:+(+/-10% bounded)}"

# ---- stage 1: world model on the accumulated dataset ----
J1=$(sbatch --parsable $SB --job-name="tr_${VAR}${IT}" --time=08:00:00 \
  --output="$RES/loop_${VAR}${IT}_train_%j.out" --error="$RES/loop_${VAR}${IT}_train_%j.err" \
  --wrap="$CONDA python -u train_rbf_pensim.py -phase -1 -data_dir $DATA \
          -tag ${VAR}_iter${IT} -n_keep $N_KEEP -n_epoch $N_EPOCH -select pivchol")
echo "  [1/3] train    -> job $J1"

# ---- stage 2: policy optimization against that model ----
J2=$(sbatch --parsable --dependency=afterok:$J1 $SB --job-name="po_${VAR}${IT}" --time=12:00:00 \
  --output="$RES/loop_${VAR}${IT}_policy_%j.out" --error="$RES/loop_${VAR}${IT}_policy_%j.err" \
  --wrap="$CONDA python -u policy_learning/cdil_policy_optimization.py \
          -model $MODEL -out $POLICY -iters $N_POLICY_ITERS $WARM_ARG")
echo "  [2/3] policy   -> job $J2  (after $J1)"

# ---- stage 3: exploration, writing back into the SAME accumulating folder ----
J3=$(sbatch --parsable --dependency=afterok:$J2 $SB --job-name="ex_${VAR}${IT}" --time=12:00:00 \
  --output="$RES/loop_${VAR}${IT}_explore_%j.out" --error="$RES/loop_${VAR}${IT}_explore_%j.err" \
  --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
          -policy $POLICY -out $DATA -tag ${VAR}_iter${IT} \
          -n $N_EXPLORE -p_dropout 0.25 $CLIP")
echo "  [3/3] explore  -> job $J3  (after $J2)"
echo
echo "watch:  squeue -u \$USER"
echo "next :  ./run_dyna_loop.sh $VAR $((IT + 1))    # once job $J3 completes"
