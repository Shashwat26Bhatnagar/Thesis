#!/bin/bash
# =====================================================================================
# Dyna loop:  train world model  ->  optimize policy  ->  explore  ->  repeat
#
#   ./run_dyna_loop.sh init                 # one-time: seed the dataset folders
#   ./run_dyna_loop.sh bnd 1                # bounded (+/-10%), iteration 1, rbf policy
#   ./run_dyna_loop.sh unb 1                # unbounded, iteration 1
#   POLICY_KIND=kan ./run_dyna_loop.sh bnd 1    # same, with the KAN policy
#
# POLICY_KIND (env var, default "rbf") selects the policy architecture:
#   rbf  Sum_of_gaussians   joint Gaussian basis over ||s-c||   (MC-PILCO's own)
#   mlp  feed-forward       fixed activations, learned weights
#   kan  Kolmogorov-Arnold  LEARNED univariate activations on edges (radial basis)
# Each kind gets its OWN dataset folder, models and policies, so the three chains
# accumulate independently and never train on each other's trajectories.
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

POLICY_KIND=${POLICY_KIND:-rbf}
case "$POLICY_KIND" in rbf|mlp|kan) ;; *) echo "POLICY_KIND must be rbf|mlp|kan" >&2; exit 1;; esac

N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
N_POLICY_ITERS=${N_POLICY_ITERS:-20}
N_EXPLORE=${N_EXPLORE:-10}

# ---------------------------------------------------------------- init ----
if [ "${1:-}" = "init" ]; then
  # one folder per (bound-variant, policy-kind) pair -- they must not share data
  for v in unb bnd; do
    for k in rbf mlp kan; do
      d="$CFG/pensim_${v}_${k}"
      mkdir -p "$d"
      cp "$BASE"/random_batch_*.csv "$BASE"/gpei_batch_*.csv "$d/"
      echo "seeded $d  ($(ls "$d"/*.csv | wc -l) CSVs)"
    done
  done
  exit 0
fi

# NOTE: do not use ${1:?...} with braces in the message -- bash closes the
# parameter expansion at the first '}', and the rest is parsed as a redirection.
if [ $# -lt 2 ]; then
    echo "usage: $0 init                    # seed the dataset folders" >&2
    echo "       $0 unb|bnd <iter>          # one Dyna iteration" >&2
    echo "       POLICY_KIND=kan $0 bnd 0   # with a specific policy arch" >&2
    exit 1
fi
VAR=$1                                             # unb | bnd
IT=$2
PREV=$((IT - 1))
TAG=${VAR}_${POLICY_KIND}                      # e.g. bnd_kan
DATA=$CFG/pensim_$TAG
[ "$VAR" = "bnd" ] && CLIP="-clip10" || CLIP=""

MODEL=$RES/rbf_model_${TAG}_iter${IT}.pt
POLICY=$RES/cdil_policy_${TAG}_iter${IT}.pt
PREV_POLICY=$RES/cdil_policy_${TAG}_iter${PREV}.pt

if [ ! -d "$DATA" ]; then
  echo "dataset folder $DATA missing -- run: $0 init" >&2
  exit 1
fi

# from iteration 1 on: warm-start the policy (standardizer always refits on the union)
WARM_ARG=""
if [ "$IT" -gt 0 ] && [ -f "$PREV_POLICY" ]; then
  WARM_ARG="-init_policy $PREV_POLICY"
fi

echo "=== $VAR / $POLICY_KIND  iteration $IT ==="
echo "  data   : $DATA  ($(ls "$DATA"/*.csv 2>/dev/null | wc -l) CSVs)"
echo "  model  : $MODEL"
echo "  policy : $POLICY   ${WARM_ARG:+(warm start)}"
echo "  explore: $N_EXPLORE episodes  ${CLIP:+(+/-10% bounded)}"
echo "  policy arch: $POLICY_KIND"
DEP1=""
if [ -n "${DEP_JOB:-}" ]; then
  DEP1="--dependency=afterok:${DEP_JOB}"
  echo "  waiting on : job ${DEP_JOB} (previous iteration's explore)"
fi

# ---- stage 1: world model on the accumulated dataset ----
J1=$(sbatch --parsable $SB $DEP1 --job-name="tr_${TAG}${IT}" --time=08:00:00 \
  --output="$RES/loop_${TAG}${IT}_train_%j.out" --error="$RES/loop_${TAG}${IT}_train_%j.err" \
  --wrap="$CONDA python -u train_rbf_pensim.py -phase -1 -data_dir $DATA \
          -tag ${TAG}_iter${IT} -n_keep $N_KEEP -n_epoch $N_EPOCH -select pivchol")
echo "  [1/3] train    -> job $J1"

# ---- stage 2: policy optimization against that model ----
J2=$(sbatch --parsable --dependency=afterok:$J1 $SB --job-name="po_${TAG}${IT}" --time=12:00:00 \
  --output="$RES/loop_${TAG}${IT}_policy_%j.out" --error="$RES/loop_${TAG}${IT}_policy_%j.err" \
  --wrap="$CONDA python -u policy_learning/cdil_policy_optimization.py \
          -model $MODEL -out $POLICY -iters $N_POLICY_ITERS \
          -policy_kind $POLICY_KIND $WARM_ARG")
echo "  [2/3] policy   -> job $J2  (after $J1)"

# ---- stage 3: exploration, writing back into the SAME accumulating folder ----
J3=$(sbatch --parsable --dependency=afterok:$J2 $SB --job-name="ex_${TAG}${IT}" --time=12:00:00 \
  --output="$RES/loop_${TAG}${IT}_explore_%j.out" --error="$RES/loop_${TAG}${IT}_explore_%j.err" \
  --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
          -policy $POLICY -out $DATA -tag ${TAG}_iter${IT} \
          -n $N_EXPLORE -p_dropout 0.25 $CLIP")
echo "  [3/3] explore  -> job $J3  (after $J2)"
echo
echo "watch:  squeue -u \$USER"
echo "next :  POLICY_KIND=$POLICY_KIND ./run_dyna_loop.sh $VAR $((IT + 1))    # once job $J3 completes"
