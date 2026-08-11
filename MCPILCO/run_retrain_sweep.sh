#!/bin/bash
# =====================================================================================
# Retrain the three phase world models with a PINNED standardizer, then sweep lambda,
# evaluate and explore. Everything lands in results_retrain/ and configdata/pensim_rt/.
#
#   ./run_retrain_sweep.sh
#   SRC=.../pensimenv LAMBDAS="0 0.03 0.1 0.3 1" ./run_retrain_sweep.sh
#
# WHY THE DATASET IS SNAPSHOTTED
# The standardizer is fitted on whatever is in the data folder at that moment, and the
# folder grows as exploration adds CSVs. Two models trained two days apart on the same
# folder saw 19953 and 25290 transitions, and pH's std went 0.0153 -> 0.0329 -- the
# same physical state then maps to a different z, so the RBF centres cover a different
# physical region and neither ||a||^2 nor W2 is comparable between them. Copying the
# CSVs to a frozen folder removes that.
#
# AND WHY PHASE 0 IS TRAINED FIRST
# Phases 1 and 2 then take -std_from phase 0, so all three share one z-space by
# construction rather than by coincidence of loading the same folder. Serial rather
# than parallel for stage 1, which costs an hour and buys certainty.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_retrain
SNAP=$CFG/pensim_rt_data          # frozen copy the models are fitted on
DATA=$CFG/pensim_rt               # where exploration writes
mkdir -p "$RES" "$SNAP" "$DATA"

SRC=${SRC:-$CFG/pensim_bnd_rbf}
LAMBDAS=${LAMBDAS:-"0 0.03 0.1 0.3 1"}
N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}
ITERS=${ITERS:-20}
TOL=${TOL:-0.05}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}
PREFIX=$RES/rt_model

# ---- freeze the dataset ----
rm -f "$SNAP"/*.csv
cp "$SRC"/*.csv "$SNAP"/
NCSV=$(ls "$SNAP"/*.csv | wc -l)
echo "snapshot: $SRC -> $SNAP  ($NCSV CSVs, frozen for the whole run)"
[ -f "$DATA/gpei_batch_0.csv" ] || cp "$CFG"/pensimenv/*.csv "$DATA/"

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

echo "phase bounds: $PB     lambdas: $LAMBDAS"

# ---- [1] phase 0 fits the standardizer ----
J0=$(sbatch --parsable $SB --job-name=rt_p0 --time=10:00:00 \
  --output="$RES/rt_p0_%j.out" --error="$RES/rt_p0_%j.err" \
  --wrap="$CONDA python -u train_rbf_pensim.py -phase 0 -data_dir $SNAP \
          -tag rt_phase0 -n_keep $N_KEEP -n_epoch $N_EPOCH -select pivchol \
          -save_dir $RES; \
          cp $RES/rbf_model_rt_phase0.pt ${PREFIX}_phase0.pt")
echo "  [1/4] phase 0 (fits stats) -> job $J0"

# ---- [2] phases 1 and 2 reuse phase 0's statistics ----
TIDS=()
for P in 1 2; do
  J=$(sbatch --parsable $SB --dependency=afterok:$J0 --job-name=rt_p$P --time=10:00:00 \
    --output="$RES/rt_p${P}_%j.out" --error="$RES/rt_p${P}_%j.err" \
    --wrap="$CONDA python -u train_rbf_pensim.py -phase $P -data_dir $SNAP \
            -tag rt_phase${P} -n_keep $N_KEEP -n_epoch $N_EPOCH -select pivchol \
            -save_dir $RES -std_from ${PREFIX}_phase0.pt; \
            cp $RES/rbf_model_rt_phase${P}.pt ${PREFIX}_phase${P}.pt")
  TIDS+=("$J"); echo "  [2/4] phase $P (-std_from phase 0) -> job $J"
done
TDEP=$(IFS=:; echo "${TIDS[*]}")

# ---- [3] one policy per lambda ----
PIDS=(); REF=""
for L in $LAMBDAS; do
  LT="lam${L//./p}"
  POL=$RES/rt_${LT}.pt
  [ "$L" = "0" ] && REF=$POL
  J=$(sbatch --parsable $SB --dependency=afterok:$TDEP --job-name="rt_$LT" --time=14:00:00 \
    --output="$RES/rt_${LT}_%j.out" --error="$RES/rt_${LT}_%j.err" \
    --wrap="$CONDA python -u policy_learning/cdil_policy_clean.py \
            -phase_prefix $PREFIX -lam $L -iters $ITERS -out $POL")
  PIDS+=("$J"); echo "  [3/4] lambda=$L -> job $J"
done
[ -n "$REF" ] || { echo "ERROR: LAMBDAS must include 0 (the reference)" >&2; exit 1; }

# ---- [4] evaluate, then one exploration episode with the winner ----
PDEP=$(IFS=:; echo "${PIDS[*]}")
JE=$(sbatch --parsable $SB --dependency=afterok:$PDEP --job-name=rt_eval --time=04:00:00 \
  --output="$RES/rt_eval_%j.out" --error="$RES/rt_eval_%j.err" \
  --wrap="$CONDA python -u evaluate_lambda_sweep.py -phase_prefix $PREFIX \
          -ref $REF -tol $TOL -cand '$RES/rt_lam*.pt' \
          -out $RES/rt_sweep.json -select_out $RES/rt_best.pt")
echo "  [4/4] evaluate -> job $JE"

sbatch --dependency=afterok:$JE $SB --job-name=rt_expl --time=06:00:00 \
  --output="$RES/rt_explore_%j.out" --error="$RES/rt_explore_%j.err" \
  --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
          -policy $RES/rt_best.pt -out $DATA -tag rt -n 1 -p_dropout 0.0"
echo "        explore (1 episode) -> queued"

echo
echo "watch : squeue -u \$USER"
echo "stats : grep -h '\[std\] obs_sd' $RES/rt_p*_*.out      # all three must MATCH"
echo "L2    : grep -h '||a||^2  first' $RES/rt_lam*_*.out"
echo "sweep : tail -25 $RES/rt_eval_*.out"
echo "rows  : head -3 $DATA/rt_batch_0.csv"
