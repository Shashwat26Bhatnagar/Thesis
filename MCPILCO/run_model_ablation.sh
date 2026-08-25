#!/bin/bash
# =====================================================================================
# Controlled comparison: ONE world model vs THREE phase world models.
#
#   ./run_model_ablation.sh              # 20 seeds per arm
#   SEEDS_N=8 ./run_model_ablation.sh    # a shorter pilot
#
# ENDPOINT: total yield collected in the REAL simulator, paired by seed.
#
# STAGE 0 REMOVES A CONFOUND, AND IT MATTERS. The existing single model
# (rbf_model_bnd_rbf_iter0.pt) and the existing phase models were fitted on different
# snapshots of a growing dataset -- 19953 vs 25290 transitions -- and their
# standardisers differ accordingly, pH's std being 0.0153 against 0.0329. The same
# physical state therefore maps to a different z in each, so a comparison between them
# would confound "one model vs three" with "two unrelated coordinate systems". Both
# arms are refitted here on ONE frozen snapshot with the standardiser PINNED to a
# single reference (-std_from), so the arms differ only in how the data is partitioned.
#
# WHAT IS HELD FIXED ACROSS ARMS: the objective and all its weights (eta, alpha_W2,
# kappa, lambda), the reward GP, the phase-keyed particle pools, the phase-keyed expert
# query, the window schedule, the number of policy iterations, the discharge pin, and
# the seed. -model reuses the identical code path as -phase_prefix, simply resolving
# every phase to the same model, so no other difference is introduced.
#
# PAIRING: seed s produces the same policy initialisation, window draw and particle
# sampling in both arms, so the two are matched and the difference can be tested
# pairwise. That is a far more sensitive design than comparing two independent means,
# because it removes the seed-to-seed variance that has dominated every comparison in
# this project so far.
#
# EVALUATION: each trained policy is rolled out deterministically (p_dropout = 0) for
# N_EVAL episodes at DIFFERENT environment seeds, so policy variance and environment
# variance can be separated.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_ablation
SNAP=$CFG/pensim_abl_data
DATA=$CFG/pensim_abl
mkdir -p "$RES" "$SNAP" "$DATA"

SEEDS_N=${SEEDS_N:-20}
SRC_DATA=${SRC_DATA:-$CFG/pensimenv}
REWARD=${REWARD:-$REPO/results_pensim/reward_model_base.pt}
STD_REF=${STD_REF:-$REPO/results_pensim/rbf_model_bnd_rbf_iter0_phase0.pt}
ETA=${ETA:-0.35}; ALPHA_W2=${ALPHA_W2:-15.0}; KAPPA=${KAPPA:-1.0}
LAM_GROWTH=${LAM_GROWTH:-0.2}
ITERS=${ITERS:-20}; N_KEEP=${N_KEEP:-800}; N_EPOCH=${N_EPOCH:-2001}
N_EVAL=${N_EVAL:-3}
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}
PHASE0_END=${PB%%,*}

[ -f "$REWARD" ] || { echo "reward model missing: $REWARD" >&2; exit 1; }
[ -f "$STD_REF" ] || { echo "standardiser reference missing: $STD_REF" >&2; exit 1; }

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"
CONDA="source /opt/conda/etc/profile.d/conda.sh; conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; cd $REPO;"

# ---- stage 0: freeze the dataset, fit all four models on it ----
if [ ! -f "$RES/abl_single.pt" ]; then
  rm -f "$SNAP"/*.csv; cp "$SRC_DATA"/*.csv "$SNAP"/
  echo "frozen snapshot: $(ls "$SNAP"/*.csv | wc -l) CSVs from $SRC_DATA"
  MIDS=()
  J=$(sbatch --parsable $SB --job-name=abl_m1 --time=10:00:00 \
    --output="$RES/model_single_%j.out" --error="$RES/model_single_%j.err" \
    --wrap="$CONDA python -u train_rbf_pensim.py -phase -1 -data_dir $SNAP \
            -tag abl_all -save_dir $RES -n_keep $N_KEEP -n_epoch $N_EPOCH \
            -select pivchol -std_from $STD_REF; \
            cp $RES/rbf_model_abl_all.pt $RES/abl_single.pt")
  MIDS+=("$J"); echo "  [0] single model      -> $J"
  for P in 0 1 2; do
    J=$(sbatch --parsable $SB --job-name=abl_m3$P --time=10:00:00 \
      --output="$RES/model_ph${P}_%j.out" --error="$RES/model_ph${P}_%j.err" \
      --wrap="$CONDA python -u train_rbf_pensim.py -phase $P -data_dir $SNAP \
              -tag abl_ph${P} -save_dir $RES -n_keep $N_KEEP -n_epoch $N_EPOCH \
              -select pivchol -std_from $STD_REF; \
              cp $RES/rbf_model_abl_ph${P}.pt $RES/abl_phase_phase${P}.pt")
    MIDS+=("$J"); echo "  [0] phase $P model     -> $J"
  done
  MDEP="--dependency=afterok:$(IFS=:; echo "${MIDS[*]}")"
else
  echo "models already present, skipping stage 0"
  MDEP=""
fi

# ---- stages 1-2: one policy set per seed per arm, then evaluate ----
echo
echo "seeds: 1..$SEEDS_N   eval episodes per policy: $N_EVAL"
for S in $(seq 1 "$SEEDS_N"); do
  # ---- arm A: three phase models ----
  AIDS=()
  for P in 0 1 2; do
    if [ "$P" = "0" ]; then LAM=$LAM_GROWTH; FIX="-fix_discharge 0"; else LAM=0.0; FIX=""; fi
    J=$(sbatch --parsable $SB $MDEP --job-name="A${S}p$P" --time=14:00:00 \
      --output="$RES/A_s${S}_p${P}_%j.out" --error="$RES/A_s${S}_p${P}_%j.err" \
      --wrap="$CONDA python -u policy_learning/rl_imit_phase.py \
              -phase_prefix $RES/abl_phase -reward_model $REWARD -phase $P \
              -eta $ETA -alpha_w2 $ALPHA_W2 -kappa $KAPPA -lam $LAM $FIX \
              -iters $ITERS -seed $S -out $RES/A_s${S}_ph${P}.pt")
    AIDS+=("$J")
  done
  ADEP=$(IFS=:; echo "${AIDS[*]}")
  for E in $(seq 0 $((N_EVAL-1))); do
    sbatch $SB --dependency=afterok:$ADEP --job-name="Ax${S}e$E" --time=04:00:00 \
      --output="$RES/A_s${S}_e${E}_%j.out" --error="$RES/A_s${S}_e${E}_%j.err" \
      --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
              -phase_policies $RES/A_s${S} -fix_discharge 0 -fix_until $PHASE0_END \
              -out $DATA -tag A_s${S}_e${E} -n 1 -p_dropout 0.0 -seed $((100*S+E))" > /dev/null
  done

  # ---- arm B: one model, same code path ----
  BIDS=()
  for P in 0 1 2; do
    if [ "$P" = "0" ]; then LAM=$LAM_GROWTH; FIX="-fix_discharge 0"; else LAM=0.0; FIX=""; fi
    J=$(sbatch --parsable $SB $MDEP --job-name="B${S}p$P" --time=14:00:00 \
      --output="$RES/B_s${S}_p${P}_%j.out" --error="$RES/B_s${S}_p${P}_%j.err" \
      --wrap="$CONDA python -u policy_learning/rl_imit_phase.py \
              -model $RES/abl_single.pt -reward_model $REWARD -phase $P \
              -eta $ETA -alpha_w2 $ALPHA_W2 -kappa $KAPPA -lam $LAM $FIX \
              -iters $ITERS -seed $S -out $RES/B_s${S}_ph${P}.pt")
    BIDS+=("$J")
  done
  BDEP=$(IFS=:; echo "${BIDS[*]}")
  for E in $(seq 0 $((N_EVAL-1))); do
    sbatch $SB --dependency=afterok:$BDEP --job-name="Bx${S}e$E" --time=04:00:00 \
      --output="$RES/B_s${S}_e${E}_%j.out" --error="$RES/B_s${S}_e${E}_%j.err" \
      --wrap="$CONDA python -u policy_learning/explore_with_policy.py \
              -phase_policies $RES/B_s${S} -fix_discharge 0 -fix_until $PHASE0_END \
              -out $DATA -tag B_s${S}_e${E} -n 1 -p_dropout 0.0 -seed $((100*S+E))" > /dev/null
  done
  echo "  seed $S: arm A ${AIDS[*]} | arm B ${BIDS[*]}"
done

echo
echo "jobs: $((SEEDS_N * 6)) policies + $((SEEDS_N * 2 * N_EVAL)) evaluations"
echo "watch  : squeue -u \$USER | wc -l"
echo "analyse: python analyse_ablation.py"
