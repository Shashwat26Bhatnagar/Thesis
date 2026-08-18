#!/bin/bash

# =====================================================================================
#
# Ten iterations of: train policy -> explore ONE episode -> retrain the world models.
#
# ./run_expfull_loop.sh            # iterations 0..9
# ./run_expfull_loop.sh 3 9        # resume at 3
#
# Configuration:
# - three phase world models, selected per window by expert time
# - objective, TWO REGIMES split at the expert's horizon:
#   t <= 150 h : -LCB_reward + alpha*relu(W2_h - eta) + lambda*||a||^2 + chance
#   t >  150 h : -LCB_reward + lambda*||a||^2 + chance
# - Windows are drawn over the FULL 1..230 h.
# - static action box alpha=1000, vessel floor alpha=1.0
# - 5-step (1 h) windows, 150 per iteration, 20 iterations
# - exploration with -cliprecipe AND BUGGY_RECIPE_CLIP=1, p_dropout 0.25
#
# IMPORTANT:
# - lambda = 0.03
# - The standardizer is pinned (-std_from) across all iterations.
#
# =====================================================================================

set -euo pipefail

cd "$(dirname "$0")"
REPO=$PWD

CFG=/home/s2892016/Thesis/deps/smpl/smpl/configdata
RES=$REPO/results_expfull
DATA=$CFG/pensim_expfull

mkdir -p "$RES" "$DATA"

FROM=${1:-0}
TO=${2:-9}

SRC_PREFIX=${SRC_PREFIX:-$REPO/results_pensim/rbf_model_bnd_rbf_iter0}

# Policy configuration
LAM=${LAM:-0.03}
ETA=${ETA:-0.23}
ALPHA_W2=${ALPHA_W2:-15.0}
KAPPA=${KAPPA:-1.0}

REWARD=${REWARD:-$REPO/results_pensim/reward_model_base.pt}

# Policy optimisation / model retraining
ITERS=${ITERS:-20}
N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-2001}

# PenSim phase boundaries
PB=${PENSIM_PHASE_BOUNDS:-47.5,72.5}

# Exploration configuration
DROPOUT=${DROPOUT:-0.25}

[ -f "$REWARD" ] || {
    echo "reward model not found: $REWARD" >&2
    exit 1
}

[ -f "${SRC_PREFIX}_phase0.pt" ] || {
    echo "models not found: ${SRC_PREFIX}_phase0.pt" >&2
    exit 1
}

if [ ! -f "$DATA/gpei_batch_0.csv" ]; then
    cp "$CFG"/pensimenv/*.csv "$DATA"/
    echo "seeded $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
fi

if [ ! -f "$RES/xf_model_iter0_phase0.pt" ]; then
    for p in 0 1 2; do
        cp "${SRC_PREFIX}_phase${p}.pt" \
           "$RES/xf_model_iter0_phase${p}.pt"
    done

    echo "iteration 0 models copied from $SRC_PREFIX"
fi

SB="--partition=Teaching --account=general-teaching --qos=teaching --cpus-per-task=4 --mem=16G"

CONDA="source /opt/conda/etc/profile.d/conda.sh; \
conda activate rvgp; \
export PENSIM_PHASE_BOUNDS='$PB'; \
cd $REPO;"

echo "=========================================================================="
echo " expfull loop: iterations $FROM..$TO"
echo "   eta=$ETA alpha_W2=$ALPHA_W2 kappa=$KAPPA lambda=$LAM"
echo "   W2 applies t<=150 h; reward applies over the full 1..230 h"
echo "   policy iters=$ITERS  dropout=$DROPOUT  clip=BUGGY_RECIPE_CLIP"
echo "   reward: $(basename "$REWARD")"
echo "   data  : $DATA ($(ls "$DATA"/*.csv | wc -l) CSVs)"
echo "=========================================================================="

DEP=""

for IT in $(seq "$FROM" "$TO"); do

    MODEL=$RES/xf_model_iter${IT}
    NEXT=$RES/xf_model_iter$((IT+1))

    POL=$RES/xf_policy_iter${IT}.pt
    PREV=$RES/xf_policy_iter$((IT-1)).pt

    WARM=""
    [ -f "$PREV" ] && WARM="-init_policy $PREV"

    D=""
    [ -n "$DEP" ] && D="--dependency=afterok:$DEP"

    echo "--- iteration $IT"

    # -------------------------------------------------------------------------
    # 1. Train policy
    # -------------------------------------------------------------------------

    J1=$(sbatch --parsable $SB $D \
        --job-name="xf${IT}" \
        --time=14:00:00 \
        --output="$RES/it${IT}_policy_%j.out" \
        --error="$RES/it${IT}_policy_%j.err" \
        --wrap="$CONDA python -u policy_learning/exp_policy_full.py \
        -phase_prefix $MODEL \
        -reward_model $REWARD \
        -lam $LAM \
        -eta $ETA \
        -alpha_w2 $ALPHA_W2 \
        -kappa $KAPPA \
        -iters $ITERS \
        -out $POL \
        $WARM")

    echo "  [1/3] policy  -> job $J1"

    # -------------------------------------------------------------------------
    # 2. Explore using the newly trained policy
    #
    # Same exploration configuration as the original script:
    #   BUGGY_RECIPE_CLIP=1
    #   -cliprecipe
    #   -p_dropout 0.25
    # -------------------------------------------------------------------------

    J2=$(sbatch --parsable $SB \
        --dependency=afterok:$J1 \
        --job-name="ex${IT}" \
        --time=06:00:00 \
        --output="$RES/it${IT}_explore_%j.out" \
        --error="$RES/it${IT}_explore_%j.err" \
        --wrap="$CONDA BUGGY_RECIPE_CLIP=1 \
        python -u policy_learning/explore_with_policy.py \
        -policy $POL \
        -cliprecipe \
        -out $DATA \
        -tag xf_iter${IT} \
        -n 1 \
        -p_dropout $DROPOUT")

    echo "  [2/3] explore -> job $J2"

    # -------------------------------------------------------------------------
    # 3. Retrain three phase world models
    # -------------------------------------------------------------------------

    if [ "$IT" -lt "$TO" ]; then

        TIDS=()

        for P in 0 1 2; do

            J=$(sbatch --parsable $SB \
                --dependency=afterok:$J2 \
                --job-name="em${IT}p$P" \
                --time=10:00:00 \
                --output="$RES/it${IT}_m${P}_%j.out" \
                --error="$RES/it${IT}_m${P}_%j.err" \
                --wrap="$CONDA python -u train_rbf_pensim.py \
                -phase $P \
                -data_dir $DATA \
                -tag xf_it$((IT+1))_p${P} \
                -save_dir $RES \
                -n_keep $N_KEEP \
                -n_epoch $N_EPOCH \
                -select pivchol \
                -std_from ${SRC_PREFIX}_phase0.pt; \
                cp $RES/rbf_model_xf_it$((IT+1))_p${P}.pt \
                ${NEXT}_phase${P}.pt")

            TIDS+=("$J")

        done

        DEP=$(IFS=:; echo "${TIDS[*]}")

        echo "  [3/3] models  -> ${TIDS[*]}"

    else

        DEP=$J2

    fi

done

echo
echo "watch : squeue -u $USER"
echo "W2/eta: grep -h 'violating' $RES/it*_policy_*.out"
echo "reward: grep -h 'reward(LCB)' $RES/it*_policy_*.out"
echo "yields: grep -h 'total yield' $RES/it*_explore_*.out"
echo "split : grep -h 'reward-only' $RES/it*_policy_*.out | tail -5"
echo "rows  : head -3 $DATA/xf_iter0_batch_0.csv"
