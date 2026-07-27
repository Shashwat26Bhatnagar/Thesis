#!/bin/bash
# Submit one job per fermentation phase (independent, run in parallel).
#   bash run_train_phases.sh              # phases 0,1,2
#   bash run_train_phases.sh 0 1 2 -1     # explicit list, -1 = all data
#
# Each phase writes its own model/metrics, so the jobs never collide:
#   results_pensim/rbf_model_phase{0,1,2}.pt
#   results_pensim/rbf_metrics_phase{0,1,2}.json
#   results_pensim/train_phase{0,1,2}_<jobid>.out

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_pensim

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then PHASES=(0 1 2); fi

N_KEEP=${N_KEEP:-800}
N_EPOCH=${N_EPOCH:-501}

for ph in "${PHASES[@]}"; do
  tag=$([ "$ph" = "-1" ] && echo "all" || echo "phase$ph")
  jid=$(sbatch --parsable \
    --job-name="rbf_${tag}" \
    --partition=Teaching \
    --account=general-teaching \
    --qos=teaching \
    --time=08:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --output="results_pensim/train_${tag}_%j.out" \
    --error="results_pensim/train_${tag}_%j.err" \
    --wrap="source /opt/conda/etc/profile.d/conda.sh; \
            conda activate rvgp; \
            cd $PWD; \
            echo \"node: \$(hostname)  phase=$ph  n_keep=$N_KEEP  n_epoch=$N_EPOCH\"; \
            echo \"start: \$(date)\"; \
            python -u train_rbf_pensim.py -phase $ph -n_keep $N_KEEP -n_epoch $N_EPOCH; \
            echo \"end: \$(date)\"")
  echo "submitted phase $ph  ->  job $jid   (log: results_pensim/train_${tag}_${jid}.out)"
done

echo
echo "watch:   squeue -u \$USER"
echo "tail:    tail -f results_pensim/train_phase*_*.out"
