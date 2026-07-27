#!/bin/bash
#SBATCH --job-name=cdil
#SBATCH --partition=Teaching
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results_pensim/cdil_%j.out
#SBATCH --error=results_pensim/cdil_%j.err
#SBATCH --account=general-teaching
#SBATCH --qos=teaching

source /opt/conda/etc/profile.d/conda.sh
conda activate rvgp
cd ~/Thesis/MCPILCO

echo "node: $(hostname)   started: $(date)"
python -u policy_learning/cdil_policy_optimization.py
echo "finished: $(date)"
