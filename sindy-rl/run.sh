#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate sindyrl
cd $HOME/Thesis/sindy-rl
export PYTHONPATH=$HOME/Thesis/sindy-rl

# Ray's raylet needs Unix domain sockets, which AFS cannot provide.
export RAY_TMPDIR=/disk/scratch/$USER/ray_$SLURM_JOB_ID
export TMPDIR=$RAY_TMPDIR
export RAY_ADDRESS=local
unset ip_head
mkdir -p "$RAY_TMPDIR" || { echo "cannot create $RAY_TMPDIR"; exit 1; }
echo "RAY_TMPDIR=$RAY_TMPDIR"

# Ray reuses /tmp/ray for its socket namespace regardless of _temp_dir.
# A job that died leaves a stale session there, and the next job tries to
# attach to it -- surfacing as "Unable to register worker with raylet".
rm -rf /tmp/ray 2>/dev/null
echo "cleared /tmp/ray on $(hostname)"

python -u sindy_rl/pbt_dyna.py
rc=$?
rm -rf "$RAY_TMPDIR"
exit $rc
