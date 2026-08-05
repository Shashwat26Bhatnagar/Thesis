#!/bin/bash
# End-to-end Dyna loop with the MLP policy.
#   ./run_mlp.sh 0        # iteration 0  (train -> policy -> explore, chained)
#   ./run_mlp.sh 1        # iteration 1, once 0 has finished
#
# Uses its own dataset folder (configdata/pensim_bnd_mlp), models and policies, so it
# never shares trajectories with the rbf or kan chains.
# Run `./run_dyna_loop.sh init` ONCE before the first iteration of any variant.
set -euo pipefail
cd "$(dirname "$0")"
IT=${1:-0}
VAR=${VAR:-bnd}                       # bnd = +/-10% clipped actions | unb = unclipped
POLICY_KIND=mlp ./run_dyna_loop.sh "$VAR" "$IT"
