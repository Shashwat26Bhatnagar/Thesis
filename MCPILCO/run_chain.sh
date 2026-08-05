#!/bin/bash
# Submit N Dyna iterations up front, each waiting on the previous iteration's
# explore stage. Every iteration saves its 10 CSVs into the variant's own folder.
#   POLICY_KIND=kan ./run_chain.sh bnd 0 6      # iterations 0..6
set -euo pipefail
cd "$(dirname "$0")"
VAR=${1:-bnd}; FROM=${2:-0}; TO=${3:-6}
POLICY_KIND=${POLICY_KIND:-rbf}
DEP=""
for IT in $(seq "$FROM" "$TO"); do
    out=$(DEP_JOB="$DEP" POLICY_KIND="$POLICY_KIND" ./run_dyna_loop.sh "$VAR" "$IT")
    echo "$out"
    DEP=$(echo "$out" | grep -oP '\[3/3\] explore  -> job \K[0-9]+')
    [ -z "$DEP" ] && { echo "could not parse explore job id -- stopping" >&2; exit 1; }
done
