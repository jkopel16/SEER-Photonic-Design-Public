#!/bin/bash
# Run v2 inverse design + FDTD verification, sequentially, for a list of
# cells. Meant for an ALREADY-ALLOCATED GPU node (no qsub) -- launch under
# nohup or screen/tmux, it runs for many hours:
#
#   cd /project/rise-batteries/Photonics_RISE
#   nohup bash scripts/run_v2_campaigns.sh > v2_campaigns.log 2>&1 &
#   tail -f v2_campaigns.log
#
# Per cell: design tiers (baseline/screen/cmaes/gradient, ~1-2 h) then
# full-numerics verification of the 20 exports (~5 h). A failed design
# skips that cell's verification but the script continues to the next
# cell. Everything appends to a summary at the end.

set -u
cd /project/rise-batteries/Photonics_RISE
export LD_LIBRARY_PATH=/project/rise-batteries/photonics-fdtd/lib:${LD_LIBRARY_PATH:-}
umask 002

PY=/project/rise-batteries/photonics-fdtd/bin/python3
BUNDLE=runs/surrogate_128_fft_nll_sweep/surrogate_bundle.pt   # v2 (NLL head)

# class  sigma   export-dir
CELLS="
jitter 0.10  runs/inverse_v2/jitter_s010
jitter 0.125 runs/inverse_v2/jitter_s0125
radius 0.25  runs/inverse_v2/radius_s025
"

declare -a SUMMARY=()

echo "$CELLS" | while read -r CLS SIG DIR; do
    [ -z "${CLS:-}" ] && continue
    echo
    echo "################ $CLS sigma=$SIG -> $DIR ################"
    date

    $PY -u -m models.inverse_design \
        --bundle "$BUNDLE" \
        --disorder-class "$CLS" --sigma "$SIG" \
        --tiers baseline screen cmaes gradient \
        --export-dir "$DIR" \
        --n-baseline 5000 \
        --n-screen 50000 --screen-keep-frac 0.2 \
        --cmaes-restarts 12 --cmaes-iters 600 --cmaes-popsize 24 \
        --grad-starts 12 --grad-steps 500 --grad-lr 0.015 \
        --kappa 0.2 --export-top 20
    if [ $? -ne 0 ]; then
        echo "!! DESIGN FAILED for $CLS s=$SIG -- skipping its verification"
        continue
    fi

    PC_MODE=FULL PC_COMPILE=1 $PY -u scripts/FDTD_solver/verify_candidates.py \
        --in-dir "$DIR"
    if [ $? -ne 0 ]; then
        echo "!! VERIFICATION FAILED for $CLS s=$SIG"
        continue
    fi
    echo ">> DONE: $CLS s=$SIG"
done

echo
echo "================ batch finished ================"
date
for DIR in runs/inverse_v2/jitter_s010 runs/inverse_v2/jitter_s0125 \
           runs/inverse_v2/radius_s025; do
    V="$DIR/verification_verdict.json"
    if [ -f "$V" ]; then
        echo "-- $DIR"
        $PY -c "import json; d=json.load(open('$V')); print('   ', {k: d[k] for k in list(d)[:6]})" 2>/dev/null \
            || echo "    (verdict unreadable)"
    else
        echo "-- $DIR : NO VERDICT (design or verify failed above)"
    fi
done
