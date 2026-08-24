#!/bin/bash
# Run the ablation suite as long sequential batches -- one batch per
# ALREADY-ALLOCATED GPU node (no qsub inside), launched under nohup/tmux.
# Modeled on scripts/run_v2_campaigns.sh.
#
#   cd /project/rise-batteries/Photonics_RISE
#   # GPU 1 (~8-12 h): every retraining ablation, then the table
#   nohup bash scripts/ablation/run_ablation_batch.sh retrains > abl_retrains.log 2>&1 &
#   # GPU 2 (~44 h): the FDTD arms (kappa0 ~7h, single-member ~7h,
#   #                random baseline ~30h) -- fully resumable
#   nohup bash scripts/ablation/run_ablation_batch.sh fdtd > abl_fdtd.log 2>&1 &
#   # or everything on one GPU (~55 h):
#   nohup bash scripts/ablation/run_ablation_batch.sh all > abl_all.log 2>&1 &
#
# Every step is idempotent: finished trainings refuse to rerun (bundle
# exists), finished/partial verifications resume from verify_cache, so
# re-launching the same batch after a kill/timeout just continues.
# DO NOT run the SAME batch on two nodes at once (out-dir races);
# 'retrains' and 'fdtd' on two different nodes is the intended split.
#
# A failed step logs and CONTINUES to the next (per-step logs also land
# in runs/ablation/*/logs/ via each script's tee).

set -u
cd /project/rise-batteries/Photonics_RISE
export LD_LIBRARY_PATH=/project/rise-batteries/photonics-fdtd/lib:${LD_LIBRARY_PATH:-}
umask 002

PY=/project/rise-batteries/photonics-fdtd/bin/python3
GROUP="${1:-all}"

step () {
    echo
    echo "################ $* ################"
    date
    "$@"
    if [ $? -ne 0 ]; then
        echo "!! STEP FAILED (continuing): $*"
    fi
}

run_retrains () {
    # Priority order: highest-information first, 4x-cost raster256 last.
    step $PY -u scripts/ablation/ablation_08_kfold_off.py
    step $PY -u scripts/ablation/ablation_09_loss_head.py       # both corners
    step $PY -u scripts/ablation/ablation_10_augment_off.py
    step $PY -u scripts/ablation/ablation_11_fft_off.py
    step $PY -u scripts/ablation/ablation_13_ensemble_size.py   # k=1,3 (+k=5 view via #8)
    step $PY -u scripts/ablation/ablation_12_naive_split.py
    step $PY -u scripts/ablation/ablation_15_cell_holdout.py
    step $PY -u scripts/ablation/ablation_18_attn_none.py
    step $PY -u scripts/ablation/ablation_19_attn_cbam.py
    step $PY -u scripts/ablation/ablation_20_attn_eca.py
    step $PY -u scripts/ablation/ablation_22_recon.py
    step $PY -u scripts/ablation/ablation_23_fft_only.py
    step $PY -u scripts/ablation/ablation_24_attn_sa4.py
    step $PY -u scripts/ablation/ablation_25_seed_replicates.py
    step $PY -u scripts/ablation/ablation_14_raster256.py       # builds samples_256.npz first
    step $PY -u scripts/ablation/ablation_21_attn_sa.py         # quadratic attn: most expensive, LAST
}

run_fdtd () {
    # Longest first would starve the short arms on a timed allocation, so:
    # the two ~7 h champion-cell arms, then the ~30 h random baseline.
    step $PY -u scripts/ablation/ablation_05_kappa0.py --verify
    step $PY -u scripts/ablation/ablation_07_single_model.py --verify
    step $PY -u scripts/ablation/ablation_26_screen_only.py --verify
    step $PY -u scripts/ablation/ablation_06_random_baseline.py --verify
}

run_wall24 () {
    # Single-GPU timed-allocation order: everything that can FINISH runs
    # first (FDTD champion-cell arms ~7 h each, then all retrains); the
    # 30 h random baseline goes last as the sacrificial tail -- it eats
    # whatever hours remain and resumes per-cell/per-solve on the next
    # allocation (relaunch this same batch, finished steps skip).
    step $PY -u scripts/ablation/ablation_05_kappa0.py --verify
    step $PY -u scripts/ablation/ablation_07_single_model.py --verify
    run_retrains
    step $PY -u scripts/ablation/ablation_06_random_baseline.py --verify
}

case "$GROUP" in
    retrains) run_retrains ;;
    fdtd)     run_fdtd ;;
    wall24)   run_wall24 ;;
    all)      run_retrains; run_fdtd ;;
    *) echo "usage: $0 [retrains|fdtd|wall24|all]"; exit 2 ;;
esac

echo
echo "================ batch '$GROUP' finished ================"
date
$PY scripts/ablation/ablation_table.py
