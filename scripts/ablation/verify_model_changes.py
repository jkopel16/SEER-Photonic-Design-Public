"""One-shot verification for the attention fixes + recon head (2026-08-12).

Checks, in order (all must print OK):
  1. ECA kernel follows the published heuristic (k=5 at C=128).
  2. PhotonicCNN shapes with recon_head=True at widths 128 and 48;
     forward() unchanged; strict state_dict round-trip through the arch
     dict; legacy arch (no new keys) still builds.
  3. Smoke train (SUBPROCESS, runs before this process touches CUDA:
     exclusive-process GPUs allow only one CUDA context): 1 member,
     2 epochs, --recon-head, then strict reload of the saved bundle
     via PhotonicCNN(**arch).
  4. Deployed bundle still reproduces its table row through
     evaluate_bundle (norm-stats gate + MAE/rho/ratio) and loads through
     SurrogateScorer (the strict, no-setdefault path).
  5. Dry runs: ablation_22 emits --recon-head; ablation_18 --wandb
     drops --no-wandb.

Usage (GPU node or CPU; GPU strongly recommended for step 3/4):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/verify_model_changes.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, DEPLOYED_BUNDLE, ENV_PY,     # noqa: E402
                    REPO, evaluate_bundle)

import torch                                                    # noqa: E402
from models.model import ECA, PhotonicCNN                       # noqa: E402

SMOKE_DIR = os.path.join(ABLATION_DIR, "_smoke_recon")
fails = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail
                                                  else ""))
    if not ok:
        fails.append(name)


# ---- 1. ECA kernel -------------------------------------------------------
check("ECA k(C=128) == 5", ECA(128).conv.kernel_size[0] == 5)
check("ECA k(C=32) == 3", ECA(32).conv.kernel_size[0] == 3)

# ---- 2. shapes + arch round-trip -----------------------------------------
for h in (128, 48):
    m = PhotonicCNN(2, h, 2, recon_head=True)
    x = torch.randn(2, 2, 128, 128)
    o = m(x)
    o2, r = m.forward_with_recon(x)
    check(f"shapes h={h}", o.shape == (2, 2) and o2.shape == (2, 2)
          and r.shape == (2, 1, 128, 128))
arch = dict(input_shape=2, hidden_units=128, output_shape=2, dropout=0.1,
            stochastic_depth=0.05, padding_mode="zeros", attention="se",
            img_size=128, recon_head=True)
m2 = PhotonicCNN(**arch)
m2.load_state_dict(PhotonicCNN(**arch).state_dict())
check("recon arch strict round-trip", True)
legacy = PhotonicCNN(input_shape=2, hidden_units=128, output_shape=2,
                     dropout=0.1, stochastic_depth=0.05)
check("legacy arch builds, no decoder", legacy.decoder is None)

# ---- 3. smoke train with recon head (BEFORE parent takes the GPU) --------
# On exclusive-process GPUs the subprocess cannot get a CUDA context if
# this process already holds one, so this must precede step 4.
shutil.rmtree(SMOKE_DIR, ignore_errors=True)
rc = subprocess.call(
    [ENV_PY, "-u", "-m", "models.model",
     "-i", os.path.join(REPO, "data", "samples_128.npz"),
     "-o", SMOKE_DIR, "--raster-only", "--fft-channel", "--nll-head",
     "--recon-head", "--ensemble", "1", "--epochs", "2", "--seed", "137",
     "--no-wandb"], cwd=REPO)
check("smoke train rc == 0", rc == 0)
bpath = os.path.join(SMOKE_DIR, "surrogate_bundle.pt")
b = torch.load(bpath, map_location="cpu", weights_only=False)
check("bundle arch records recon_head", b["arch"].get("recon_head") is True)
mr = PhotonicCNN(**b["arch"])
mr.load_state_dict(b["state_dicts"][0])
check("recon bundle strict reload", True)
check("decoder weights present",
      any(k.startswith("decoder.") for k in b["state_dicts"][0]))
shutil.rmtree(SMOKE_DIR, ignore_errors=True)

# ---- 4. deployed bundle unchanged -----------------------------------------
out = evaluate_bundle(DEPLOYED_BUNDLE, out_json="/dev/null",
                      label="compat_check")
row = out["rows"]["test/tta_on"]
check("deployed metrics reproduce",
      abs(row["mae"] - 0.005440) < 5e-6
      and abs(row["rho_pooled"] - 0.701) < 5e-3
      and abs(row["rms_s_over_rmse"] - 1.005) < 5e-3,
      f"mae {row['mae']:.6f} rho {row['rho_pooled']:.3f} "
      f"ratio {row['rms_s_over_rmse']:.3f}")
from models.inverse_design import SurrogateScorer               # noqa: E402
dev = "cuda" if torch.cuda.is_available() else "cpu"
s = SurrogateScorer(DEPLOYED_BUNDLE, dev, use_tta=True, kappa=0.2,
                    batch_size=64, calibration=None)
check("SurrogateScorer strict load", len(s.models) == 5)

# ---- 5. driver dry runs ---------------------------------------------------
def dry(script, extra, want_in, want_out=()):
    p = subprocess.run([ENV_PY, os.path.join(REPO, "scripts", "ablation",
                                             script), "--dry-run"] + extra,
                       cwd=REPO, capture_output=True, text=True)
    txt = p.stdout + p.stderr
    ok = (p.returncode == 0 and all(w in txt for w in want_in)
          and not any(w in txt for w in want_out))
    check(f"dry-run {script} {' '.join(extra)}", ok)


dry("ablation_22_recon.py", [], ["--recon-head", "--no-wandb"])
dry("ablation_18_attn_none.py", ["--wandb"], ["--attention none"],
    want_out=["--no-wandb"])

print()
if fails:
    print("FAILURES:", ", ".join(fails))
    raise SystemExit(1)
print("ALL CHECKS PASSED")
