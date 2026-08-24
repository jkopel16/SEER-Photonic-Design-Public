"""Benchmark surrogate model inference speed across batch sizes.

Usage:
    python scripts/bench_model_timing.py
    python scripts/bench_model_timing.py --n-samples 100
    python scripts/bench_model_timing.py --device cpu
    python scripts/bench_model_timing.py --batch-sizes 1 32 128
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "FDTD_solver"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "models"))

from inverse_design import SurrogateScorer


class Tee:
    def __init__(self, log_path):
        self.file = open(log_path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Benchmark surrogate model inference speed.")
    ap.add_argument("--bundle",
                    default=os.path.join(_REPO_ROOT, "runs",
                                         "surrogate_128_fft_nll_sweep",
                                         "surrogate_bundle.pt"))
    ap.add_argument("--samples-dir",
                    default=os.path.join(_REPO_ROOT, "data", "samples"))
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 32, 64, 128, 256])
    ap.add_argument("--inference-iterations", type=int, default=50,
                    help="Repeated runs for batch=1 timing statistics.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "bench_logs"))
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def load_samples(samples_dir, n):
    """Load up to n sample .npz files. Returns (holes_list, a_super) tuples."""
    fnames = sorted(
        f for f in os.listdir(samples_dir) if f.endswith(".npz"))
    if not fnames:
        raise FileNotFoundError(f"No .npz files found in {samples_dir}")
    rng = np.random.default_rng(42)
    chosen = rng.choice(fnames, size=min(n, len(fnames)), replace=False)
    entries = []
    for fn in chosen:
        d = np.load(os.path.join(samples_dir, fn), allow_pickle=False)
        holes = d["holes_xyr_nm"]
        entries.append({
            "path": fn,
            "holes": [(float(x), float(y), float(r)) for x, y, r in holes],
            "a_super_nm": float(d["a_super_nm"]),
            "E": float(d["E"]),
            "sigma": float(d["sigma"]) if "sigma" in d.files else -1,
            "cls": str(d["disorder_class"]) if "disorder_class" in d.files else "?",
        })
    return entries


def main():
    args = parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"bench_model_timing_{ts}.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    print(f"=== {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    import torch
    device = torch.device(args.device) if args.device else torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Bundle: {args.bundle}")

    scorer = SurrogateScorer(args.bundle, device, use_tta=True, kappa=1.0,
                             batch_size=max(args.batch_sizes))
    print(f"Model ensemble: {len(scorer.models)} members")

    samples = load_samples(args.samples_dir, args.n_samples)
    print(f"Samples: {len(samples)} loaded from {args.samples_dir}\n")

    holes_list = [s["holes"] for s in samples]
    a_super = samples[0]["a_super_nm"]

    # warm-up
    _ = scorer.score_holes(holes_list[:2], a_super)

    print("=== Inference timing ===\n")

    results = {}
    for bs in sorted(args.batch_sizes):
        if bs == 1:
            times = []
            for _ in range(args.inference_iterations):
                t0 = time.perf_counter()
                _ = scorer.score_holes(holes_list[:1], a_super)
                times.append((time.perf_counter() - t0) * 1000)
            mean_ms = float(np.mean(times))
            std_ms = float(np.std(times))
            results[bs] = {"mean_ms": mean_ms, "std_ms": std_ms}
        else:
            n_batches = max(1, len(holes_list) // bs)
            t0 = time.perf_counter()
            for i in range(n_batches):
                batch = holes_list[i * bs:(i + 1) * bs]
                _ = scorer.score_holes(batch, a_super)
            elapsed = time.perf_counter() - t0
            per_sample = elapsed / (n_batches * bs) * 1000
            results[bs] = {"mean_ms": per_sample, "std_ms": 0.0}

        rate = 1000 / results[bs]["mean_ms"] if results[bs]["mean_ms"] > 0 else 0
        print(f"  batch_size={bs:3d}  "
              f"{results[bs]['mean_ms']:7.2f} ms  "
              f"({rate:7.0f} samples/s)")

    print()
    total_batch = max(1, min(bs for bs in args.batch_sizes if bs >= 32))
    n_batches = max(1, len(holes_list) // total_batch)
    t0 = time.perf_counter()
    for i in range(n_batches):
        batch = holes_list[i * total_batch:(i + 1) * total_batch]
        _ = scorer.score_holes(batch, a_super)
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"Total over {len(samples)} samples (batch={total_batch}): "
          f"{total_ms:.1f} ms")

    print(f"\n=== Sample summary ===")
    print(f"  {'file':>16s}  {'class':>6s}  {'sigma':>5s}  {'E':>6s}")
    for s in samples:
        print(f"  {s['path']:>16s}  {s['cls']:>6s}  "
              f"{s['sigma']:5.2f}  {s['E']:6.4f}")

    tee.close()
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
