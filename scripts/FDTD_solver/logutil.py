"""
logutil.py
----------
Everything printed must also land on disk: idle SCC nodes wipe terminal
history, so a run whose output only went to the screen never happened.

Usage (first lines of every driver):

    from logutil import tee
    log_path = tee("timing_test", out_dir)      # OUT_DIR/logs/...

After this, ALL stdout+stderr (prints, warnings, tracebacks) are mirrored
line-buffered into OUT_DIR/logs/<name>_<UTC timestamp>.log.  faulthandler
is armed so even hard crashes (segfault, CUDA abort) leave a stack trace.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time


class _Tee:
    def __init__(self, stream, fh):
        self._s = stream
        self._f = fh

    def write(self, data):
        self._s.write(data)
        self._f.write(data)
        self._f.flush()          # crash-safe: every line hits disk

    def flush(self):
        self._s.flush()
        self._f.flush()

    def isatty(self):
        return False

    def fileno(self):            # some libs (faulthandler) want this
        return self._f.fileno()


def tee(name, out_dir):
    """Mirror stdout+stderr into out_dir/logs/<name>_<timestamp>.log and
    return the log path."""
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    path = os.path.join(log_dir, f"{name}_{stamp}.log")
    fh = open(path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    faulthandler.enable(file=fh)
    print(f"[log] mirroring all output to {path}")
    return path
