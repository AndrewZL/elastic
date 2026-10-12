#!/usr/bin/env python3
"""Train the LIBERO-10-R stride policy on frozen pi0.5 (docker/pi05/compose.yml).

Flags default to configs/libero.yaml (or --config PATH); anything passed on the
command line wins. Remaining flags go to scripts/pi05/sac_stride_libero.py:

  python scripts/pi05/train.py --output_dir data/libero/runs/<name>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch import ROOT, apply_recipe_defaults, compose_env, extract_config_argv, host_env, load_recipe  # noqa: E402

TRAIN_SCRIPT = ROOT / "scripts" / "pi05" / "sac_stride_libero.py"


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    argv, config_path = extract_config_argv(raw, default=ROOT / "configs" / "libero.yaml")
    if not ("--help" in argv or "-h" in argv):
        argv = apply_recipe_defaults(argv, load_recipe(config_path))
    env = compose_env(host_env())
    return subprocess.call([sys.executable, str(TRAIN_SCRIPT), *argv], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
