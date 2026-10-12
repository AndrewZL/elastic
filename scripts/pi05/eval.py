#!/usr/bin/env python3
"""Held-out LIBERO-10-R eval: learned stride plus the paper baselines.

``--paper-baselines`` adds normal, best-of-N and budget-matched fixed; defaults come from
configs/libero.yaml and remaining flags go to scripts/pi05/run_libero_eval.py.

  python scripts/pi05/eval.py --paper-baselines \
      --stride-run elastic=data/libero/runs/<name>/checkpoints/last_stride_learner.pt
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch import ROOT, extract_config_argv, has_flag, host_env, load_recipe  # noqa: E402

EVAL_SCRIPT = ROOT / "scripts" / "pi05" / "run_libero_eval.py"


def expand(argv: list[str], recipe: dict) -> list[str]:
    out = list(argv)
    if "--paper-baselines" in out:
        out.remove("--paper-baselines")
        for flag in ("--normal-baseline", "--parallel-baseline", "--fixed-baseline"):
            if flag not in out:
                out.append(flag)
    for flag, key in (("--trials", "eval_trials"), ("--num-tasks", "n_tasks"), ("--suite", "suite")):
        if not has_flag(out, flag) and recipe.get(key) is not None:
            out += [flag, str(recipe[key])]
    if not has_flag(out, "--vgps-checkpoint") and recipe.get("vgps_checkpoint"):
        out += ["--vgps-checkpoint", str(recipe["vgps_checkpoint"])]
    return out


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    argv, config_path = extract_config_argv(raw, default=ROOT / "configs" / "libero.yaml")
    if not ("--help" in argv or "-h" in argv):
        argv = expand(argv, load_recipe(config_path))
    return subprocess.call([sys.executable, str(EVAL_SCRIPT), *argv], cwd=ROOT, env=host_env())


if __name__ == "__main__":
    raise SystemExit(main())
