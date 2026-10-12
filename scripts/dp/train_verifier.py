#!/usr/bin/env python3
"""Train a DuelingVerifier (MSE or Cal-QL) from configs/verifiers/{task}.yaml.

  python scripts/dp/train_verifier.py pusht
  python scripts/dp/train_verifier.py square --dry_run
  python scripts/dp/train_verifier.py pusht --set epochs=1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from elastic.config import apply_set, load_yaml  # noqa: E402
from verifiers.dueling import train_from_cfg  # noqa: E402

TASKS = ("pusht", "pusht_obs", "can_paired", "can_reverse", "square")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=TASKS)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Overlay key=value on the task YAML. Repeatable.",
    )
    a = p.parse_args()
    cfg_path = ROOT / "configs" / "verifiers" / f"{a.task}.yaml"
    cfg = apply_set(load_yaml(cfg_path), a.set)
    print(f"config: {cfg_path}")
    if a.dry_run:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(OmegaConf.create(cfg)))
        return
    os.chdir(ROOT)
    train_from_cfg(cfg)


if __name__ == "__main__":
    main()
