"""Shared host-side helpers for the pi0.5 LIBERO launchers (docker/pi05/compose.yml)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from elastic.config import data_root, load_yaml  # noqa: E402

COMPOSE_FILE = ROOT / "docker" / "pi05" / "compose.yml"
# Mounts: runs dir (both containers), ELASTIC_DATA read-only (server), repo (both).
RUNS_CONTAINER = Path("/runs")
DATA_CONTAINER = Path("/data")
REPO_CONTAINER = Path("/elastic")


def runs_host_dir() -> Path:
    return Path(os.environ.get("ELASTIC_RUNS", ROOT / "data" / "libero")).expanduser().resolve()


def to_container_path(host_path: Path, writable: bool = False) -> Path:
    """Host path -> the path the containers see. Outputs (``writable``) must be under the runs dir."""
    p = Path(host_path).expanduser().resolve()
    mounts = [(runs_host_dir(), RUNS_CONTAINER)]
    if not writable:
        mounts += [(data_root().expanduser().resolve(), DATA_CONTAINER), (ROOT.resolve(), REPO_CONTAINER)]
    for host, container in mounts:
        try:
            return container / p.relative_to(host)
        except ValueError:
            pass
    if writable:
        raise ValueError(f"output {p} must be under the runs dir {runs_host_dir()} (set ELASTIC_RUNS)")
    raise ValueError(f"{p} is not under {runs_host_dir()} (ELASTIC_RUNS), {data_root()} (ELASTIC_DATA) or the repo {ROOT}")


def compose_cmd(*args: str, project: str | None = None) -> list[str]:
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    if project:
        cmd += ["-p", project]
    return cmd + list(args)


def compose_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.setdefault("ELASTIC_RUNS", str(runs_host_dir()))
    env.setdefault("WANDB_MODE", "disabled")
    runs_host_dir().mkdir(parents=True, exist_ok=True)
    return env


def host_env() -> dict[str, str]:
    """Env for the host orchestrator (needs ELASTIC and openpi_client's msgpack helpers)."""
    env = dict(os.environ)
    extra = [str(ROOT), str(ROOT / "backends" / "openpi" / "packages" / "openpi-client" / "src")]
    env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def load_recipe(path: Path | str) -> dict:
    return load_yaml(path)


def extract_config_argv(argv: Sequence[str], *, default: Path | str) -> tuple[list[str], Path]:
    """Remove ``--config PATH`` from argv; return (rest, recipe path)."""
    args = list(argv)
    path = Path(default)
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 >= len(args):
            raise ValueError("--config requires a path")
        path = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]
    return args, path


def has_flag(argv: Sequence[str], flag: str) -> bool:
    return flag in argv or any(arg.startswith(flag + "=") for arg in argv)


_RECIPE_FLAGS = (
    "demo_trials", "online_trials", "online_rotate_every", "num_samples", "baseline_num_steps", "latency_penalty",
    "parallel_penalty", "terminal_bonus", "no_finish_penalty", "verifier_bias", "k_subsample_draws",
    "offline_pretrain_steps", "online_updates_per_step", "min_online_transitions", "suite", "vgps_checkpoint",
)


def apply_recipe_defaults(argv: Sequence[str], recipe: Mapping[str, Any]) -> list[str]:
    """Append ``--key value`` from the recipe for every flag the user did not pass."""
    out = list(argv)
    if recipe.get("task_ids") is not None and not has_flag(out, "--task_ids"):
        ids = recipe["task_ids"]
        out += ["--task_ids", ids if isinstance(ids, str) else ",".join(str(x) for x in ids)]
    for key in _RECIPE_FLAGS:
        if key in recipe and not has_flag(out, f"--{key}"):
            out += [f"--{key}", str(recipe[key])]
    for key in ("second_pass_counterfactuals", "all_stopped_counterfactuals"):
        if key in recipe and not has_flag(out, f"--{key}") and not has_flag(out, f"--no-{key}"):
            out.append(f"--{key}" if recipe[key] else f"--no-{key}")
    return out
