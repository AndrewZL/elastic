"""Diffusion Policy checkpoint loading, runner construction and reward calibration."""

import datetime
import json
import os
import pathlib
import re
import sys
from typing import Any, cast, Optional, Tuple
import dill
import hydra
import torch
from omegaconf import open_dict

from elastic.config import resolve_repo_path


def ensure_diffusion_policy_importable() -> None:
    """Prefer the bundled checkout when ``diffusion_policy`` is not already on the path."""
    bundled = pathlib.Path(__file__).resolve().parents[2] / "backends" / "diffusion_policy"
    if bundled.is_dir():
        entry = str(bundled)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _load_diffusion_policy_from_checkpoint(checkpoint: str, output_dir: str, device: torch.device):
    """Load a DP checkpoint as a frozen policy with the adaptive stride sampler."""
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # The adaptive stride sampler needs DDIM.
    cfg.policy.noise_scheduler["_target_"] = "diffusers.schedulers.scheduling_ddim.DDIMScheduler"
    with open_dict(cfg.policy.noise_scheduler):
        if "variance_type" in cfg.policy.noise_scheduler:
            del cfg.policy.noise_scheduler["variance_type"]
    cfg.policy.num_inference_steps = 0

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace = cast(Any, workspace)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    from elastic.dp.sampling.policies import adopt_policy

    adopt_policy(policy)

    policy.to(device)
    policy.eval()
    return policy, cfg


def _build_pusht_runner(args, cfg, output_dir: str, num_action_samples: int):
    """Build a PushT runner using n_obs_steps/n_action_steps from the checkpoint cfg."""
    n_obs_steps = int(cfg.get("n_obs_steps", 8))
    n_action_steps = int(cfg.get("n_action_steps", 8))
    os.makedirs(output_dir, exist_ok=True)
    n_vis = int(getattr(args, "n_vis", 0) or 0)

    from envs.pusht import PushTKeypointsResetVectorRunner

    return PushTKeypointsResetVectorRunner(
        output_dir=output_dir,
        n_train=0,
        n_test=args.n_envs,
        n_test_vis=n_vis,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        test_start_seed=args.seed,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_action_samples=num_action_samples,
        shape_type="tee",
        include_obstacle=args.obstacle,
    )


def _build_robomimic_runner(args, cfg, output_dir: str, num_action_samples: int, gamma: float):
    """Build a robomimic runner; abs_action comes from the checkpoint cfg unless --abs_action is set."""
    from envs.robomimic import RobomimicResetVectorRunner

    task_cfg = cfg.get("task", {}) if cfg else {}
    shape_meta = task_cfg.get("shape_meta", None)
    if shape_meta is not None:
        obs_keys = None
    else:
        obs_keys = (
            [k.strip() for k in args.robomimic_obs_keys.split(",")] if args.robomimic_obs_keys else None
        )

    if getattr(args, "abs_action", None) is not None:
        abs_action = args.abs_action
    else:
        abs_action = bool(task_cfg.get("abs_action", False))
    print(f"abs_action: {abs_action}")
    n_vis = int(getattr(args, "n_vis", 0) or 0)

    return RobomimicResetVectorRunner(
        output_dir=output_dir,
        dataset_path=args.robomimic_dataset,
        obs_keys=obs_keys,
        shape_meta=shape_meta,
        n_train=0,
        n_test=args.n_envs,
        n_test_vis=n_vis,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        test_start_seed=getattr(args, "seed", 10000),
        num_action_samples=num_action_samples,
        gamma=gamma,
        abs_action=abs_action,
    )


def build_runner(args, cfg, output_dir: str, policy, gamma: float, n_envs: Optional[int] = None):
    """Build a runner that reuses an already-loaded diffusion ``policy`` / ``cfg``."""
    import copy

    ensure_diffusion_policy_importable()
    num_action_samples = args.samples
    runner_args = args
    if n_envs is not None:
        runner_args = copy.copy(args)
        runner_args.n_envs = int(n_envs)

    if getattr(runner_args, "robomimic_dataset", None):
        runner = _build_robomimic_runner(runner_args, cfg, output_dir, num_action_samples, gamma)
    else:
        runner = _build_pusht_runner(runner_args, cfg, output_dir, num_action_samples)

    runner.set_policy(policy)
    return runner


def build_eval_runner(args, train_runner, policy, gamma: float, device):
    """Held-out eval runner that shares the train policy."""
    ensure_diffusion_policy_importable()
    eval_n = min(int(args.eval_n_envs), int(args.eval_episodes))
    eval_out = pathlib.Path(train_runner.output_dir) / "eval_holdout"
    eval_out.mkdir(parents=True, exist_ok=True)
    cfg = getattr(train_runner, "_dp_cfg", None)
    if cfg is None:
        _, cfg = _load_diffusion_policy_from_checkpoint(
            _resolve_checkpoint(args), str(eval_out), device
        )
    return build_runner(args, cfg, str(eval_out), policy, gamma, n_envs=eval_n)


def build_policy_and_runner(args, output_dir: str, gamma: float) -> Tuple[Any, Any]:
    """Return (policy, runner): PushT by default, robomimic when --robomimic_dataset is set."""
    ensure_diffusion_policy_importable()
    device = torch.device("cuda:0")
    checkpoint = _resolve_checkpoint(args)
    policy, cfg = _load_diffusion_policy_from_checkpoint(checkpoint, output_dir, device)
    runner = build_runner(args, cfg, output_dir, policy, gamma)
    runner._dp_cfg = cfg
    return policy, runner


def _resolve_checkpoint(args) -> str:
    """Return the checkpoint path, applying defaults if not explicitly set."""
    if args.checkpoint:
        return args.checkpoint
    if args.robomimic_dataset:
        raise ValueError("--checkpoint is required for robomimic mode (no default checkpoint)")
    raise ValueError("--checkpoint is required")


# Robomimic tasks in match order: can_reverse datasets live under paths that also
# contain "can_paired" (e.g. can_paired_from_can_reverse), so check it first.
_ROBOMIMIC_TASK_PATTERNS = (
    ("can_reverse", r"can_reverse"),
    ("can_paired", r"can_paired"),
    ("square_ph", r"square[/_-]ph"),
    ("square_mh", r"square[/_-]mh"),
)
_PUSHT_OBS_TOKENS = ("pusht_obs", "pusht-obs", "obs_keypoints", "_obs_lowdim", "obstacle")


def infer_task(checkpoint: str, robomimic_dataset: Optional[str], obstacle: bool) -> str:
    """Task slug (pusht, pusht_obs, can_paired, ...) from the checkpoint/dataset paths.

    For robomimic the dataset path decides; the checkpoint is only a fallback.
    """
    ck = (checkpoint or "").lower()
    if robomimic_dataset:
        ds = robomimic_dataset.lower()
        for text in (ds, f"{ck}\n{ds}"):
            for slug, pattern in _ROBOMIMIC_TASK_PATTERNS:
                if re.search(pattern, text):
                    return slug
        raise ValueError(
            "Could not infer robomimic task from paths (expected can_paired, can_reverse, "
            f"square_mh or square_ph). Got dataset={robomimic_dataset!r} checkpoint={checkpoint!r}"
        )
    if obstacle or any(tok in ck for tok in _PUSHT_OBS_TOKENS):
        return "pusht_obs"
    return "pusht"


def extract_run_tag(checkpoint_path: str) -> str:
    """Short run tag, e.g. ``.../13.29.25_train_diffusion_unet_lowdim_can_lowdim/checkpoints/x.ckpt`` -> ``unet_lowdim_can``."""
    if not checkpoint_path:
        return "run"
    segment = checkpoint_path.split("/")[-3]
    segment = re.sub(r"^\d+\.\d+\.\d+_train_", "", segment)
    segment = re.sub(r"^diffusion_", "", segment)
    segment = re.sub(r"_lowdim$", "", segment)
    return segment


def resolve_output_dir(args) -> str:
    """Build the output directory name from args, or return args.output_dir if set."""
    if args.output_dir:
        return args.output_dir
    checkpoint = _resolve_checkpoint(args)
    run_tag = extract_run_tag(checkpoint)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/sac_stride_{args.latency}_{args.parallel_penalty}_{run_tag}_{timestamp}"


def auto_calibrate_reward_coefficients(args) -> Optional[dict]:
    """Rescale args' reward coefficients from Can_paired units to this task's verifier units, if enabled.

    Reads ``<task>_calibrated_from_can_paired_S{samples}.json``; returns a summary dict, or None when disabled.
    """
    if not getattr(args, "auto_calibrate_penalties", False):
        return None

    checkpoint = _resolve_checkpoint(args)
    task_slug = infer_task(
        checkpoint=checkpoint,
        robomimic_dataset=getattr(args, "robomimic_dataset", None),
        obstacle=bool(getattr(args, "obstacle", False)),
    )
    task_name = task_slug
    if task_slug == "can_paired":
        return {
            "task": task_name,
            "task_slug": task_slug,
            "status": "reference_task_no_scaling",
        }

    cal_dir = resolve_repo_path(getattr(args, "calibrated_penalty_dir", "configs/calibration"))
    cal_file = cal_dir / f"{task_slug}_calibrated_from_can_paired_S{int(args.samples)}.json"
    if not cal_file.exists():
        raise SystemExit(
            f"missing calibration file {cal_file}; set calibrated_penalty_dir or "
            "auto_calibrate_penalties: false"
        )

    with open(cal_file, "r", encoding="utf-8") as f:
        cal = json.load(f)
    ref = cal.get("reference", {})
    tgt = cal.get("target", {})
    mu_ref = float(ref["mu"])
    sigma_ref = float(ref["sigma"])
    mu_tgt = float(tgt["mu"])
    sigma_tgt = float(tgt["sigma"])
    if sigma_ref <= 0.0 or sigma_tgt <= 0.0:
        return {
            "task": task_name,
            "task_slug": task_slug,
            "status": "invalid_sigma",
            "path": str(cal_file),
        }

    scale = sigma_ref / sigma_tgt
    args.terminal_bonus = float(args.terminal_bonus) * scale
    args.latency = float(args.latency) * scale
    args.parallel_penalty = float(args.parallel_penalty) * scale
    args.no_finish_penalty = float(args.no_finish_penalty) * scale
    args.verifier_bias = (mu_ref + float(args.verifier_bias)) * (sigma_tgt / sigma_ref) - mu_tgt

    out = {
        "task": task_name,
        "task_slug": task_slug,
        "status": "applied",
        "path": str(cal_file),
        "scale": scale,
        "sigma_ref": sigma_ref,
        "sigma_tgt": sigma_tgt,
        "mu_ref": mu_ref,
        "mu_tgt": mu_tgt,
    }

    return out
