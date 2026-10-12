#!/usr/bin/env python3
"""ELASTIC / fixed-stride / budget-random → metrics.json.

  python scripts/dp/evaluate.py --config configs/eval/pusht_obs.yaml \\
      --stride_checkpoint CKPT --output_dir OUT

``--config`` sets defaults from a YAML and explicit flags override it."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import types

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from elastic.config import DATA_PATH_KEYS, load_yaml, resolve_data_path  # noqa: E402
from elastic.dp.runner import build_policy_and_runner  # noqa: E402
from elastic.dp.eval.episode import DEFAULT_SUCCESS_THRESHOLD  # noqa: E402
from elastic.core.actor import StrideActorCritic
from elastic.dp.sampling.heuristic import ContactProximityHeuristicSelector
from elastic.dp.sampling.selectors import (  # noqa: E402
    BudgetRandomStrideSelector,
    DenoiseStrideSelector,
    FixedStrideSelector,
)
from elastic.dp.eval.evaluate import paper_metrics, run_selector_eval  # noqa: E402
from verifiers.advantage import AdvantageVerifier  # noqa: E402

# Must match training's STRIDE_BUCKETS; the actor is continuous, buckets only feed logging and baselines.
STRIDE_BUCKETS = [5, 10, 20, 30, 40, 50, 75, 100]


def _parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="Eval YAML, e.g. configs/eval/{task}.yaml")
    p.add_argument("--checkpoint", "-c", default=None)
    p.add_argument("--stride_checkpoint", "-s", default=None)
    p.add_argument("--fixed_stride", type=int, default=None)
    p.add_argument("--random_baseline", action="store_true")
    p.add_argument("--target_inference_steps", type=float, default=None)
    p.add_argument("--target_sample_usage", type=float, default=None)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--n_env_seeds", type=int, default=0)
    p.add_argument("--n_policy_seeds", type=int, default=1)
    p.add_argument("--policy_seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--n_vis", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--success_threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    p.add_argument("--obstacle", action="store_true")
    p.add_argument("--robomimic_dataset", default=None)
    p.add_argument("--robomimic_obs_keys", default=None)
    p.add_argument("--use_verifier_reward", action="store_true")
    p.add_argument("--q_path", default=None)
    p.add_argument(
        "--parallel_penalty",
        type=float,
        default=None,
        help="Parallel penalty (used by --heuristic).",
    )
    p.add_argument("--heuristic", action="store_true")
    p.add_argument("--heuristic_task", default=None, help="pusht / can_paired / square")
    p.add_argument(
        "--heuristic_par_source",
        default="task",
        choices=["task", "v_s", "fixed"],
    )
    p.add_argument(
        "--heuristic_can_par_mode",
        default=None,
        choices=["pulse", "soft_lift"],
    )
    p.add_argument("--heuristic_can_pulse_hold", type=int, default=2)
    p.add_argument("--heuristic_can_pulse_decay", type=int, default=6)
    p.add_argument("--heuristic_can_pulse_start_prox", type=float, default=0.5)
    p.add_argument(
        "--verifier_noise_std",
        type=float,
        default=0.0,
        help="Additive N(0, σ²) on verifier advantage at eval (match train σ).",
    )
    p.add_argument("--latency", type=float, default=None, help="Optional heuristic budget lookup")
    config, _ = p.parse_known_args()
    if config.config:
        cfg = load_yaml(config.config)
        unknown = sorted(set(cfg) - set(vars(config)))
        if unknown:
            p.error(f"unknown keys in {config.config}: {unknown}")
        for key in DATA_PATH_KEYS:
            if cfg.get(key):
                cfg[key] = resolve_data_path(cfg[key])
        p.set_defaults(**cfg)
    args = p.parse_args()
    if not args.checkpoint:
        p.error("--checkpoint is required (or set it in --config)")
    return args


def _set_video_path_worker(env, video_path):
    from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper

    if hasattr(env, "env") and isinstance(env.env, VideoRecordingWrapper):
        env.env.file_path = video_path


def _stop_video_recording_worker(env):
    from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper

    if hasattr(env, "env") and isinstance(env.env, VideoRecordingWrapper):
        wrapper = env.env
        wrapper.file_path = None
        wrapper.video_recoder.stop()


def _install_video_paths(runner, output_dir: str, start_idx: int, n_vis: int, episodes: int):
    if n_vis <= 0:
        return
    import dill

    video_dir = pathlib.Path(output_dir) / "media"
    video_dir.mkdir(parents=True, exist_ok=True)
    n_envs = int(runner.n_envs)
    physical = int(getattr(runner.env, "num_envs", n_envs))
    fns = []
    for i in range(n_envs):
        ep_idx = start_idx + i
        path = str(video_dir / f"episode_{ep_idx}.mp4") if ep_idx < n_vis and ep_idx < episodes else None
        fns.append(dill.dumps(lambda env, p=path: _set_video_path_worker(env, p)))
    if len(fns) < physical:
        fns.extend(
            dill.dumps(lambda env: _set_video_path_worker(env, None))
            for _ in range(physical - len(fns))
        )
    else:
        fns = fns[:physical]
    runner.env.call_each("run_dill_function", args_list=[(x,) for x in fns])


def _stop_finished_videos(runner, start_idx: int, newly_done, n_vis: int):
    if n_vis <= 0 or newly_done is None or not np.any(newly_done):
        return
    import dill

    n_envs = int(runner.n_envs)
    physical = int(getattr(runner.env, "num_envs", n_envs))
    fns = []
    for i in range(physical):
        ep_idx = start_idx + i
        stop = i < n_envs and bool(newly_done[i]) and ep_idx < n_vis
        if stop:
            fns.append(dill.dumps(lambda env: _stop_video_recording_worker(env)))
        else:
            fns.append(dill.dumps(lambda env: None))
    runner.env.call_each("run_dill_function", args_list=[(x,) for x in fns])


def _load_stride_policy(ckpt_path: str, policy, runner, samples: int, device: torch.device):
    obs = runner.reset()
    obs_robot = runner.prepare_obs(obs)
    with torch.no_grad():
        np_obs_dict = runner._build_obs_dict(obs_robot)
        obs_dict = {k: torch.from_numpy(v).to(device=device) for k, v in np_obs_dict.items()}
        t_max = int(policy.noise_scheduler.config.num_train_timesteps)
        action_dict = policy.predict_action(obs_dict, num_inference_steps=t_max)
    obs_features = action_dict["obs_features"]
    obs_features_dim = int(np.prod(obs_features.shape[1:]))

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["policy_state_dict"]
    sample_obs_dim = int(state_dict["sample_encoder_actor.0.weight"].shape[1])
    hidden = int(state_dict["sample_encoder_actor.0.weight"].shape[0])
    stride_policy = StrideActorCritic(
        state_dim=obs_features_dim,
        sample_dim=sample_obs_dim,
        num_samples=samples,
        hidden=hidden,
    ).to(device)
    stride_policy.load_state_dict(state_dict)
    stride_policy.eval()
    return stride_policy


def main():
    a = _parse()
    n_env_seeds = int(a.n_env_seeds or 0)
    n_policy_seeds = int(a.n_policy_seeds or 1)
    env_cap = n_env_seeds if n_env_seeds > 0 else int(a.episodes)
    n_envs = min(int(a.n_envs), env_cap)
    if a.output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        a.output_dir = f"results/eval_{stamp}"
    pathlib.Path(a.output_dir).mkdir(parents=True, exist_ok=True)

    env_args = types.SimpleNamespace(
        checkpoint=a.checkpoint,
        obstacle=bool(a.obstacle),
        seed=int(a.seed),
        n_envs=n_envs,
        max_steps=int(a.max_steps),
        samples=int(a.samples),
        robomimic_dataset=a.robomimic_dataset,
        robomimic_obs_keys=a.robomimic_obs_keys,
        n_vis=int(a.n_vis),
    )
    device = torch.device(a.device)
    policy, runner = build_policy_and_runner(env_args, output_dir=a.output_dir, gamma=1.0)

    t_max = int(policy.noise_scheduler.config.num_train_timesteps)
    action_range = (0.0, float(t_max))
    n_actions = len(STRIDE_BUCKETS)
    samples = int(a.samples)

    scoring_fn = None
    need_verifier = bool(a.use_verifier_reward) or (
        bool(a.heuristic) and str(a.heuristic_par_source) == "v_s"
    )
    if need_verifier:
        if not a.q_path:
            raise SystemExit("--q_path is required with --use_verifier_reward or heuristic par_source=v_s")
        verifier = AdvantageVerifier(
            a.q_path, device, noise_std=float(a.verifier_noise_std or 0.0)
        )
        if verifier.noise_std > 0:
            print(f"AdvantageVerifier: additive noise_std={verifier.noise_std:g}")

        want_v = bool(a.heuristic) and str(a.heuristic_par_source) == "v_s"

        def scoring_fn(obs_snapshot, action_B_S):
            adv = runner.verifier_advantage(verifier, obs_snapshot, action_B_S, device)
            if not want_v:
                return adv
            with torch.no_grad():
                b, k, tdim, d = action_B_S.shape
                flat_obs = runner.flatten_obs_for_verifier(obs_snapshot, device)
                flat_act = torch.from_numpy(action_B_S).to(device).float().reshape(b * k, tdim, d)
                v = verifier.compute_v(flat_obs, flat_act).view(b, k).mean(dim=1).cpu().numpy()
            return {"advantage": adv, "v": v}

    if a.heuristic:
        if a.target_inference_steps is None or a.target_sample_usage is None:
            raise SystemExit(
                "--heuristic requires --target_inference_steps and --target_sample_usage"
            )
        htask = a.heuristic_task or ("pusht" if a.obstacle else None)
        if not htask:
            raise SystemExit("--heuristic_task is required")
        can_mode = a.heuristic_can_par_mode
        if can_mode is None and htask in {"can_paired", "can_reverse", "can"}:
            can_mode = "soft_lift" if a.heuristic_par_source == "v_s" else "pulse"
        selector = ContactProximityHeuristicSelector(
            task=htask,
            device=device,
            samples_per_env=samples,
            n_actions=n_actions,
            t_max=t_max,
            action_dim=int(policy.action_dim),
            stride_buckets=STRIDE_BUCKETS,
            action_range=action_range,
            latency=a.latency,
            parallel_penalty=a.parallel_penalty,
            target_inference_steps=float(a.target_inference_steps),
            target_sample_usage=float(a.target_sample_usage),
            par_source=a.heuristic_par_source,
            can_par_mode=can_mode or "pulse",
            can_pulse_hold=int(a.heuristic_can_pulse_hold),
            can_pulse_decay=int(a.heuristic_can_pulse_decay),
            can_pulse_start_prox=float(a.heuristic_can_pulse_start_prox),
            include_obstacle=bool(a.obstacle),
        )
        method = f"heuristic_{htask}_{a.heuristic_par_source}"
    elif a.random_baseline:
        if a.target_inference_steps is None or a.target_sample_usage is None:
            raise SystemExit(
                "--random_baseline requires --target_inference_steps and --target_sample_usage"
            )
        selector = BudgetRandomStrideSelector(
            device=device,
            target_inference_steps=float(a.target_inference_steps),
            target_sample_usage=float(a.target_sample_usage),
            t_max=t_max,
            samples_per_env=samples,
            action_dim=int(policy.action_dim),
            n_actions=n_actions,
            stride_buckets=STRIDE_BUCKETS,
            action_range=action_range,
        )
        method = (
            f"random L={a.target_inference_steps:g} P={a.target_sample_usage:g}"
        )
    elif a.fixed_stride is not None:
        selector = FixedStrideSelector(
            stride=int(a.fixed_stride),
            device=device,
            n_actions=n_actions,
            t_max=t_max,
            action_dim=int(policy.action_dim),
            samples_per_env=samples,
            stride_buckets=STRIDE_BUCKETS,
            action_range=action_range,
        )
        method = f"fixed_stride={a.fixed_stride}"
    else:
        if not a.stride_checkpoint:
            raise SystemExit(
                "--stride_checkpoint is required unless --fixed_stride / --random_baseline / --heuristic"
            )
        stride_policy = _load_stride_policy(
            a.stride_checkpoint, policy, runner, samples, device
        )
        selector = DenoiseStrideSelector(
            policy=stride_policy,
            device=device,
            n_actions=n_actions,
            t_max=t_max,
            action_dim=int(policy.action_dim),
            samples_per_env=samples,
            stride_buckets=STRIDE_BUCKETS,
            action_range=action_range,
        )
        method = f"elastic {a.stride_checkpoint}"

    def on_batch_start(start_idx: int):
        _install_video_paths(runner, a.output_dir, start_idx, int(a.n_vis), int(a.episodes))

    def on_newly_done(start_idx: int, newly_done):
        _stop_finished_videos(runner, start_idx, newly_done, int(a.n_vis))

    def on_batch_end(batch_L, batch_P):
        if hasattr(selector, "record_actual_usage") and batch_L:
            selector.record_actual_usage(float(np.mean(batch_L)), float(np.mean(batch_P)))

    print(f"Eval {method} episodes={a.episodes} n_envs={n_envs} → {a.output_dir}")
    vis_n = int(a.n_vis)
    heldout = run_selector_eval(
        runner,
        selector,
        episodes=int(a.episodes) if n_env_seeds <= 0 else n_env_seeds,
        eval_seed=int(a.seed),
        scoring_fn=scoring_fn,
        success_threshold=float(a.success_threshold),
        max_denoise_steps=t_max,
        on_batch_start=on_batch_start if vis_n > 0 else None,
        on_batch_end=on_batch_end if (a.random_baseline or a.heuristic) else None,
        on_newly_done=on_newly_done if vis_n > 0 else None,
        n_vis=vis_n,
        trial_histories_dir=str(pathlib.Path(a.output_dir) / "trial_histories")
        if vis_n > 0
        else None,
        n_env_seeds=n_env_seeds if n_env_seeds > 0 else None,
        n_policy_seeds=n_policy_seeds,
        policy_seed=int(a.policy_seed),
    )
    metrics = paper_metrics(heldout)
    metrics.update(
        method=method,
        episodes=int(a.episodes),
        n_envs=n_envs,
        samples=samples,
        seed=int(a.seed),
        success_threshold=float(a.success_threshold),
        checkpoint=a.checkpoint,
        stride_checkpoint=a.stride_checkpoint,
        fixed_stride=a.fixed_stride,
        random_baseline=bool(a.random_baseline),
        heuristic=bool(a.heuristic),
        heuristic_task=a.heuristic_task,
        heuristic_par_source=a.heuristic_par_source if a.heuristic else None,
        verifier_noise_std=float(a.verifier_noise_std or 0.0),
        n_env_seeds=n_env_seeds or None,
        n_policy_seeds=n_policy_seeds,
        policy_seed=int(a.policy_seed),
    )
    out = pathlib.Path(a.output_dir) / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(
        f"coverage={metrics['coverage_mean']:.4f} success={metrics['success_rate']:.3f} "
        f"L={metrics['avg_inference_steps']:.3f} P={metrics['avg_sample_usage']:.3f} "
        f"n={metrics['coverage_n_episodes']} → {out}"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
