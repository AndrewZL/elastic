"""Held-out eval rollouts: coverage / success plus budget-matched L and P."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm

from elastic.dp.eval.episode import (
    DEFAULT_SUCCESS_THRESHOLD,
    env_finished,
    summarize_per_env_compute,
    update_max_coverage,
)
from elastic.dp.sampling.selectors import DenoiseStrideSelector


def _chunk_latency_and_parallel(
    transitions, n_envs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-env L (last active denoising step + 1) and P (mean active samples over active steps) for one chunk."""
    if not transitions:
        z = np.zeros((n_envs,), dtype=np.float32)
        return z, z

    active_stack = np.stack([tr["active"] for tr in transitions], axis=0)  # (steps, B, S)
    any_active = active_stack.any(axis=2)

    step_idx = np.where(any_active, np.arange(any_active.shape[0])[:, None], -1)
    latency = (np.max(step_idx, axis=0) + 1).astype(np.float32)
    latency = np.where(any_active.any(axis=0), latency, 0.0).astype(np.float32)

    parallel_counts = active_stack.sum(axis=2).astype(np.float64)
    masked = np.where(any_active, parallel_counts, np.nan)
    with np.errstate(all="ignore"):
        parallel = np.nanmean(masked, axis=0)
    parallel = np.nan_to_num(parallel, nan=0.0).astype(np.float32)
    return latency, parallel


def seed_policy_rng(seed: int) -> None:
    """Reseed Python / NumPy / Torch for one policy-seed replicate."""
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_nested_passes(passes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack policy-seed passes over the same env inits into policy-seed / env-mean metrics."""
    cov_kn = np.stack(
        [np.asarray(p["coverages"], dtype=np.float64) for p in passes], axis=0
    )
    k, n = cov_kn.shape
    pol_means = cov_kn.mean(axis=1)
    env_means = cov_kn.mean(axis=0)
    flat = cov_kn.reshape(-1)
    mean_L = float(np.mean([p["mean_latency"] for p in passes]))
    mean_P = float(np.mean([p["mean_sample_usage"] for p in passes]))
    std_L = float(np.mean([p["std_latency"] for p in passes]))
    std_P = float(np.mean([p["std_sample_usage"] for p in passes]))
    policy_sem = (
        float(np.std(pol_means, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
    )
    env_sem = float(np.std(env_means, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    out = dict(passes[0])
    out.update(
        {
            "mean_coverage": float(np.mean(flat)),
            "success_rate": float(np.mean([p["success_rate"] for p in passes])),
            "mean_latency": mean_L,
            "std_latency": std_L,
            "mean_sample_usage": mean_P,
            "std_sample_usage": std_P,
            "mean_latency_until_success_or_horizon": mean_L,
            "mean_sample_usage_until_success_or_horizon": mean_P,
            "mean_latency_to_success": float(
                np.nanmean([p["mean_latency_to_success"] for p in passes])
            ),
            "mean_sample_usage_to_success": float(
                np.nanmean([p["mean_sample_usage_to_success"] for p in passes])
            ),
            "n_episodes": int(flat.size),
            "n_chunks": int(sum(int(p.get("n_chunks") or 0) for p in passes)),
            "coverages": flat.tolist(),
            "n_env_seeds": int(n),
            "n_policy_seeds": int(k),
            "coverage_by_policy": pol_means.tolist(),
            "coverage_by_env": env_means.tolist(),
            "coverage_policy_sem": policy_sem,
            "coverage_env_sem": env_sem,
            "coverages_kn": cov_kn.tolist(),
        }
    )
    return out


def paper_metrics(heldout: Dict[str, Any]) -> Dict[str, Any]:
    """Keys consumed by ``eval_jobs.write_summary`` / ``recipe.read_metrics_LS``."""
    cov = np.asarray(heldout.get("coverages") or [], dtype=np.float64)
    n = int(heldout.get("n_episodes") or cov.size)
    mean_cov = float(heldout["mean_coverage"])
    mean_L = float(heldout["mean_latency"])
    mean_P = float(heldout["mean_sample_usage"])
    out = {
        "coverage_mean": mean_cov,
        "mean_coverage": mean_cov,
        "success_rate": float(heldout["success_rate"]),
        "avg_inference_steps": mean_L,
        "std_inference_steps": float(heldout["std_latency"]),
        "avg_sample_usage": mean_P,
        "std_sample_usage": float(heldout["std_sample_usage"]),
        "avg_inference_steps_until_success_or_horizon": mean_L,
        "std_inference_steps_until_success_or_horizon": float(heldout["std_latency"]),
        "avg_sample_usage_until_success_or_horizon": mean_P,
        "std_sample_usage_until_success_or_horizon": float(heldout["std_sample_usage"]),
        "avg_inference_steps_to_success": float(heldout["mean_latency_to_success"]),
        "avg_sample_usage_to_success": float(heldout["mean_sample_usage_to_success"]),
        "coverage_n_episodes": n,
        "n": n,
        "n_chunks": int(heldout.get("n_chunks") or 0),
    }
    if cov.size:
        std = float(np.std(cov))
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        out["coverage_std"] = std
        out["coverage_sem"] = sem
        out["coverage_95ci_low"] = mean_cov - 1.96 * sem
        out["coverage_95ci_high"] = mean_cov + 1.96 * sem
        out["coverages"] = [float(x) for x in cov.tolist()]
    for key in (
        "n_env_seeds",
        "n_policy_seeds",
        "coverage_by_policy",
        "coverage_by_env",
        "coverage_policy_sem",
        "coverage_env_sem",
        "coverages_kn",
    ):
        if key in heldout:
            out[key] = heldout[key]
    pol_sem = heldout.get("coverage_policy_sem")
    if pol_sem is not None:
        out["coverage_95ci_low"] = mean_cov - 1.96 * float(pol_sem)
        out["coverage_95ci_high"] = mean_cov + 1.96 * float(pol_sem)
    return out


def run_selector_eval(
    runner,
    selector,
    *,
    episodes: int,
    eval_seed: int,
    scoring_fn: Optional[Callable] = None,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    max_denoise_steps: Optional[int] = None,
    show_pbar: bool = True,
    on_batch_start: Optional[Callable[[int], None]] = None,
    on_batch_end: Optional[Callable[[Sequence[float], Sequence[float]], None]] = None,
    on_newly_done: Optional[Callable[[int, np.ndarray], None]] = None,
    n_vis: int = 0,
    trial_histories_dir: Optional[str] = None,
    n_env_seeds: Optional[int] = None,
    n_policy_seeds: int = 1,
    policy_seed: int = 0,
) -> Dict[str, Any]:
    """Roll out ``episodes`` with an already-built stride selector.

    With ``n_policy_seeds`` > 1, reruns ``n_env_seeds`` fixed inits, reseeding the policy RNG each time.
    """
    k_pol = int(n_policy_seeds or 1)
    if k_pol > 1:
        n_env = int(n_env_seeds or episodes)
        passes = []
        for k in range(k_pol):
            seed_policy_rng(int(policy_seed) + k)
            passes.append(
                run_selector_eval(
                    runner,
                    selector,
                    episodes=n_env,
                    eval_seed=eval_seed,
                    scoring_fn=scoring_fn,
                    success_threshold=success_threshold,
                    max_denoise_steps=max_denoise_steps,
                    show_pbar=show_pbar,
                    on_batch_start=on_batch_start,
                    on_batch_end=on_batch_end,
                    on_newly_done=on_newly_done,
                    n_vis=0,
                    trial_histories_dir=None,
                    n_env_seeds=None,
                    n_policy_seeds=1,
                    policy_seed=0,
                )
            )
        return _aggregate_nested_passes(passes)

    empty = {
        "mean_coverage": float("nan"),
        "success_rate": float("nan"),
        "mean_latency": float("nan"),
        "std_latency": float("nan"),
        "mean_sample_usage": float("nan"),
        "std_sample_usage": float("nan"),
        "mean_latency_until_success_or_horizon": float("nan"),
        "mean_sample_usage_until_success_or_horizon": float("nan"),
        "mean_latency_to_success": float("nan"),
        "mean_sample_usage_to_success": float("nan"),
        "n_episodes": 0,
        "n_chunks": 0,
        "coverages": [],
    }
    if episodes <= 0:
        return empty

    t_max = int(getattr(selector, "t_max", 0) or 0)
    denoise_cap = int(t_max if max_denoise_steps is None else max_denoise_steps)
    n_action_steps = int(getattr(runner, "n_action_steps", 8) or 8)
    max_steps = int(getattr(runner, "max_steps", 400) or 400)
    max_chunks = max(1, int(np.ceil(max_steps / max(1, n_action_steps))))

    coverages: list = []
    chunk_latencies: list = []
    ep_latencies: list = []
    ep_parallels: list = []

    n_envs = int(runner.n_envs)
    collected = 0
    pbar = tqdm.tqdm(total=episodes, desc="Eval", disable=not show_pbar)
    thresh = float(success_threshold)

    try:
        while collected < episodes:
            if on_batch_start is not None:
                on_batch_start(collected)
            obs = runner.reset_with_seed(collected, eval_seed)
            obs_robot = runner.prepare_obs(obs)
            max_coverage = np.zeros((n_envs,), dtype=np.float32)
            ep_lat_sum = np.zeros((n_envs,), dtype=np.float64)
            ep_par_sum = np.zeros((n_envs,), dtype=np.float64)
            ep_chunks = np.zeros((n_envs,), dtype=np.float64)
            ep_lat_hist = [[] for _ in range(n_envs)]
            ep_par_hist = [[] for _ in range(n_envs)]
            done_ever = np.zeros((n_envs,), dtype=bool)

            for _ in range(max_chunks):
                selector.reset_episode()
                selector.set_current_max_coverage(max_coverage)
                if hasattr(selector, "set_heuristic_obs"):
                    h_obs = obs if isinstance(obs, np.ndarray) else obs_robot
                    if isinstance(h_obs, np.ndarray):
                        selector.set_heuristic_obs(h_obs)
                _obs_for_scoring = obs
                step_scoring = None
                if scoring_fn is not None:
                    def step_scoring(_obs, _action_B_S, _captured=_obs_for_scoring):
                        scores = scoring_fn(_captured, _action_B_S)
                        if (
                            hasattr(selector, "set_v_scores")
                            and isinstance(scores, dict)
                            and scores.get("v") is not None
                        ):
                            selector.set_v_scores(np.asarray(scores["v"]).reshape(-1))
                        if isinstance(scores, dict):
                            return scores["advantage"]
                        return scores

                step_out = runner.step(
                    obs_robot,
                    step_scoring,
                    timestep_selector=selector,
                    timestep_selector_mode="delta",
                    max_denoise_steps=denoise_cap,
                )
                (next_obs, reward, done, _info) = step_out[0]
                update_max_coverage(max_coverage, reward, n_envs)

                transitions = selector.pop_transitions()
                lat_B, par_B = _chunk_latency_and_parallel(transitions, n_envs)
                alive = ~done_ever
                for b in np.where(alive)[0]:
                    chunk_latencies.append(float(lat_B[b]))
                    ep_lat_sum[b] += float(lat_B[b])
                    ep_par_sum[b] += float(par_B[b])
                    ep_chunks[b] += 1.0
                    if (collected + int(b)) < int(n_vis):
                        ep_lat_hist[b].append(float(lat_B[b]))
                        ep_par_hist[b].append(float(par_B[b]))

                finished = env_finished(done, max_coverage, success_threshold=thresh)
                newly_done = finished & ~done_ever
                done_ever |= finished
                if on_newly_done is not None and np.any(newly_done):
                    on_newly_done(collected, newly_done)
                obs = next_obs
                obs_robot = runner.prepare_obs(obs)
                if bool(np.all(done_ever)):
                    break

            take = min(n_envs, episodes - collected)
            batch_L: List[float] = []
            batch_P: List[float] = []
            hist_dir = Path(trial_histories_dir) if trial_histories_dir else None
            if hist_dir is not None:
                hist_dir.mkdir(parents=True, exist_ok=True)
            for i in range(take):
                coverages.append(float(max_coverage[i]))
                n_c = max(float(ep_chunks[i]), 1.0)
                lat = float(ep_lat_sum[i] / n_c)
                par = float(ep_par_sum[i] / n_c)
                ep_latencies.append(lat)
                ep_parallels.append(par)
                batch_L.append(lat)
                batch_P.append(par)
                ep_idx = collected + i
                if hist_dir is not None and ep_idx < int(n_vis):
                    np.savez_compressed(
                        hist_dir / f"trial_{ep_idx:05d}.npz",
                        trial_idx=np.int32(ep_idx),
                        trial_max_coverage=np.float32(max_coverage[i]),
                        episode_inference_steps_history=np.asarray(
                            ep_lat_hist[i], dtype=np.float32
                        ),
                        episode_sample_usage_history=np.asarray(
                            ep_par_hist[i], dtype=np.float32
                        ),
                    )
            if on_batch_end is not None:
                on_batch_end(batch_L, batch_P)
            collected += take
            pbar.update(take)
    finally:
        pbar.close()

    cov = np.asarray(coverages, dtype=np.float64)
    success = cov >= thresh
    summary = summarize_per_env_compute(ep_latencies, ep_parallels, success)
    mean_L = summary["avg_inference_steps_until_success_or_horizon"]
    std_L = summary["std_inference_steps_until_success_or_horizon"]
    mean_P = summary["avg_sample_usage_until_success_or_horizon"]
    std_P = summary["std_sample_usage_until_success_or_horizon"]
    return {
        "mean_coverage": float(np.mean(cov)) if cov.size else float("nan"),
        "success_rate": float(np.mean(success.astype(np.float64))) if success.size else float("nan"),
        "mean_latency": mean_L,
        "std_latency": std_L,
        "mean_sample_usage": mean_P,
        "std_sample_usage": std_P,
        "mean_latency_until_success_or_horizon": mean_L,
        "mean_sample_usage_until_success_or_horizon": mean_P,
        "mean_latency_to_success": summary["avg_inference_steps_to_success"],
        "mean_sample_usage_to_success": summary["avg_sample_usage_to_success"],
        "mean_latency_ep": mean_L,
        "mean_sample_usage_ep": mean_P,
        "n_episodes": int(cov.size),
        "n_chunks": int(len(chunk_latencies)),
        "coverages": coverages,
    }


def run_heldout_eval(
    runner,
    stride_policy,
    *,
    t_max: int,
    n_actions: int,
    action_range,
    samples: int,
    episodes: int,
    eval_seed: int,
    device: torch.device,
    scoring_fn: Optional[Callable] = None,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    stride_buckets=None,
    max_denoise_steps: Optional[int] = None,
    show_pbar: bool = True,
) -> Dict[str, Any]:
    """Training-time wrapper: ELASTIC policy on a held-out runner."""
    was_training = stride_policy.training
    stride_policy.eval()
    selector = DenoiseStrideSelector(
        policy=stride_policy,
        device=device,
        n_actions=n_actions,
        t_max=t_max,
        action_dim=int(runner.policy.action_dim),
        samples_per_env=samples,
        stride_buckets=stride_buckets,
        action_range=action_range,
    )
    try:
        return run_selector_eval(
            runner,
            selector,
            episodes=episodes,
            eval_seed=eval_seed,
            scoring_fn=scoring_fn,
            success_threshold=success_threshold,
            max_denoise_steps=max_denoise_steps,
            show_pbar=show_pbar,
        )
    finally:
        if was_training:
            stride_policy.train()
