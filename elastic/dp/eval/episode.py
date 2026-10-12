"""Episode-finished latch and equal-weight per-env L/P accounting.

Gym ``done`` is TimeLimit-only, so an env also finishes once its running max coverage
reaches ``success_threshold``. L/P are averaged per env, then across envs with equal weight.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_SUCCESS_THRESHOLD = 0.999


def flatten_env_vec(x, n_envs: int, dtype=None) -> np.ndarray:
    """Broadcast/reshape an env-step vector to ``(n_envs,)``."""
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    flat = np.reshape(arr, -1)
    if flat.size == 1 and n_envs != 1:
        out = np.broadcast_to(flat, (n_envs,)).copy()
    elif flat.size == n_envs:
        out = np.array(flat, copy=True)
    else:
        raise ValueError(f"env vector length {flat.size} != n_envs {n_envs}")
    return out


def update_max_coverage(
    max_coverage: np.ndarray,
    reward,
    n_envs: Optional[int] = None,
) -> np.ndarray:
    """In-place running max of env reward/coverage. Returns ``max_coverage``."""
    n = int(max_coverage.shape[0] if n_envs is None else n_envs)
    reward_B = flatten_env_vec(reward, n, dtype=np.float32)
    np.maximum(max_coverage, reward_B, out=max_coverage)
    return max_coverage


def env_finished(
    done,
    max_coverage: np.ndarray,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    n_envs: Optional[int] = None,
) -> np.ndarray:
    """Per-env finished mask: gym done OR running max coverage >= threshold."""
    n = int(max_coverage.shape[0] if n_envs is None else n_envs)
    done_B = flatten_env_vec(done, n).astype(bool, copy=False)
    cov = np.asarray(max_coverage, dtype=np.float64).reshape(-1)
    if cov.size != n:
        raise ValueError(f"max_coverage length {cov.size} != n_envs {n}")
    return done_B | (cov >= float(success_threshold))


def summarize_per_env_compute(
    per_env_L: np.ndarray,
    per_env_P: np.ndarray,
    succeeded: np.ndarray,
) -> dict:
    """Equal-weight means of per-env L/P (all envs, and successful-only)."""
    L = np.asarray(per_env_L, dtype=np.float64).reshape(-1)
    P = np.asarray(per_env_P, dtype=np.float64).reshape(-1)
    ok = np.asarray(succeeded, dtype=bool).reshape(-1)
    if L.size != P.size or L.size != ok.size:
        raise ValueError("per-env L, P, and succeeded must have the same length")

    def _mean_std(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(x)), float(np.std(x))

    all_L, all_L_std = _mean_std(L)
    all_P, all_P_std = _mean_std(P)
    succ_L, succ_L_std = _mean_std(L[ok])
    succ_P, succ_P_std = _mean_std(P[ok])
    return {
        "avg_inference_steps_until_success_or_horizon": all_L,
        "std_inference_steps_until_success_or_horizon": all_L_std,
        "avg_sample_usage_until_success_or_horizon": all_P,
        "std_sample_usage_until_success_or_horizon": all_P_std,
        "avg_inference_steps_to_success": succ_L,
        "std_inference_steps_to_success": succ_L_std,
        "avg_sample_usage_to_success": succ_P,
        "std_sample_usage_to_success": succ_P_std,
        "success_window_n_episodes": int(np.sum(ok)),
    }
