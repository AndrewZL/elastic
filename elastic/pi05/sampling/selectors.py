from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any

import numpy as np
import torch

from elastic.core import action as _action
from elastic.core.action import FLOW_MIN_STRIDE
from elastic.pi05.constants import DIFFUSION_T_MAX
from elastic.pi05.constants import PAUSE_ACTION_THRESHOLD


def _to_numpy(value: Any, dtype=None) -> np.ndarray:
    if value is None:
        return value
    if isinstance(value, np.ndarray):
        out = value
    elif isinstance(value, torch.Tensor):
        out = value.detach().cpu().numpy()
    else:
        import jax  # lazy: host-side launchers import this module without jax

        out = np.asarray(jax.device_get(value))
    return out.astype(dtype) if dtype is not None else out


def action_to_stride(action, action_range: tuple[float, float], pause_threshold: float = PAUSE_ACTION_THRESHOLD):
    """Actor action -> stride in normalized flow time (float32); 0 where paused."""
    stride = _action.action_to_stride(action, action_range, pause_threshold, min_stride=FLOW_MIN_STRIDE)
    return stride if isinstance(stride, torch.Tensor) else stride.astype(np.float32)


def delta_to_action(delta, action_range: tuple[float, float], pause_threshold: float = PAUSE_ACTION_THRESHOLD):
    """Flow-time stride -> actor action; stride <= 0 -> +1 (pause)."""
    return _action.delta_to_action(delta, action_range, pause_threshold, min_stride=FLOW_MIN_STRIDE)


@dataclass
class SelectorTransition:
    state_obs: np.ndarray
    sample_obs: np.ndarray
    next_state_obs: np.ndarray
    next_sample_obs: np.ndarray
    action: np.ndarray
    done: np.ndarray
    active: np.ndarray
    running: np.ndarray
    timestep: np.ndarray
    next_timestep: np.ndarray


class StrideSelectorBase:
    """Selector contract shared by fixed, random and learned stride policies.

    Selectors receive flow times ``t`` in [1, 0] and return positive strides in the same
    units (10 steps = stride 0.1); ``max_steps`` caps model evaluations per chunk. With
    ``use_extended_obs`` False, per-sample obs match the Diffusion Policy layout.
    """

    def __init__(
        self,
        *,
        samples_per_env: int,
        max_steps: int,
        action_dim: int,
        action_range: tuple[float, float] | None = None,
        pause_action_threshold: float = PAUSE_ACTION_THRESHOLD,
        use_extended_obs: bool = True,
    ):
        self.samples_per_env = int(samples_per_env)
        self.max_steps = int(max_steps)
        self.action_dim = int(action_dim)
        self.action_range = action_range or (0.0, 1.0)
        self.pause_action_threshold = float(pause_action_threshold)
        self.use_extended_obs = bool(use_extended_obs)
        self._transitions: list[SelectorTransition] = []
        self._stopped_mask: np.ndarray | None = None
        self._stopped_mask_before_action: np.ndarray | None = None
        self._last_running_mask: np.ndarray | None = None
        self._last_raw_action: np.ndarray | None = None
        self._last_delta: np.ndarray | None = None
        self._last_score_magnitude: np.ndarray | None = None
        self._last_x0_dev: np.ndarray | None = None
        self._last_mid_feature: np.ndarray | None = None
        self._mid_feature_dim = 0

    def reset_episode(self) -> None:
        self._transitions = []
        self._stopped_mask = None
        self._stopped_mask_before_action = None
        self._last_running_mask = None
        self._last_raw_action = None
        self._last_delta = None
        self._last_score_magnitude = None
        self._last_x0_dev = None
        self._last_mid_feature = None

    def pop_transitions(self) -> list[dict[str, np.ndarray]]:
        transitions = [transition.__dict__ for transition in self._transitions]
        self._transitions = []
        return transitions

    def _ensure_stopped_mask(self, timestep: np.ndarray) -> tuple[int, int]:
        flat = timestep.reshape(-1)
        k = max(self.samples_per_env, 1)
        if flat.size % k != 0:
            raise ValueError(f"timestep size {flat.size} must be divisible by samples_per_env={k}")
        b = flat.size // k
        if self._stopped_mask is None or self._stopped_mask.shape != (b, k):
            self._stopped_mask = np.zeros((b, k), dtype=np.bool_)
        return b, k

    def _apply_no_restart_constraint(self, proposed_delta: np.ndarray, timestep: np.ndarray) -> np.ndarray:
        timestep = _to_numpy(timestep, np.float32).reshape(-1)
        b, k = self._ensure_stopped_mask(timestep)
        stopped = self._stopped_mask
        if stopped is None:
            raise RuntimeError("stopped mask was not initialized")
        t_bk = timestep.reshape(b, k)
        running = (t_bk > 1e-6) & (~stopped)
        self._last_running_mask = running.copy()
        self._stopped_mask_before_action = stopped.copy()

        out = np.asarray(proposed_delta, dtype=np.float32).reshape(b, k)
        out[~running] = 0.0
        out = np.clip(out, 0.0, t_bk)
        self._stopped_mask = stopped | (running & (out == 0))
        self._last_delta = out.reshape(-1).copy()
        return out.reshape(-1)

    def _x0_dev(self, x0_pred: np.ndarray | None, timestep: np.ndarray) -> np.ndarray | None:
        if x0_pred is None or self.samples_per_env <= 1:
            return None
        x0 = _to_numpy(x0_pred, np.float32)
        b, k = self._ensure_stopped_mask(timestep)
        x0_flat = x0[..., : self.action_dim].reshape(b, k, -1)
        centered = x0_flat - x0_flat.mean(axis=1, keepdims=True)
        raw = np.linalg.norm(centered, axis=-1).reshape(b * k, 1)
        t_norm = _to_numpy(timestep, np.float32).reshape(b * k, 1).clip(0.0, 1.0)
        return raw * np.cos(math.pi / 2.0 * t_norm).clip(min=1e-3)

    def _obs_from(
        self,
        sample: Any,
        timestep: Any,
        global_cond: Any,
        *,
        score_magnitude: Any = None,
        x0_dev: Any = None,
        mid_feature_pooled: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_np = _to_numpy(sample, np.float32)
        timestep_np = _to_numpy(timestep, np.float32).reshape(-1)
        global_np = _to_numpy(global_cond, np.float32)
        b, k = self._ensure_stopped_mask(timestep_np)
        n = b * k

        if sample_np.shape[0] != n:
            raise ValueError(f"sample batch {sample_np.shape[0]} does not match B*K={n}")
        x_flat = sample_np[..., : self.action_dim].reshape(n, -1)
        t_norm = timestep_np.reshape(n, 1).astype(np.float32)

        if self._last_delta is None:
            moving = (timestep_np.reshape(b, k) > 1e-6).astype(np.float32)
        else:
            moving = (self._last_delta.reshape(b, k) != 0).astype(np.float32)
        active_frac = (moving.sum(axis=1, keepdims=True) / max(k, 1)).repeat(k, axis=1).reshape(n, 1)

        stopped = self._stopped_mask
        if stopped is None:
            raise RuntimeError("stopped mask was not initialized")
        stopped_flat = stopped.reshape(n, 1).astype(np.float32)
        if self.use_extended_obs:
            score = (
                _to_numpy(score_magnitude, np.float32).reshape(n, 1)
                if score_magnitude is not None
                else np.zeros((n, 1), dtype=np.float32)
            )
            x0d = (
                _to_numpy(x0_dev, np.float32).reshape(n, 1)
                if x0_dev is not None
                else np.zeros((n, 1), dtype=np.float32)
            )
            mid = (
                _to_numpy(mid_feature_pooled, np.float32).reshape(n, -1)
                if mid_feature_pooled is not None
                else np.zeros((n, self._mid_feature_dim), dtype=np.float32)
            )
            if mid_feature_pooled is not None:
                self._mid_feature_dim = int(mid.shape[-1])
            sample_obs = np.concatenate([x_flat, mid, t_norm, active_frac, stopped_flat, score, x0d], axis=-1)
        else:
            sample_obs = np.concatenate([x_flat, t_norm, active_frac, stopped_flat], axis=-1)

        if global_np.ndim == 2 and global_np.shape[0] == n:
            state_obs = global_np.reshape(b, k, -1)
        elif global_np.ndim == 2 and global_np.shape[0] == b:
            state_obs = np.repeat(global_np[:, None, :], k, axis=1)
        elif global_np.ndim == 2 and global_np.shape[0] == 1 and b == 1:
            state_obs = np.repeat(global_np[:, None, :], k, axis=1)
        else:
            state_obs = global_np.reshape(b, k, -1)
        return state_obs.astype(np.float32), sample_obs.reshape(b, k, -1).astype(np.float32)

    def on_transition(
        self,
        *,
        sample: Any,
        timestep: Any,
        global_cond: Any,
        action: Any,
        next_sample: Any,
        next_timestep: Any,
        score_magnitude: Any = None,
        x0_pred: Any = None,
        mid_feature_pooled: Any = None,
        **_: Any,
    ) -> None:
        timestep_np = _to_numpy(timestep, np.float32).reshape(-1)
        next_timestep_np = _to_numpy(next_timestep, np.float32).reshape(-1)
        if self.use_extended_obs:
            x0_dev = self._x0_dev(_to_numpy(x0_pred, np.float32) if x0_pred is not None else None, timestep_np)
            self._last_score_magnitude = _to_numpy(score_magnitude, np.float32) if score_magnitude is not None else None
            self._last_x0_dev = x0_dev
            self._last_mid_feature = (
                _to_numpy(mid_feature_pooled, np.float32) if mid_feature_pooled is not None else None
            )
        else:
            x0_dev = None
            self._last_score_magnitude = None
            self._last_x0_dev = None
            self._last_mid_feature = None

        stopped_after = None if self._stopped_mask is None else self._stopped_mask.copy()
        if self._stopped_mask_before_action is not None:
            self._stopped_mask = self._stopped_mask_before_action.copy()
        state_obs, sample_obs = self._obs_from(
            sample,
            timestep_np,
            global_cond,
            score_magnitude=score_magnitude if self.use_extended_obs else None,
            x0_dev=x0_dev,
            mid_feature_pooled=mid_feature_pooled if self.use_extended_obs else None,
        )
        if stopped_after is not None:
            self._stopped_mask = stopped_after
        next_state_obs, next_sample_obs = self._obs_from(next_sample, next_timestep_np, global_cond)

        b, k = state_obs.shape[:2]
        action_np = _to_numpy(action, np.float32)
        action_norm = delta_to_action(action_np, self.action_range, self.pause_action_threshold)
        if self._last_raw_action is not None:
            action_norm = self._last_raw_action
            self._last_raw_action = None
        action_norm = action_norm.reshape(b, k, -1).astype(np.float32)

        running = self._last_running_mask
        if running is None:
            running = np.zeros((b, k), dtype=np.bool_)
        done = next_timestep_np.reshape(b, k) <= 1e-6
        active = (timestep_np.reshape(b, k) > next_timestep_np.reshape(b, k) + 1e-8) & running
        self._transitions.append(
            SelectorTransition(
                state_obs=state_obs,
                sample_obs=sample_obs,
                next_state_obs=next_state_obs,
                next_sample_obs=next_sample_obs,
                action=action_norm,
                done=done.astype(np.bool_),
                active=active.astype(np.bool_),
                running=running.astype(np.bool_),
                timestep=timestep_np.reshape(b, k).astype(np.float32),
                next_timestep=next_timestep_np.reshape(b, k).astype(np.float32),
            )
        )

    def __call__(self, *, sample: Any, timestep: Any, global_cond: Any, **kwargs: Any):
        raise NotImplementedError


class FixedStrideSelector(StrideSelectorBase):
    def __init__(self, *, stride: float, active_samples_per_env: int | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.stride = float(stride)
        self.active_samples_per_env = active_samples_per_env

    def __call__(self, *, sample: Any, timestep: Any, global_cond: Any, **_: Any):
        timestep_np = _to_numpy(timestep, np.float32).reshape(-1)
        b, k = self._ensure_stopped_mask(timestep_np)
        active = k if self.active_samples_per_env is None else min(int(self.active_samples_per_env), k)
        proposed = np.zeros((b, k), dtype=np.float32)
        proposed[:, :active] = self.stride
        return self._apply_no_restart_constraint(proposed, timestep_np)


class BudgetRandomStrideSelector(StrideSelectorBase):
    def __init__(self, *, target_steps: float, target_samples: float, deficit_gain: float = 0.7, **kwargs: Any):
        super().__init__(**kwargs)
        self.target_steps = float(target_steps)
        self.target_samples = float(target_samples)
        self.deficit_gain = float(deficit_gain)
        self._deficit_steps = 0.0
        self._deficit_samples = 0.0
        self._episode_stride: float | None = None
        self._episode_active: np.ndarray | None = None

    def record_actual_usage(self, actual_steps: float, actual_samples: float) -> None:
        if np.isfinite(actual_steps):
            self._deficit_steps += self.target_steps - float(actual_steps)
        if np.isfinite(actual_samples):
            self._deficit_samples += self.target_samples - float(actual_samples)

    def reset_episode(self) -> None:
        super().reset_episode()
        effective_steps = np.clip(
            self.target_steps + self.deficit_gain * self._deficit_steps, 1.0, float(self.max_steps)
        )
        effective_samples = np.clip(
            self.target_samples + self.deficit_gain * self._deficit_samples, 1.0, float(self.samples_per_env)
        )
        n_steps = int(np.floor(effective_steps) + (np.random.random() < (effective_steps % 1.0)))
        n_steps = max(1, min(n_steps, self.max_steps))
        self._episode_stride = 1.0 / float(n_steps)
        n_active = int(np.floor(effective_samples) + (np.random.random() < (effective_samples % 1.0)))
        n_active = max(1, min(n_active, self.samples_per_env))
        self._episode_active = np.random.permutation(self.samples_per_env)[:n_active]

    def __call__(self, *, sample: Any, timestep: Any, global_cond: Any, **_: Any):
        if self._episode_stride is None or self._episode_active is None:
            self.reset_episode()
        timestep_np = _to_numpy(timestep, np.float32).reshape(-1)
        b, k = self._ensure_stopped_mask(timestep_np)
        proposed = np.zeros((b, k), dtype=np.float32)
        proposed[:, self._episode_active] = float(self._episode_stride)
        return self._apply_no_restart_constraint(proposed, timestep_np)


class PerChunkRandomStrideSelector(StrideSelectorBase):
    """One fixed stride per chunk, drawn from integer buckets of a 100-step schedule (stride = bucket / 100)."""

    def __init__(
        self,
        *,
        step_options: list[float],
        weights: list[float] | None = None,
        diffusion_t_max: float = DIFFUSION_T_MAX,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not step_options:
            raise ValueError("step_options must be non-empty")
        self.step_options = np.asarray(step_options, dtype=np.float32)
        if np.any(self.step_options <= 0):
            raise ValueError(f"step_options must be positive, got {step_options}")
        self.diffusion_t_max = float(diffusion_t_max)
        if weights is None:
            self.weights = np.ones_like(self.step_options, dtype=np.float64) / len(self.step_options)
        else:
            raw = np.asarray(weights, dtype=np.float64)
            if raw.shape != self.step_options.shape:
                raise ValueError(f"weights shape {raw.shape} does not match step_options {self.step_options.shape}")
            self.weights = raw / np.sum(raw)
        self._episode_stride: float | None = None

    def reset_episode(self) -> None:
        super().reset_episode()
        stride_bucket = float(np.random.choice(self.step_options, p=self.weights))
        self._episode_stride = stride_bucket / max(self.diffusion_t_max, 1.0)

    def __call__(self, *, sample: Any, timestep: Any, global_cond: Any, **_: Any):
        if self._episode_stride is None:
            self.reset_episode()
        timestep_np = _to_numpy(timestep, np.float32).reshape(-1)
        b, k = self._ensure_stopped_mask(timestep_np)
        proposed = np.full((b, k), float(self._episode_stride), dtype=np.float32)
        return self._apply_no_restart_constraint(proposed, timestep_np)


class LearnedStrideSelector(StrideSelectorBase):
    def __init__(self, *, policy: torch.nn.Module, device: str | torch.device = "cpu", deterministic: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self._logged_obs_align = False

    def _align_obs_to_policy(self, state_obs: np.ndarray, sample_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Zero-pad obs up to ``policy.{state,sample}_dim``; never truncates.

        A fresh selector has ``_mid_feature_dim`` 0, so early steps can emit narrower obs
        than the policy was trained on.
        """
        policy = self.policy
        if not hasattr(policy, "sample_dim") or not hasattr(policy, "state_dim"):
            return state_obs, sample_obs
        t_sd = int(policy.sample_dim)
        t_st = int(policy.state_dim)
        d_in = int(sample_obs.shape[-1])
        if d_in > t_sd:
            raise ValueError(
                f"LearnedStrideSelector sample_obs last dim {d_in} exceeds policy.sample_dim {t_sd}; "
                "truncation is not allowed."
            )
        if d_in < t_sd:
            if not self._logged_obs_align:
                logging.debug(
                    "LearnedStrideSelector: padding sample_obs width %d -> %d for SAC (selector mid state)",
                    d_in,
                    t_sd,
                )
                self._logged_obs_align = True
            pad = np.zeros((*sample_obs.shape[:-1], t_sd - d_in), dtype=np.float32)
            sample_obs = np.concatenate([sample_obs, pad], axis=-1)
        st_in = int(state_obs.shape[-1])
        if t_st > 0:
            if st_in > t_st:
                raise ValueError(
                    f"LearnedStrideSelector state_obs last dim {st_in} exceeds policy.state_dim {t_st}; "
                    "truncation is not allowed."
                )
            if st_in < t_st:
                pad = np.zeros((*state_obs.shape[:-1], t_st - st_in), dtype=np.float32)
                state_obs = np.concatenate([state_obs, pad], axis=-1)
        return state_obs, sample_obs

    def __call__(self, *, sample: Any, timestep: Any, global_cond: Any, **_: Any):
        timestep_np = _to_numpy(timestep, np.float32).reshape(-1)
        state_obs, sample_obs = self._obs_from(
            sample,
            timestep_np,
            global_cond,
            score_magnitude=self._last_score_magnitude if self.use_extended_obs else None,
            x0_dev=self._last_x0_dev if self.use_extended_obs else None,
            mid_feature_pooled=self._last_mid_feature if self.use_extended_obs else None,
        )
        state_obs, sample_obs = self._align_obs_to_policy(state_obs, sample_obs)
        with torch.no_grad():
            state_t = torch.from_numpy(state_obs).to(self.device)
            sample_t = torch.from_numpy(sample_obs).to(self.device)
            action, _, _ = self.policy.sample_action(state_t, sample_t, deterministic=self.deterministic)
        raw_action = action.detach().cpu().numpy().astype(np.float32)
        self._last_raw_action = raw_action
        proposed = action_to_stride(raw_action, self.action_range, self.pause_action_threshold)
        delta = self._apply_no_restart_constraint(proposed, timestep_np)
        running = self._last_running_mask
        if running is not None:
            raw_action = raw_action.reshape(running.shape[0], running.shape[1], -1)
            raw_action[~running] = 1.0
            self._last_raw_action = raw_action
        return delta
