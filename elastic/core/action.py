"""Stride action <-> denoising stride.

a ∈ [-1, 1]; a >= ``pause_threshold`` pauses the sample (stride 0), otherwise a maps linearly
onto [max(low, min_stride), high]. Diffusion Policy strides are integer DDIM timesteps;
pi0.5 strides are continuous flow time.
"""
import numpy as np
import torch

FLOW_MIN_STRIDE = 1e-3


def _active_range(action_range, min_stride: float) -> tuple[float, float]:
    low, high = action_range
    active_low = max(float(low), float(min_stride))
    return active_low, max(float(high), active_low)


def action_to_stride(action, action_range, pause_threshold: float = 0.98, active_eps: float = 1e-3, min_stride: float = 1.0):
    """Continuous stride; 0 where paused."""
    active_low, active_high = _active_range(action_range, min_stride)
    active_upper = pause_threshold - active_eps
    if isinstance(action, torch.Tensor):
        a = action.float()
        active_frac = torch.clamp((a - (-1.0)) / max(1e-8, (active_upper - (-1.0))), 0.0, 1.0)
        stride = active_low + active_frac * (active_high - active_low)
        return torch.where(a >= pause_threshold, torch.zeros_like(stride), stride)
    a = np.asarray(action, dtype=np.float32)
    active_frac = np.clip((a - (-1.0)) / max(1e-8, (active_upper - (-1.0))), 0.0, 1.0)
    stride = active_low + active_frac * (active_high - active_low)
    return np.where(a >= pause_threshold, 0.0, stride)


def action_to_delta(action, action_range, pause_threshold: float = 0.98, active_eps: float = 1e-3):
    """Integer DDIM timestep jump (Diffusion Policy); 0 where paused."""
    stride = action_to_stride(action, action_range, pause_threshold, active_eps)
    if isinstance(stride, torch.Tensor):
        return torch.round(stride).long()
    return np.round(stride).astype(np.int64)


def delta_to_action(delta, action_range, pause_threshold: float = 0.98, active_eps: float = 1e-3, min_stride: float = 1.0):
    """Inverse map, for replay rows of strides the actor did not choose. delta <= 0 -> +1 (pause)."""
    active_low, active_high = _active_range(action_range, min_stride)
    denom = max(1e-8, active_high - active_low)
    active_upper = pause_threshold - active_eps
    if isinstance(delta, torch.Tensor):
        d = delta.float()
        active_frac = torch.clamp((d - active_low) / denom, 0.0, 1.0)
        active_action = (-1.0) + active_frac * (active_upper - (-1.0))
        return torch.where(d <= 0, torch.full_like(active_action, 1.0), active_action)
    d = np.asarray(delta, dtype=np.float32)
    active_frac = np.clip((d - active_low) / denom, 0.0, 1.0)
    active_action = (-1.0) + active_frac * (active_upper - (-1.0))
    return np.where(d <= 0, 1.0, active_action)
