"""Stride selectors for Diffusion Policy eval schedules."""
import math

from elastic.config import load_dp_defaults
import torch
import numpy as np


from elastic.core.action import (
    action_to_delta,
    delta_to_action,
)
from elastic.core.actor import StrideActorCritic

DEFAULT_PAUSE_ACTION_THRESHOLD = float(load_dp_defaults().get("pause_action_threshold", 0.7))


class StrideSelectorBase:
    """Interface for Stride Selectors."""

    def __init__(
        self,
        device,
        n_actions=None,
        t_max=None,
        action_dim=None,
        samples_per_env=1,
        stride_buckets=None,
        action_range=None,
        pause_action_threshold: float = DEFAULT_PAUSE_ACTION_THRESHOLD,
    ):
        self.device = device
        self.n_actions = n_actions
        self.t_max = t_max
        self.action_dim = action_dim
        self.samples_per_env = samples_per_env
        self.stride_buckets = None
        if stride_buckets is not None:
            self.stride_buckets = torch.tensor(stride_buckets, device=device, dtype=torch.long)
        self.action_range = action_range
        self.pause_action_threshold = float(pause_action_threshold)
        self._transitions = []
        self._stopped_mask = None  # (B, S): True once a sample has emitted delta_t=0.
        self._stopped_mask_before_action = None
        self._last_running_mask = None

    def set_current_max_coverage(self, max_coverage):
        """No-op."""
        pass

    def reset_episode(self):
        self._transitions = []
        self._stopped_mask = None
        self._stopped_mask_before_action = None
        self._last_running_mask = None
        if hasattr(self, "_last_delta"):
            self._last_delta = None
        if hasattr(self, "_last_raw_action"):
            self._last_raw_action = None

    def _ensure_stopped_mask(self, timestep: torch.Tensor):
        N = timestep.numel()
        S = self.samples_per_env if self.samples_per_env > 0 else 1
        B = N // S
        if self._stopped_mask is None or self._stopped_mask.shape != (B, S):
            self._stopped_mask = torch.zeros((B, S), dtype=torch.bool, device=self.device)

    def _apply_no_restart_constraint(
        self, proposed_delta_t: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-way stopping: once a sample emits delta_t=0 it stays paused.

        Returns the constrained (N,) deltas and the (B, S) running mask before this step.
        """
        self._ensure_stopped_mask(timestep)

        N = timestep.numel()
        S = self.samples_per_env if self.samples_per_env > 0 else 1
        B = N // S

        t_bk = timestep.reshape(B, S)
        stopped_mask_current = self._stopped_mask
        if stopped_mask_current is None:
            raise RuntimeError("stopped mask was not initialized")
        running_mask = (t_bk > 0) & (~stopped_mask_current)
        self._last_running_mask = running_mask.clone()
        self._stopped_mask_before_action = stopped_mask_current.clone()

        out = proposed_delta_t.reshape(-1).long()
        out_bk = out.reshape(B, S)
        out_bk[~running_mask] = 0

        newly_stopped = running_mask & (out_bk == 0)
        self._stopped_mask = stopped_mask_current | newly_stopped

        out = out_bk.reshape(-1)
        self._last_delta = out.clone()
        return out, running_mask

    def pop_transitions(self):
        out = self._transitions
        self._transitions = []
        return out

    def _obs_from(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.t_max is None or self.action_dim is None:
            raise ValueError("t_max and action_dim must be set")

        N = sample.shape[0]
        S = int(self.samples_per_env if self.samples_per_env is not None else 1)
        B = N // S
        self._ensure_stopped_mask(timestep)

        if sample.ndim >= 3 and sample.shape[-1] >= self.action_dim:
            sample = sample[..., : self.action_dim]

        x_flat = sample.reshape(N, -1)
        t = timestep.reshape(N, 1).float() / max(1.0, float(self.t_max - 1))
        features = [x_flat.float(), t]

        if hasattr(self, "_last_delta") and self._last_delta is not None:
            last_delta = self._last_delta.reshape(B, S).float()
            was_moving = (last_delta != 0).float()
        else:
            was_moving = (timestep.reshape(B, S) > 0).float()
        n_active_frac = was_moving.sum(dim=1, keepdim=True) / max(S, 1)
        n_active_frac = n_active_frac.expand(B, S).reshape(N, 1)

        features.append(n_active_frac)

        stopped_mask_current = self._stopped_mask
        if stopped_mask_current is None:
            raise RuntimeError("stopped mask was not initialized")
        stopped = stopped_mask_current.reshape(N, 1).float()

        features.append(stopped)

        sample_obs = torch.cat(features, dim=-1)
        sample_obs = sample_obs.reshape(B, S, -1)

        state_obs = global_cond
        state_obs = state_obs.unsqueeze(1).expand(B, S, -1)

        return state_obs, sample_obs

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError

    def on_transition(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        action: torch.Tensor,
        next_sample: torch.Tensor,
        next_timestep: torch.Tensor,
        **kwargs,
    ):
        if self.t_max is None or self.action_dim is None:
            return

        stopped_mask_after_action = self._stopped_mask
        self._stopped_mask = self._stopped_mask_before_action
        state_obs, sample_obs = self._obs_from(sample, timestep, global_cond)
        self._stopped_mask = stopped_mask_after_action
        next_state_obs, next_sample_obs = self._obs_from(next_sample, next_timestep, global_cond)

        B, S, _ = sample_obs.shape

        action_range = self.action_range
        if action_range is None:
            raise ValueError("action_range must be set for transition normalization")
        action_val = delta_to_action(
            action.float(),
            action_range=action_range,
            pause_threshold=self.pause_action_threshold,
        )
        if not isinstance(action_val, torch.Tensor):
            action_val = torch.from_numpy(action_val).to(device=action.device)

        # Log the raw actor action, not the quantized delta, so the critic sees the pause region.
        if hasattr(self, "_last_raw_action") and self._last_raw_action is not None:
            action_val = self._last_raw_action.to(action.device)
            self._last_raw_action = None

        if action_val.ndim == 1:
            action_val = action_val.reshape(B, S, -1)
        elif action_val.ndim == 2 and action_val.shape[0] == B * S:
            action_val = action_val.reshape(B, S, -1)

        if action_val.shape[:2] != (B, S):
            action_val = action_val.reshape(B, S, -1)

        done = next_timestep == 0
        active = (timestep > 0) & (timestep != next_timestep)
        if self._last_running_mask is None:
            raise RuntimeError("running mask was not initialized")
        running = self._last_running_mask.reshape(-1)

        done = done.reshape(B, S)
        active = active.reshape(B, S)
        active = active & running.reshape(B, S)
        running = running.reshape(B, S)

        active_np = active.detach().to("cpu").numpy().astype(np.bool_)
        running_np = running.detach().to("cpu").numpy().astype(np.bool_)

        # Partially denoised actions for plotting; NaN for samples that did not move this step.
        _ns = next_sample.detach().to("cpu")
        _ns_action = _ns[..., : self.action_dim]
        _ns_flat = _ns_action.reshape(B, S, -1).numpy().astype(np.float32)
        next_sample_action_np = np.where(active_np[:, :, None], _ns_flat, np.nan)

        timestep_np = timestep.detach().to("cpu").reshape(B, S).numpy().astype(np.int32)
        next_timestep_np = next_timestep.detach().to("cpu").reshape(B, S).numpy().astype(np.int32)

        self._transitions.append(
            {
                "state_obs": state_obs.detach().to("cpu").numpy().astype(np.float32),
                "sample_obs": sample_obs.detach().to("cpu").numpy().astype(np.float32),
                "next_state_obs": next_state_obs.detach().to("cpu").numpy().astype(np.float32),
                "next_sample_obs": next_sample_obs.detach().to("cpu").numpy().astype(np.float32),
                "action": action_val.detach().to("cpu").numpy().astype(np.float32),
                "done": done.detach().to("cpu").numpy().astype(np.bool_),
                "active": active_np,  # sample advanced this step
                "running": running_np,  # sample was eligible to act
                "env_id": None,
                "next_sample_action": next_sample_action_np,
                "timestep": timestep_np,
                "next_timestep": next_timestep_np,
            }
        )


class DenoiseStrideSelector(StrideSelectorBase):
    """Learned selector: maps the actor's continuous action to delta_t and logs per-step transitions."""

    def __init__(
        self,
        policy: StrideActorCritic,
        device: torch.device,
        n_actions: int,
        t_max: int,
        action_dim: int,
        samples_per_env: int = 1,
        stride_buckets=None,
        action_range=None,
    ):
        super().__init__(
            device,
            n_actions,
            t_max,
            action_dim,
            samples_per_env,
            stride_buckets=stride_buckets,
            action_range=action_range,
        )
        self.policy = policy
        self._last_action = None
        self._last_delta = None

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        **kwargs,
    ):
        with torch.no_grad():
            state_obs, sample_obs = self._obs_from(sample, timestep, global_cond)

            action, _, _ = self.policy.sample_action(state_obs, sample_obs)

            self._last_raw_action = action  # (B, S, A) in [-1, 1]

            assert (
                self.action_range is not None
            ), "Currently only supports action_range scaling. Add support for raw actions if needed."
            proposed_delta_t = action_to_delta(
                action,
                action_range=self.action_range,
                pause_threshold=self.pause_action_threshold,
            )
            if not isinstance(proposed_delta_t, torch.Tensor):
                proposed_delta_t = torch.from_numpy(proposed_delta_t).to(device=action.device)
            proposed_delta_t = proposed_delta_t.long().reshape(-1)
            delta_t, running_mask = self._apply_no_restart_constraint(proposed_delta_t, timestep)

            # Already-stopped samples are logged with the canonical stop action.
            stop_raw = 1.0
            running_flat = running_mask.reshape(-1)
            raw_flat = self._last_raw_action.reshape(-1, self._last_raw_action.shape[-1])
            raw_flat[~running_flat] = stop_raw
            self._last_raw_action = raw_flat.reshape_as(self._last_raw_action)

            return delta_t


class FixedStrideSelector(StrideSelectorBase):
    def __init__(
        self,
        stride,
        device,
        samples_per_env=1,
        active_samples_per_env=None,
        n_actions=None,
        t_max=None,
        action_dim=None,
        stride_buckets=None,
        action_range=None,
        **kwargs,
    ):
        super().__init__(
            device,
            n_actions,
            t_max,
            action_dim,
            samples_per_env,
            stride_buckets=stride_buckets,
            action_range=action_range,
        )
        self.stride = stride
        self.active_samples_per_env = (
            active_samples_per_env if active_samples_per_env is not None else samples_per_env
        )
        self._active_indices = None

    def reset_episode(self):
        super().reset_episode()
        self._active_indices = None

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        **kwargs,
    ):
        N = sample.shape[0]
        S = self.samples_per_env

        act = self.active_samples_per_env if self.active_samples_per_env is not None else S
        if act > S:
            raise ValueError("active_samples_per_env cannot exceed samples_per_env")

        if self._active_indices is None:
            self._active_indices = torch.arange(act, device=self.device, dtype=torch.long)

        per_env = torch.zeros((S,), device=self.device, dtype=torch.long)
        per_env[self._active_indices] = self.stride
        proposed = per_env.repeat(N // S)
        out, _ = self._apply_no_restart_constraint(proposed, timestep)
        return out


class PerChunkRandomStrideSelector(StrideSelectorBase):
    """Picks one random stride per action chunk (at reset_episode), shared by all samples."""

    def __init__(
        self,
        stride_options,
        device,
        samples_per_env=1,
        n_actions=None,
        t_max=None,
        action_dim=None,
        stride_buckets=None,
        action_range=None,
        weights=None,
        **kwargs,
    ):
        super().__init__(
            device,
            n_actions,
            t_max,
            action_dim,
            samples_per_env,
            stride_buckets=stride_buckets,
            action_range=action_range,
        )
        self.stride_options = list(stride_options)
        if weights is not None:
            w = np.array(weights, dtype=np.float64)
            self.weights = w / w.sum()
        else:
            self.weights = None
        self._current_stride = None

    def reset_episode(self):
        super().reset_episode()
        self._current_stride = int(np.random.choice(self.stride_options, p=self.weights))

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        **kwargs,
    ):
        N = sample.shape[0]

        if self._current_stride is None:
            self._current_stride = int(np.random.choice(self.stride_options))

        proposed = torch.full((N,), self._current_stride, device=self.device, dtype=torch.long)
        out, _ = self._apply_no_restart_constraint(proposed, timestep)
        return out


class RandomStrideSelector(StrideSelectorBase):
    """Random stride selector for collecting diverse offline data."""

    def __init__(
        self,
        device,
        n_actions=None,
        t_max=None,
        action_dim=None,
        samples_per_env=1,
        stride_buckets=None,
        action_range=None,
        neg_prob: float = 0.1,
        zero_prob: float = 0.0,
    ):
        super().__init__(
            device,
            n_actions,
            t_max,
            action_dim,
            samples_per_env,
            stride_buckets=stride_buckets,
            action_range=action_range,
        )

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        **kwargs,
    ):
        N = sample.shape[0]

        t_flat = timestep.reshape(-1)

        candidate_mask = t_flat > 0

        if self.stride_buckets is not None:
            buckets = self.stride_buckets.cpu().numpy().tolist()
        else:
            max_t = int(self.t_max) if self.t_max is not None else 1
            buckets = list(range(1, max(2, max_t + 1)))

        choices = np.random.choice(buckets, size=(N,))
        choices = choices.astype(np.int64)

        out = torch.from_numpy(choices).to(device=self.device, dtype=torch.long)

        finished_mask = t_flat == 0
        out[finished_mask] = 0

        if candidate_mask.any():
            cand_idx = torch.nonzero(candidate_mask, as_tuple=False).reshape(-1)
            cand_out = out[cand_idx]
            if torch.all(cand_out < 0):
                out[cand_idx[0]] = 1

        out, _ = self._apply_no_restart_constraint(out, timestep)
        return out


class BudgetRandomStrideSelector(StrideSelectorBase):
    """Random baseline matching a target's expected compute: L steps x P active samples.

    Each episode rounds L and P stochastically to floor/ceil and picks a random active subset.
    ``record_actual_usage`` feeds back the realized usage so the running mean converges to target.
    """

    def __init__(
        self,
        device,
        target_inference_steps: float,
        target_sample_usage: float,
        t_max: int,
        samples_per_env: int,
        action_dim: int,
        n_actions=None,
        stride_buckets=None,
        action_range=None,
        deficit_gain: float = 0.7,
    ):
        super().__init__(
            device,
            n_actions,
            t_max,
            action_dim,
            samples_per_env,
            stride_buckets=stride_buckets,
            action_range=action_range,
        )
        self._target_L = float(target_inference_steps)
        self._target_P = float(target_sample_usage)
        self._episode_stride = None
        self._episode_active_indices = None
        # Positive deficit = under-spending so far.
        self._deficit_L = 0.0
        self._deficit_P = 0.0
        self._deficit_gain = float(deficit_gain)
        print(
            f"[BudgetRandomStrideSelector] target L={target_inference_steps:.2f}, "
            f"P={target_sample_usage:.2f} — stochastic floor/ceil + cross-episode deficit correction "
            f"(gain={deficit_gain:.2f})"
        )

    def record_actual_usage(self, actual_L: float, actual_P: float):
        """Accumulate (target - actual) so later episodes compensate for over- or under-spend."""
        if np.isfinite(actual_L):
            self._deficit_L += self._target_L - actual_L
        if np.isfinite(actual_P):
            self._deficit_P += self._target_P - actual_P

    def reset_episode(self):
        super().reset_episode()
        S = self.samples_per_env
        if self.t_max is None:
            raise ValueError("t_max must be set")
        t_max = int(self.t_max)

        effective_L = self._target_L + self._deficit_gain * self._deficit_L
        effective_P = self._target_P + self._deficit_gain * self._deficit_P

        effective_L = float(np.clip(effective_L, 1.0, float(t_max)))
        effective_P = float(np.clip(effective_P, 1.0, float(S)))

        L_lo = max(1, int(math.floor(effective_L)))
        L_hi = max(1, int(math.ceil(effective_L)))
        frac_L = effective_L - math.floor(effective_L)
        n_steps = L_hi if (np.random.random() < frac_L) else L_lo
        n_steps = max(1, min(n_steps, t_max))
        # Guarantees exactly n_steps UNet calls.
        self._episode_stride = math.ceil((t_max - 1) / n_steps)

        P_lo = max(1, int(math.floor(effective_P)))
        P_hi = max(1, int(math.ceil(effective_P)))
        P_lo = min(P_lo, S)
        P_hi = min(P_hi, S)
        frac_P = effective_P - math.floor(effective_P)
        active = P_hi if (np.random.random() < frac_P) else P_lo

        perm = torch.randperm(S, device=self.device)
        self._episode_active_indices = perm[:active]

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        **kwargs,
    ):
        N = sample.shape[0]
        S = self.samples_per_env

        if self._episode_stride is None or self._episode_active_indices is None:
            self.reset_episode()
        if self._episode_stride is None or self._episode_active_indices is None:
            raise RuntimeError("episode stride state was not initialized")

        per_env = torch.zeros((S,), device=self.device, dtype=torch.long)
        per_env[self._episode_active_indices] = self._episode_stride
        proposed = per_env.repeat(N // S)
        out, _ = self._apply_no_restart_constraint(proposed, timestep)
        return out
