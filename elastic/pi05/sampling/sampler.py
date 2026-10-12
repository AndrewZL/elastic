"""Adaptive stride denoising for pi0.5: selector construction and the sampling paths.

Learned eval runs a fused JAX loop, fixed schedules one jitted trace, anything else a
host-side loop that calls the selector between denoise steps.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.shared import nnx_utils

from elastic.pi05.config import StrideConfig
from elastic.pi05.constants import DIFFUSION_T_MAX
from elastic.pi05.sampling.jax_actor import JaxStrideActorSpec, load_jax_stride_actor
from elastic.pi05.training.learner import load_stride_policy
from elastic.pi05.sampling.selectors import (
    BudgetRandomStrideSelector,
    FixedStrideSelector,
    LearnedStrideSelector,
    PerChunkRandomStrideSelector,
)

MAX_DENOISE_STEPS = 30


class StrideSampler:
    """Samples S candidate chunks with adaptive strides. ``trainer`` (an OnlineTrainer) supplies
    the live learner when the stride policy trains online."""

    def __init__(self, model, cfg: StrideConfig, *, num_samples: int, num_steps: int, trainer=None):
        self._model = model
        self.trainer = trainer
        self._num_samples = int(num_samples)
        self._baseline_num_steps = int(num_steps)
        self._max_denoise_steps = MAX_DENOISE_STEPS
        self._selector_kind = cfg.stride_selector
        self._fixed_stride = float(cfg.stride_fixed_stride)
        self._target_steps = float(cfg.stride_target_steps)
        self._target_samples = float(cfg.stride_target_samples)
        self._random_step_options = cfg.step_options()
        self._random_step_weights = cfg.step_weights()
        self._diffusion_t_max = DIFFUSION_T_MAX
        self._stride_checkpoint = cfg.stride_checkpoint
        self._stride_device = cfg.stride_device
        self._use_extended_obs = bool(cfg.stride_use_extended_obs)
        self._sparse_active_denoising = bool(cfg.stride_sparse_active_denoising)
        self._online_train = cfg.online_train
        self._sample_fixed_stride_trace = nnx_utils.module_jit(
            model.sample_actions_fixed_stride_trace, static_argnames=("num_samples", "num_steps")
        )
        self._stride_denoise_step = nnx_utils.module_jit(model.stride_denoise_step)
        self._sample_learned_stride_jax = nnx_utils.module_jit(
            model.sample_actions_learned_stride_jax,
            static_argnames=("actor_spec", "num_samples", "max_denoise_steps", "use_extended_obs", "use_sparse_active_denoising"),
        )
        self._jax_stride_actor_spec: JaxStrideActorSpec | None = None
        self._jax_stride_actor_params = None
        self._checkpoint_stride_policy = None
        self._selector_cache: Any = None
        self._selector_cache_key: tuple[Any, ...] | None = None

    @property
    def selector_kind(self) -> str:
        return self._selector_kind

    @property
    def fixed_stride(self) -> float:
        return self._fixed_stride

    def sample(self, sample_rng, observation: _model.Observation, selector) -> dict[str, Any]:
        if (
            self._selector_kind == "learned"
            and self._stride_checkpoint is not None
            and not self._online_train
        ):
            if self.uses_jax_selector():
                return self._sample_actions_learned_jax(sample_rng, observation)

        fixed_stride = self._fixed_schedule_stride(selector)
        if fixed_stride is not None and self._sample_fixed_stride_trace is not None:
            num_steps = max(1, min(self._max_denoise_steps, int(np.ceil(1.0 / max(float(fixed_stride), 1e-6)))))
            sampled = self._sample_fixed_stride_trace(
                sample_rng,
                observation,
                fixed_stride=float(fixed_stride),
                num_steps=num_steps,
                num_samples=self._num_samples,
            )
            self._record_fixed_trace(selector, sampled)
            return sampled

        return self._sample_actions_stepwise_fast(sample_rng, observation, selector)

    def uses_jax_selector(self) -> bool:
        return (
            self._selector_kind == "learned"
            and self._stride_checkpoint is not None
            and not self._online_train
            and self._sample_learned_stride_jax is not None
        )

    def _ensure_jax_learned_selector(self) -> tuple[JaxStrideActorSpec, Any]:
        if self._jax_stride_actor_spec is None or self._jax_stride_actor_params is None:
            self._jax_stride_actor_spec, self._jax_stride_actor_params = load_jax_stride_actor(
                self._stride_checkpoint,
                device=self._stride_device,
            )
        return self._jax_stride_actor_spec, self._jax_stride_actor_params

    def _sample_actions_learned_jax(self, sample_rng, observation: _model.Observation) -> dict[str, Any]:
        spec, params = self._ensure_jax_learned_selector()
        if spec.num_samples != self._num_samples:
            raise ValueError(f"Learned STRIDE checkpoint expects num_samples={spec.num_samples}, got {self._num_samples}")
        if not self._use_extended_obs:
            ah = int(self._model.action_horizon)
            ad = int(self._model.action_dim)
            expected_sd = ah * ad + 3
            if int(spec.sample_dim) != expected_sd:
                raise ValueError(
                    "stride_use_extended_obs=False requires a STRIDE actor trained with minimal observations: "
                    f"expected sample_dim={expected_sd} (action_horizon*action_dim + t/active/stopped), "
                    f"got sample_dim={spec.sample_dim}. Re-export the JAX actor from a minimal-obs Torch policy, "
                    "or keep stride_use_extended_obs=True for extended-feature checkpoints."
                )

        sampled = self._sample_learned_stride_jax(
            sample_rng,
            observation,
            actor_params=params,
            actor_spec=spec,
            num_samples=self._num_samples,
            max_denoise_steps=self._max_denoise_steps,
            use_extended_obs=self._use_extended_obs,
            use_sparse_active_denoising=self._sparse_active_denoising,
        )
        info = sampled["info"]
        steps = info["adaptive_steps"]
        steps_i = int(np.asarray(jax.device_get(steps)))
        active_counts = info["active_counts"]
        return {
            "actions": sampled["actions"],
            "info": {
                "adaptive_steps": steps_i,
                "default_num_steps": self._baseline_num_steps,
                "max_model_evals": self._max_denoise_steps,
                "active_counts": [int(x) for x in np.asarray(jax.device_get(active_counts[:steps_i])).reshape(-1)],
                "final_timestep": np.asarray(jax.device_get(info["final_timestep"]), dtype=np.float32),
                "selector_backend": "jax",
            },
        }

    def _sample_actions_stepwise_fast(self, sample_rng, observation: _model.Observation, selector) -> dict[str, Any]:
        if hasattr(selector, "reset_episode"):
            selector.reset_episode()

        context, x_t, t_idx = self._model.init_adaptive_denoise_state(
            sample_rng,
            observation,
            num_samples=self._num_samples,
        )
        prefix_context_n = context["prefix_context_n"]
        active_counts: list[int] = []
        transitions = 0
        eps = 1e-6

        while transitions < self._max_denoise_steps:
            candidate = np.asarray(jax.device_get(t_idx > eps), dtype=np.bool_)
            if not bool(np.any(candidate)):
                break

            out = selector(sample=x_t, timestep=t_idx, global_cond=prefix_context_n)
            delta = jnp.asarray(out, dtype=jnp.float32)
            if delta.ndim == 0:
                delta = jnp.broadcast_to(delta, t_idx.shape)
            delta = delta.reshape(t_idx.shape)
            delta = jnp.where(t_idx > eps, jnp.clip(delta, 0.0, t_idx), 0.0)
            active = np.asarray(jax.device_get(delta > 0), dtype=np.bool_)

            if not bool(np.any(active)):
                if hasattr(selector, "on_transition"):
                    selector.on_transition(
                        sample=x_t,
                        timestep=t_idx,
                        global_cond=prefix_context_n,
                        action=delta,
                        next_sample=x_t,
                        next_timestep=t_idx,
                        score_magnitude=None,
                        x0_pred=x_t,
                    )
                break

            x_next, next_t_idx, trace = self._stride_denoise_step(context, x_t, t_idx, delta)
            if hasattr(selector, "on_transition"):
                selector.on_transition(
                    sample=trace["sample"],
                    timestep=trace["timestep"],
                    global_cond=prefix_context_n,
                    action=trace["action"],
                    next_sample=trace["next_sample"],
                    next_timestep=trace["next_timestep"],
                    score_magnitude=trace["score_magnitude"],
                    x0_pred=trace["x0_pred"],
                    mid_feature_pooled=trace["mid_feature_pooled"],
                )

            active_counts.append(int(np.asarray(jax.device_get(trace["active_count"]))))
            x_t = x_next
            t_idx = next_t_idx
            transitions += 1

        info: dict[str, Any] = {
            "adaptive_steps": transitions,
            "default_num_steps": self._baseline_num_steps,
            "max_model_evals": self._max_denoise_steps,
            "active_counts": active_counts,
            "final_timestep": np.asarray(jax.device_get(t_idx), dtype=np.float32),
        }
        return {
            "actions": x_t,
            "info": info,
        }

    def _fixed_schedule_stride(self, selector) -> float | None:
        if isinstance(selector, FixedStrideSelector):
            stride = self._fixed_stride if self._fixed_stride > 0.0 else 1.0 / max(float(self._baseline_num_steps), 1.0)
            return float(stride)
        if isinstance(selector, PerChunkRandomStrideSelector):
            selector.reset_episode()
            return float(selector._episode_stride) if selector._episode_stride is not None else None
        return None

    def _record_fixed_trace(self, selector, sampled: dict[str, Any]) -> None:
        info = sampled["info"]
        trace = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), info["trace"])
        global_cond = np.asarray(jax.device_get(info["prefix_context_n"]), dtype=np.float32)
        for idx in range(int(info["adaptive_steps"])):
            selector(
                sample=trace["sample"][idx],
                timestep=trace["timestep"][idx],
                global_cond=global_cond,
            )
            selector.on_transition(
                sample=trace["sample"][idx],
                timestep=trace["timestep"][idx],
                global_cond=global_cond,
                action=trace["action"][idx],
                next_sample=trace["next_sample"][idx],
                next_timestep=trace["next_timestep"][idx],
                score_magnitude=trace["score_magnitude"][idx],
                x0_pred=trace["x0_pred"][idx],
                mid_feature_pooled=trace["mid_feature_pooled"][idx],
            )
        sampled["info"] = {
            **info,
            "active_counts": [int(x) for x in np.asarray(jax.device_get(info["active_counts"])).reshape(-1)],
            "final_timestep": np.asarray(jax.device_get(info["final_timestep"]), dtype=np.float32),
            "prefix_context": np.asarray(jax.device_get(info["prefix_context"]), dtype=np.float32),
        }

    def selector(self) -> Any:
        """Return the cached selector, rebuilt only when learned readiness or the policy object changes.

        Rebuilding per infer is slow (``policy.to(device)``); reuse is safe because each
        denoising pass calls ``reset_episode()``.
        """
        ready = self.learned_selector_ready()
        key = (
            ready,
            self._selector_kind,
            self._stride_checkpoint,
            id(self.trainer.learner) if self.trainer is not None and self.trainer.learner is not None else None,
            id(self._checkpoint_stride_policy) if self._checkpoint_stride_policy is not None else None,
        )
        if self._selector_cache is None or self._selector_cache_key != key:
            self._selector_cache = self._build_selector()
            self._selector_cache_key = key
        return self._selector_cache

    def _build_selector(self):
        common = {
            "samples_per_env": self._num_samples,
            "max_steps": self._max_denoise_steps,
            "action_dim": int(self._model.action_dim),
            "action_range": (0.0, 1.0),
            "use_extended_obs": self._use_extended_obs,
        }
        if self.learned_selector_ready() and self._selector_kind in ("budget_random", "learned"):
            return LearnedStrideSelector(policy=self.trainer.learner.policy, device=self._stride_device, **common)
        if self._selector_kind == "fixed":
            stride = self._fixed_stride if self._fixed_stride > 0.0 else 1.0 / max(float(self._baseline_num_steps), 1.0)
            return FixedStrideSelector(stride=stride, **common)
        if self._selector_kind == "per_chunk_random":
            return PerChunkRandomStrideSelector(
                step_options=self._random_step_options,
                weights=self._random_step_weights or None,
                diffusion_t_max=self._diffusion_t_max,
                **common,
            )
        if self._selector_kind == "budget_random" or (self._selector_kind == "learned" and self._stride_checkpoint is None):
            return BudgetRandomStrideSelector(
                target_steps=self._target_steps,
                target_samples=self._target_samples,
                **common,
            )
        if self._selector_kind == "learned":
            if self._stride_checkpoint is None:
                raise ValueError("--stride-checkpoint is required when --stride-selector learned")
            if self._checkpoint_stride_policy is None:
                self._checkpoint_stride_policy = load_stride_policy(self._stride_checkpoint, device=self._stride_device)
            if not self._use_extended_obs:
                ah = int(self._model.action_horizon)
                ad = int(self._model.action_dim)
                expected_sd = ah * ad + 3
                sd = int(self._checkpoint_stride_policy.sample_dim)
                if sd != expected_sd:
                    raise ValueError(
                        "stride_use_extended_obs=False requires a STRIDE actor trained with minimal observations: "
                        f"expected sample_dim={expected_sd} (action_horizon*action_dim + t/active/stopped), "
                        f"got sample_dim={sd}. Train or export a minimal-obs policy, or keep stride_use_extended_obs=True."
                    )
            return LearnedStrideSelector(policy=self._checkpoint_stride_policy, device=self._stride_device, **common)
        raise ValueError(f"Unknown stride selector {self._selector_kind!r}")

    def learned_selector_ready(self) -> bool:
        return self.trainer is not None and self.trainer.learned_selector_ready()

