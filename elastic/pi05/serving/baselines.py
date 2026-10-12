"""pi0.5 baselines served next to ELASTIC: plain flow matching and best-of-N with V-GPS."""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from verifiers.vgps import extract_vgps_image, extract_vgps_prompt, load_vgps_agent_and_text_processor


class NormalPolicy(_policy.Policy):
    """Upstream single-sample policy; adds the timing and stride_info keys the LIBERO client logs."""

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        outputs = super().infer(obs, noise=noise)
        infer_ms = float(outputs["policy_timing"]["infer_ms"])
        outputs["policy_timing"].update(flow_ms=infer_ms, verifier_ms=0.0, total_ms=infer_ms)
        outputs["stride_info"] = {
            "selector": "normal",
            "adaptive_steps": int(self._sample_kwargs.get("num_steps", 10)),
            "avg_active_samples": 1.0,
        }
        return outputs


class ParallelPolicy(_base_policy.BasePolicy):
    """Best-of-N: S chunks sampled in parallel, V-GPS picks one (softmax over Q at ``action_temp``, argmax if <= 0)."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not hasattr(model, "sample_actions_parallel"):
            raise ValueError(f"{type(model).__name__} does not support parallel action sampling.")

        sample_kwargs = dict(sample_kwargs or {})
        self._vgps_checkpoint = sample_kwargs.pop("vgps_checkpoint", None)
        self._vgps_config_path = sample_kwargs.pop("vgps_config_path", None)
        self._action_temp = float(sample_kwargs.pop("action_temp", 1.0))
        self._include_vgps_candidate_chunks = bool(sample_kwargs.pop("include_vgps_candidate_chunks", False))
        # V-GPS encoder input side; must match verifier training (256 for LIBERO).
        self._vgps_image_size = int(sample_kwargs.pop("vgps_image_size", 256))

        self._sample_actions = nnx_utils.module_jit(model.sample_actions_parallel, static_argnames=("num_samples",))
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = jax.random.key(0) if rng is None else rng
        self._sample_kwargs = sample_kwargs
        self._metadata = metadata or {}
        self._init_vgps()

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Copy: transforms may modify inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        sample_kwargs = dict(self._sample_kwargs)
        if "num_samples" in sample_kwargs:
            sample_kwargs["num_samples"] = int(sample_kwargs["num_samples"])

        self._rng, sample_rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"][0, ...],
            "actions": self._sample_actions(sample_rng, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        outputs = jax.tree.map(np.asarray, outputs)
        outputs = self._output_transform(outputs)

        verifier_start = time.monotonic()
        action_index, q_value, q_values = self._select_vgps_action(obs, outputs["actions"])
        verifier_time = time.monotonic() - verifier_start
        if self._include_vgps_candidate_chunks:
            outputs["vgps_candidate_chunks"] = np.copy(np.asarray(outputs["actions"]))
        outputs["actions"] = outputs["actions"][action_index]
        outputs["q_values"] = q_values
        outputs["q_value"] = float(q_value)
        outputs["action_index"] = int(action_index)

        infer_ms = model_time * 1000
        verifier_ms = verifier_time * 1000
        num_steps = int(sample_kwargs.get("num_steps", self._sample_kwargs.get("num_steps", 10)))
        num_samples = int(sample_kwargs.get("num_samples", self._sample_kwargs.get("num_samples", 1)))
        outputs["policy_timing"] = {
            "infer_ms": infer_ms,
            "flow_ms": infer_ms,
            "verifier_ms": verifier_ms,
            "total_ms": infer_ms + verifier_ms,
        }
        outputs["stride_info"] = {
            "selector": "parallel",
            "adaptive_steps": num_steps,
            "avg_active_samples": float(num_samples),
        }
        return outputs

    def _init_vgps(self) -> None:
        self._vgps_agent, self._vgps_text_processor = load_vgps_agent_and_text_processor(
            self._vgps_checkpoint, self._vgps_config_path
        )

    def _select_vgps_action(self, obs: dict, actions: np.ndarray) -> tuple[int, float, np.ndarray]:
        image = extract_vgps_image(obs, vgps_image_size=self._vgps_image_size)
        prompt = extract_vgps_prompt(obs)
        values = self._vgps_agent.get_q_values(
            observations={"image": np.repeat(image[None, ...], actions.shape[0], axis=0)},
            goals={
                "language": np.repeat(
                    self._vgps_text_processor.encode([prompt])[0:1],
                    actions.shape[0],
                    axis=0,
                )
            },
            actions=actions[:, 0, :7],
        )
        # jaxrl_m may return () or (1, 1) for a batch of 1.
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape != (actions.shape[0],):
            raise ValueError(f"Expected V-GPS values shape {(actions.shape[0],)}, got {values.shape}")

        logging.debug(
            "V-GPS values: max=%.2f min=%.2f mean=%.2f",
            values.max(),
            values.min(),
            values.mean(),
        )
        if self._action_temp > 0:
            self._rng, action_rng = jax.random.split(self._rng)
            action_index = int(jax.random.categorical(action_rng, jnp.asarray(values) / self._action_temp))
        else:
            action_index = int(np.argmax(values))
        return action_index, float(values[action_index]), values

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
