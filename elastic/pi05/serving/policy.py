"""ELASTIC on pi0.5: S candidate chunks denoised with adaptive strides, one picked by V-GPS.

When training online, each infer's selector transitions become replay rows and SAC updates.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi_client import base_policy as _base_policy

from elastic.pi05.config import StrideConfig
from elastic.pi05.training.online import OnlineTrainer, TransitionRecorder, WandbLogger
from elastic.pi05.training.replay import pack_stride_infer_meta_row
from elastic.pi05.sampling.sampler import MAX_DENOISE_STEPS, StrideSampler
from elastic.pi05.sampling.selectors import FixedStrideSelector
from verifiers.vgps import VGPSActionVerifier


class StrideAdaptivePolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        cfg: StrideConfig,
        *,
        num_samples: int,
        num_steps: int,
        vgps_checkpoint: str,
        vgps_config_path: str | None = None,
        action_temp: float = 1.0,
        vgps_image_size: int = 256,
        include_vgps_candidate_chunks: bool = False,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        metadata: dict[str, Any] | None = None,
        rng: at.KeyArrayLike | None = None,
    ):
        self._cfg = cfg
        self._model = model
        self._num_samples = int(num_samples)
        self._baseline_num_steps = int(num_steps)
        self._include_vgps_candidate_chunks = bool(include_vgps_candidate_chunks)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._metadata = metadata or {}
        self._rng = jax.random.key(0) if rng is None else rng

        self._wandb = WandbLogger(cfg, {"num_samples": self._num_samples, "baseline_num_steps": self._baseline_num_steps,
                                        **{k: v for k, v in vars(cfg).items() if k.startswith("stride_")}})
        self._recorder = TransitionRecorder(cfg.stride_replay_dir)
        self._trainer = (
            OnlineTrainer(cfg, num_samples=self._num_samples, wandb_logger=self._wandb) if cfg.online_train else None
        )
        self._sampler = StrideSampler(
            model, cfg, num_samples=self._num_samples, num_steps=self._baseline_num_steps, trainer=self._trainer
        )
        logging.info("Initializing V-GPS verifier...")
        self._verifier = VGPSActionVerifier(
            checkpoint=vgps_checkpoint,
            config_path=vgps_config_path,
            action_temp=action_temp,
            rng=self._rng,
            vgps_image_size=vgps_image_size,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        meta = dict(self._metadata)
        # Lets the LIBERO launchers confirm they reached the server for their run.
        if self._cfg.stride_wandb_dir:
            meta["stride_output_dir_name"] = Path(self._cfg.stride_wandb_dir).name
        elif self._cfg.stride_checkpoint_dir is not None:
            meta["stride_output_dir_name"] = Path(self._cfg.stride_checkpoint_dir).parent.name
        return meta

    def flush_stride_training_state(self) -> None:
        """Called by the server on client disconnect and exit."""
        if self._trainer is not None:
            self._trainer.flush()
        self._wandb.commit()

    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        state = np.asarray(inputs["state"][0, ...])
        observation = _model.Observation.from_dict(inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        selector = None if self._sampler.uses_jax_selector() else self._sampler.selector()
        start = time.monotonic()
        sampled = self._sampler.sample(sample_rng, observation, selector)
        infer_ms = (time.monotonic() - start) * 1000
        info = sampled["info"]

        candidates = np.asarray(sampled["actions"])
        outputs = self._output_transform({"state": state, "actions": candidates})
        verifier_start = time.monotonic()
        final_t = np.asarray(info.get("final_timestep", np.array([])), dtype=np.float32).reshape(-1)
        finished = final_t <= 1e-6 if final_t.shape == (outputs["actions"].shape[0],) else None
        action_index, q_value, q_values = self._verifier.select(obs, outputs["actions"], eligible_mask=finished)
        verifier_ms = (time.monotonic() - verifier_start) * 1000
        selected_finished = finished is not None and bool(finished[int(action_index)])
        if finished is not None and not selected_finished:
            logging.warning(
                "Selected chunk did not finish denoising (index %d, t=%.5g, finished fraction %.2f).",
                int(action_index), float(final_t[int(action_index)]), float(np.mean(finished)),
            )

        active_counts = info.get("active_counts", [])
        avg_active = float(np.mean(active_counts)) if active_counts else 0.0
        finished_fraction = float(np.mean(finished)) if finished is not None else 0.0
        transitions = selector.pop_transitions() if selector is not None else []
        train_info: dict[str, float] = {}
        if transitions:
            meta = {
                "adaptive_steps": float(int(info.get("adaptive_steps", 0))),
                "avg_active_samples": avg_active,
                "finished_fraction": finished_fraction,
                "num_transitions": float(len(transitions)),
                "baseline_num_steps": float(int(info.get("default_num_steps", self._baseline_num_steps))),
                "selected_finished": float(selected_finished),
            }
            infer_idx = self._recorder.record(transitions, q_values, action_index, meta)
            if self._trainer is not None:
                train_info = self._trainer.step(
                    transitions,
                    q_values,
                    infer_idx=infer_idx,
                    meta_row=pack_stride_infer_meta_row(**meta),
                    redenoise=lambda stride: self._redenoise(stride, observation, obs, state),
                )

        if self._include_vgps_candidate_chunks:
            outputs["vgps_candidate_chunks"] = np.copy(candidates)
        outputs["actions"] = outputs["actions"][action_index]
        outputs["q_values"] = q_values
        outputs["q_value"] = float(q_value)
        outputs["action_index"] = int(action_index)
        outputs["stride_info"] = {
            "selector": self._sampler.selector_kind,
            "selector_backend": info.get("selector_backend", "torch" if self._sampler.selector_kind == "learned" else "python"),
            "adaptive_steps": int(info.get("adaptive_steps", 0)),
            "baseline_num_steps": int(info.get("default_num_steps", self._baseline_num_steps)),
            "max_model_evals": int(info.get("max_model_evals", MAX_DENOISE_STEPS)),
            "fixed_stride": self._sampler.fixed_stride if self._sampler.selector_kind == "fixed" else None,
            "avg_active_samples": avg_active,
            "active_counts": active_counts,
            "num_transitions": len(transitions),
            "online_train": self._trainer is not None,
            "learned_selector_ready": self._sampler.learned_selector_ready(),
            "online_replay_size": self._trainer.replay_size if self._trainer is not None else 0,
            "train_metrics": train_info,
            "final_timestep": final_t.tolist(),
            "finished_fraction": finished_fraction,
            "selected_finished": selected_finished,
            "all_candidates_unfinished": finished is not None and not bool(np.any(finished)),
        }
        outputs["policy_timing"] = {
            "infer_ms": infer_ms,
            "flow_ms": infer_ms,
            "verifier_ms": verifier_ms,
            "total_ms": infer_ms + verifier_ms,
        }
        self._wandb.log_infer(
            q_values=q_values,
            selected_q=q_value,
            action_index=action_index,
            stride_info=outputs["stride_info"],
            infer_ms=infer_ms,
            total_ms=infer_ms + verifier_ms,
        )
        return outputs

    def _redenoise(self, stride: float, observation: _model.Observation, obs: dict, state: np.ndarray):
        """Second pass: re-denoise all S candidates at a fixed ``stride`` (no env step) and score them."""
        selector = FixedStrideSelector(
            stride=stride,
            samples_per_env=self._num_samples,
            max_steps=MAX_DENOISE_STEPS,
            action_dim=int(self._model.action_dim),
            action_range=(0.0, 1.0),
            use_extended_obs=self._cfg.stride_use_extended_obs,
        )
        self._rng, rng = jax.random.split(self._rng)
        sampled = self._sampler.sample(rng, observation, selector)
        transitions = selector.pop_transitions()
        if not transitions:
            return None
        actions = self._output_transform({"actions": np.asarray(sampled["actions"]), "state": state})["actions"]
        return self._verifier.score_actions(obs, np.asarray(actions)), transitions
