"""ELASTIC settings for the pi0.5 policy server (``--stride-*`` flags of ``elastic.pi05.serving.serve``)."""
from __future__ import annotations

import dataclasses
from typing import Literal

from elastic.pi05 import constants as C


@dataclasses.dataclass
class StrideConfig:
    stride_selector: Literal["fixed", "per_chunk_random", "budget_random", "learned"] = "fixed"
    stride_checkpoint: str | None = None
    stride_device: str = "cpu"
    # False matches the Diffusion Policy obs layout; True adds a mid-layer pool, score norm and
    # x0 deviation. Checkpoints must match.
    stride_use_extended_obs: bool = C.STRIDE_USE_EXTENDED_OBS
    # Compact active samples before denoising; buckets 1/2/4/8/S keep JAX shapes static.
    stride_sparse_active_denoising: bool = True

    # Flow-time stride for the fixed selector; 0 means 1 / --num-steps.
    stride_fixed_stride: float = 0.0
    stride_target_steps: float = 10.0
    stride_target_samples: float = 10.0
    stride_random_step_options: str = C.RANDOM_STEP_OPTIONS
    stride_random_step_weights: str = ""
    # Output dir for per-infer transition records (stride_step_*.npz).
    stride_replay_dir: str | None = None

    stride_checkpoint_dir: str | None = None
    stride_offline_replay: str | None = None
    stride_offline_pretrain_steps: int = C.OFFLINE_PRETRAIN_STEPS
    stride_online_updates_per_step: int = C.ONLINE_UPDATES_PER_STEP
    stride_min_online_transitions: int = C.ONLINE_MIN_TRANSITIONS
    stride_offline_batch_frac: float = C.OFFLINE_BATCH_FRAC
    stride_batch_size: int = C.BATCH_SIZE
    stride_replay_capacity: int = C.STRIDE_REPLAY_CAPACITY
    stride_checkpoint_every: int = C.STRIDE_CHECKPOINT_EVERY

    stride_latency_penalty: float = C.LATENCY_PENALTY
    stride_parallel_penalty: float = C.PARALLEL_PENALTY
    stride_terminal_bonus: float = C.TERMINAL_BONUS
    stride_no_finish_penalty: float = C.NO_FINISH_PENALTY
    stride_verifier_bias: float = C.VERIFIER_BIAS

    stride_k_subsample_counterfactuals: bool = C.K_SUBSAMPLE_ADD_COUNTERFACTUALS
    stride_k_subsample_draws: int = C.K_SUBSAMPLE_DRAWS
    stride_second_pass_counterfactuals: bool = C.SECOND_PASS_COUNTERFACTUALS
    stride_second_pass_every: int = C.SECOND_PASS_EVERY
    stride_all_stopped_counterfactuals: bool = C.ALL_STOPPED_COUNTERFACTUALS

    # Empty project disables W&B.
    stride_wandb_project: str | None = C.DEFAULT_STRIDE_WANDB_PROJECT
    stride_wandb_name: str | None = None
    stride_wandb_dir: str | None = None

    @property
    def online_train(self) -> bool:
        return self.stride_checkpoint_dir is not None or self.stride_offline_replay is not None

    def step_options(self) -> list[float]:
        return _float_list(self.stride_random_step_options)

    def step_weights(self) -> list[float]:
        weights = _float_list(self.stride_random_step_weights)
        if weights and len(weights) != len(self.step_options()):
            raise ValueError("stride_random_step_weights must match stride_random_step_options")
        return weights


def _float_list(raw) -> list[float]:
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).replace(" ", "").split(",") if x]
