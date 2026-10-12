"""Online ELASTIC training inside the pi0.5 policy server.

``OnlineTrainer`` turns each infer's transitions into replay rows and SAC updates;
``TransitionRecorder`` saves raw per-infer records for building an offline replay.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import tqdm
import wandb

from elastic.pi05 import constants as C
from elastic.pi05.config import StrideConfig
from elastic.pi05.training.learner import SACLearnerConfig, StrideSACLearner
from elastic.pi05.training.replay import NumpyReplayBuffer, load_replay_npz
from elastic.pi05.training.reward import TransitionAccumulator
from elastic.pi05.sampling.sampler import MAX_DENOISE_STEPS
from elastic.pi05.sampling.selectors import action_to_stride
from elastic.pi05.training.counterfactuals import (
    add_all_stopped_counterfactuals,
    add_k_subsample_counterfactuals,
    add_second_pass_counterfactuals,
    finite_q_mean,
    q_scores_to_advantage,
    q_scores_to_advantage_with_baseline,
)

# Robot-episode id stored on replay rows; LIBERO runs leave it unset.
EPISODE_IDX = -1

Transitions = list[dict[str, np.ndarray]]
# stride -> (V-GPS scores of the re-denoised candidates, their transitions), or None.
Redenoise = Callable[[float], "tuple[np.ndarray, Transitions] | None"]


class WandbLogger:
    def __init__(self, cfg: StrideConfig, config: dict[str, Any]):
        self.step = 0
        self.run = None
        project = (cfg.stride_wandb_project or "").strip()
        if not project:
            logging.info("Weights & Biases disabled (empty stride_wandb_project).")
            return
        self.run = wandb.init(
            project=project,
            name=cfg.stride_wandb_name or None,
            dir=cfg.stride_wandb_dir or None,
            config=config,
            reinit=True,
            settings=wandb.Settings(_disable_stats=True),
        )

    def log(self, row: dict[str, Any], *, step: int | None = None, commit: bool | None = None) -> None:
        if self.run is not None:
            self.run.log(row, step=self.step if step is None else step, commit=commit)

    def log_pretrain(self, *, step: int, total: int, metrics: dict[str, float], wandb_step: int) -> None:
        """Sparse W&B points during offline critic pretrain."""
        every = max(50, total // 25)
        if step > 0 and (step + 1) % every != 0 and step + 1 != total:
            return
        row = {"offline_pretrain/step": step + 1, "offline_pretrain/total_steps": total}
        row.update({f"offline_pretrain/{k}": float(v) for k, v in _scalars(metrics).items()})
        self.log(row, step=wandb_step)

    def log_infer(self, *, q_values, selected_q, action_index, stride_info, infer_ms, total_ms) -> None:
        if self.run is not None:
            # Unfinished candidates carry -inf from the verifier; stats ignore them.
            q = np.asarray(q_values, dtype=np.float32)
            q = q[np.isfinite(q)]
            row = {
                "stride/action_chunks": self.step,
                "stride/adaptive_steps": int(stride_info.get("adaptive_steps", 0)),
                "stride/avg_active_samples": float(stride_info.get("avg_active_samples", 0.0)),
                "stride/finished_fraction": float(stride_info.get("finished_fraction", 0.0)),
                "stride/selected_finished": bool(stride_info.get("selected_finished", False)),
                "stride/num_transitions": int(stride_info.get("num_transitions", 0)),
                "stride/online_replay_size": int(stride_info.get("online_replay_size", 0)),
                "stride/learned_selector_ready": bool(stride_info.get("learned_selector_ready", False)),
                "policy/infer_ms": float(infer_ms),
                "policy/total_ms": float(total_ms),
                "verifier/q_selected": float(selected_q),
                "verifier/q_mean": float(q.mean()) if q.size else 0.0,
                "verifier/q_max": float(q.max()) if q.size else 0.0,
                "verifier/q_min": float(q.min()) if q.size else 0.0,
                "verifier/q_std": float(q.std()) if q.size else 0.0,
                "verifier/action_index": int(action_index),
                **{f"train/{k}": float(v) for k, v in _scalars(stride_info.get("train_metrics") or {}).items()},
            }
            # commit=False avoids a sync per infer; flush() commits.
            self.log(row, commit=False)
        self.step += 1

    def commit(self) -> None:
        if self.run is not None:
            try:
                self.run.log({}, commit=True)
            except Exception:
                logging.exception("wandb commit failed")


def _scalars(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
    }


class TransitionRecorder:
    """Numbers infers that produced transitions and, with ``replay_dir``, saves each as stride_step_*.npz."""

    def __init__(self, replay_dir: str | None):
        self.replay_dir = Path(replay_dir) if replay_dir else None
        if self.replay_dir is not None:
            self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.num_infers = 0

    def record(self, transitions: Transitions, q_values, action_index: int, meta: dict[str, float]) -> int:
        """Returns this infer's index."""
        infer_idx = self.num_infers
        self.num_infers += 1
        if self.replay_dir is not None:
            payload = {
                "verifier_scores": np.asarray(q_values, dtype=np.float32),
                "action_index": np.asarray([action_index], dtype=np.int32),
                "robot_episode_idx": np.asarray([EPISODE_IDX], dtype=np.int32),
                "stride_infer_idx_in_episode": np.asarray([infer_idx], dtype=np.int32),
                **{f"stride_{k}": np.asarray([v], dtype=np.float32) for k, v in meta.items()},
            }
            for key in transitions[0]:
                payload[f"transition_{key}"] = np.stack([tr[key] for tr in transitions])
            np.savez_compressed(self.replay_dir / f"stride_step_{infer_idx:08d}.npz", **payload)
        return infer_idx


class OnlineTrainer:
    def __init__(self, cfg: StrideConfig, *, num_samples: int, wandb_logger: WandbLogger):
        self.cfg = cfg
        self.num_samples = int(num_samples)
        self.wandb = wandb_logger
        self.checkpoint_dir = Path(cfg.stride_checkpoint_dir) if cfg.stride_checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.replay: NumpyReplayBuffer | None = None
        self.learner: StrideSACLearner | None = None
        self.accumulator: TransitionAccumulator | None = None
        self.online_step = 0
        self.did_offline_pretrain = False
        self.best_reward_mean = -float("inf")
        self.last_train_metrics: dict[str, float] = {}
        self.cf_avg_stride = float(C.CF_INITIAL_AVG_STRIDE)
        self.episode_strides: list[float] = []

        self.offline_replay: NumpyReplayBuffer | None = None
        if cfg.stride_offline_replay is not None:
            logging.info("Loading offline replay from %s ...", cfg.stride_offline_replay)
            self.offline_replay = load_replay_npz(cfg.stride_offline_replay)
            logging.info("Offline replay ready (%d transitions).", len(self.offline_replay))
            if self.wandb.run is not None:
                self.wandb.log({"offline_replay/num_transitions": len(self.offline_replay)}, step=0)
                # Offline pretrain logs from the current step; keep it off step 0.
                self.wandb.step = max(self.wandb.step, 1)

    @property
    def replay_size(self) -> int:
        return len(self.replay) if self.replay is not None else 0

    def learned_selector_ready(self) -> bool:
        return self.learner is not None and (
            self.did_offline_pretrain or self.replay_size >= self.cfg.stride_min_online_transitions
        )

    def step(
        self,
        transitions: Transitions,
        q_values: np.ndarray,
        *,
        infer_idx: int,
        meta_row: np.ndarray | None,
        redenoise: Redenoise,
    ) -> dict[str, float]:
        """One infer's worth of replay rows and SAC updates."""
        if not transitions:
            return {}
        cfg = self.cfg
        self._ensure_state(transitions)
        self._track_strides(transitions)

        # Second pass: V(s) baseline for the advantage, and optionally its own counterfactual rows.
        second_pass_added = 0.0
        baseline_q = None
        if self.num_samples >= 2:
            stride = max(
                1.0 / MAX_DENOISE_STEPS,
                float(np.random.uniform(self.cf_avg_stride * C.CF_STRIDE_SCALE_MIN, self.cf_avg_stride * C.CF_STRIDE_SCALE_MAX)),
            )
            result = redenoise(stride)
            if result is not None:
                cf_q, cf_transitions = result
                baseline_q = finite_q_mean(cf_q)
                if cfg.stride_second_pass_counterfactuals and infer_idx % max(1, cfg.stride_second_pass_every) == 0:
                    second_pass_added = float(add_second_pass_counterfactuals(
                        cf_transitions,
                        self.replay,
                        q_scores_to_advantage_with_baseline(cf_q.reshape(1, -1), baseline_q),
                        cf_stride_norm=stride,
                        **self._reward_kwargs(),
                        robot_episode_idx=EPISODE_IDX,
                        stride_infer_meta_row=meta_row,
                        use_extended_obs=cfg.stride_use_extended_obs,
                    ))

        q_row = np.asarray(q_values, dtype=np.float32).reshape(1, -1)
        if baseline_q is not None:
            advantages = q_scores_to_advantage_with_baseline(q_row, baseline_q)
        else:
            logging.warning("No second-pass baseline; falling back to the same-pass mean advantage.")
            advantages = q_scores_to_advantage(q_row)
        stats = self.accumulator.add_to_replay(
            transitions,
            self.replay,
            advantages,
            robot_episode_idx=EPISODE_IDX,
            stride_infer_idx_in_episode=int(infer_idx),
            stride_infer_meta=meta_row,
        )
        k_added = 0
        if cfg.stride_k_subsample_counterfactuals:
            k_added = add_k_subsample_counterfactuals(
                transitions,
                self.replay,
                advantages,
                **self._reward_kwargs(),
                num_draws=cfg.stride_k_subsample_draws,
                robot_episode_idx=EPISODE_IDX,
                stride_infer_meta_row=meta_row,
                use_extended_obs=cfg.stride_use_extended_obs,
            )
        stopped_added = 0
        if cfg.stride_all_stopped_counterfactuals:
            stopped_added = add_all_stopped_counterfactuals(
                transitions,
                self.replay,
                no_finish_penalty=cfg.stride_no_finish_penalty,
                latency_penalty=cfg.stride_latency_penalty,
                parallel_penalty=cfg.stride_parallel_penalty,
                robot_episode_idx=EPISODE_IDX,
                stride_infer_meta_row=meta_row,
                use_extended_obs=cfg.stride_use_extended_obs,
            )
        metrics = {
            "reward_mean": stats.reward_mean,
            "verifier_quality_mean": stats.verifier_quality_mean,
            "latency_cost_mean": stats.latency_cost_mean,
            "parallel_cost_mean": stats.parallel_cost_mean,
            "finished_fraction": stats.finished_fraction,
            "k_subsample_added": float(k_added),
            "all_stopped_added": float(stopped_added),
            "second_pass_added": second_pass_added,
        }
        if len(self.replay) < cfg.stride_min_online_transitions:
            return metrics
        for _ in range(max(0, cfg.stride_online_updates_per_step)):
            self.last_train_metrics = self.learner.train_step(
                self.replay,
                offline_replay=self.offline_replay,
                offline_fraction=cfg.stride_offline_batch_frac,
                online_step=self.online_step,
            )
        metrics.update(self.last_train_metrics)
        self.online_step += 1
        self._maybe_checkpoint(float(metrics.get("reward_mean", -float("inf"))))
        return metrics

    def flush(self) -> None:
        """Save last_stride_learner.pt and last_online_replay.npz (client disconnect, SIGTERM, exit)."""
        try:
            if self.checkpoint_dir is not None:
                if self.learner is not None:
                    torch.save(self.learner.state_dict(), self.checkpoint_dir / "last_stride_learner.pt")
                if self.replay is not None:
                    self.replay.save(self.checkpoint_dir / "last_online_replay.npz")
        finally:
            if self.episode_strides:
                self.cf_avg_stride = float(np.mean(self.episode_strides))
            self.episode_strides = []

    def _reward_kwargs(self) -> dict[str, float]:
        cfg = self.cfg
        return dict(
            terminal_bonus=cfg.stride_terminal_bonus,
            latency_penalty=cfg.stride_latency_penalty,
            parallel_penalty=cfg.stride_parallel_penalty,
            no_finish_penalty=cfg.stride_no_finish_penalty,
            verifier_bias=cfg.stride_verifier_bias,
        )

    def _track_strides(self, transitions: Transitions) -> None:
        actions = np.stack([tr["action"] for tr in transitions])
        active = np.stack([tr["active"] for tr in transitions]).astype(np.bool_)
        if actions.ndim == 4:
            actions = actions.squeeze(-1)
        if np.any(active):
            strides = action_to_stride(actions, (0.0, 1.0))[active]
            self.episode_strides.extend(strides.reshape(-1).astype(np.float64).tolist())

    def _ensure_state(self, transitions: Transitions) -> None:
        """Build replay, accumulator and learner from the first transitions; run offline pretrain once."""
        if self.replay is not None:
            return
        cfg = self.cfg
        first = transitions[0]
        state_dim = int(first["state_obs"].shape[-1])
        sample_dim = int(first["sample_obs"].shape[-1])
        self.replay = NumpyReplayBuffer(
            capacity=cfg.stride_replay_capacity,
            action_shape=tuple(first["action"].shape[1:]),
            state_dim=state_dim,
            sample_dim=sample_dim,
            num_samples=self.num_samples,
        )
        obs_dim = state_dim + self.num_samples * sample_dim
        if self.offline_replay is not None and self.offline_replay.obs.shape[-1] != obs_dim:
            raise ValueError(f"offline replay obs dim {self.offline_replay.obs.shape[-1]} != online obs dim {obs_dim}")
        self.accumulator = TransitionAccumulator(n_envs=1, num_action_samples=self.num_samples, **self._reward_kwargs())
        if cfg.stride_checkpoint is not None:
            self.learner = StrideSACLearner.from_checkpoint(cfg.stride_checkpoint, device=cfg.stride_device)
        else:
            self.learner = StrideSACLearner(
                SACLearnerConfig(
                    state_dim=state_dim,
                    sample_dim=sample_dim,
                    num_samples=self.num_samples,
                    batch_size=cfg.stride_batch_size,
                    offline_fraction=cfg.stride_offline_batch_frac,
                ),
                device=cfg.stride_device,
            )
        if self.offline_replay is not None and len(self.offline_replay) > 0:
            n = max(0, cfg.stride_offline_pretrain_steps)
            base = self.wandb.step
            for i in tqdm.tqdm(range(n), desc="Pretraining", leave=True):
                self.last_train_metrics = self.learner.train_step(self.offline_replay, update_actor=False)
                self.wandb.log_pretrain(step=i, total=n, metrics=self.last_train_metrics, wandb_step=base + i)
            if n > 0:
                self.wandb.log(
                    {"offline_pretrain/done": 1,
                     **{f"offline_pretrain/final_{k}": v for k, v in _scalars(self.last_train_metrics).items()}},
                    step=base + n,
                )
                self.wandb.step = base + n + 1
            self.did_offline_pretrain = True

    def _maybe_checkpoint(self, reward_mean: float) -> None:
        if self.checkpoint_dir is None:
            return
        if np.isfinite(reward_mean) and reward_mean > self.best_reward_mean:
            self.best_reward_mean = float(reward_mean)
            torch.save(self.learner.state_dict(), self.checkpoint_dir / "best_stride_policy.pt")
            self.replay.save(self.checkpoint_dir / "best_online_replay.npz")
        every = self.cfg.stride_checkpoint_every
        if every > 0 and self.online_step % every == 0:
            torch.save(self.learner.state_dict(), self.checkpoint_dir / f"stride_policy_step{self.online_step:08d}.pt")
            self.replay.save(self.checkpoint_dir / f"online_replay_step{self.online_step:08d}.npz")
