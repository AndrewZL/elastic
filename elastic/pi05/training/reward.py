from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from elastic.pi05.constants import (
    LATENCY_PENALTY,
    NO_FINISH_PENALTY,
    PARALLEL_PENALTY,
    TERMINAL_BONUS,
    VERIFIER_BIAS,
)


def relabel_rewards_full(
    reward_components: np.ndarray,
    *,
    latency_penalty: float,
    parallel_penalty: float,
    terminal_bonus: float,
    saved_terminal_bonus: float,
    saved_verifier_bias: float = 0.0,
    verifier_bias: float,
    no_finish_penalty: float = NO_FINISH_PENALTY,
    saved_no_finish_penalty: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute rewards and ``reward_components[:, 0]`` (best quality) from stored components.

    Decodes finished rows as ``raw_score = rc0 / saved_terminal_bonus - saved_verifier_bias``.
    No-finish rows keep ``saved_no_finish_penalty`` (default ``NO_FINISH_PENALTY``) as best quality.
    """
    if saved_terminal_bonus <= 0 or terminal_bonus <= 0:
        raise ValueError("saved_terminal_bonus and terminal_bonus must be positive")
    nf_saved = float(NO_FINISH_PENALTY if saved_no_finish_penalty is None else saved_no_finish_penalty)
    rc = np.asarray(reward_components, dtype=np.float32)
    core = rc[:, :3] if rc.ndim == 2 and rc.shape[1] > 3 else rc
    best_old = core[:, 0].astype(np.float64)
    latency = core[:, 1].astype(np.float64)
    parallel_excess = core[:, 2].astype(np.float64)
    finished = best_old > nf_saved + 0.5
    raw_score = np.zeros_like(best_old, dtype=np.float64)
    raw_score[finished] = best_old[finished] / float(saved_terminal_bonus) - float(saved_verifier_bias)
    best_new = np.full_like(best_old, float(no_finish_penalty), dtype=np.float64)
    best_new[finished] = (raw_score[finished] + float(verifier_bias)) * float(terminal_bonus)
    rewards = (
        best_new - float(latency_penalty) * latency - float(parallel_penalty) * parallel_excess
    ).astype(np.float32)
    return rewards, best_new.astype(np.float32)


def _pad_flat_obs_to_replay(vec: np.ndarray, obs_dim: int) -> np.ndarray:
    """Zero-pad flattened obs to ``obs_dim``; a longer vector is a layout bug and raises."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size > obs_dim:
        raise ValueError(
            f"Flattened obs has length {v.size}, larger than replay obs_dim={obs_dim}. "
            "Truncation is not allowed; fix STRIDE obs construction or replay sizing."
        )
    if v.size == obs_dim:
        return v
    return np.pad(v, (0, obs_dim - v.size))


def _discounted_returns(rewards_t_b: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards_t_b, dtype=np.float32)
    running = np.zeros(rewards_t_b.shape[1:], dtype=np.float32)
    for idx in range(rewards_t_b.shape[0] - 1, -1, -1):
        running = rewards_t_b[idx].astype(np.float32) + float(gamma) * running
        returns[idx] = running
    return returns


@dataclass
class RewardStats:
    success_proxy: float
    reward_mean: float
    verifier_quality_mean: float
    latency_cost_mean: float
    parallel_cost_mean: float
    finished_fraction: float
    winner_indices: np.ndarray
    reward_components: np.ndarray


class TransitionAccumulator:
    """Centralized team reward for adaptive pi0.5 candidate denoising."""

    def __init__(
        self,
        *,
        n_envs: int,
        num_action_samples: int,
        terminal_bonus: float = TERMINAL_BONUS,
        latency_penalty: float = LATENCY_PENALTY,
        parallel_penalty: float = PARALLEL_PENALTY,
        gamma: float = 1.0,
        no_finish_penalty: float = NO_FINISH_PENALTY,
        verifier_bias: float = VERIFIER_BIAS,
    ):
        self.n_envs = int(n_envs)
        self.num_action_samples = int(num_action_samples)
        self.terminal_bonus = float(terminal_bonus)
        self.latency_penalty = float(latency_penalty)
        self.parallel_penalty = float(parallel_penalty)
        self.gamma = float(gamma)
        self.no_finish_penalty = float(no_finish_penalty)
        self.verifier_bias = float(verifier_bias)
        self.last_stats: RewardStats | None = None
        self._logged_replay_obs_align = False

    def compute_returns(
        self,
        transitions: list[dict[str, np.ndarray]],
        verifier_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, RewardStats]:
        if not transitions:
            raise ValueError("Cannot compute rewards without selector transitions")

        scores = np.asarray(verifier_scores, dtype=np.float32)
        if scores.shape != (self.n_envs, self.num_action_samples):
            raise ValueError(f"Expected verifier scores {(self.n_envs, self.num_action_samples)}, got {scores.shape}")

        finished = np.stack([transition["done"] for transition in transitions]).astype(np.bool_)
        active = np.stack([transition["active"] for transition in transitions]).astype(np.float32)
        running = np.stack([transition["running"] for transition in transitions]).astype(np.float32)
        t_count, b_count, k_count = active.shape
        if (b_count, k_count) != (self.n_envs, self.num_action_samples):
            raise ValueError(
                f"Transition shape {(b_count, k_count)} does not match "
                f"{(self.n_envs, self.num_action_samples)}"
            )

        finished_any = np.any(finished, axis=0)
        has_finished = np.any(finished_any, axis=1)
        quality = (scores + self.verifier_bias) * self.terminal_bonus
        finished_quality = np.where(finished_any, quality, -np.inf)
        best_quality = np.where(has_finished, np.max(finished_quality, axis=1), self.no_finish_penalty)
        winner = np.where(has_finished, np.argmax(finished_quality, axis=1), np.argmax(quality, axis=1))

        l_k = active.sum(axis=0)
        latency_steps = np.max(l_k, axis=1)
        sum_l = l_k.sum(axis=1)
        safe_latency = np.maximum(latency_steps, 1.0)
        norm_excess = np.maximum(sum_l - latency_steps, 0.0) / safe_latency
        latency_cost = self.latency_penalty * latency_steps
        parallel_cost = self.parallel_penalty * norm_excess
        terminal_reward = best_quality - latency_cost - parallel_cost

        any_active = np.any(active > 0, axis=2)
        last_active = np.max(np.arange(t_count)[:, None] * any_active, axis=0)
        rewards_t_b = np.zeros((t_count, self.n_envs, 1), dtype=np.float32)
        rewards_t_b[last_active, np.arange(self.n_envs), 0] = terminal_reward.astype(np.float32)
        returns_t_b = _discounted_returns(rewards_t_b, self.gamma)

        any_running = np.any(running > 0, axis=2)
        last_running = np.max(np.arange(t_count)[:, None] * any_running, axis=0)
        dones_t_b = (np.arange(t_count)[:, None] >= last_running[None]).astype(np.float32)[..., None]

        success_proxy = float(np.mean(has_finished))
        components = np.stack(
            [best_quality.astype(np.float32), latency_steps.astype(np.float32), norm_excess.astype(np.float32)], axis=1
        )
        stats = RewardStats(
            success_proxy=success_proxy,
            reward_mean=float(np.mean(terminal_reward)),
            verifier_quality_mean=float(np.mean(best_quality.astype(np.float64))),
            latency_cost_mean=float(np.mean(latency_cost)),
            parallel_cost_mean=float(np.mean(parallel_cost)),
            finished_fraction=float(np.mean(has_finished)),
            winner_indices=winner.astype(np.int32),
            reward_components=components,
        )
        self.last_stats = stats
        return returns_t_b, dones_t_b, stats

    def add_to_replay(
        self,
        transitions: list[dict[str, np.ndarray]],
        replay,
        verifier_scores: np.ndarray,
        *,
        robot_episode_success_log: np.ndarray | None = None,
        robot_episode_idx: int = -1,
        stride_infer_idx_in_episode: int = -1,
        stride_infer_meta: np.ndarray | None = None,
    ) -> RewardStats:
        """``robot_episode_success_log`` only fills ``reward_components[:, 3]``; it never enters the reward."""
        returns_t_b, dones_t_b, stats = self.compute_returns(transitions, verifier_scores)
        es_vec = (
            np.ones((self.n_envs,), dtype=np.float32)
            if robot_episode_success_log is None
            else np.clip(np.asarray(robot_episode_success_log, dtype=np.float32).reshape(self.n_envs), 0.0, 1.0)
        )
        meta_row = (
            np.zeros((replay.stride_infer_meta.shape[-1],), dtype=np.float32)
            if stride_infer_meta is None
            else np.asarray(stride_infer_meta, dtype=np.float32).reshape(-1)
        )
        traj_pair = np.asarray([int(robot_episode_idx), int(stride_infer_idx_in_episode)], dtype=np.int64)
        for t_idx, transition in enumerate(transitions):
            state = transition["state_obs"][:, 0, :]
            sample = transition["sample_obs"].reshape(self.n_envs, -1)
            next_state = transition["next_state_obs"][:, 0, :]
            next_sample = transition["next_sample_obs"].reshape(self.n_envs, -1)
            obs = np.concatenate([state, sample], axis=-1)
            next_obs = np.concatenate([next_state, next_sample], axis=-1)
            group_running = np.any(transition["running"], axis=1)
            for env_idx in range(self.n_envs):
                if group_running[env_idx] or t_idx == len(transitions) - 1:
                    raw_o = np.asarray(obs[env_idx], dtype=np.float32).reshape(-1)
                    raw_no = np.asarray(next_obs[env_idx], dtype=np.float32).reshape(-1)
                    o = _pad_flat_obs_to_replay(raw_o, replay.obs_dim)
                    no = _pad_flat_obs_to_replay(raw_no, replay.obs_dim)
                    if not self._logged_replay_obs_align and (
                        raw_o.size < replay.obs_dim or raw_no.size < replay.obs_dim
                    ):
                        logging.debug(
                            "Padded online replay obs to obs_dim=%d (variable STRIDE mid features).",
                            replay.obs_dim,
                        )
                        self._logged_replay_obs_align = True
                    comp = np.concatenate(
                        [stats.reward_components[env_idx], np.asarray([es_vec[env_idx]], dtype=np.float32)],
                        axis=-1,
                    )
                    replay.add(
                        obs=o,
                        next_obs=no,
                        action=transition["action"][env_idx],
                        reward=returns_t_b[t_idx, env_idx],
                        done=dones_t_b[t_idx, env_idx],
                        reward_components=comp,
                        traj_info=traj_pair,
                        lifecycle=int(t_idx),
                        stride_infer_meta=meta_row,
                    )
        return stats
