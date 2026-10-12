"""Replay buffer for Diffusion Policy stride training.

Rows are flat observations [state | S x sample]; the next observation lives in
the following slot, so the slot at ``pos`` is never sampled once the buffer wraps.
"""
from __future__ import annotations

import pickle
from typing import NamedTuple

import numpy as np
import torch

from elastic.dp.training.reward import REWARD_COMPONENT_DIM, relabel_buffer_rewards


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class SplitReplayBuffer:
    n_envs = 1

    def __init__(self, capacity: int, obs_dim: int, num_samples: int, state_dim: int, device):
        self.buffer_size = int(capacity)
        self.state_dim = int(state_dim)
        self.obs_shape = (int(obs_dim),)
        self.action_dim = int(num_samples)
        self.device = torch.device(device)
        self.state_obs = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.sample_obs = np.zeros((self.buffer_size, int(obs_dim) - self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, 1, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.reward_components = np.zeros((self.buffer_size, REWARD_COMPONENT_DIM), dtype=np.float32)
        # [:, 0] episode index, [:, 1] outer env step within the episode.
        self.traj_info = np.full((self.buffer_size, 2), -1, dtype=np.int32)
        self.pos = 0
        self.full = False

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray) -> None:
        obs = np.asarray(obs).reshape(-1)
        next_obs = np.asarray(next_obs).reshape(-1)
        nxt = (self.pos + 1) % self.buffer_size
        self.state_obs[self.pos] = obs[: self.state_dim]
        self.sample_obs[self.pos] = obs[self.state_dim :]
        self.state_obs[nxt] = next_obs[: self.state_dim]
        self.sample_obs[nxt] = next_obs[self.state_dim :]
        self.actions[self.pos] = np.array(action.reshape((self.n_envs, self.action_dim)))
        self.rewards[self.pos] = np.array(reward)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if self.full:
            inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(inds)

    def _get_samples(self, inds: np.ndarray) -> ReplayBufferSamples:
        env = np.random.randint(0, high=self.n_envs, size=(len(inds),))
        obs = np.concatenate([self.state_obs[inds], self.sample_obs[inds]], axis=-1)
        return ReplayBufferSamples(
            torch.tensor(obs, device=self.device),
            torch.tensor(self.actions[inds, env, :], device=self.device),
            torch.tensor(self.rewards[inds, env].reshape(-1, 1), device=self.device),
        )


def load_offline_buffers(path, buffer, offline_buffer, state_dim, reward_cfg, log=print):
    """Load a collected buffer pickle into ``buffer`` and its first rows into the frozen ``offline_buffer``.

    Rewards are relabeled when the reward coefficients differ from those used at collection.
    """
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    obs = loaded["observations"]
    n = obs.shape[0]
    size = buffer.buffer_size
    flat = obs.reshape(n, -1)
    buffer.state_obs[:n] = flat[:, :state_dim]
    buffer.sample_obs[:n] = flat[:, state_dim:]
    # Next-obs slot of the last row, so sampling never reads an unwritten row.
    buffer.state_obs[n % size] = flat[-1, :state_dim]
    buffer.sample_obs[n % size] = flat[-1, state_dim:]
    buffer.actions[:n] = loaded["actions"][:n]
    buffer.rewards[:n] = loaded["rewards"][:n]
    buffer.pos = int(loaded.get("pos", 0))
    buffer.full = bool(loaded.get("full", False))
    rc = loaded["reward_components"][:n]
    if rc.shape[1] != REWARD_COMPONENT_DIM:
        raise ValueError(f"reward_components width {rc.shape[1]}, expected {REWARD_COMPONENT_DIM}")
    buffer.reward_components[:n] = rc
    if loaded.get("traj_info") is not None:
        buffer.traj_info[:n] = loaded["traj_info"][:n]

    saved = {
        "latency_penalty": loaded["saved_latency_penalty"],
        "parallel_penalty": loaded["saved_parallel_penalty"],
        "terminal_bonus": loaded["saved_terminal_bonus"],
        "verifier_bias": loaded["saved_verifier_bias"],
        "no_finish_penalty": loaded["saved_no_finish_penalty"],
        "advantage_scale": float(loaded.get("saved_advantage_scale", 1.0)),
    }
    relabel = any(abs(float(saved[k]) - float(reward_cfg[k])) > 1e-8 for k in saved)

    def _relabel(buf):
        relabel_buffer_rewards(
            buf,
            reward_cfg["latency_penalty"],
            reward_cfg["parallel_penalty"],
            reward_cfg["terminal_bonus"],
            reward_cfg["verifier_bias"],
            reward_cfg["no_finish_penalty"],
            new_advantage_scale=reward_cfg["advantage_scale"],
            saved_terminal_bonus=saved["terminal_bonus"],
        )

    if relabel:
        log(f"Relabeling buffer rewards: saved {saved} -> new {reward_cfg}")
        _relabel(buffer)
    else:
        log("Buffer reward hyperparameters match; no relabeling needed.")

    off_size = offline_buffer.buffer_size
    n_off = min(n, off_size)
    offline_buffer.state_obs[:n_off] = buffer.state_obs[:n_off]
    offline_buffer.sample_obs[:n_off] = buffer.sample_obs[:n_off]
    offline_buffer.state_obs[n_off % off_size] = buffer.state_obs[n_off - 1]
    offline_buffer.sample_obs[n_off % off_size] = buffer.sample_obs[n_off - 1]
    offline_buffer.actions[:n_off] = buffer.actions[:n_off]
    offline_buffer.rewards[:n_off] = buffer.rewards[:n_off]
    offline_buffer.reward_components[:n_off] = buffer.reward_components[:n_off]
    offline_buffer.traj_info[:n_off] = buffer.traj_info[:n_off]
    offline_buffer.pos = n_off % off_size
    offline_buffer.full = n_off >= off_size
    if relabel:
        _relabel(offline_buffer)
    log(f"Offline buffer: {offline_buffer.size()} transitions (full={offline_buffer.full})")
