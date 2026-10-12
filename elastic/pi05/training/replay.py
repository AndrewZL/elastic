from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from elastic.pi05.constants import REPLAY_REWARD_COMPONENT_DIM
from elastic.pi05.constants import STRIDE_INFER_META_DIM


def pack_stride_infer_meta_row(
    *,
    adaptive_steps: float = 0.0,
    avg_active_samples: float = 0.0,
    finished_fraction: float = 0.0,
    num_transitions: float = 0.0,
    baseline_num_steps: float = 0.0,
    selected_finished: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        [
            float(adaptive_steps),
            float(avg_active_samples),
            float(finished_fraction),
            float(num_transitions),
            float(baseline_num_steps),
            float(selected_finished),
        ],
        dtype=np.float32,
    ).reshape(STRIDE_INFER_META_DIM)


def stride_infer_meta_row_from_npz(data: dict[str, np.ndarray]) -> np.ndarray:
    """Recover a ``stride_infer_meta`` row from keys saved in ``stride_step_*.npz`` (defaults if missing)."""
    def _f32(key: str, default: float = 0.0) -> float:
        if key not in data:
            return float(default)
        return float(np.asarray(data[key], dtype=np.float32).reshape(-1)[0])

    return pack_stride_infer_meta_row(
        adaptive_steps=_f32("stride_adaptive_steps", 0.0),
        avg_active_samples=_f32("stride_avg_active_samples", 0.0),
        finished_fraction=_f32("stride_finished_fraction", 0.0),
        num_transitions=_f32("stride_num_transitions", 0.0),
        baseline_num_steps=_f32("stride_baseline_num_steps", 0.0),
        selected_finished=_f32("stride_selected_finished", 0.0),
    )


@dataclass
class ReplayBatch:
    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    reward_components: np.ndarray
    traj_info: np.ndarray
    lifecycle: np.ndarray
    stride_infer_meta: np.ndarray


class NumpyReplayBuffer:
    """Uniform replay with split state/sample storage; the learner sees ``[state_obs | sample_obs_1..K]``.

    ``traj_info`` holds (robot episode, infer index in episode or a synthetic-row marker);
    ``lifecycle`` is the transition index within that infer's denoising rollout.
    """

    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int | None = None,
        action_shape: tuple[int, ...],
        state_dim: int | None = None,
        sample_dim: int | None = None,
        num_samples: int | None = None,
    ):
        self.capacity = int(capacity)
        self.action_shape = tuple(action_shape)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.sample_dim = None if sample_dim is None else int(sample_dim)
        self.num_samples = None if num_samples is None else int(num_samples)

        if obs_dim is None:
            if self.state_dim is None or self.sample_dim is None or self.num_samples is None:
                raise ValueError("Provide obs_dim or all of state_dim, sample_dim, and num_samples")
            obs_dim = self.state_dim + self.num_samples * self.sample_dim
        self.obs_dim = int(obs_dim)

        if self.state_dim is None and self.sample_dim is not None and self.num_samples is not None:
            self.state_dim = self.obs_dim - self.num_samples * self.sample_dim
        if self.sample_dim is None and self.state_dim is not None and self.num_samples is not None:
            sample_width = self.obs_dim - self.state_dim
            if sample_width % max(self.num_samples, 1) != 0:
                raise ValueError(f"obs_dim/state_dim do not divide into num_samples={self.num_samples}")
            self.sample_dim = sample_width // max(self.num_samples, 1)

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.reward_components = np.zeros((self.capacity, REPLAY_REWARD_COMPONENT_DIM), dtype=np.float32)
        self.traj_info = np.full((self.capacity, 2), -1, dtype=np.int64)
        self.lifecycle = np.zeros((self.capacity,), dtype=np.int32)
        self.stride_infer_meta = np.zeros((self.capacity, STRIDE_INFER_META_DIM), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(
        self,
        *,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        reward_components: np.ndarray | None = None,
        traj_info: np.ndarray | None = None,
        lifecycle: int | np.ndarray = 0,
        stride_infer_meta: np.ndarray | None = None,
    ) -> None:
        idx = self.pos
        self.obs[idx] = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim)
        self.actions[idx] = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        self.rewards[idx] = np.asarray(reward, dtype=np.float32).reshape(1)
        self.dones[idx] = np.asarray(done, dtype=np.float32).reshape(1)
        if reward_components is not None:
            self.reward_components[idx] = np.asarray(reward_components, dtype=np.float32).reshape(
                REPLAY_REWARD_COMPONENT_DIM
            )
        if traj_info is not None:
            self.traj_info[idx] = np.asarray(traj_info, dtype=np.int64).reshape(2)
        self.lifecycle[idx] = int(np.asarray(lifecycle).reshape(-1)[0])
        if stride_infer_meta is not None:
            self.stride_infer_meta[idx] = np.asarray(stride_infer_meta, dtype=np.float32).reshape(
                STRIDE_INFER_META_DIM
            )
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> ReplayBatch:
        size = len(self)
        if size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        idx = np.random.randint(0, size, size=int(batch_size))
        return self._get_samples(idx)

    def _get_samples(self, idx: np.ndarray) -> ReplayBatch:
        return ReplayBatch(
            obs=self.obs[idx],
            next_obs=self.next_obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
            reward_components=self.reward_components[idx],
            traj_info=self.traj_info[idx],
            lifecycle=self.lifecycle[idx],
            stride_infer_meta=self.stride_infer_meta[idx],
        )

    def split_obs(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.state_dim is None or self.sample_dim is None or self.num_samples is None:
            raise ValueError("Cannot split observations without state_dim, sample_dim, and num_samples")
        obs = np.asarray(obs, dtype=np.float32)
        state = obs[..., : self.state_dim]
        sample = obs[..., self.state_dim :].reshape(*obs.shape[:-1], self.num_samples, self.sample_dim)
        return state, sample

    def state_dict(self) -> dict[str, Any]:
        size = len(self)
        return {
            "capacity": self.capacity,
            "pos": self.pos,
            "full": self.full,
            "obs_dim": self.obs_dim,
            "state_dim": -1 if self.state_dim is None else self.state_dim,
            "sample_dim": -1 if self.sample_dim is None else self.sample_dim,
            "num_samples": -1 if self.num_samples is None else self.num_samples,
            "action_shape": np.asarray(self.action_shape, dtype=np.int64),
            "obs": self.obs[:size],
            "next_obs": self.next_obs[:size],
            "actions": self.actions[:size],
            "rewards": self.rewards[:size],
            "dones": self.dones[:size],
            "reward_components": self.reward_components[:size],
            "traj_info": self.traj_info[:size],
            "lifecycle": self.lifecycle[:size],
            "stride_infer_meta": self.stride_infer_meta[:size],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> NumpyReplayBuffer:
        obs = np.asarray(state["obs"], dtype=np.float32)
        actions = np.asarray(state["actions"], dtype=np.float32)
        state_dim = _optional_int(state.get("state_dim", -1))
        sample_dim = _optional_int(state.get("sample_dim", -1))
        num_samples = _optional_int(state.get("num_samples", -1))
        buffer = cls(
            capacity=int(state["capacity"]),
            obs_dim=int(state.get("obs_dim", obs.shape[-1])),
            action_shape=actions.shape[1:],
            state_dim=state_dim,
            sample_dim=sample_dim,
            num_samples=num_samples,
        )
        size = obs.shape[0]
        buffer.obs[:size] = obs
        buffer.next_obs[:size] = np.asarray(state["next_obs"], dtype=np.float32)
        buffer.actions[:size] = actions
        buffer.rewards[:size] = np.asarray(state["rewards"], dtype=np.float32)
        buffer.dones[:size] = np.asarray(state["dones"], dtype=np.float32)
        rc = np.asarray(
            state.get("reward_components", np.zeros((size, REPLAY_REWARD_COMPONENT_DIM))),
            dtype=np.float32,
        )
        if rc.ndim == 2 and rc.shape[1] == 3:
            rc = np.concatenate([rc, np.ones((rc.shape[0], 1), dtype=np.float32)], axis=1)
        buffer.reward_components[:size] = rc.reshape(size, REPLAY_REWARD_COMPONENT_DIM)
        buffer.traj_info[:size] = np.asarray(state.get("traj_info", np.full((size, 2), -1)), dtype=np.int64)
        buffer.lifecycle[:size] = np.asarray(state.get("lifecycle", np.zeros((size,))), dtype=np.int32)
        tmp_meta = np.zeros((size, STRIDE_INFER_META_DIM), dtype=np.float32)
        if "stride_infer_meta" in state:
            sim = np.asarray(state["stride_infer_meta"], dtype=np.float32)
            if sim.ndim == 2 and sim.shape[1] == STRIDE_INFER_META_DIM:
                n_r = min(int(size), int(sim.shape[0]))
                tmp_meta[:n_r] = sim[:n_r]
        buffer.stride_infer_meta[:size] = tmp_meta
        buffer.pos = int(state["pos"])
        buffer.full = bool(state["full"])
        return buffer

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.state_dict())


def _optional_int(value: Any) -> int | None:
    value = int(np.asarray(value).reshape(-1)[0])
    return None if value < 0 else value


def load_replay_npz(path: str | Path) -> NumpyReplayBuffer:
    data = dict(np.load(path, allow_pickle=True))
    state = {key: value.item() if value.shape == () else value for key, value in data.items()}
    return NumpyReplayBuffer.from_state_dict(state)


def save_replay_npz(
    replay: NumpyReplayBuffer,
    path: str | Path,
    *,
    extra_metadata: dict[str, np.ndarray] | None = None,
) -> None:
    state = replay.state_dict()
    if extra_metadata:
        state.update(extra_metadata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state)
