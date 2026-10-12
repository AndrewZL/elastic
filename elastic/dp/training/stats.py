"""Per-episode training statistics for ``elastic.dp.training.train`` (logging only).

The one value that feeds back into training is ``EpisodeStats.finish()["episode_avg_stride"]``,
which centers the second-pass counterfactual stride.
"""
import numpy as np

from elastic.core.action import action_to_stride


class EpisodeStats:
    """Accumulates chunk statistics over one training episode batch."""

    def __init__(self, action_range, pause_threshold):
        self.action_range = action_range
        self.pause_threshold = float(pause_threshold)
        self.reset()

    def reset(self):
        self.strides, self.inference_steps, self.samples, self.min_strides = [], [], [], []

    def add_chunk(self, transitions, alive):
        """Record one chunk's strides, latency (sequential steps) and active-sample counts."""
        actions = np.stack([t["action"] for t in transitions])
        active = np.stack([t["active"] for t in transitions])
        T = active.shape[0]
        if actions.ndim == 4:
            actions = actions.squeeze(-1)

        any_active = np.any(active, axis=2)
        last_active = np.max(np.arange(T)[:, None] * any_active, axis=0)
        latency = np.where(np.any(any_active, axis=0), last_active + 1, np.nan)

        active_vals = actions[active]
        if active_vals.size > 0:
            self.strides.append(active_vals.flatten())
        self.inference_steps.append(np.where(alive, latency, np.nan))
        min_s = np.where(any_active, np.min(np.where(active, actions, np.inf), axis=2), np.nan)
        self.min_strides.append(np.where(alive[None, :], min_s, np.nan).flatten())
        usage = np.where(any_active, np.sum(active, axis=2), np.nan)
        self.samples.append(np.where(alive[None, :], usage, np.nan).flatten())

    def _to_stride(self, actions):
        return np.asarray(
            action_to_stride(actions, self.action_range, pause_threshold=self.pause_threshold), dtype=np.float32
        )

    def finish(self):
        """Episode summary, or None when no chunk had an active sample."""
        if not self.strides:
            return None
        strides = self._to_stride(np.concatenate(self.strides))
        min_strides = self._to_stride(np.concatenate(self.min_strides))
        self.last_strides = strides
        return {
            "episode_avg_stride": float(np.mean(strides)),
            "episode_var_stride": float(np.var(strides)),
            "episode_avg_min_stride": float(np.nanmean(min_strides[min_strides > 0])),
            "episode_avg_latency": float(np.nanmean(np.concatenate(self.inference_steps))),
            "episode_avg_sample_usage": float(np.nanmean(np.concatenate(self.samples))),
        }
