"""Team reward for multi-sample denoising and in-place reward relabeling."""
from __future__ import annotations

import numpy as np
from typing import Optional


# reward_components columns: max advantage among finished samples (0 if none),
# any-finished flag, L_B = max active denoise steps, norm_excess = (sum_l - L) / max(L, 1).
REWARD_RC_MAX_ADV = 0
REWARD_RC_HAS_FINISHED = 1
REWARD_RC_L = 2
REWARD_RC_NORM_EXCESS = 3
REWARD_COMPONENT_DIM = 4


def relabel_buffer_rewards(
    buffer,
    new_latency_penalty: float,
    new_parallel_penalty: float,
    new_terminal_bonus: float,
    new_verifier_bias: float,
    new_no_finish_penalty: float,
    new_advantage_scale: float = 1.0,
    saved_terminal_bonus: Optional[float] = None,
):
    """Recompute buffer rewards in-place from stored reward_components with new coefficients."""
    if not hasattr(buffer, "reward_components") or buffer.reward_components is None:
        raise ValueError(
            "Buffer has no reward_components. Re-collect demos with store_raw_components=True."
        )
    if saved_terminal_bonus is None or saved_terminal_bonus <= 0:
        raise ValueError(
            "saved_terminal_bonus is required to relabel rewards when terminal_bonus changes."
        )
    n = buffer.size()
    rc = buffer.reward_components[:n]
    ncol = int(rc.shape[1])
    if ncol != REWARD_COMPONENT_DIM:
        raise ValueError(
            f"reward_components must have width {REWARD_COMPONENT_DIM}, got {ncol}. "
            "Reload buffer with current code or re-save after collect."
        )

    max_adv = rc[:, REWARD_RC_MAX_ADV].astype(np.float32)
    has_f = rc[:, REWARD_RC_HAS_FINISHED].astype(np.float32)
    L_B = rc[:, REWARD_RC_L].astype(np.float64)
    norm_excess = rc[:, REWARD_RC_NORM_EXCESS].astype(np.float64)

    best_new = has_f * (
        max_adv * float(new_advantage_scale) + float(new_verifier_bias)
    ) * float(new_terminal_bonus) + (1.0 - has_f) * float(new_no_finish_penalty)
    new_reward = (
        best_new - float(new_latency_penalty) * L_B - float(new_parallel_penalty) * norm_excess
    )
    buffer.rewards[:n, 0] = new_reward.astype(np.float32)


def _discounted_returns(rewards_t_b_k, gamma):
    """Discounted return G_t = r_t + gamma * G_{t+1} over time axis 0."""
    returns = np.zeros_like(rewards_t_b_k, dtype=np.float32)
    running = np.zeros(rewards_t_b_k.shape[1:], dtype=np.float32)
    for t in range(rewards_t_b_k.shape[0] - 1, -1, -1):
        running = rewards_t_b_k[t].astype(np.float32) + gamma * running
        returns[t] = running
    return returns


class TransitionAccumulator:
    """Centralized team reward for multi-sample denoising, written to replay as MC returns.

    Per env: R = Q_win - alpha * L - beta * max(sum_k l_k - L, 0) / L, where Q_win is the best
    finished quality (or the no-finish penalty) and L the longest sample's active steps. The
    reward lands at the env's last active denoise step.
    """

    def __init__(
        self,
        n_envs: int,
        num_action_samples: int,
        terminal_bonus: float,
        latency_penalty: float,
        parallel_penalty: float,
        gamma: float = 1.0,
        no_finish_penalty: float = -30.0,
        verifier_bias: float = 0.9,
        advantage_scale: float = 1.0,
        store_raw_components: bool = False,
    ):
        self.n_envs = n_envs
        self.num_action_samples = num_action_samples
        self.terminal_bonus = terminal_bonus
        self.latency_penalty = latency_penalty
        self.parallel_penalty = parallel_penalty
        self.gamma = float(gamma)
        self.no_finish_penalty = float(no_finish_penalty)
        self.verifier_bias = float(verifier_bias)
        self.advantage_scale = float(advantage_scale)
        self.store_raw_components = store_raw_components
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        self.reset_env_state()

    def reset_env_state(self):
        self.env_reward_accumulator = np.zeros((self.n_envs,), dtype=np.float32)
        self.env_latency_cost_accumulator = np.zeros((self.n_envs,), dtype=np.float32)
        self.env_parallel_cost_accumulator = np.zeros((self.n_envs,), dtype=np.float32)
        self.max_coverage = np.zeros((self.n_envs,), dtype=np.float32)
        self._traj_step = 0
        self.reset_denoise_episode_state()

    def reset_denoise_episode_state(self):
        self.denoise_transitions = []

    def reset_accumulator(self):
        """Return episode stats and reset accumulators."""
        avg_max_coverage = np.mean(self.max_coverage)
        total_return = np.sum(self.env_reward_accumulator)
        avg_latency_cost = np.mean(self.env_latency_cost_accumulator)
        avg_parallel_cost = np.mean(self.env_parallel_cost_accumulator)
        self._episode_idx = getattr(self, "_episode_idx", 0) + 1
        self.reset_env_state()
        return avg_max_coverage, total_return, avg_latency_cost, avg_parallel_cost

    def compute_rewards(
        self,
        transitions,
        buffer,
        advantage_override,
        max_coverage=None,
        env_mask=None,
    ):
        if len(transitions) == 0:
            return 0

        quality_B_S = (
            advantage_override * self.advantage_scale + self.verifier_bias
        ) * self.terminal_bonus

        finished_mask = np.stack([t["done"] for t in transitions]).astype(np.bool_)
        finished_denoising = np.any(finished_mask, axis=0)
        actives = np.stack([t["active"] for t in transitions]).astype(np.float32)
        T, B, S = actives.shape

        has_finished = np.any(finished_denoising, axis=1)
        finished_quality = np.where(finished_denoising, quality_B_S, -np.inf)
        best_quality_B = np.where(
            has_finished, np.max(finished_quality, axis=1), self.no_finish_penalty
        )

        adv_B_S = np.asarray(advantage_override, dtype=np.float64)
        adv_finished = np.where(finished_denoising, adv_B_S, -np.inf)
        max_adv_finished_B = np.where(
            has_finished, np.max(adv_finished, axis=1), 0.0
        ).astype(np.float32)

        assert max_coverage is not None, "max_coverage is required"
        max_coverage_B = max_coverage

        any_active_T_B = np.any(actives > 0, axis=2)
        last_active_idx_env = np.max(np.arange(T)[:, None] * any_active_T_B, axis=0)
        l_k_B_S = actives.sum(axis=0)
        L_B = np.max(l_k_B_S, axis=1)

        sum_l_B = l_k_B_S.sum(axis=1)
        excess_compute_B = (sum_l_B - L_B).clip(min=0)
        # Avoid 0/0 when no sample advanced in an env (L_B == 0).
        safe_L_B = np.maximum(L_B, 1.0)
        group_parallel_cost_B = (
            self.parallel_penalty * excess_compute_B / safe_L_B
        )

        latency_cost_B = self.latency_penalty * L_B
        parallel_cost_B = group_parallel_cost_B

        team_terminal_reward_B = best_quality_B - latency_cost_B - parallel_cost_B

        if env_mask is not None:
            alive = np.asarray(env_mask, dtype=bool).reshape(-1)
            if alive.size != B:
                raise ValueError(f"env_mask length {alive.size} != n_envs {B}")
            latency_cost_B = np.where(alive, latency_cost_B, 0.0)
            parallel_cost_B = np.where(alive, parallel_cost_B, 0.0)
            team_terminal_reward_B = np.where(alive, team_terminal_reward_B, 0.0)
        else:
            alive = None

        vec_rewards_T_B = np.zeros((T, B, 1), dtype=np.float32)
        vec_rewards_T_B[last_active_idx_env, np.arange(B), 0] += team_terminal_reward_B

        mc_returns_T_B = _discounted_returns(vec_rewards_T_B, self.gamma)

        self.env_latency_cost_accumulator += latency_cost_B
        self.env_parallel_cost_accumulator += parallel_cost_B
        self.env_reward_accumulator += team_terminal_reward_B

        self.max_coverage = np.maximum(self.max_coverage, max_coverage_B)

        raw_components_B = None
        if self.store_raw_components:
            norm_excess_B = (excess_compute_B / safe_L_B).astype(np.float32)
            has_f = has_finished.astype(np.float32)
            raw_components_B = np.stack(
                [
                    max_adv_finished_B,
                    has_f,
                    L_B.astype(np.float32),
                    norm_excess_B,
                ],
                axis=1,
            )

        return self.transfer_transitions(
            transitions, mc_returns_T_B, buffer, raw_components_B, env_mask=alive
        )

    def transfer_transitions(
        self, transitions, mc_returns, buffer, raw_components_B=None, env_mask=None
    ):
        added_count = 0
        B = self.n_envs
        episode_idx = getattr(self, "_episode_idx", 0)
        alive = (
            np.ones((B,), dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for t, tr_data in enumerate(transitions):
            # State obs is shared across S; sample obs is concatenated across S for the centralized critic.
            state_col = tr_data["state_obs"][:, 0, :]
            sample_col = tr_data["sample_obs"].reshape(B, -1)
            obs_np = np.concatenate([state_col, sample_col], axis=-1)

            next_state_col = tr_data["next_state_obs"][:, 0, :]
            next_sample_col = tr_data["next_sample_obs"].reshape(B, -1)
            next_obs_np = np.concatenate([next_state_col, next_sample_col], axis=-1)

            step_team_return = mc_returns[t]

            group_active = np.any(tr_data["running"], axis=1)
            for i in range(B):
                if not alive[i]:
                    continue
                if group_active[i] or t == len(transitions) - 1:
                    buffer.add(
                        obs_np[i : i + 1],
                        next_obs_np[i : i + 1],
                        tr_data["action"][i : i + 1],
                        step_team_return[i : i + 1],
                    )
                    slot = (buffer.pos - 1) % buffer.buffer_size
                    if raw_components_B is not None and hasattr(buffer, "reward_components"):
                        buffer.reward_components[slot] = raw_components_B[i]
                    if hasattr(buffer, "traj_info"):
                        buffer.traj_info[slot] = [episode_idx, self._traj_step]
                    added_count += 1

        self._traj_step += 1
        self.reset_denoise_episode_state()
        return added_count
