"""Diffusion Policy batch sampling (PER online + decaying offline mix) around the shared SAC update.

The polyak target copy is only saved in checkpoints; the loss never reads it.
"""
from __future__ import annotations

import types

import numpy as np
import torch

from elastic.core.sac import SACConfig, StrideSAC, split_obs

_UNPREFIXED = {"critic_loss", "actor_loss", "entropy", "q1_mean", "q2_mean", "alpha"}
_DEBUG = {"alpha_lp_per_sample_mean", "alpha_lp_plus_target_mean", "alpha_raw_grad", "q_disagreement_data", "q_disagreement_pi"}


def _log_name(key: str) -> str:
    if key in _UNPREFIXED:
        return key
    if key in _DEBUG or key.startswith("policy_"):
        return f"debug/{key}"
    return f"train/{key}"


class DiffusionPolicySAC:
    def __init__(
        self,
        *,
        policy,
        target,
        optimizer_actor,
        optimizer_critic,
        log_alpha,
        alpha_optimizer,
        buffer,
        offline_buffer,
        online_priority,
        pause_action_threshold,
        device,
        batch_size,
        num_samples,
        obs_features_dim,
        total_steps,
        offline_batch_frac,
        offline_decay_steps,
        per_alpha,
        per_beta_start,
        per_beta_end,
        per_eps,
        target_entropy,
        tau,
    ):
        self.sac = StrideSAC(
            SACConfig(num_samples=num_samples, target_entropy=target_entropy, pause_threshold=float(pause_action_threshold)),
            policy, optimizer_actor, optimizer_critic, log_alpha, alpha_optimizer,
        )
        self.policy = policy
        self.target = target
        self.buffer = buffer
        self.offline_buffer = offline_buffer
        self.online_priority = online_priority
        self.device = device
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.obs_features_dim = obs_features_dim
        self.total_steps = total_steps
        self.offline_batch_frac = offline_batch_frac
        self.offline_decay_steps = offline_decay_steps
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_eps = per_eps
        self.tau = tau

    def _sample_online(self, n: int, online_step: int):
        """PER draw of ``n`` online rows -> (indices, IS weights (n, 1), stats)."""
        size = self.buffer.size()
        if size <= 0:
            raise ValueError("Cannot sample online replay from empty buffer")
        progress = min(max(float(online_step) / float(max(self.total_steps - 1, 1)), 0.0), 1.0)
        beta = self.per_beta_start + (self.per_beta_end - self.per_beta_start) * progress
        priority = np.maximum(self.online_priority[:size], self.per_eps)
        probs = np.power(priority, self.per_alpha)
        probs_sum = float(np.sum(probs))
        if (not np.isfinite(probs_sum)) or probs_sum <= 0.0:
            probs = np.full((size,), 1.0 / float(size), dtype=np.float32)
        else:
            probs = probs / probs_sum
        idx = np.random.choice(size, size=n, replace=True, p=probs).astype(np.int64)
        is_w = np.power(size * probs[idx], -beta)
        is_w /= np.max(is_w)
        stats = {"per_beta": beta, "per_isw_mean": float(np.mean(is_w)), "per_priority_mean": float(np.mean(priority[idx]))}
        return idx, torch.from_numpy(is_w.astype(np.float32)).to(self.device).unsqueeze(-1), stats

    def _offline_count(self, online_step: int) -> int:
        if self.offline_buffer.size() <= 0:
            return 0
        frac = self.offline_batch_frac
        if self.offline_decay_steps > 0:
            frac = frac * max(0.0, 1.0 - float(online_step) / float(self.offline_decay_steps))
        return int(self.batch_size * frac)

    def train_step(self, step_idx, update_actor=True, online_step=0):
        alpha = self.sac.alpha
        self.buffer.sample(self.batch_size)

        n_offline = self._offline_count(online_step)
        n_online = self.batch_size - n_offline
        idx, is_w, per_stats = self._sample_online(n_online, online_step)
        data = self.buffer._get_samples(idx)
        if n_offline > 0:
            off = self.offline_buffer.sample(n_offline)
            data = types.SimpleNamespace(
                observations=torch.cat([data.observations, off.observations], dim=0),
                actions=torch.cat([data.actions, off.actions], dim=0),
                rewards=torch.cat([data.rewards, off.rewards], dim=0),
            )
            is_w = torch.cat([is_w, torch.ones((n_offline, 1), device=self.device)], dim=0)

        state, sample = split_obs(data.observations, self.obs_features_dim, self.num_samples)
        metrics, q1, q2 = self.sac.update(
            state, sample, data.actions, data.rewards, is_weights=is_w, update_actor=update_actor, alpha=alpha
        )

        with torch.no_grad():
            td_err = 0.5 * (q1 - data.rewards).abs() + 0.5 * (q2 - data.rewards).abs()
            self.online_priority[idx] = np.maximum(td_err[:n_online].squeeze(-1).detach().cpu().numpy(), self.per_eps)

        with torch.no_grad():
            for p, tp in zip(self.policy.parameters(), self.target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        metrics.update(
            per_stats,
            actor_lr=self.sac.actor_opt.param_groups[0]["lr"],
            critic_lr=self.sac.critic_opt.param_groups[0]["lr"],
        )
        return {_log_name(k): v for k, v in metrics.items()}
