"""SAC update for the stride actor-critic, shared by every backend.

The critic regresses Monte-Carlo chunk returns directly (no TD bootstrap, no target network).
Batch sampling belongs to the backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from elastic.core.actor import StrideActorCritic


@dataclass
class SACConfig:
    num_samples: int
    target_entropy: float
    pause_threshold: float
    stopped_index: int = -1
    critic_clip: float = 1.0
    actor_clip: float = 2.0
    log_alpha_max: float = 3.0


def make_optimizers(policy: StrideActorCritic, *, actor_lr: float, critic_lr: float, alpha_lr: float | None, device):
    """Adam (eps 1e-4) on the actor and critic halves, and on log_alpha (initialised to -4) unless ``alpha_lr`` is None."""
    actor_opt = torch.optim.Adam(policy.actor_parameters(), lr=actor_lr, eps=1e-4)
    critic_opt = torch.optim.Adam(policy.critic_parameters(), lr=critic_lr, eps=1e-4)
    log_alpha = torch.tensor([-4.0], device=device, requires_grad=True)
    alpha_opt = None if alpha_lr is None else torch.optim.Adam([log_alpha], lr=alpha_lr, eps=1e-4)
    return actor_opt, critic_opt, log_alpha, alpha_opt


def split_obs(obs: torch.Tensor, state_dim: int, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat replay rows [state | S x sample] -> (state (B, S, D_state), sample (B, S, D_sample))."""
    b = obs.shape[0]
    sample = obs[:, state_dim:].reshape(b, num_samples, -1)
    if state_dim == 0:
        state = obs.new_zeros(b, num_samples, 0)
    else:
        state = obs[:, :state_dim].reshape(b, 1, -1).expand(-1, num_samples, -1)
    return state, sample


def _frac(values: torch.Tensor, pred) -> float:
    return float(pred(values).float().mean().item()) if values.numel() > 0 else 0.0


class StrideSAC:
    def __init__(
        self,
        cfg: SACConfig,
        policy: StrideActorCritic,
        actor_opt: torch.optim.Optimizer,
        critic_opt: torch.optim.Optimizer,
        log_alpha: torch.Tensor,
        alpha_opt: torch.optim.Optimizer | None,
    ):
        self.cfg = cfg
        self.policy = policy
        self.actor_params = policy.actor_parameters()
        self.critic_params = policy.critic_parameters()
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.log_alpha = log_alpha
        self.alpha_opt = alpha_opt

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def update(
        self,
        state: torch.Tensor,
        sample: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        *,
        is_weights: torch.Tensor | None = None,
        update_actor: bool = True,
        alpha: float | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """One critic (+ actor + alpha) step. Returns (metrics, q1, q2) with q1/q2 from before the step.

        ``alpha`` defaults to the current value; pass the one read before batch sampling
        when it must match an earlier read exactly.
        """
        cfg = self.cfg
        alpha = self.alpha if alpha is None else alpha
        if actions.ndim == 2:
            actions = actions.unsqueeze(-1)

        q1 = self.policy.forward_critic1(state, sample, actions)
        q2 = self.policy.forward_critic2(state, sample, actions)
        critic1_per = (q1 - rewards).pow(2)
        critic2_per = (q2 - rewards).pow(2)
        if is_weights is not None:
            critic1_per = critic1_per * is_weights
            critic2_per = critic2_per * is_weights
        critic_loss = critic1_per.mean() + critic2_per.mean()
        with torch.no_grad():
            q_disagreement_data = float((q1 - q2).abs().mean().item())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_params, max_norm=cfg.critic_clip)
        self.critic_opt.step()

        m = {
            "critic_loss": critic_loss.item(),
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "entropy": 0.0,
            "entropy_cash_value": 0.0,
            "actor_grad_norm": 0.0,
            "min_q_pi_mean": 0.0,
            "mu_mean": 0.0,
            "mu_std_across_batch": 0.0,
            "mu_std_per_sample": 0.0,
            "q_disagreement_pi": 0.0,
            "alpha_lp_per_sample_mean": 0.0,
            "alpha_lp_plus_target_mean": 0.0,
            "alpha_raw_grad": 0.0,
            "policy_action_abs_mean": 0.0,
            "policy_action_std": 0.0,
            **{f"{p}_frac_{band}_band": 0.0 for p in ("policy", "policy_mu") for band in ("low", "pause", "mid")},
        }
        min_q_pi = torch.tensor(0.0)
        if update_actor:
            for p in self.critic_params:
                p.requires_grad = False
            running = sample[:, :, cfg.stopped_index] < 0.5
            new_action, log_prob, mu = self.policy.sample_action(state, sample, running_mask=running)
            q1_pi = self.policy.forward_critic1(state, sample, new_action)
            q2_pi = self.policy.forward_critic2(state, sample, new_action)
            min_q_pi = torch.min(q1_pi, q2_pi)
            with torch.no_grad():
                m["q_disagreement_pi"] = float((q1_pi - q2_pi).abs().mean().item())
            # Rows with no running sample carry no actor signal. Normalizing by the fixed S
            # keeps the entropy scale (and alpha's target) independent of how many still run.
            any_running = running.any(dim=1).float()
            n_rows = any_running.sum().clamp(min=1.0)
            log_prob_norm = log_prob / float(cfg.num_samples)
            actor_loss = ((alpha * log_prob_norm - min_q_pi).squeeze(-1) * any_running).sum() / n_rows
            self.actor_opt.zero_grad()
            actor_loss.backward()
            m["actor_grad_norm"] = clip_grad_norm_(self.actor_params, max_norm=cfg.actor_clip).item()
            self.actor_opt.step()
            for p in self.critic_params:
                p.requires_grad = True
            m["actor_loss"] = actor_loss.item()

            if self.alpha_opt is not None:
                inner = (log_prob_norm + cfg.target_entropy).detach() * any_running.unsqueeze(-1)
                m["alpha_lp_per_sample_mean"] = float(
                    (log_prob_norm * any_running.unsqueeze(-1)).sum().item() / n_rows.item()
                )
                m["alpha_lp_plus_target_mean"] = float(inner.sum().item() / n_rows.item())
                alpha_loss = -(self.log_alpha * inner).sum() / n_rows
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                m["alpha_raw_grad"] = float(self.log_alpha.grad.item()) if self.log_alpha.grad is not None else 0.0
                self.alpha_opt.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(max=cfg.log_alpha_max)
                m["alpha_loss"] = alpha_loss.item()

            entropy = -log_prob.mean()
            m["entropy"] = entropy.item()
            m["entropy_cash_value"] = alpha * entropy.item()
            m["mu_mean"] = mu.mean().item()
            m["mu_std_across_batch"] = mu.std().item()
            m["mu_std_per_sample"] = mu.std(dim=0).mean().item()
            with torch.no_grad():
                mask = running.unsqueeze(-1).expand_as(new_action)
                for prefix, acts in (("policy", new_action[mask]), ("policy_mu", torch.tanh(mu)[mask])):
                    m[f"{prefix}_frac_low_band"] = _frac(acts, lambda x: x <= -0.9)
                    m[f"{prefix}_frac_pause_band"] = _frac(acts, lambda x: x >= cfg.pause_threshold)
                    m[f"{prefix}_frac_mid_band"] = _frac(acts, lambda x: (x > -0.1) & (x < 0.1))
                running_actions = new_action[mask]
                if running_actions.numel() > 0:
                    m["policy_action_abs_mean"] = float(running_actions.abs().mean().item())
                    m["policy_action_std"] = float(running_actions.std(unbiased=False).item())

        m.update(
            alpha=alpha,
            q1_mean=q1.mean().item(),
            q2_mean=q2.mean().item(),
            min_q_pi_mean=min_q_pi.mean().item(),
            q_overestimation=(min_q_pi.mean() - q1.mean()).item(),
            q_bias=(q1.mean() - rewards.mean()).item(),
            q_disagreement_data=q_disagreement_data,
        )
        return m, q1, q2
