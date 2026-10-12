"""ELASTIC stride actor-critic over the S parallel denoising samples of one chunk, shared by every backend.

The actor emits one tanh-squashed stride action per sample; the twin critic scores the joint
(state, samples, actions). ``state_dim == 0`` is blind mode (no state input).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

ATTN_HEADS = 4


def _encoder(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())


def _film(hidden: int, final_relu: bool) -> nn.Sequential:
    layers = [nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * hidden)]
    return nn.Sequential(*layers, nn.ReLU()) if final_relu else nn.Sequential(*layers)


class StrideActorCritic(nn.Module):
    """``film_relu`` adds a ReLU after the FiLM projection (pi0.5 checkpoints were trained with it).
    It has no parameters, so it does not show in the state dict and must travel in the config."""

    # Parameter order fixes the optimizer state layout of saved checkpoints.
    CRITIC_MODULES = (
        "ln_state_critic", "ln_sample_critic", "state_encoder_critic", "sample_encoder_critic",
        "critic_film", "critic_item_encoder", "critic_cross_attn", "decoder_critic",
        "critic1_head", "critic2_head",
    )
    ACTOR_MODULES = (
        "ln_state_actor", "ln_sample_actor", "state_encoder_actor", "actor_film",
        "sample_encoder_actor", "actor_cross_attn", "decoder_actor", "actor_mu", "actor_log_std",
    )

    def __init__(
        self,
        *,
        state_dim: int,
        sample_dim: int,
        num_samples: int,
        hidden: int,
        action_dim: int = 1,
        film_relu: bool = False,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.sample_dim = int(sample_dim)
        self.num_samples = int(num_samples)
        self.action_dim = int(action_dim)
        self.film_relu = bool(film_relu)
        self.blind_mode = self.state_dim == 0
        self.log_std_min = -20.0
        self.log_std_max = -2.0
        net_state_dim = max(self.state_dim, 1)
        state_feats = 0 if self.blind_mode else hidden

        self.ln_state_actor = nn.LayerNorm(net_state_dim)
        self.ln_sample_actor = nn.LayerNorm(self.sample_dim)
        self.ln_state_critic = nn.LayerNorm(net_state_dim)
        self.ln_sample_critic = nn.LayerNorm(self.sample_dim)

        self.state_encoder_actor = _encoder(net_state_dim, hidden)
        self.actor_film = _film(hidden, self.film_relu)
        self.critic_film = _film(hidden, self.film_relu)
        self.sample_encoder_actor = _encoder(self.sample_dim, hidden)
        self.actor_cross_attn = nn.MultiheadAttention(hidden, ATTN_HEADS, batch_first=True)
        self.decoder_actor = _encoder(state_feats + 3 * hidden, hidden)
        self.actor_mu = nn.Linear(hidden, self.action_dim)
        self.actor_log_std = nn.Linear(hidden, self.action_dim)
        if self.blind_mode:
            nn.init.constant_(self.actor_mu.bias, -3.0)

        self.state_encoder_critic = _encoder(net_state_dim, hidden)
        self.sample_encoder_critic = _encoder(self.sample_dim, hidden)
        self.critic_item_encoder = _encoder(hidden + self.action_dim, hidden)
        self.critic_cross_attn = nn.MultiheadAttention(hidden, ATTN_HEADS, batch_first=True)
        self.decoder_critic = _encoder(state_feats + 3 * hidden, hidden)
        self.critic1_head = nn.Linear(hidden, 1)
        self.critic2_head = nn.Linear(hidden, 1)

    def critic_parameters(self):
        return [p for name in self.CRITIC_MODULES for p in getattr(self, name).parameters()]

    def actor_parameters(self):
        return [p for name in self.ACTOR_MODULES for p in getattr(self, name).parameters()]

    def _check_joint_shapes(self, state_obs: torch.Tensor, sample_obs: torch.Tensor) -> None:
        if state_obs.ndim != 3 or sample_obs.ndim != 3:
            raise ValueError(f"expected (B, S, D) inputs, got state={tuple(state_obs.shape)}, sample={tuple(sample_obs.shape)}")
        if state_obs.shape[1] != sample_obs.shape[1]:
            raise ValueError(f"state/sample S mismatch: {state_obs.shape[1]} vs {sample_obs.shape[1]}")
        if sample_obs.shape[1] != self.num_samples:
            raise ValueError(f"expected S={self.num_samples}, got S={sample_obs.shape[1]}")

    @staticmethod
    def _apply_film(film: nn.Module, z_state: torch.Tensor, z_sample: torch.Tensor) -> torch.Tensor:
        gamma_raw, beta_raw = torch.chunk(film(z_state), 2, dim=-1)
        gamma = torch.tanh(gamma_raw).unsqueeze(1)
        beta = torch.tanh(beta_raw).unsqueeze(1)
        return z_sample * (1.0 + gamma) + beta

    def _encode_actor_joint(self, state_obs: torch.Tensor, sample_obs: torch.Tensor) -> torch.Tensor:
        self._check_joint_shapes(state_obs, sample_obs)
        B, S, _ = sample_obs.shape
        if self.blind_mode:
            z_sample = self.sample_encoder_actor(self.ln_sample_actor(sample_obs))
            z_query = z_sample
            parts = []
        else:
            z_state = self.state_encoder_actor(self.ln_state_actor(state_obs[:, 0, :]))
            z_sample = self.sample_encoder_actor(self.ln_sample_actor(sample_obs))
            z_sample = self._apply_film(self.actor_film, z_state, z_sample)
            z_query = z_sample + z_state.unsqueeze(1)
            parts = [z_state.unsqueeze(1).expand(B, S, -1)]
        z_ctx, _ = self.actor_cross_attn(query=z_query, key=z_sample, value=z_sample, need_weights=False)
        z_ctx = z_ctx + z_sample
        z_pool = z_ctx.mean(dim=1, keepdim=True).expand(B, S, -1)
        z_item = torch.cat([*parts, z_ctx, z_pool, z_ctx - z_pool], dim=-1)
        return self.decoder_actor(z_item)

    def _encode_critic_joint(self, state_obs: torch.Tensor, sample_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self._check_joint_shapes(state_obs, sample_obs)
        if action.ndim != 3 or action.shape[1] != self.num_samples or action.shape[2] != self.action_dim:
            raise ValueError(f"expected action (B, {self.num_samples}, {self.action_dim}), got {tuple(action.shape)}")
        if self.blind_mode:
            z_sample = self.sample_encoder_critic(self.ln_sample_critic(sample_obs))
            z_item = self.critic_item_encoder(torch.cat([z_sample, action], dim=-1))
            z_query = z_item.mean(dim=1, keepdim=True)
            parts = []
        else:
            z_state = self.state_encoder_critic(self.ln_state_critic(state_obs[:, 0, :]))
            z_sample = self.sample_encoder_critic(self.ln_sample_critic(sample_obs))
            z_sample = self._apply_film(self.critic_film, z_state, z_sample)
            z_item = self.critic_item_encoder(torch.cat([z_sample, action], dim=-1))
            z_query = z_state.unsqueeze(1)
            parts = [z_state]
        z_attn, _ = self.critic_cross_attn(query=z_query, key=z_item, value=z_item, need_weights=False)
        z_joint = torch.cat([*parts, z_attn.squeeze(1), z_item.mean(dim=1), z_item.max(dim=1).values], dim=-1)
        return self.decoder_critic(z_joint)

    def forward_actor(self, state_obs: torch.Tensor, sample_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._encode_actor_joint(state_obs, sample_obs)
        log_std = torch.clamp(self.actor_log_std(z), self.log_std_min, self.log_std_max)
        return self.actor_mu(z), log_std

    def forward_critic1(self, state_obs: torch.Tensor, sample_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic1_head(self._encode_critic_joint(state_obs, sample_obs, action))

    def forward_critic2(self, state_obs: torch.Tensor, sample_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic2_head(self._encode_critic_joint(state_obs, sample_obs, action))

    def sample_action(
        self,
        state_obs: torch.Tensor,
        sample_obs: torch.Tensor,
        *,
        deterministic: bool = False,
        running_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob summed over samples, mu).

        ``running_mask`` (B, S): stopped samples get the canonical stop action +1 and
        drop out of the log-prob, so they carry no actor or entropy gradient.
        """
        mu, log_std = self.forward_actor(state_obs, sample_obs)
        dist = Normal(mu, log_std.exp())
        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)
        log_prob_per = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        if running_mask is not None:
            mask = running_mask.to(dtype=torch.bool, device=a.device).unsqueeze(-1).expand_as(a)
            a = torch.where(mask, a, torch.ones_like(a).detach())
            log_prob_per = log_prob_per * mask.float()
        log_prob = log_prob_per.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return a, log_prob, mu
