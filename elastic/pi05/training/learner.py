"""pi0.5 batch sampling and checkpoints around the shared SAC update.

Online and offline replay are both sampled uniformly; the offline share of each
batch decays linearly to 0 over ``offline_decay_steps`` online steps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch

from elastic.core.actor import StrideActorCritic
from elastic.core.sac import SACConfig, StrideSAC, make_optimizers, split_obs
from elastic.pi05.constants import BATCH_SIZE, PAUSE_ACTION_THRESHOLD

# The stopped flag's position inside a pi0.5 sample observation block.
SOBS_STOPPED = -3


@dataclass
class SACLearnerConfig:
    state_dim: int
    sample_dim: int
    num_samples: int
    hidden: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 1e-3
    batch_size: int = BATCH_SIZE
    target_entropy: float = -2.0
    offline_fraction: float = 0.2
    offline_decay_steps: int = 5000
    film_relu: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SACLearnerConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(raw).items() if k in names})


def build_policy(config: SACLearnerConfig) -> StrideActorCritic:
    return StrideActorCritic(
        state_dim=config.state_dim,
        sample_dim=config.sample_dim,
        num_samples=config.num_samples,
        hidden=config.hidden,
        film_relu=config.film_relu,
    )


def load_stride_policy(path: str, *, device: str | torch.device = "cpu") -> StrideActorCritic:
    state = torch.load(path, map_location=device)
    policy = build_policy(SACLearnerConfig.from_dict(state["config"]))
    policy.load_state_dict(state["policy"])
    return policy.to(device).eval()


class StrideSACLearner:
    def __init__(self, config: SACLearnerConfig, *, device: str | torch.device = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.policy = build_policy(config).to(self.device)
        actor_opt, critic_opt, log_alpha, alpha_opt = make_optimizers(
            self.policy, actor_lr=config.actor_lr, critic_lr=config.critic_lr, alpha_lr=config.alpha_lr, device=self.device
        )
        self.sac = StrideSAC(
            SACConfig(
                num_samples=config.num_samples,
                target_entropy=config.target_entropy,
                pause_threshold=PAUSE_ACTION_THRESHOLD,
                stopped_index=SOBS_STOPPED,
            ),
            self.policy, actor_opt, critic_opt, log_alpha, alpha_opt,
        )

    def _offline_fraction(self, override: float | None, online_step: int) -> float:
        base = self.config.offline_fraction if override is None else float(override)
        decay = max(0.0, 1.0 - float(online_step) / max(float(self.config.offline_decay_steps), 1.0))
        return max(0.0, base * decay)

    def train_step(
        self,
        replay,
        *,
        offline_replay=None,
        offline_fraction: float | None = None,
        online_step: int = 0,
        update_actor: bool = True,
    ) -> dict[str, float]:
        fraction = self._offline_fraction(offline_fraction, online_step)
        batch_size = self.config.batch_size
        if offline_replay is None or len(offline_replay) == 0 or fraction <= 0.0:
            batch = replay.sample(batch_size)
        else:
            offline_n = max(0, min(int(batch_size * fraction), batch_size - 1))
            online, offline = replay.sample(batch_size - offline_n), offline_replay.sample(offline_n)
            batch = type(online)(**{
                f.name: np.concatenate([getattr(online, f.name), getattr(offline, f.name)], axis=0)
                for f in fields(online)
            })
        return self.train_batch(batch, update_actor=update_actor)

    def train_batch(self, batch, *, update_actor: bool = True) -> dict[str, float]:
        def t(x):
            return torch.from_numpy(x).float().to(self.device)

        state, sample = split_obs(t(batch.obs), self.config.state_dim, self.config.num_samples)
        metrics, _, _ = self.sac.update(state, sample, t(batch.actions), t(batch.rewards), update_actor=update_actor)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "policy": self.policy.state_dict(),
            "actor_optimizer": self.sac.actor_opt.state_dict(),
            "critic_optimizer": self.sac.critic_opt.state_dict(),
            "alpha_optimizer": self.sac.alpha_opt.state_dict(),
            "log_alpha": self.sac.log_alpha.detach().cpu(),
        }

    @classmethod
    def from_checkpoint(cls, path: str, *, device: str | torch.device = "cpu") -> "StrideSACLearner":
        state = torch.load(path, map_location=device)
        learner = cls(SACLearnerConfig.from_dict(state["config"]), device=device)
        learner.policy.load_state_dict(state["policy"])
        learner.sac.actor_opt.load_state_dict(state["actor_optimizer"])
        learner.sac.critic_opt.load_state_dict(state["critic_optimizer"])
        learner.sac.alpha_opt.load_state_dict(state["alpha_optimizer"])
        with torch.no_grad():
            learner.sac.log_alpha.copy_(state["log_alpha"].to(learner.device))
        return learner
