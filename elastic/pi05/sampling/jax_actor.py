from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from elastic.core.action import FLOW_MIN_STRIDE
from elastic.core.actor import StrideActorCritic
from elastic.pi05.constants import PAUSE_ACTION_THRESHOLD
from elastic.pi05.training.learner import load_stride_policy


@dataclass(frozen=True)
class JaxStrideActorSpec:
    state_dim: int
    sample_dim: int
    num_samples: int
    hidden: int
    action_dim: int = 1
    num_heads: int = 4
    film_relu: bool = True

    @property
    def blind_mode(self) -> bool:
        return self.state_dim == 0


def load_jax_stride_actor(path: str, *, device: str | torch.device = "cpu") -> tuple[JaxStrideActorSpec, dict[str, Any]]:
    """Load actor-only JAX params from a Torch stride SAC checkpoint."""
    return jax_actor_from_torch_policy(load_stride_policy(path, device=device))


def jax_actor_from_torch_policy(policy: StrideActorCritic) -> tuple[JaxStrideActorSpec, dict[str, Any]]:
    spec = JaxStrideActorSpec(
        state_dim=int(policy.state_dim),
        sample_dim=int(policy.sample_dim),
        num_samples=int(policy.num_samples),
        hidden=int(policy.actor_mu.in_features),
        action_dim=int(policy.action_dim),
        film_relu=bool(policy.film_relu),
    )
    sd = policy.state_dict()
    params = {
        "ln_state_actor": _layer_norm(sd, "ln_state_actor"),
        "ln_sample_actor": _layer_norm(sd, "ln_sample_actor"),
        "state_encoder_actor": _mlp(sd, "state_encoder_actor"),
        "sample_encoder_actor": _mlp(sd, "sample_encoder_actor"),
        "actor_film": _mlp(sd, "actor_film"),
        "actor_cross_attn": _mha(sd, "actor_cross_attn"),
        "decoder_actor": _mlp(sd, "decoder_actor"),
        "actor_mu": _linear(sd, "actor_mu"),
    }
    return spec, params


def jax_actor_deterministic_action(
    params: dict[str, Any],
    spec: JaxStrideActorSpec,
    state_obs: jax.Array,
    sample_obs: jax.Array,
    running_mask: jax.Array | None = None,
) -> jax.Array:
    """Deterministic Torch-equivalent action: tanh(actor_mu(...))."""
    z = _encode_actor_joint(params, spec, state_obs, sample_obs)
    mu = _linear_apply(z, params["actor_mu"])
    action = jnp.tanh(mu)
    if running_mask is not None:
        mask = running_mask.astype(bool)[..., None]
        action = jnp.where(mask, action, jnp.ones_like(action))
    return action


def action_to_stride_jax(
    action: jax.Array,
    action_range: tuple[float, float],
    pause_threshold: float = PAUSE_ACTION_THRESHOLD,
    active_eps: float = 1e-3,
) -> jax.Array:
    """JAX twin of ``elastic.core.action.action_to_stride`` in flow time (min stride ``FLOW_MIN_STRIDE``)."""
    low, high = action_range
    active_low = max(float(low), FLOW_MIN_STRIDE)
    active_high = max(float(high), active_low)
    active_upper = float(pause_threshold) - float(active_eps)
    active_frac = jnp.clip((action + 1.0) / max(1e-8, active_upper + 1.0), 0.0, 1.0)
    stride = active_low + active_frac * (active_high - active_low)
    return jnp.where(action >= pause_threshold, jnp.zeros_like(stride), stride).astype(jnp.float32)


def _encode_actor_joint(
    params: dict[str, Any],
    spec: JaxStrideActorSpec,
    state_obs: jax.Array,
    sample_obs: jax.Array,
) -> jax.Array:
    z_sample = _mlp_apply(_layer_norm_apply(sample_obs, params["ln_sample_actor"]), params["sample_encoder_actor"])
    if spec.blind_mode:
        z_ctx = _mha_apply(z_sample, z_sample, z_sample, params["actor_cross_attn"], spec.num_heads)
        z_ctx = z_ctx + z_sample
        z_pool = jnp.broadcast_to(jnp.mean(z_ctx, axis=1, keepdims=True), z_ctx.shape)
        z_relative = z_ctx - z_pool
        z_item = jnp.concatenate([z_ctx, z_pool, z_relative], axis=-1)
    else:
        z_state_in = state_obs[:, 0, :]
        z_state = _mlp_apply(_layer_norm_apply(z_state_in, params["ln_state_actor"]), params["state_encoder_actor"])
        gamma_raw, beta_raw = jnp.split(_mlp_apply(z_state, params["actor_film"], final_relu=spec.film_relu), 2, axis=-1)
        gamma = jnp.tanh(gamma_raw)[:, None, :]
        beta = jnp.tanh(beta_raw)[:, None, :]
        z_sample = z_sample * (1.0 + gamma) + beta
        z_query = z_sample + z_state[:, None, :]
        z_ctx = _mha_apply(z_query, z_sample, z_sample, params["actor_cross_attn"], spec.num_heads)
        z_ctx = z_ctx + z_sample
        z_state_rep = jnp.broadcast_to(z_state[:, None, :], z_ctx.shape)
        z_pool = jnp.broadcast_to(jnp.mean(z_ctx, axis=1, keepdims=True), z_ctx.shape)
        z_relative = z_ctx - z_pool
        z_item = jnp.concatenate([z_state_rep, z_ctx, z_pool, z_relative], axis=-1)
    return _mlp_apply(z_item, params["decoder_actor"])


def _layer_norm_apply(x: jax.Array, params: dict[str, jax.Array], eps: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * params["weight"] + params["bias"]


def _linear_apply(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    return x @ params["weight"].T + params["bias"]


def _mlp_apply(x: jax.Array, layers: tuple[dict[str, jax.Array], dict[str, jax.Array]], final_relu: bool = True) -> jax.Array:
    x = _linear_apply(jax.nn.relu(_linear_apply(x, layers[0])), layers[1])
    return jax.nn.relu(x) if final_relu else x


def _mha_apply(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    params: dict[str, jax.Array],
    num_heads: int,
) -> jax.Array:
    hidden = query.shape[-1]
    head_dim = hidden // int(num_heads)
    q_w, k_w, v_w = jnp.split(params["in_proj_weight"], 3, axis=0)
    q_b, k_b, v_b = jnp.split(params["in_proj_bias"], 3, axis=0)
    q = query @ q_w.T + q_b
    k = key @ k_w.T + k_b
    v = value @ v_w.T + v_b

    def split_heads(x):
        b, s, h = x.shape
        return x.reshape(b, s, int(num_heads), head_dim).transpose(0, 2, 1, 3)

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
    weights = jax.nn.softmax((qh @ jnp.swapaxes(kh, -1, -2)) / np.sqrt(float(head_dim)), axis=-1)
    attended = (weights @ vh).transpose(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], hidden)
    return _linear_apply(attended, params["out_proj"])


def _tensor(value: torch.Tensor) -> jax.Array:
    return jnp.asarray(value.detach().cpu().numpy(), dtype=jnp.float32)


def _linear(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, jax.Array]:
    return {"weight": _tensor(sd[f"{prefix}.weight"]), "bias": _tensor(sd[f"{prefix}.bias"])}


def _layer_norm(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, jax.Array]:
    return {"weight": _tensor(sd[f"{prefix}.weight"]), "bias": _tensor(sd[f"{prefix}.bias"])}


def _mlp(sd: dict[str, torch.Tensor], prefix: str) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    return _linear(sd, f"{prefix}.0"), _linear(sd, f"{prefix}.2")


def _mha(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, Any]:
    return {
        "in_proj_weight": _tensor(sd[f"{prefix}.in_proj_weight"]),
        "in_proj_bias": _tensor(sd[f"{prefix}.in_proj_bias"]),
        "out_proj": _linear(sd, f"{prefix}.out_proj"),
    }
