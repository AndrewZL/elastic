"""pi0.5 with the adaptive flow-matching stride samplers.

``adopt_model`` swaps an upstream ``Pi0`` to ``ElasticPi0`` (methods only, params untouched).
Swap before any ``nnx_utils.module_jit`` so the jitted graph records this class.
"""
from typing import Any

import einops
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.shared import array_typing as at

from elastic.pi05.sampling.jax_actor import JaxStrideActorSpec, action_to_stride_jax, jax_actor_deterministic_action


class ElasticPi0(Pi0):
    def sample_actions_parallel(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        num_samples: int = 10,
    ) -> _model.Actions:
        """Sample multiple action chunks for a single observation using one shared prefix pass."""
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        if observation.state.shape[0] != 1:
            raise ValueError(
                f"sample_actions_parallel expects a single observation with batch size 1, got {observation.state.shape[0]}"
            )

        dt = -1.0 / num_steps

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions_prefix)

        prefix_mask_n = jnp.broadcast_to(prefix_mask, (num_samples,) + prefix_mask.shape[1:])
        prefix_tokens_len = prefix_tokens.shape[1]
        kv_cache_n = jax.tree.map(
            lambda x: jnp.broadcast_to(x, x.shape[:1] + (num_samples,) + x.shape[2:]),
            kv_cache,
        )
        observation_n = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_samples,) + x.shape[1:])
            if hasattr(x, "shape") and x.shape[:1] == (1,)
            else x,
            observation,
        )

        noise = jax.random.normal(rng, (num_samples, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation_n, x_t, jnp.broadcast_to(time, num_samples)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask_n, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                num_samples,
                suffix_tokens.shape[1],
                prefix_tokens_len + suffix_tokens.shape[1],
            )

            positions_suffix = jnp.sum(prefix_mask_n, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            prefix_out, suffix_out = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache_n,
                adarms_cond=[None, adarms_cond],
            )[0]
            assert prefix_out is None

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_fixed_stride_trace(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        fixed_stride: float | at.Float[at.Array, ""] = 0.1,
        num_steps: int = 10,
        num_samples: int = 10,
    ):
        """Fixed-stride denoising in one jitted loop, returning the per-step selector trace."""
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        if observation.state.shape[0] != 1:
            raise ValueError(
                f"sample_actions_fixed_stride_trace expects a single observation with batch size 1, got {observation.state.shape[0]}"
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions_prefix)

        prefix_mask_n = jnp.broadcast_to(prefix_mask, (num_samples,) + prefix_mask.shape[1:])
        prefix_tokens_len = prefix_tokens.shape[1]
        kv_cache_n = jax.tree.map(
            lambda x: jnp.broadcast_to(x, x.shape[:1] + (num_samples,) + x.shape[2:]),
            kv_cache,
        )
        observation_n = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_samples,) + x.shape[1:])
            if hasattr(x, "shape") and x.shape[:1] == (1,)
            else x,
            observation,
        )

        prefix_weights = prefix_mask.astype(prefix_tokens.dtype)[..., None]
        prefix_context = jnp.sum(prefix_tokens * prefix_weights, axis=1) / jnp.maximum(
            jnp.sum(prefix_weights, axis=1), 1.0
        )
        prefix_context_n = jnp.broadcast_to(prefix_context, (num_samples,) + prefix_context.shape[1:])

        x_t = jax.random.normal(rng, (num_samples, self.action_horizon, self.action_dim))
        t_idx = jnp.ones((num_samples,), dtype=jnp.float32)
        stride = jnp.asarray(fixed_stride, dtype=jnp.float32)
        eps = jnp.asarray(1e-6, dtype=jnp.float32)

        def step(carry, _):
            x_t, t_idx = carry
            candidate = t_idx > eps
            delta = jnp.where(candidate, jnp.clip(jnp.broadcast_to(stride, t_idx.shape), 0.0, t_idx), 0.0)
            next_t_idx = t_idx - delta
            active = candidate & (next_t_idx != t_idx)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation_n, x_t, t_idx)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask_n, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                num_samples,
                suffix_tokens.shape[1],
                prefix_tokens_len + suffix_tokens.shape[1],
            )

            positions_suffix = jnp.sum(prefix_mask_n, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            prefix_out, suffix_out = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache_n,
                adarms_cond=[None, adarms_cond],
            )[0]
            assert prefix_out is None

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            dt = (next_t_idx - t_idx)[:, None, None]
            x_next = x_t + dt * v_t
            x_next = jnp.where(active[:, None, None], x_next, x_t)
            x0_pred = x_t - t_idx[:, None, None] * v_t
            score_magnitude = jnp.linalg.norm(v_t.reshape(v_t.shape[0], -1), axis=-1)
            action_features = jnp.mean(suffix_out[:, -self.action_horizon :], axis=1)
            trace = {
                "sample": x_t,
                "timestep": t_idx,
                "action": delta,
                "next_sample": x_next,
                "next_timestep": next_t_idx,
                "score_magnitude": score_magnitude,
                "x0_pred": x0_pred,
                "mid_feature_pooled": action_features,
                "active_count": jnp.sum(active.astype(jnp.int32)),
            }
            return (x_next, next_t_idx), trace

        (x_0, final_timestep), trace = jax.lax.scan(step, (x_t, t_idx), xs=None, length=num_steps)
        info = {
            "adaptive_steps": num_steps,
            "default_num_steps": num_steps,
            "max_model_evals": num_steps,
            "active_counts": trace["active_count"],
            "final_timestep": final_timestep,
            "prefix_context": prefix_context,
            "prefix_context_n": prefix_context_n,
            "trace": trace,
        }
        return {"actions": x_0, "info": info}

    def init_adaptive_denoise_state(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_samples: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ):
        """Prepare reusable pi0.5 context for Python-controlled adaptive denoising."""
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        if observation.state.shape[0] != 1:
            raise ValueError(
                f"init_adaptive_denoise_state expects a single observation with batch size 1, got {observation.state.shape[0]}"
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions_prefix)

        prefix_mask_n = jnp.broadcast_to(prefix_mask, (num_samples,) + prefix_mask.shape[1:])
        kv_cache_n = jax.tree.map(
            lambda x: jnp.broadcast_to(x, x.shape[:1] + (num_samples,) + x.shape[2:]),
            kv_cache,
        )
        observation_n = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_samples,) + x.shape[1:])
            if hasattr(x, "shape") and x.shape[:1] == (1,)
            else x,
            observation,
        )

        prefix_weights = prefix_mask.astype(prefix_tokens.dtype)[..., None]
        prefix_context = jnp.sum(prefix_tokens * prefix_weights, axis=1) / jnp.maximum(
            jnp.sum(prefix_weights, axis=1), 1.0
        )
        prefix_context_n = jnp.broadcast_to(prefix_context, (num_samples,) + prefix_context.shape[1:])

        if noise is None:
            x_t = jax.random.normal(rng, (num_samples, self.action_horizon, self.action_dim))
        else:
            x_t = jnp.asarray(noise)
            if x_t.ndim == 3 and x_t.shape[0] == 1:
                x_t = jnp.broadcast_to(x_t, (num_samples,) + x_t.shape[1:])
            if x_t.shape != (num_samples, self.action_horizon, self.action_dim):
                raise ValueError(
                    "noise must have shape "
                    f"{(num_samples, self.action_horizon, self.action_dim)} or "
                    f"{(1, self.action_horizon, self.action_dim)}, got {x_t.shape}"
                )

        context = {
            "observation_n": observation_n,
            "prefix_mask_n": prefix_mask_n,
            "kv_cache_n": kv_cache_n,
            "prefix_context": prefix_context,
            "prefix_context_n": prefix_context_n,
        }
        t_idx = jnp.ones((num_samples,), dtype=jnp.float32)
        return context, x_t, t_idx

    def stride_denoise_step(
        self,
        context: dict,
        x_t: at.Float[at.Array, "b ah ad"],
        t_idx: at.Float[at.Array, "b"],
        delta: at.Float[at.Array, "b"],
    ):
        """One jittable adaptive denoising step for any stride selector."""
        observation_n = context["observation_n"]
        prefix_mask_n = context["prefix_mask_n"]
        kv_cache_n = context["kv_cache_n"]

        delta = jnp.asarray(delta, dtype=jnp.float32).reshape(t_idx.shape)
        candidate = t_idx > 1e-6
        delta = jnp.where(candidate, jnp.clip(delta, 0.0, t_idx), 0.0)
        next_t_idx = t_idx - delta
        active = candidate & (next_t_idx != t_idx)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation_n, x_t, t_idx)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask_n, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions_suffix = jnp.sum(prefix_mask_n, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        prefix_out, suffix_out = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_suffix,
            kv_cache=kv_cache_n,
            adarms_cond=[None, adarms_cond],
        )[0]
        assert prefix_out is None

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        dt = (next_t_idx - t_idx)[:, None, None]
        x_next = x_t + dt * v_t
        x_next = jnp.where(active[:, None, None], x_next, x_t)
        x0_pred = x_t - t_idx[:, None, None] * v_t
        score_magnitude = jnp.linalg.norm(v_t.reshape(v_t.shape[0], -1), axis=-1)
        action_features = jnp.mean(suffix_out[:, -self.action_horizon :], axis=1)
        trace = {
            "sample": x_t,
            "timestep": t_idx,
            "action": delta,
            "next_sample": x_next,
            "next_timestep": next_t_idx,
            "score_magnitude": score_magnitude,
            "x0_pred": x0_pred,
            "mid_feature_pooled": action_features,
            "active_count": jnp.sum(active.astype(jnp.int32)),
        }
        return x_next, next_t_idx, trace

    def sample_actions_learned_stride_jax(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        actor_params: dict[str, Any],
        actor_spec: JaxStrideActorSpec,
        num_samples: int,
        max_denoise_steps: int,
        use_extended_obs: bool = True,
        use_sparse_active_denoising: bool = True,
    ):
        """Learned-stride sampler in one compiled graph (prefix, actor and denoise loop)."""
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        if observation.state.shape[0] != 1:
            raise ValueError(
                "sample_actions_learned_stride_jax expects a single observation "
                f"with batch size 1, got {observation.state.shape[0]}"
            )
        if actor_spec.num_samples != int(num_samples):
            raise ValueError(f"actor checkpoint expects K={actor_spec.num_samples}, got K={num_samples}")

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions_prefix)

        prefix_mask_n = jnp.broadcast_to(prefix_mask, (num_samples,) + prefix_mask.shape[1:])
        kv_cache_n = jax.tree.map(
            lambda x: jnp.broadcast_to(x, x.shape[:1] + (num_samples,) + x.shape[2:]),
            kv_cache,
        )
        observation_n = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_samples,) + x.shape[1:])
            if hasattr(x, "shape") and x.shape[:1] == (1,)
            else x,
            observation,
        )

        prefix_weights = prefix_mask.astype(prefix_tokens.dtype)[..., None]
        prefix_context = jnp.sum(prefix_tokens * prefix_weights, axis=1) / jnp.maximum(
            jnp.sum(prefix_weights, axis=1), 1.0
        )
        prefix_context_n = jnp.broadcast_to(prefix_context, (num_samples,) + prefix_context.shape[1:])

        x_t = jax.random.normal(rng, (num_samples, self.action_horizon, self.action_dim))
        t_idx = jnp.ones((num_samples,), dtype=jnp.float32)
        tail_dim = 5 if use_extended_obs else 3
        mid_dim = int(actor_spec.sample_dim - self.action_horizon * self.action_dim - tail_dim)
        if mid_dim < 0:
            raise ValueError(
                f"actor sample_dim={actor_spec.sample_dim} is too small for "
                f"action_horizon={self.action_horizon}, action_dim={self.action_dim}, "
                f"use_extended_obs={use_extended_obs} (tail_dim={tail_dim})"
            )

        def fit_mid(mid):
            if mid_dim == 0:
                return jnp.zeros((mid.shape[0], 0), dtype=jnp.float32)
            if mid.shape[-1] >= mid_dim:
                return mid[:, :mid_dim]
            pad = jnp.zeros((mid.shape[0], mid_dim - mid.shape[-1]), dtype=mid.dtype)
            return jnp.concatenate([mid, pad], axis=-1)

        def selector_obs(x, t, stopped, prev_delta, score, x0d, mid):
            x_flat = x[..., : self.action_dim].reshape(num_samples, -1)
            moving = (prev_delta.reshape(1, num_samples) != 0).astype(jnp.float32)
            active_frac = jnp.broadcast_to(
                jnp.sum(moving, axis=1, keepdims=True) / max(num_samples, 1),
                (1, num_samples),
            ).reshape(num_samples, 1)
            t_col = t.reshape(num_samples, 1)
            stop_col = stopped.reshape(num_samples, 1).astype(jnp.float32)
            if use_extended_obs:
                sample_obs = jnp.concatenate(
                    [
                        x_flat,
                        fit_mid(mid),
                        t_col,
                        active_frac,
                        stop_col,
                        score.reshape(num_samples, 1),
                        x0d.reshape(num_samples, 1),
                    ],
                    axis=-1,
                )
            else:
                sample_obs = jnp.concatenate([x_flat, t_col, active_frac, stop_col], axis=-1)
            state_obs = prefix_context_n.reshape(1, num_samples, -1)
            return state_obs.astype(jnp.float32), sample_obs.reshape(1, num_samples, -1).astype(jnp.float32)

        def denoise_step_dense(x, t, delta):
            delta = jnp.asarray(delta, dtype=jnp.float32).reshape(t.shape)
            candidate = t > 1e-6
            delta = jnp.where(candidate, jnp.clip(delta, 0.0, t), 0.0)
            next_t = t - delta
            active = candidate & (next_t != t)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation_n, x, t)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask_n, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions_suffix = jnp.sum(prefix_mask_n, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            prefix_out, suffix_out = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache_n,
                adarms_cond=[None, adarms_cond],
            )[0]
            assert prefix_out is None

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            dt = (next_t - t)[:, None, None]
            x_next = x + dt * v_t
            x_next = jnp.where(active[:, None, None], x_next, x)
            x0_pred = x - t[:, None, None] * v_t
            score_magnitude = jnp.linalg.norm(v_t.reshape(v_t.shape[0], -1), axis=-1)
            action_features = jnp.mean(suffix_out[:, -self.action_horizon :], axis=1)
            active_count = jnp.sum(active.astype(jnp.int32))
            return x_next, next_t, x0_pred, score_magnitude, action_features, active_count

        bucket_sizes = tuple([size for size in (1, 2, 4, 8) if size < int(num_samples)] + [int(num_samples)])
        sample_rows = jnp.arange(num_samples)

        def take_observation_rows(obs, idx):
            return jax.tree.map(
                lambda y: jnp.take(y, idx, axis=0)
                if hasattr(y, "shape") and len(y.shape) >= 1 and y.shape[0] == num_samples
                else y,
                obs,
            )

        def take_kv_rows(cache, idx):
            return jax.tree.map(
                lambda y: jnp.take(y, idx, axis=1)
                if hasattr(y, "shape") and len(y.shape) >= 2 and y.shape[1] == num_samples
                else y,
                cache,
            )

        def scatter_rows(current, idx, valid, updates):
            one_hot = (sample_rows[None, :] == idx[:, None]) & valid[:, None]
            weights = one_hot.astype(updates.dtype)
            for _ in range(updates.ndim - 1):
                weights = weights[..., None]
            scattered = jnp.sum(weights * updates[:, None, ...], axis=0)
            update_mask = jnp.any(one_hot, axis=0)
            for _ in range(current.ndim - 1):
                update_mask = update_mask[..., None]
            return jnp.where(update_mask, scattered, current)

        def denoise_step_bucket(bucket_size, x, t, delta, active_count):
            active = delta > 0.0
            idx = jnp.nonzero(active, size=bucket_size, fill_value=0)[0]
            valid = jnp.arange(bucket_size) < active_count
            x_b = jnp.take(x, idx, axis=0)
            t_b = jnp.take(t, idx, axis=0)
            delta_b = jnp.take(delta, idx, axis=0)
            next_t_b = t_b - delta_b

            observation_b = take_observation_rows(observation_n, idx)
            prefix_mask_b = jnp.take(prefix_mask_n, idx, axis=0)
            kv_cache_b = take_kv_rows(kv_cache_n, idx)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation_b, x_b, t_b)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask_b, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions_suffix = jnp.sum(prefix_mask_b, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            prefix_out, suffix_out = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache_b,
                adarms_cond=[None, adarms_cond],
            )[0]
            assert prefix_out is None

            v_t_b = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            dt_b = (next_t_b - t_b)[:, None, None]
            x_next_b = x_b + dt_b * v_t_b
            x0_pred_b = x_b - t_b[:, None, None] * v_t_b
            score_b = jnp.linalg.norm(v_t_b.reshape(v_t_b.shape[0], -1), axis=-1)
            mid_b = jnp.mean(suffix_out[:, -self.action_horizon :], axis=1)
            return (
                scatter_rows(x, idx, valid, x_next_b),
                scatter_rows(t, idx, valid, next_t_b),
                scatter_rows(x, idx, valid, x0_pred_b),
                scatter_rows(jnp.zeros_like(t), idx, valid, score_b),
                scatter_rows(jnp.zeros((num_samples, mid_b.shape[-1]), dtype=mid_b.dtype), idx, valid, mid_b),
                active_count,
            )

        def denoise_step_sparse(x, t, delta):
            delta = jnp.asarray(delta, dtype=jnp.float32).reshape(t.shape)
            candidate = t > 1e-6
            delta = jnp.where(candidate, jnp.clip(delta, 0.0, t), 0.0)
            active_count = jnp.sum((delta > 0.0).astype(jnp.int32))
            bucket_index = jnp.sum(active_count > jnp.asarray(bucket_sizes[:-1], dtype=jnp.int32))
            branches = tuple(
                lambda operand, bucket_size=bucket_size: denoise_step_bucket(bucket_size, *operand)
                for bucket_size in bucket_sizes
            )
            return jax.lax.switch(bucket_index, branches, (x, t, delta, active_count))

        def denoise_step(x, t, delta):
            if use_sparse_active_denoising:
                return denoise_step_sparse(x, t, delta)
            return denoise_step_dense(x, t, delta)

        def x0_dev(x0_pred, timestep):
            x0_flat = x0_pred[..., : self.action_dim].reshape(1, num_samples, -1)
            centered = x0_flat - jnp.mean(x0_flat, axis=1, keepdims=True)
            raw = jnp.linalg.norm(centered, axis=-1).reshape(num_samples)
            scale = jnp.maximum(jnp.cos(jnp.pi / 2.0 * timestep), 1e-3)
            return raw * scale

        active_counts0 = jnp.zeros((max_denoise_steps,), dtype=jnp.int32)
        stopped0 = jnp.zeros((num_samples,), dtype=jnp.bool_)
        prev_delta0 = jnp.where(t_idx > 1e-6, 1.0, 0.0).astype(jnp.float32)
        score0 = jnp.zeros((num_samples,), dtype=jnp.float32)
        x0d0 = jnp.zeros((num_samples,), dtype=jnp.float32)
        mid0 = jnp.zeros((num_samples, mid_dim), dtype=jnp.float32)

        def cond(carry):
            x, t, stopped, prev_delta, score, x0d, mid, active_counts, step_i = carry
            del x, prev_delta, score, x0d, mid, active_counts
            return (step_i < max_denoise_steps) & jnp.any((t > 1e-6) & (~stopped))

        def body(carry):
            x, t, stopped, prev_delta, score, x0d, mid, active_counts, step_i = carry
            running = (t > 1e-6) & (~stopped)
            state_obs, sample_obs = selector_obs(x, t, stopped, prev_delta, score, x0d, mid)
            raw_action = jax_actor_deterministic_action(
                actor_params,
                actor_spec,
                state_obs,
                sample_obs,
                running_mask=running.reshape(1, num_samples),
            ).reshape(num_samples)
            proposed = action_to_stride_jax(raw_action, (0.0, 1.0)).reshape(num_samples)
            delta = jnp.where(running, jnp.clip(proposed, 0.0, t), 0.0)
            stopped_next = stopped | (running & (delta == 0.0))
            any_active = jnp.any(delta > 0.0)

            def do_step(_):
                x_next, t_next, x0, next_score, next_mid_raw, active_count = denoise_step(x, t, delta)
                return (
                    x_next,
                    t_next,
                    next_score.astype(jnp.float32),
                    x0_dev(x0, t),
                    fit_mid(next_mid_raw.astype(jnp.float32)),
                    active_count.astype(jnp.int32),
                    step_i + jnp.asarray(1, dtype=jnp.int32),
                )

            def skip_step(_):
                return x, t, score, x0d, mid, jnp.asarray(0, dtype=jnp.int32), step_i

            x_next, t_next, next_score, next_x0d, next_mid, active_count, step_next = jax.lax.cond(
                any_active, do_step, skip_step, operand=None
            )
            active_counts = active_counts.at[step_i].set(active_count)
            return x_next, t_next, stopped_next, delta, next_score, next_x0d, next_mid, active_counts, step_next

        final = jax.lax.while_loop(
            cond,
            body,
            (x_t, t_idx, stopped0, prev_delta0, score0, x0d0, mid0, active_counts0, jnp.asarray(0, dtype=jnp.int32)),
        )
        x_final, t_final, _, _, _, _, _, active_counts, steps = final
        return {
            "actions": x_final,
            "info": {
                "adaptive_steps": steps,
                "default_num_steps": 0,
                "max_model_evals": max_denoise_steps,
                "active_counts": active_counts,
                "final_timestep": t_final,
            },
        }

def adopt_model(model):
    if type(model) is ElasticPi0:
        return model
    if type(model) is not Pi0:
        raise TypeError(f"no adaptive stride sampler for {type(model).__name__}")
    model.__class__ = ElasticPi0
    return model
