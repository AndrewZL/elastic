"""Adaptive DDIM stride sampling shared by the Diffusion Policy wrappers."""
import torch
from typing import Callable, Optional


def _as_long_timestep_tensor(t, batch_size: int, device: torch.device):
    if torch.is_tensor(t):
        t_tensor = t.to(device=device)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor[None].expand(batch_size)
        return t_tensor.long()
    return torch.full((batch_size,), int(t), device=device, dtype=torch.long)


def _ddim_jump_step(scheduler, model_output, timestep, next_timestep, sample):
    """DDIM (eta=0) jump x_t -> x_{t_next} for scalar or per-sample timesteps."""
    bsz = sample.shape[0]
    device = sample.device
    t = _as_long_timestep_tensor(timestep, bsz, device)
    t_next = _as_long_timestep_tensor(next_timestep, bsz, device)

    t = t.clamp_min(0)
    t_next = t_next.clamp_min(0)

    alphas_cumprod = scheduler.alphas_cumprod
    if not isinstance(alphas_cumprod, torch.Tensor):
        alphas_cumprod = torch.tensor(alphas_cumprod, device=device, dtype=sample.dtype)
    else:
        alphas_cumprod = alphas_cumprod.to(device=device, dtype=sample.dtype)

    alpha_t = alphas_cumprod[t]
    alpha_next = alphas_cumprod[t_next]

    view_shape = (bsz,) + (1,) * (sample.ndim - 1)
    alpha_t_b = alpha_t.view(view_shape)
    alpha_next_b = alpha_next.view(view_shape)

    pred_type = getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon")
    if pred_type == "epsilon":
        pred_epsilon = model_output
        pred_original_sample = (
            sample - (1 - alpha_t_b).sqrt() * pred_epsilon
        ) / alpha_t_b.sqrt().clamp_min(1e-8)
    elif pred_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_t_b.sqrt() * pred_original_sample) / (
            1 - alpha_t_b
        ).sqrt().clamp_min(1e-8)
    elif pred_type == "v_prediction":
        pred_original_sample = alpha_t_b.sqrt() * sample - (1 - alpha_t_b).sqrt() * model_output
        pred_epsilon = alpha_t_b.sqrt() * model_output + (1 - alpha_t_b).sqrt() * sample
    else:
        raise ValueError(f"Unsupported prediction type {pred_type}")

    # Mirror DDIMScheduler thresholding/clipping for numerical parity with scheduler.step.
    config = getattr(scheduler, "config", None)
    if config is not None and getattr(config, "thresholding", False):
        dynamic_thresholding_ratio = float(getattr(config, "dynamic_thresholding_ratio", 0.995))
        sample_max_value = float(getattr(config, "sample_max_value", 1.0))

        flat = pred_original_sample.reshape(bsz, -1)
        abs_flat = flat.abs()
        s = torch.quantile(abs_flat, dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1.0, max=sample_max_value)
        s_view = s.view((bsz,) + (1,) * (pred_original_sample.ndim - 1))
        pred_original_sample = torch.clamp(pred_original_sample, -s_view, s_view) / s_view

    if config is not None and getattr(config, "clip_sample", False):
        clip_range = float(getattr(config, "clip_sample_range", 1.0))
        pred_original_sample = pred_original_sample.clamp(-clip_range, clip_range)

    # Backtracking (t_next > t) re-noises with fresh epsilon: x_{t_next} ~ q(x_{t_next} | x_0_pred).
    is_backtrack = t_next > t
    if is_backtrack.any():
        noise = torch.randn_like(pred_epsilon)
        mask = is_backtrack.reshape(view_shape).float()
        pred_epsilon = mask * noise + (1 - mask) * pred_epsilon

    prev_sample = (
        alpha_next_b.sqrt() * pred_original_sample + (1 - alpha_next_b).sqrt() * pred_epsilon
    )
    return prev_sample


def adaptive_denoise_loop(
    trajectory: torch.Tensor,
    condition_data: torch.Tensor,
    condition_mask: torch.Tensor,
    timestep_selector,
    timestep_selector_mode: str,
    max_denoise_steps: int,
    max_stride: int,
    model_predict_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    scheduler,
    obs_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive DDIM loop: the selector picks each sample's next timestep.

    model_predict_fn receives (traj_sub, t_sub, active_idx) and returns model_output_sub.
    Returns (trajectory, done_mask).
    """
    if max_denoise_steps is None:
        raise ValueError("max_denoise_steps must be specified when using timestep_selector")
    if timestep_selector_mode not in ("absolute", "delta"):
        raise ValueError(
            f"timestep_selector_mode must be 'absolute' or 'delta', got {timestep_selector_mode!r}"
        )

    start_t = max_stride - 1
    t = torch.full((trajectory.shape[0],), start_t, device=trajectory.device, dtype=torch.long)

    n = 0
    while n < max_denoise_steps:
        trajectory[condition_mask] = condition_data[condition_mask]

        candidate_mask = t > 0
        if not torch.any(candidate_mask):
            break

        out = timestep_selector(
            sample=trajectory,
            timestep=t,
            global_cond=obs_features,
        )
        if isinstance(out, torch.Tensor):
            out_tensor = out.detach().to(device=trajectory.device)
            if out_tensor.ndim == 0:
                out_tensor = out_tensor[None].expand_as(t)
            out_tensor = out_tensor.long()
        else:
            out_tensor = torch.full_like(t, int(out), dtype=torch.long)

        if timestep_selector_mode == "absolute":
            next_t = out_tensor
            next_t = next_t.clamp_min(0)
            next_t = torch.minimum(next_t, t)
            same = (next_t == t) & (t > 0)
            next_t = torch.where(same, torch.clamp(t - 1, min=0), next_t)
        else:
            delta_t = out_tensor
            next_t = t - delta_t
            next_t = next_t.clamp(min=0, max=max_stride - 1)

        # Finished samples keep their timestep, so a selector cannot re-noise them.
        next_t = torch.where(candidate_mask, next_t, t)

        is_active_step = candidate_mask & (next_t != t)
        if not torch.any(is_active_step):
            if hasattr(timestep_selector, "on_transition"):
                timestep_selector.on_transition(
                    sample=trajectory,
                    timestep=t,
                    global_cond=obs_features,
                    action=out_tensor,
                    next_sample=trajectory,
                    next_timestep=t,
                )
            break

        active_idx = torch.nonzero(is_active_step).squeeze(-1)
        traj_in = trajectory[active_idx]
        t_in = t[active_idx]

        model_output_sub = model_predict_fn(traj_in, t_in, active_idx)

        prev_trajectory = trajectory.clone()
        next_t_in = next_t[active_idx]
        trajectory[active_idx] = _ddim_jump_step(
            scheduler=scheduler,
            model_output=model_output_sub,
            timestep=t_in,
            next_timestep=next_t_in,
            sample=traj_in,
        )

        if hasattr(timestep_selector, "on_transition"):
            timestep_selector.on_transition(
                sample=prev_trajectory,
                timestep=t,
                global_cond=obs_features,
                action=out_tensor,
                next_sample=trajectory,
                next_timestep=next_t,
            )

        t = next_t
        n += 1

    trajectory[condition_mask] = condition_data[condition_mask]
    done_mask = t == 0
    return trajectory, done_mask


def unet_eps(out):
    """UNet noise prediction. Upstream returns a tensor; some forks return (eps, feature)."""
    return out[0] if isinstance(out, tuple) else out


def fixed_denoise_loop(
    model, scheduler, trajectory, condition_data, condition_mask,
    local_cond, global_cond, num_inference_steps, generator=None, **kwargs,
):
    """Upstream fixed DDIM schedule (``scheduler.timesteps``)."""
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        trajectory[condition_mask] = condition_data[condition_mask]
        model_output = unet_eps(model(trajectory, t, local_cond=local_cond, global_cond=global_cond))
        trajectory = scheduler.step(
            model_output, t, trajectory, generator=generator, **kwargs
        ).prev_sample
    trajectory[condition_mask] = condition_data[condition_mask]
    return trajectory


def make_model_predict_fn(model, condition_data, local_cond, global_cond):
    """UNet call on the active rows of a (B*S)-row trajectory.

    Conditioning may be per env (B rows); active sample i then reads env i // S.
    """
    def predict(traj_in, t_in, active_idx):
        base_batch = condition_data.shape[0]
        if global_cond is not None:
            base_batch = global_cond.shape[0]
        if condition_data.shape[0] != base_batch:
            repeat_factor = condition_data.shape[0] // base_batch
            env_active_idx = torch.div(active_idx, repeat_factor, rounding_mode="floor")
        else:
            env_active_idx = active_idx

        def sub(cond):
            if cond is None:
                return None
            if cond.shape[0] == condition_data.shape[0]:
                return cond[active_idx]
            return cond[env_active_idx]

        return unet_eps(model(traj_in, t_in, local_cond=sub(local_cond), global_cond=sub(global_cond)))

    return predict


def stride_conditional_sample(
    policy,
    condition_data,
    condition_mask,
    local_cond=None,
    global_cond=None,
    generator=None,
    num_inference_steps=None,
    timestep_selector=None,
    timestep_selector_mode: str = "absolute",
    max_denoise_steps: Optional[int] = None,
    max_stride: Optional[int] = None,
    obs_features=None,
    **kwargs,
):
    """``conditional_sample`` for both DP policies: upstream fixed schedule, or the adaptive loop with a selector.

    Returns (trajectory, done_mask).
    """
    trajectory = torch.randn(
        size=condition_data.shape,
        dtype=condition_data.dtype,
        device=condition_data.device,
        generator=generator,
    )
    scheduler = policy.noise_scheduler
    if timestep_selector is None:
        if num_inference_steps is None:
            num_inference_steps = policy.num_inference_steps
        if int(num_inference_steps) <= 0:
            raise ValueError("num_inference_steps must be > 0 without a timestep_selector")
        trajectory = fixed_denoise_loop(
            policy.model, scheduler, trajectory, condition_data, condition_mask,
            local_cond, global_cond, num_inference_steps, generator, **kwargs,
        )
        done_mask = torch.ones(trajectory.shape[0], dtype=torch.bool, device=trajectory.device)
        return trajectory, done_mask

    if max_stride is None:
        max_stride = scheduler.config.num_train_timesteps
    return adaptive_denoise_loop(
        trajectory=trajectory,
        condition_data=condition_data,
        condition_mask=condition_mask,
        timestep_selector=timestep_selector,
        timestep_selector_mode=timestep_selector_mode,
        max_denoise_steps=max_denoise_steps,
        max_stride=int(max_stride),
        model_predict_fn=make_model_predict_fn(policy.model, condition_data, local_cond, global_cond),
        scheduler=scheduler,
        obs_features=obs_features if obs_features is not None else global_cond,
    )
