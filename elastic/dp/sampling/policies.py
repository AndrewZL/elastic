"""Upstream Diffusion Policy checkpoints with the adaptive stride sampler swapped in after load."""
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from typing import Dict, Optional

from elastic.dp.sampling.sampler import stride_conditional_sample


class ElasticUnetLowdimPolicy(DiffusionUnetLowdimPolicy):
    """Upstream lowdim policy with the adaptive stride sampler. Construction is upstream's."""

    def conditional_sample(self, condition_data, condition_mask, **kwargs):
        return stride_conditional_sample(self, condition_data, condition_mask, **kwargs)

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        num_inference_steps=None,
        timestep_selector=None,
        timestep_selector_mode: str = "absolute",
        max_denoise_steps: Optional[int] = None,
        num_action_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Sample actions; with num_action_samples > 1, outputs are shaped (B, num_action_samples, ...)."""
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        if num_action_samples > 1:
            cond_data = cond_data.repeat_interleave(num_action_samples, dim=0)
            cond_mask = cond_mask.repeat_interleave(num_action_samples, dim=0)

        obs_features = self.encode_obs_features(nobs)

        with torch.no_grad():
            nsample, done_mask = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                num_inference_steps=num_inference_steps,
                timestep_selector=timestep_selector,
                timestep_selector_mode=timestep_selector_mode,
                max_denoise_steps=max_denoise_steps,
                max_stride=self.noise_scheduler.num_train_timesteps,
                obs_features=obs_features,
                **self.kwargs,
            )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        if num_action_samples > 1:
            action = action.reshape(B, num_action_samples, *action.shape[1:])
            action_pred = action_pred.reshape(B, num_action_samples, *action_pred.shape[1:])
            if done_mask is not None:
                done_mask = done_mask.reshape(B, num_action_samples)

        result = {
            "action": action,
            "action_pred": action_pred,
            "obs_features": obs_features,
            "done_mask": done_mask,
        }

        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            if num_action_samples > 1:
                obs_pred = obs_pred.reshape(B, num_action_samples, *obs_pred.shape[1:])
                action_obs_pred = action_obs_pred.reshape(
                    B, num_action_samples, *action_obs_pred.shape[1:]
                )

            result["action_obs_pred"] = action_obs_pred
            result["obs_pred"] = obs_pred

        return result

    def encode_obs_features(self, nobs: torch.Tensor) -> torch.Tensor:
        """Stride-selector obs features, (B, n_obs_steps * obs_dim): the normalized obs (no encoder)."""
        return nobs[:, : self.n_obs_steps].reshape(nobs.shape[0], -1)


class ElasticUnetImagePolicy(DiffusionUnetImagePolicy):
    """Upstream image policy with the adaptive stride sampler. Construction is upstream's."""

    def conditional_sample(self, condition_data, condition_mask, **kwargs):
        return stride_conditional_sample(self, condition_data, condition_mask, **kwargs)

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        num_inference_steps=None,
        batch_idx=None,
        timestep_selector=None,
        timestep_selector_mode: str = "absolute",
        max_denoise_steps: Optional[int] = None,
        num_action_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Sample actions; with num_action_samples > 1, outputs are shaped (B, num_action_samples, ...)."""
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        if batch_idx is not None:
            nobs = dict_apply(nobs, lambda x: x[batch_idx])
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        obs_features = nobs_features.reshape(B, -1)

        if num_action_samples > 1:
            cond_data = cond_data.repeat_interleave(num_action_samples, dim=0)
            cond_mask = cond_mask.repeat_interleave(num_action_samples, dim=0)

        nsample, done_mask = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            num_inference_steps=num_inference_steps,
            timestep_selector=timestep_selector,
            timestep_selector_mode=timestep_selector_mode,
            max_denoise_steps=max_denoise_steps,
            max_stride=self.noise_scheduler.config.num_train_timesteps,
            obs_features=obs_features,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        if num_action_samples > 1:
            action = action.reshape(B, num_action_samples, *action.shape[1:])
            action_pred = action_pred.reshape(B, num_action_samples, *action_pred.shape[1:])
            if done_mask is not None:
                done_mask = done_mask.reshape(B, num_action_samples)

        result = {
            "action": action,
            "action_pred": action_pred,
            "obs_features": obs_features,
            "done_mask": done_mask,
        }
        return result

    def encode_obs_features(self, nobs: dict) -> torch.Tensor:
        """Flattened obs-encoder output, (B, n_obs_steps * obs_feature_dim)."""
        To = self.n_obs_steps
        value = next(iter(nobs.values()))
        B = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        return self.obs_encoder(this_nobs).reshape(B, -1)


_ELASTIC_CLASSES = {
    DiffusionUnetLowdimPolicy: ElasticUnetLowdimPolicy,
    DiffusionUnetImagePolicy: ElasticUnetImagePolicy,
}


def adopt_policy(policy):
    """Give a hydra-built upstream policy the adaptive stride sampler, in place.

    The subclasses add methods only, so swapping the class keeps every
    parameter, buffer and attribute that upstream construction produced.
    """
    cls = _ELASTIC_CLASSES.get(type(policy))
    if cls is None:
        raise TypeError(f"no adaptive stride sampler for {type(policy).__name__}")
    policy.__class__ = cls
    return policy
