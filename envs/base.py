"""Shared vectorized runner: sample S candidate chunks per step, score them with a
``scoring_fn`` (verifier advantage), and execute the best finished candidate.
Task runners override ``prepare_obs``, ``_build_obs_dict`` and ``_to_env_action``.
"""
from typing import Optional

import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper


def wrap_env(inner, *, fps, crf, n_obs_steps, n_action_steps, max_steps, **video_kwargs):
    """Video recording plus multi-step chunking around a single env."""
    return MultiStepWrapper(
        VideoRecordingWrapper(
            inner,
            video_recoder=VideoRecorder.create_h264(
                fps=fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=crf,
                thread_type="FRAME",
                thread_count=1,
            ),
            file_path=None,
            **video_kwargs,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
    )


def _batch_size(obs) -> int:
    if isinstance(obs, dict):
        return next(iter(obs.values())).shape[0]
    return obs.shape[0]


class ResetVectorRunner(BaseLowdimRunner):
    """Base class. Subclasses build ``self.env`` (an AsyncVectorEnv) and set
    ``n_envs``, ``max_k``, ``n_obs_steps`` and ``n_action_steps``."""

    policy = None

    def set_policy(self, policy):
        self.policy = policy

    def reset(self):
        assert self.policy is not None
        seed = np.random.randint(0, 1000000)
        seeds = [seed + b for b in range(self.n_envs)]
        self.env.seed(seeds)
        self.policy.reset()
        return self.env.reset()

    def reset_with_seed(self, start_ep_idx: int, base_seed: int):
        """Deterministic reset: env i gets seed base_seed + start_ep_idx + i."""
        assert self.policy is not None
        seeds = [base_seed + start_ep_idx + i for i in range(self.n_envs)]
        self.env.seed(seeds)
        self.policy.reset()
        return self.env.reset()

    def prepare_obs(self, raw_obs):
        """Raw env obs -> obs for ``step``. Identity unless overridden."""
        return raw_obs

    def _build_obs_dict(self, obs):
        raise NotImplementedError

    def _to_env_action(self, actions):
        """Policy-space action -> env action. Identity unless overridden."""
        return actions

    def flatten_obs_for_verifier(self, raw_obs, device):
        """(B*S, F) float32 raw obs (before ``prepare_obs``), repeated per candidate.

        Dict obs are concatenated in insertion order, the layout the verifiers were trained on.
        """
        S = self.max_k
        if isinstance(raw_obs, dict):
            raw_obs = np.concatenate([v for v in raw_obs.values()], axis=-1)
        t_obs = torch.from_numpy(raw_obs).to(device).float()
        B = t_obs.shape[0]
        flat = t_obs.reshape(B, -1)
        return flat.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)

    def verifier_advantage(self, verifier, raw_obs, action_B_S, device) -> np.ndarray:
        """(B, S) verifier advantage of each candidate chunk."""
        with torch.no_grad():
            B, S = action_B_S.shape[0], self.max_k
            T, D = action_B_S.shape[-2:]
            flat_obs = self.flatten_obs_for_verifier(raw_obs, device)
            flat_action = torch.from_numpy(action_B_S).to(device).float().reshape(B * S, T, D)
            return verifier.compute_advantage(flat_obs, flat_action).view(B, S).cpu().numpy()

    def _predict(self, obs, timestep_selector, timestep_selector_mode, max_denoise_steps):
        device = self.policy.device
        obs_dict = dict_apply(
            self._build_obs_dict(obs), lambda x: torch.from_numpy(x).to(device=device)
        )
        return self.policy.predict_action(
            obs_dict,
            num_action_samples=self.max_k,
            timestep_selector=timestep_selector,
            timestep_selector_mode=timestep_selector_mode,
            max_denoise_steps=max_denoise_steps,
        )

    def sim_step(
        self,
        obs,
        timestep_selector=None,
        timestep_selector_mode: str = "absolute",
        max_denoise_steps: Optional[int] = None,
    ):
        """Sample S candidates without stepping the envs."""
        assert self.policy is not None
        result = self._predict(obs, timestep_selector, timestep_selector_mode, max_denoise_steps)
        return result["action"].detach().cpu().numpy(), result

    def step(
        self,
        obs,
        scoring_fn,
        timestep_selector=None,
        timestep_selector_mode: str = "absolute",
        max_denoise_steps: Optional[int] = None,
        pre_env_step_hook=None,
    ):
        assert self.policy is not None
        B = _batch_size(obs)
        S = self.max_k

        result = self._predict(obs, timestep_selector, timestep_selector_mode, max_denoise_steps)
        action_B_S = result["action"].detach().cpu().numpy()
        if S == 1 and action_B_S.ndim == 3:
            action_B_S = action_B_S[:, None, ...]

        done_mask_B_S = None
        if result.get("done_mask") is not None:
            done_mask_B_S = result["done_mask"].detach().cpu().numpy()
            if S == 1 and done_mask_B_S.ndim == 1:
                done_mask_B_S = done_mask_B_S[:, None]

        if scoring_fn is not None:
            exec_scores = scoring_fn(obs, action_B_S)
        else:
            exec_scores = np.zeros((B, S), dtype=np.float32)

        # NumPy argmax picks the first index on ties.
        if done_mask_B_S is not None:
            valid_mask = done_mask_B_S.astype(bool)
            masked_scores = np.where(valid_mask, exec_scores, -np.inf)
            has_valid = np.any(valid_mask, axis=1)
            best_idx = np.where(
                has_valid,
                np.argmax(masked_scores, axis=1),
                np.argmax(exec_scores, axis=1),
            )
        else:
            best_idx = np.argmax(exec_scores, axis=1)
        chosen_actions_B = action_B_S[np.arange(B), best_idx]

        # Returned actions stay in policy space.
        env_actions = self._to_env_action(chosen_actions_B)
        if pre_env_step_hook is not None:
            pre_env_step_hook(self.env)
        logical_res = self.env.step(env_actions)
        return logical_res, chosen_actions_B, action_B_S, exec_scores
