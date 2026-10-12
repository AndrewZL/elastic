"""PushT keypoints runner, with the optional obstacle (``pusht_obs``)."""
import pathlib
import uuid
from typing import List

import dill
import numpy as np
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper

from envs.base import ResetVectorRunner, wrap_env
from envs.pusht_obs_keypoints_env import PushTObsKeypointsEnv as PushTKeypointsEnv


class PushTKeypointsResetVectorRunner(ResetVectorRunner):
    """PushT keypoints runner. Obs carry an (obs, mask) layout, plus the obstacle when enabled."""

    def __init__(
        self,
        output_dir,
        keypoint_visible_rate=1.0,
        render_size=96,
        n_train=10,
        n_train_vis=3,
        train_start_seed=0,
        n_test=22,
        n_test_vis=6,
        legacy_test=False,
        test_start_seed=10000,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        n_latency_steps=0,
        fps=10,
        crf=22,
        agent_keypoints=False,
        past_action=False,
        tqdm_interval_sec=5.0,
        num_action_samples=1,
        gamma=0.99,
        n_envs=None,
        shape_type=None,
        include_obstacle=None,
        obstacle_radius=42,
        obstacle_blocking_prob=0.8,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        self.n_envs = n_envs

        self.max_k = num_action_samples
        n_physical_envs = n_envs

        if shape_type is None:
            shape_type = "tee"
        shape_type = str(shape_type)
        if include_obstacle is None:
            include_obstacle = False
        include_obstacle = bool(include_obstacle)

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        kp_kwargs = PushTKeypointsEnv.generate_keypoint_manager_params()

        def env_fn():
            return wrap_env(
                PushTKeypointsEnv(
                    legacy=legacy_test,
                    keypoint_visible_rate=keypoint_visible_rate,
                    agent_keypoints=agent_keypoints,
                    render_size=render_size,
                    local_keypoint_map=kp_kwargs["local_keypoint_map"],
                    color_map=kp_kwargs["color_map"],
                    shape_type=shape_type,
                    include_obstacle=include_obstacle,
                    include_obstacle_pos=True,
                    obstacle_radius=obstacle_radius,
                    obstacle_blocking_prob=obstacle_blocking_prob,
                ),
                fps=fps,
                crf=crf,
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_steps=max_steps,
            )

        env_fns = [env_fn] * n_physical_envs
        env_seeds: List[int] = []
        env_prefixs: List[str] = []
        env_init_fn_dills: List[bytes] = []

        def create_init_fn(seed, enable_render):
            def init_fn(env):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", str(uuid.uuid4()) + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            return init_fn

        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(create_init_fn(seed, enable_render)))

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(create_init_fn(seed, enable_render)))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills

        assert len(self.env_fns) == len(self.env_init_fn_dills)
        self.env.call_each("run_dill_function", args_list=[(x,) for x in self.env_init_fn_dills])
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.num_action_samples = num_action_samples
        self.gamma = gamma

    def prepare_obs(self, raw_obs):
        """Drop the obstacle columns from the interleaved (obs, mask) layout."""
        assert self.policy is not None
        Do = self.policy.obs_dim
        F = int(raw_obs.shape[-1])
        if F == 2 * Do:
            return raw_obs
        elif F >= 2 * Do + 2:
            return np.concatenate(
                [raw_obs[:, :, :Do], raw_obs[:, :, Do + 2 : Do + 2 + Do]], axis=-1
            )
        else:
            raise ValueError(
                f"Obs last-dim {F} too small for obs_dim={Do} (need at least {2*Do})"
            )

    def _build_obs_dict(self, obs: np.ndarray):
        assert self.policy is not None
        Do = obs.shape[-1] // 2
        return {
            "obs": obs[..., : self.n_obs_steps, :Do].astype(np.float32),
            "obs_mask": obs[..., : self.n_obs_steps, Do:] > 0.5,
        }
