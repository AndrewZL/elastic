"""Robomimic parallel env runner for SAC-stride training."""

import collections
import os
import pathlib
import uuid
from typing import List

import dill
import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from envs.base import ResetVectorRunner, wrap_env


def _create_robomimic_lowdim_env(env_meta, obs_keys, *, render_offscreen: bool = False):
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=bool(render_offscreen),
        use_image_obs=False,
    )
    return env


def _create_robomimic_image_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


class RobomimicResetVectorRunner(ResetVectorRunner):
    """Parallel env runner for SAC-stride training on robomimic tasks."""

    def __init__(
        self,
        output_dir,
        dataset_path,
        obs_keys=None,
        shape_meta=None,
        n_train=0,
        n_train_vis=0,
        train_start_idx=0,
        n_test=22,
        n_test_vis=0,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        n_latency_steps=0,
        render_obs_key="agentview_image",
        render_hw=(256, 256),
        render_camera_name="agentview",
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        num_action_samples=1,
        gamma=0.99,
        n_envs=None,
    ):
        super().__init__(output_dir)

        os.environ.setdefault("MUJOCO_GL", "osmesa")

        if shape_meta is not None:
            self.image_mode = True
        elif obs_keys is not None:
            self.image_mode = False
        else:
            raise ValueError("Must provide either obs_keys (lowdim) or shape_meta (image)")

        dataset_path = os.path.expanduser(dataset_path)
        if n_envs is None:
            n_envs = max(n_train + n_test, 1)
        self.n_envs = n_envs
        self.max_k = num_action_samples

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        rotation_transformer = None
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        wrap_kwargs = dict(
            fps=fps,
            crf=crf,
            n_obs_steps=env_n_obs_steps,
            n_action_steps=env_n_action_steps,
            max_steps=max_steps,
            steps_per_render=steps_per_render,
        )

        # The main-process dummy env never renders: offscreen GL contexts there
        # conflict with the workers and hang AsyncVectorEnv init.
        if self.image_mode:
            env_meta["env_kwargs"]["use_object_obs"] = False

            def make_image_env(enable_render=True):
                import os

                from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
                    RobomimicImageWrapper,
                )

                os.environ.setdefault("MUJOCO_GL", "osmesa")
                robomimic_env = _create_robomimic_image_env(
                    env_meta=env_meta, shape_meta=shape_meta, enable_render=enable_render
                )
                robomimic_env.env.hard_reset = False
                return wrap_env(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    **wrap_kwargs,
                )

            def dummy_env_fn():
                return make_image_env(enable_render=False)

            env_fns = [make_image_env] * n_envs
        else:
            _n_vis_render = int(n_test_vis)

            def make_lowdim_env(render_offscreen):
                import os

                from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import (
                    RobomimicLowdimWrapper,
                )

                os.environ.setdefault("MUJOCO_GL", "osmesa")
                robomimic_env = _create_robomimic_lowdim_env(
                    env_meta=env_meta, obs_keys=obs_keys, render_offscreen=render_offscreen
                )
                return wrap_env(
                    RobomimicLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=obs_keys,
                        init_state=None,
                        render_hw=render_hw,
                        render_camera_name=render_camera_name,
                    ),
                    **wrap_kwargs,
                )

            def make_lowdim_env_fn(worker_index: int):
                off = worker_index < _n_vis_render

                def env_fn(render_offscreen=off):
                    return make_lowdim_env(render_offscreen)

                return env_fn

            def dummy_env_fn():
                return make_lowdim_env(False)

            env_fns = [make_lowdim_env_fn(i) for i in range(n_envs)]
        env_seeds: List[int] = []
        env_prefixs: List[str] = []
        env_init_fn_dills: List[bytes] = []

        with h5py.File(dataset_path, "r") as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f"data/demo_{train_idx}/states"][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", str(uuid.uuid4()) + ".mp4"
                        )
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append("train/")
                env_init_fn_dills.append(dill.dumps(init_fn))

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", str(uuid.uuid4()) + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        self.env = env
        self.env_meta = env_meta
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills

        assert len(self.env_fns) == len(self.env_init_fn_dills)
        self.env.call_each(
            "run_dill_function", args_list=[(x,) for x in self.env_init_fn_dills]
        )

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.num_action_samples = num_action_samples
        self.gamma = gamma
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.obs_keys = obs_keys
        self.shape_meta = shape_meta

    def _build_obs_dict(self, obs):
        """Convert prepared obs to the dict format the diffusion policy expects."""
        assert self.policy is not None
        if self.image_mode:
            return dict_apply(obs, lambda x: x.astype(np.float32))
        else:
            return {
                "obs": obs[:, : self.n_obs_steps].astype(np.float32),
            }

    def _to_env_action(self, actions):
        # abs_action checkpoints predict rotation_6d; the env takes axis_angle.
        if self.abs_action and self.rotation_transformer is not None:
            return self._undo_transform_action(actions)
        return actions

    def _undo_transform_action(self, action):
        """Convert rotation_6d back to axis_angle for robomimic abs_action mode."""
        if self.rotation_transformer is None:
            return action
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # Dual arm: (N, 20) -> (N, 2, 10) per arm -> (N, 14).
            action = action.reshape(-1, 2, 10)
            action_arms = []
            for i in range(2):
                d_rot = action.shape[-1] - 4
                pos = action[:, i, :3]
                rot = action[:, i, 3 : 3 + d_rot]
                gripper = action[:, i, [-1]]
                rot = self.rotation_transformer.inverse(rot)
                action_arms.append(np.concatenate([pos, rot, gripper], axis=-1))
            action = np.concatenate(action_arms, axis=-1)
            return action.reshape(raw_shape[:-1] + (action.shape[-1],))
        else:
            d_rot = raw_shape[-1] - 4  # total - 3 pos - 1 gripper
            pos = action[..., :3]
            rot = action[..., 3 : 3 + d_rot]
            gripper = action[..., [-1]]
            rot = self.rotation_transformer.inverse(rot)
            uaction = np.concatenate([pos, rot, gripper], axis=-1)
            return uaction.reshape(raw_shape[:-1] + (uaction.shape[-1],))
