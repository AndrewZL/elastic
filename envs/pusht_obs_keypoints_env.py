"""Keypoint-observation variant of the PushT obstacle env."""
from typing import Dict, Optional
from gym import spaces
from envs.pusht_obs_env import PushTObsEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np


class PushTObsKeypointsEnv(PushTObsEnv):
    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_size=96,
        keypoint_visible_rate=1.0,
        agent_keypoints=False,
        draw_keypoints=False,
        reset_to_state=None,
        render_action=True,
        local_keypoint_map: Dict[str, np.ndarray] = None,
        color_map: Optional[Dict[str, np.ndarray]] = None,
        shape_type="tee",
        include_obstacle: bool = False,
        include_obstacle_pos: bool = False,
        obstacle_radius: int = 42,
        obstacle_blocking_prob: float = 0.8,
    ):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action,
            shape_type=shape_type,
            include_obstacle=include_obstacle,
            obstacle_radius=obstacle_radius,
            obstacle_blocking_prob=obstacle_blocking_prob,
        )
        ws = self.window_size

        if local_keypoint_map is None:
            kp_kwargs = self.generate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs["local_keypoint_map"]
            color_map = kp_kwargs["color_map"]

        Dblockkps = np.prod(local_keypoint_map["block"].shape)
        Dagentkps = np.prod(local_keypoint_map["agent"].shape)
        Dagentpos = 2
        Dobstaclepos = 2

        Do = Dblockkps
        if agent_keypoints:
            Do += Dagentkps
        else:
            Do += Dagentpos
        if include_obstacle and include_obstacle_pos:
            Do += Dobstaclepos
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        high[Do:] = 1.0

        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float64)

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.include_obstacle_pos = include_obstacle_pos
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map, color_map=color_map
        )
        self.draw_kp_map = None

    @classmethod
    def generate_keypoint_manager_params(cls):
        env = PushTObsEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        obj_map = {"block": self.block}
        if self.agent_keypoints:
            obj_map["agent"] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
        kps = np.concatenate(list(kp_map.values()), axis=0)

        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:, None], 2, axis=1)

        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {"block": vis_kps[: len(kp_map["block"])]}
        if self.agent_keypoints:
            draw_kp_map["agent"] = vis_kps[len(kp_map["block"]) :]
        self.draw_kp_map = draw_kp_map

        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([obs, agent_pos])
            obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])
        if self.include_obstacle and self.include_obstacle_pos:
            obstacle_pos = np.array(self.obstacle.position)
            obs = np.concatenate([obs, obstacle_pos])
            obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])

        obs = np.concatenate([obs, obs_mask.astype(obs.dtype)], axis=0)
        return obs

    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(img, self.draw_kp_map, radius=int(img.shape[0] / 96))
        return img
