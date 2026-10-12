"""PushT env with an optional fixed circular obstacle (``pusht_obs``) and alternative block shapes."""
import collections
import copy

import cv2
import gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
import skimage.transform as st
from gym import spaces
from pymunk.vec2d import Vec2d

from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        elif isinstance(shape, pymunk.shapes.Circle):
            pos = body.local_to_world(shape.offset)
            geoms.append(sg.Point(pos).buffer(shape.radius))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    geom = geom.buffer(0)
    return geom


class PushTObsEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=96,
        reset_to_state=None,
        shape_type="tee",
        include_obstacle=True,
        obstacle_radius=42,
        obstacle_blocking_prob=0.8,
    ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512
        self.render_size = render_size
        self.sim_hz = 100
        self.k_p, self.k_v = 100, 20
        self.control_hz = self.metadata["video.frames_per_second"]
        self.legacy = legacy
        self.shape_type = shape_type
        self.include_obstacle = include_obstacle
        self.obstacle_radius = obstacle_radius
        self.obstacle_blocking_prob = obstacle_blocking_prob

        # agent_pos, block_pos, block_angle[, obstacle_pos]
        obs_dim = 5 + (2 if include_obstacle else 0)
        low = np.zeros(obs_dim, dtype=np.float64)
        high = np.array(
            [ws, ws, ws, ws, np.pi * 2] + ([ws, ws] if include_obstacle else []),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(obs_dim,),
            dtype=np.float64,
        )

        # Action is the agent's target position (PD setpoint).
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def _sample_obstacle_position(self, rs, block_pos, block_angle, agent_pos):
        goal_pos = np.array(self.goal_pose[:2], dtype=np.float64)
        block_pos = np.array(block_pos, dtype=np.float64)
        agent_pos = np.array(agent_pos, dtype=np.float64)
        obstacle_geom_radius = self.obstacle_radius + 4
        goal_geom = pymunk_to_shapely(
            self._get_goal_pose_body(self.goal_pose),
            self.block.shapes,
        )
        block_geom = pymunk_to_shapely(
            self._get_goal_pose_body(np.array([block_pos[0], block_pos[1], block_angle])),
            self.block.shapes,
        )

        segment = goal_pos - block_pos
        seg_norm = np.linalg.norm(segment)
        if seg_norm < 1e-6:
            segment = np.array([1.0, 0.0], dtype=np.float64)
            seg_norm = 1.0
        segment_dir = segment / seg_norm
        segment_perp = np.array([-segment_dir[1], segment_dir[0]], dtype=np.float64)

        hard_case = rs.rand() < self.obstacle_blocking_prob
        margin = self.obstacle_radius + 8

        # Rejection sampling keeps episodes feasible while biasing toward true blocking layouts.
        for _ in range(64):
            t = rs.uniform(0.25, 0.85)
            on_path = block_pos + t * segment

            if hard_case:
                offset_mag = rs.uniform(8.0, 42.0)
            else:
                offset_mag = rs.uniform(70.0, 140.0)

            side = rs.choice([-1.0, 1.0])
            jitter = rs.uniform(-14.0, 14.0, size=2)
            candidate = on_path + side * offset_mag * segment_perp + jitter

            candidate = np.clip(candidate, margin, self.window_size - margin)

            d_block = np.linalg.norm(candidate - block_pos)
            d_agent = np.linalg.norm(candidate - agent_pos)
            candidate_geom = sg.Point(candidate).buffer(obstacle_geom_radius)

            if d_block < (self.obstacle_radius + 48):
                continue
            if candidate_geom.intersects(goal_geom):
                continue
            if candidate_geom.intersects(block_geom):
                continue
            if d_agent < (self.obstacle_radius + 26):
                continue

            return candidate

        fallback = np.array([150.0, 150.0], dtype=np.float64)
        return fallback

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            pos_agent = np.array([rs.randint(50, 450), rs.randint(50, 450)], dtype=np.float64)
            goal_xy = self.goal_pose[:2]
            for _ in range(32):
                pos_block = np.array([rs.randint(100, 400), rs.randint(100, 400)], dtype=np.float64)
                if np.linalg.norm(pos_block - goal_xy) >= 90:
                    break
            rot_block = rs.randn() * 2 * np.pi - np.pi
            state_parts = [
                pos_agent[0],
                pos_agent[1],
                pos_block[0],
                pos_block[1],
                rot_block,
            ]
            if self.include_obstacle:
                pos_obstacle = self._sample_obstacle_position(
                    rs,
                    pos_block,
                    rot_block,
                    pos_agent,
                )
                state_parts.extend([pos_obstacle[0], pos_obstacle[1]])
            state = np.array(state_parts, dtype=np.float64)
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * dt

                self.space.step(dt)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return TeleopAgent(act)

    def _get_obs(self):
        obs = [
            *self.agent.position,
            *self.block.position,
            self.block.angle % (2 * np.pi),
        ]
        if self.include_obstacle:
            obs.extend(self.obstacle.position)
        return np.array(obs)

    def _get_goal_pose_body(self, pose):
        mass = 1
        if self.shape_type == "tee":
            inertia = pymunk.moment_for_box(mass, (50, 100))
            body = pymunk.Body(mass, inertia)
            body.position = pose[:2].tolist()
            body.angle = pose[2]
        elif self.shape_type == "square":
            side = 40
            inertia = pymunk.moment_for_box(mass, (side, side))
            body = pymunk.Body(mass, inertia)
            body.position = pose[:2].tolist()
            body.angle = pose[2]
        elif self.shape_type == "circle":
            inertia = pymunk.moment_for_circle(mass, 0, 30)
            body = pymunk.Body(mass, inertia)
            body.position = pose[:2].tolist()
            body.angle = pose[2]
        elif self.shape_type == "x":
            inertia = pymunk.moment_for_box(mass, (80, 80))
            body = pymunk.Body(mass, inertia)
            body.position = pose[:2].tolist()
            body.angle = pose[2]
        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "obstacle_pose": (
                np.array(self.obstacle.position) if self.include_obstacle else np.zeros(2)
            ),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            if isinstance(shape, pymunk.shapes.Poly):
                goal_points = [
                    pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface)
                    for v in shape.get_vertices()
                ]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(canvas, self.goal_color, goal_points)
            elif isinstance(shape, pymunk.shapes.Circle):
                pos = pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(shape.offset), draw_options.surface
                )
                radius = int(shape.radius)
                pygame.draw.circle(canvas, self.goal_color, pos, radius)
        self.space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8 / 96 * self.render_size)
                thickness = int(1 / 96 * self.render_size)
                cv2.drawMarker(
                    img,
                    coord,
                    color=(255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def get_state(self):
        """Return an exact dynamic-state snapshot for counterfactual branches."""
        state = {
            "agent_position": np.asarray(self.agent.position, dtype=np.float64),
            "agent_velocity": np.asarray(self.agent.velocity, dtype=np.float64),
            "agent_angle": float(self.agent.angle),
            "agent_angular_velocity": float(self.agent.angular_velocity),
            "block_position": np.asarray(self.block.position, dtype=np.float64),
            "block_velocity": np.asarray(self.block.velocity, dtype=np.float64),
            "block_angle": float(self.block.angle),
            "block_angular_velocity": float(self.block.angular_velocity),
            "latest_action": (
                None if self.latest_action is None else np.asarray(self.latest_action).copy()
            ),
            "n_contact_points": int(self.n_contact_points),
            "np_random_state": copy.deepcopy(self.np_random.bit_generator.state),
        }
        if self.include_obstacle:
            state["obstacle_position"] = np.asarray(
                self.obstacle.position, dtype=np.float64
            )
        return state

    def reset_to(self, state):
        """Restore a snapshot returned by :meth:`get_state`."""
        self.agent.position = np.asarray(state["agent_position"]).tolist()
        self.agent.velocity = np.asarray(state["agent_velocity"]).tolist()
        self.agent.angle = state["agent_angle"]
        self.agent.angular_velocity = state["agent_angular_velocity"]
        self.block.position = np.asarray(state["block_position"]).tolist()
        self.block.velocity = np.asarray(state["block_velocity"]).tolist()
        self.block.angle = state["block_angle"]
        self.block.angular_velocity = state["block_angular_velocity"]
        if self.include_obstacle and "obstacle_position" in state:
            self.obstacle.position = np.asarray(state["obstacle_position"]).tolist()
            self.space.reindex_shapes_for_body(self.obstacle)
        self.latest_action = (
            None
            if state["latest_action"] is None
            else np.asarray(state["latest_action"]).copy()
        )
        self.n_contact_points = int(state["n_contact_points"])
        self.np_random.bit_generator.state = copy.deepcopy(state["np_random_state"])
        return self._get_obs()

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        if self.include_obstacle and len(state) >= 7:
            pos_obstacle = state[5:7]
            self.obstacle.position = pos_obstacle
            self.space.reindex_shapes_for_body(self.obstacle)
        # Setting the angle rotates about the CoM and moves the geometric position,
        # so the order matters; legacy data set position first.
        if self.legacy:
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        if self.include_obstacle and len(state) >= 7:
            pos_obstacle = state[5:7]
            self.obstacle.position = pos_obstacle
            self.space.reindex_shapes_for_body(self.obstacle)

        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(translation=self.goal_pose[:2], rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2], rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(matrix=tf_img_obj.params @ tf_obj_new.params)
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) + [tf_img_new.rotation]
        )
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        self.agent = self.add_circle((256, 400), 15)
        if self.shape_type == "tee":
            self.block = self.add_tee((256, 300), 0)
        elif self.shape_type == "square":
            self.block = self.add_box((256, 300), 0, 40)
        elif self.shape_type == "circle":
            self.block = self.add_pushable_circle((256, 300), 30)
        elif self.shape_type == "x":
            self.block = self.add_x_shape((256, 300), 0)
        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")

        if self.include_obstacle:
            self.obstacle = self.add_fixed_circle((150, 150), self.obstacle_radius)
        else:
            self.obstacle = None

        self.goal_color = pygame.Color("LightGreen")
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)

        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_fixed_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("red")
        self.space.add(body, shape)
        return body

    def add_box(self, position, angle, side):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (side, side))
        body = pymunk.Body(mass, inertia)
        body.position = position
        body.angle = angle
        shape = pymunk.Poly.create_box(body, (side, side))
        shape.color = pygame.Color("LightSlateGray")
        self.space.add(body, shape)
        return body

    def add_pushable_circle(self, position, radius):
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("LightSlateGray")
        self.space.add(body, shape)
        return body

    def add_x_shape(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 4
        vertices1 = [
            (-scale / 2, -length * scale / 2),
            (scale / 2, -length * scale / 2),
            (scale / 2, length * scale / 2),
            (-scale / 2, length * scale / 2),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)

        vertices2 = [
            (-length * scale / 2, -scale / 2),
            (length * scale / 2, -scale / 2),
            (length * scale / 2, scale / 2),
            (-length * scale / 2, scale / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)

        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)

        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
