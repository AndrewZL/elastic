"""Privileged contact/proximity heuristic baseline for adaptive test-time compute.

Scores ``u_seq``, ``u_par`` in [0, 1] interpolate per-chunk denoising steps L and active samples P
within a band around a target (L, P), matched to ELASTIC's cost at the same penalties.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from elastic.dp.sampling.selectors import StrideSelectorBase


class _OnlineMeanStd:
    """Welford online mean/std for mapping V(s) → u_par."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        for v in np.asarray(values, dtype=np.float64).reshape(-1):
            if not np.isfinite(v):
                continue
            self.count += 1
            delta = float(v) - self.mean
            self.mean += delta / float(self.count)
            self.m2 += delta * (float(v) - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(1e-3, math.sqrt(self.m2 / float(self.count - 1)))


def v_scores_to_u_par(
    v: np.ndarray,
    norm: _OnlineMeanStd,
    warmup: int = 20,
    z_scale: float = 2.0,
) -> np.ndarray:
    """Map V(s) to u_par in [0, 1] by online z-score (low V -> more parallel); 0.5 during warmup."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if norm.count < int(warmup) or not np.isfinite(norm.mean):
        return np.full(v.shape[0], 0.5, dtype=np.float32)
    scale = max(1e-3, float(z_scale) * float(norm.std))
    z = (v - float(norm.mean)) / scale
    return _clip01(0.5 - 0.5 * z)


# Mode ids for logging: bit1 = u_seq >= 0.5, bit0 = u_par >= 0.5.
MODE_LOW_SEQ_LOW_PAR = 0
MODE_LOW_SEQ_HIGH_PAR = 1
MODE_HIGH_SEQ_LOW_PAR = 2
MODE_HIGH_SEQ_HIGH_PAR = 3

MODE_NAMES = {
    MODE_LOW_SEQ_LOW_PAR: "low_seq_low_par",
    MODE_LOW_SEQ_HIGH_PAR: "low_seq_high_par",
    MODE_HIGH_SEQ_LOW_PAR: "high_seq_low_par",
    MODE_HIGH_SEQ_HIGH_PAR: "high_seq_high_par",
}

# Measured ELASTIC (L, P) on Can_paired, keyed by (latency, parallel_penalty).
CAN_PAIRED_OPS_BUDGET: Dict[Tuple[float, float], Tuple[float, float]] = {
    (0.01, 0.05): (4.5078, 1.8565),
    (0.05, 0.05): (2.9355, 2.3608),
    (0.10, 0.10): (2.5555, 1.3436),
    (0.20, 0.10): (2.3317, 1.3643),
    (0.35, 0.20): (2.2622, 1.2047),
    (0.50, 0.20): (2.2245, 1.1559),
}


def _normalize_task(task: str) -> str:
    t = (task or "").strip().lower().replace("-", "_")
    if t in {"pusht", "pusht_obs", "push_t"}:
        return "pusht"
    if t in {"can", "can_paired", "can_reverse"}:
        return "can_paired"
    if t in {"square", "square_mh", "square_ph"}:
        return "square"
    raise ValueError(
        f"Unknown heuristic task {task!r}; expected pusht / can_paired / square"
    )


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def resolve_ops_budget(
    latency: Optional[float],
    parallel_penalty: Optional[float],
    target_L: Optional[float] = None,
    target_P: Optional[float] = None,
) -> Tuple[float, float, str]:
    """Resolve (target_L, target_P) from explicit targets or penalty table lookup."""
    if target_L is not None and target_P is not None:
        return float(target_L), float(target_P), "explicit_targets"

    if latency is None or parallel_penalty is None:
        raise ValueError(
            "Heuristic budget requires either (--target_inference_steps and "
            "--target_sample_usage) or (--latency and --parallel_penalty)"
        )

    lat = float(latency)
    par = float(parallel_penalty)
    for (la, pa), (L, P) in CAN_PAIRED_OPS_BUDGET.items():
        if abs(la - lat) < 1e-6 and abs(pa - par) < 1e-6:
            return float(L), float(P), f"ops_table({la:g},{pa:g})"

    best = None
    best_d = float("inf")
    for (la, pa), (L, P) in CAN_PAIRED_OPS_BUDGET.items():
        d = (math.log1p(la) - math.log1p(lat)) ** 2 + (pa - par) ** 2
        if d < best_d:
            best_d = d
            best = (float(L), float(P), f"ops_nearest({la:g},{pa:g})")
    assert best is not None
    return best


def extract_pusht_features(
    obs: np.ndarray, include_obstacle: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (agent_block_dist, block_center) from PushT keypoint obs."""
    if obs.ndim != 3:
        raise ValueError(f"PushT obs expected (B,T,F), got {obs.shape}")
    bsz, _, feat = obs.shape
    do = feat // 2
    numeric = obs[:, -1, :do].astype(np.float64)
    if include_obstacle:
        agent = numeric[:, -4:-2]
        block_flat = numeric[:, :-4]
    else:
        agent = numeric[:, -2:]
        block_flat = numeric[:, :-2]
    block_kps = block_flat.reshape(bsz, -1, 2)
    dists = np.linalg.norm(block_kps - agent[:, None, :], axis=-1)
    min_dist = dists.min(axis=1)
    block_center = block_kps.mean(axis=1)
    return min_dist.astype(np.float32), block_center.astype(np.float32)


def extract_robomimic_features(
    obs: np.ndarray,
    object_dim: int = 14,
    eef_dim: int = 3,
    quat_dim: int = 4,
    grip_dim: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ee_obj_dist, obj_pos, obj_z, gripper_width) from flat robomimic obs."""
    if obs.ndim != 3:
        raise ValueError(f"Robomimic obs expected (B,T,F), got {obs.shape}")
    flat = obs[:, -1, :].astype(np.float64)
    need = object_dim + eef_dim + quat_dim + grip_dim
    if flat.shape[-1] < need:
        raise ValueError(
            f"Robomimic obs feature dim {flat.shape[-1]} < expected {need} "
            f"(object+eef+quat+grip)"
        )
    obj = flat[:, :object_dim]
    grip = flat[
        :,
        object_dim + eef_dim + quat_dim : object_dim + eef_dim + quat_dim + grip_dim,
    ]
    obj_pos = obj[:, :3]
    to_eef = obj[:, 7:10]
    ee_obj_dist = np.linalg.norm(to_eef, axis=-1)
    obj_z = obj_pos[:, 2]
    gripper_width = np.abs(grip).sum(axis=-1)
    return (
        ee_obj_dist.astype(np.float32),
        obj_pos.astype(np.float32),
        obj_z.astype(np.float32),
        gripper_width.astype(np.float32),
    )


def compute_need_scores(
    task: str,
    *,
    agent_block_dist: Optional[np.ndarray] = None,
    coverage: Optional[np.ndarray] = None,
    ee_obj_dist: Optional[np.ndarray] = None,
    obj_z: Optional[np.ndarray] = None,
    init_obj_z: Optional[np.ndarray] = None,
    gripper_width: Optional[np.ndarray] = None,
    obj_pos: Optional[np.ndarray] = None,
    pusht_d_far: float = 80.0,
    pusht_coverage_done: float = 0.85,
    can_d_far: float = 0.20,
    can_lift_delta: float = 0.03,
    can_lift_scale: float = 0.04,
    square_d_far: float = 0.20,
    square_lift_delta: float = 0.02,
    square_lift_scale: float = 0.03,
    square_insert_xy_radius: float = 0.08,
    square_peg_xy: Tuple[float, float] = (0.23, 0.10),
) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous ``u_seq``, ``u_par`` ∈ [0, 1], shape (B,)."""
    task_n = _normalize_task(task)

    if task_n == "pusht":
        if agent_block_dist is None or coverage is None:
            raise ValueError("PushT heuristic requires agent_block_dist and coverage")
        d = np.asarray(agent_block_dist, dtype=np.float32)
        cov = np.asarray(coverage, dtype=np.float32)
        u_seq = _clip01(1.0 - d / float(pusht_d_far))
        u_par = _clip01(1.0 - cov / float(pusht_coverage_done))
        return u_seq, u_par

    if ee_obj_dist is None or obj_z is None or init_obj_z is None:
        raise ValueError("Robomimic heuristic requires ee_obj_dist, obj_z, init_obj_z")

    d = np.asarray(ee_obj_dist, dtype=np.float32)
    z = np.asarray(obj_z, dtype=np.float32)
    z0 = np.asarray(init_obj_z, dtype=np.float32)

    if task_n == "can_paired":
        lift = _clip01((z - z0 - float(can_lift_delta)) / max(1e-6, float(can_lift_scale)))
        prox = _clip01(1.0 - d / float(can_d_far))
        u_seq = prox * (1.0 - lift)
        # Pulse mode (can_par_mode="pulse") overrides this u_par in the selector.
        u_par = np.maximum(prox, lift)
        return u_seq, u_par

    lift = _clip01((z - z0 - float(square_lift_delta)) / max(1e-6, float(square_lift_scale)))
    prox = _clip01(1.0 - d / float(square_d_far))
    if obj_pos is None:
        insert = np.zeros_like(lift, dtype=np.float32)
    else:
        xy = np.asarray(obj_pos, dtype=np.float32)[:, :2]
        peg = np.asarray(square_peg_xy, dtype=np.float32).reshape(1, 2)
        peg_dist = np.linalg.norm(xy - peg, axis=-1)
        insert = _clip01(1.0 - peg_dist / float(square_insert_xy_radius)) * lift
    u_seq = np.maximum(prox * (1.0 - lift), insert)
    u_par = (1.0 - lift) * (1.0 - prox)
    return u_seq.astype(np.float32), u_par.astype(np.float32)


def flags_to_mode(need_seq: np.ndarray, need_par: np.ndarray) -> np.ndarray:
    return (np.asarray(need_seq, dtype=np.int32) << 1) | np.asarray(need_par, dtype=np.int32)


def pulse_from_age(
    age: np.ndarray,
    hold: int = 2,
    decay: int = 6,
) -> np.ndarray:
    """Map pulse age in chunks (-1 = not started) to u_par: 1 for ``hold`` chunks, then linear decay over ``decay``."""
    age = np.asarray(age, dtype=np.float32)
    hold = max(0, int(hold))
    decay = max(1, int(decay))
    u = np.zeros_like(age, dtype=np.float32)
    active = age >= 0
    if not np.any(active):
        return u
    a = age[active]
    peak = a < hold
    mid = (a >= hold) & (a < hold + decay)
    u_active = np.zeros_like(a, dtype=np.float32)
    u_active[peak] = 1.0
    u_active[mid] = 1.0 - (a[mid] - hold) / float(decay)
    u[active] = np.clip(u_active, 0.0, 1.0)
    return u


def scores_to_mode(u_seq: np.ndarray, u_par: np.ndarray) -> np.ndarray:
    return flags_to_mode(u_seq >= 0.5, u_par >= 0.5)


class ContactProximityHeuristicSelector(StrideSelectorBase):
    """Contact/proximity heuristic: scores interpolate L/P in [target * lo_frac, target * hi_frac].

    Deficit correction (as in ``BudgetRandomStrideSelector``) keeps realized mean L/P near target.
    """

    def __init__(
        self,
        task: str,
        device,
        samples_per_env: int = 4,
        n_actions=None,
        t_max=None,
        action_dim=None,
        stride_buckets=None,
        action_range=None,
        latency: Optional[float] = None,
        parallel_penalty: Optional[float] = None,
        target_inference_steps: Optional[float] = None,
        target_sample_usage: Optional[float] = None,
        lo_frac: float = 0.5,
        hi_frac: float = 1.5,
        deficit_gain: float = 0.7,
        pusht_d_far: float = 80.0,
        pusht_coverage_done: float = 0.85,
        can_d_far: float = 0.20,
        can_lift_delta: float = 0.03,
        can_lift_scale: float = 0.04,
        par_source: str = "task",
        v_warmup: int = 20,
        v_z_scale: float = 2.0,
        can_par_mode: str = "pulse",
        can_pulse_hold: int = 2,
        can_pulse_decay: int = 6,
        can_pulse_start_prox: float = 0.5,
        square_d_far: float = 0.20,
        square_lift_delta: float = 0.02,
        square_lift_scale: float = 0.03,
        square_insert_xy_radius: float = 0.08,
        square_peg_xy: Tuple[float, float] = (0.23, 0.10),
        include_obstacle: bool = False,
        **kwargs,
    ):
        super().__init__(
            device,
            n_actions,
            t_max,
            action_dim,
            samples_per_env,
            stride_buckets=stride_buckets,
            action_range=action_range,
        )
        self.task = _normalize_task(task)
        self.lo_frac = float(lo_frac)
        self.hi_frac = float(hi_frac)
        self.deficit_gain = float(deficit_gain)

        self.target_L, self.target_P, self.budget_source = resolve_ops_budget(
            latency=latency,
            parallel_penalty=parallel_penalty,
            target_L=target_inference_steps,
            target_P=target_sample_usage,
        )
        self.L_lo = max(1.0, self.target_L * self.lo_frac)
        self.L_hi = max(self.L_lo + 1e-3, self.target_L * self.hi_frac)
        self.P_lo = max(1.0, self.target_P * self.lo_frac)
        self.P_hi = max(
            self.P_lo + 1e-3,
            min(float(samples_per_env), self.target_P * self.hi_frac),
        )

        self.pusht_d_far = float(pusht_d_far)
        self.pusht_coverage_done = float(pusht_coverage_done)
        self.can_d_far = float(can_d_far)
        self.can_lift_delta = float(can_lift_delta)
        self.can_lift_scale = float(can_lift_scale)
        self.par_source = str(par_source).strip().lower()
        if self.par_source not in {"task", "v_s", "fixed"}:
            raise ValueError(
                f"par_source must be 'task', 'v_s', or 'fixed', got {par_source!r}"
            )
        self.v_warmup = int(v_warmup)
        self.v_z_scale = float(v_z_scale)
        self.can_par_mode = str(can_par_mode).strip().lower()
        if self.can_par_mode not in {"pulse", "soft_lift"}:
            raise ValueError(
                f"can_par_mode must be 'pulse' or 'soft_lift', got {can_par_mode!r}"
            )
        self.can_pulse_hold = int(can_pulse_hold)
        self.can_pulse_decay = int(can_pulse_decay)
        self.can_pulse_start_prox = float(can_pulse_start_prox)
        self.square_d_far = float(square_d_far)
        self.square_lift_delta = float(square_lift_delta)
        self.square_lift_scale = float(square_lift_scale)
        self.square_insert_xy_radius = float(square_insert_xy_radius)
        self.square_peg_xy = tuple(square_peg_xy)
        self.include_obstacle = bool(include_obstacle)

        self._obs = None
        self._coverage = None
        self._init_obj_z = None
        self._post_lift_age = None  # (B,) chunks since pulse start; -1 = not started
        self._v_scores = None
        self._v_norm = _OnlineMeanStd()
        self._last_modes = None
        self._last_u_seq = None
        self._last_u_par = None
        self._last_v = None
        self._deficit_L = 0.0
        self._deficit_P = 0.0
        self._u_seq_ema = 0.5
        self._u_par_ema = 0.5
        self._u_ema_count = 0

        print(
            f"[ContactProximityHeuristic] task={self.task} budget={self.budget_source} "
            f"target_L={self.target_L:.3f} target_P={self.target_P:.3f} "
            f"L∈[{self.L_lo:.2f},{self.L_hi:.2f}] P∈[{self.P_lo:.2f},{self.P_hi:.2f}] "
            f"par_source={self.par_source} can_par_mode={self.can_par_mode} "
            f"deficit_gain={self.deficit_gain:.2f}"
        )

    def set_current_max_coverage(self, max_coverage):
        if max_coverage is None:
            self._coverage = None
            return
        self._coverage = np.asarray(max_coverage, dtype=np.float32).reshape(-1)

    def set_heuristic_obs(self, obs: np.ndarray):
        if isinstance(obs, dict):
            raise TypeError(
                "ContactProximityHeuristicSelector expects a flat ndarray obs; "
                "got dict (image mode not supported)"
            )
        self._obs = np.asarray(obs)
        if self.task not in {"can_paired", "square"}:
            return

        _, _, obj_z, _ = extract_robomimic_features(self._obs)
        b = int(obj_z.shape[0])
        if self._init_obj_z is None:
            self._init_obj_z = obj_z.copy()
        if self._post_lift_age is None or self._post_lift_age.shape[0] != b:
            self._post_lift_age = np.full(b, -1, dtype=np.int32)

        if (
            self.par_source == "task"
            and self.task == "can_paired"
            and self.can_par_mode == "pulse"
        ):
            dist, _, obj_z, _ = extract_robomimic_features(self._obs)
            prox = _clip01(1.0 - dist / max(1e-6, self.can_d_far))
            lifted = obj_z > (self._init_obj_z + self.can_lift_delta)
            start = (prox >= self.can_pulse_start_prox) | lifted
            age = self._post_lift_age
            # Once started, keep ticking even if z drops after placing.
            started = age >= 0
            newly = (~started) & start
            age[started] = age[started] + 1
            age[newly] = 0
            self._post_lift_age = age

    def set_v_scores(self, v_scores: np.ndarray):
        """Cache V(s) per env for ``par_source='v_s'``."""
        v = np.asarray(v_scores, dtype=np.float32).reshape(-1)
        self._v_scores = v
        self._v_norm.update(v)

    def record_actual_usage(self, actual_L: float, actual_P: float):
        """Cross-episode deficit correction toward target_L / target_P."""
        if np.isfinite(actual_L):
            self._deficit_L += self.target_L - float(actual_L)
        if np.isfinite(actual_P):
            self._deficit_P += self.target_P - float(actual_P)

    def reset_episode(self):
        super().reset_episode()
        self._last_modes = None
        self._last_u_seq = None
        self._last_u_par = None

    def _apply_par_source(
        self, u_seq: np.ndarray, u_par: np.ndarray, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.par_source == "fixed":
            u_par = np.full(batch_size, 0.5, dtype=np.float32)
        elif self.par_source == "v_s":
            if self._v_scores is None or self._v_scores.shape[0] != batch_size:
                u_par = np.full(batch_size, 0.5, dtype=np.float32)
                self._last_v = None
            else:
                self._last_v = self._v_scores.copy()
                u_par = v_scores_to_u_par(
                    self._v_scores,
                    self._v_norm,
                    warmup=self.v_warmup,
                    z_scale=self.v_z_scale,
                )
        return u_seq, u_par

    def _scores(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._obs is None:
            z = np.zeros(batch_size, dtype=np.float32)
            return self._apply_par_source(z, z.copy(), batch_size)

        if self.task == "pusht":
            dist, _ = extract_pusht_features(self._obs, include_obstacle=self.include_obstacle)
            cov = self._coverage
            if cov is None:
                cov = np.zeros(batch_size, dtype=np.float32)
            elif cov.shape[0] != batch_size:
                cov = np.resize(cov, batch_size)
            u_seq, u_par = compute_need_scores(
                "pusht",
                agent_block_dist=dist,
                coverage=cov,
                pusht_d_far=self.pusht_d_far,
                pusht_coverage_done=self.pusht_coverage_done,
            )
            return self._apply_par_source(u_seq, u_par, batch_size)

        dist, obj_pos, obj_z, grip_w = extract_robomimic_features(self._obs)
        if self._init_obj_z is None:
            self._init_obj_z = obj_z.copy()
        init_z = self._init_obj_z
        if init_z.shape[0] != batch_size:
            init_z = np.resize(init_z, batch_size)
        u_seq, u_par = compute_need_scores(
            self.task,
            ee_obj_dist=dist,
            obj_z=obj_z,
            init_obj_z=init_z,
            gripper_width=grip_w,
            obj_pos=obj_pos,
            can_d_far=self.can_d_far,
            can_lift_delta=self.can_lift_delta,
            can_lift_scale=self.can_lift_scale,
            square_d_far=self.square_d_far,
            square_lift_delta=self.square_lift_delta,
            square_lift_scale=self.square_lift_scale,
            square_insert_xy_radius=self.square_insert_xy_radius,
            square_peg_xy=self.square_peg_xy,
        )
        if (
            self.par_source == "task"
            and self.task == "can_paired"
            and self.can_par_mode == "pulse"
        ):
            if self._post_lift_age is None or self._post_lift_age.shape[0] != batch_size:
                age = np.full(batch_size, -1, dtype=np.int32)
            else:
                age = self._post_lift_age
            u_par = pulse_from_age(
                age, hold=self.can_pulse_hold, decay=self.can_pulse_decay
            )
        return self._apply_par_source(u_seq, u_par, batch_size)

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        **kwargs,
    ):
        n = sample.shape[0]
        k = int(self.samples_per_env if self.samples_per_env else 1)
        b = n // k
        if self.t_max is None:
            raise ValueError("t_max must be set for continuous heuristic")

        u_seq, u_par = self._scores(b)
        self._last_u_seq = u_seq
        self._last_u_par = u_par
        self._last_modes = scores_to_mode(u_seq, u_par)

        u_seq_m = float(np.mean(u_seq)) if u_seq.size else 0.5
        u_par_m = float(np.mean(u_par)) if u_par.size else 0.5
        self._u_ema_count += 1
        alpha = 1.0 / float(min(self._u_ema_count, 50))
        self._u_seq_ema = (1.0 - alpha) * self._u_seq_ema + alpha * u_seq_m
        self._u_par_ema = (1.0 - alpha) * self._u_par_ema + alpha * u_par_m

        eff_L = self.target_L + self.deficit_gain * self._deficit_L
        eff_P = self.target_P + self.deficit_gain * self._deficit_P
        t_max = int(self.t_max)
        L_lo = max(1.0, eff_L * self.lo_frac)
        L_hi = max(L_lo + 1e-3, eff_L * self.hi_frac)
        P_lo = max(1.0, eff_P * self.lo_frac)
        P_hi = max(P_lo + 1e-3, min(float(k), eff_P * self.hi_frac))
        L_lo = float(np.clip(L_lo, 1.0, float(t_max)))
        L_hi = float(np.clip(max(L_hi, L_lo + 1e-3), 1.0, float(t_max)))
        P_lo = float(np.clip(P_lo, 1.0, float(k)))
        P_hi = float(np.clip(max(P_hi, P_lo + 1e-3), 1.0, float(k)))

        L_t = L_lo + u_seq * (L_hi - L_lo)
        P_t = P_lo + u_par * (P_hi - P_lo)
        # Recenter on the EMA score means so E[L], E[P] hit the targets without flattening the schedule.
        L_expected = L_lo + self._u_seq_ema * (L_hi - L_lo)
        P_expected = P_lo + self._u_par_ema * (P_hi - P_lo)
        L_t = L_t + (eff_L - L_expected)
        P_t = P_t + (eff_P - P_expected)
        L_t = np.clip(L_t, 1.0, float(t_max))
        P_t = np.clip(P_t, 1.0, float(k))

        per_env = torch.zeros((b, k), device=self.device, dtype=torch.long)
        for i in range(b):
            n_steps = int(max(1, min(t_max, round(float(L_t[i])))))
            stride_i = int(math.ceil((t_max - 1) / float(n_steps)))
            n_active = int(max(1, min(k, round(float(P_t[i])))))
            per_env[i, :n_active] = stride_i

        proposed = per_env.reshape(-1)
        out, _ = self._apply_no_restart_constraint(proposed, timestep)
        return out
