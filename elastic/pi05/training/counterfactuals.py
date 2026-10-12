from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from elastic.pi05.constants import (
    ALL_STOPPED_COUNTERFACTUALS,
    K_SUBSAMPLE_ADD_COUNTERFACTUALS,
    K_SUBSAMPLE_DRAWS,
    LATENCY_PENALTY,
    NO_FINISH_PENALTY,
    PARALLEL_PENALTY,
    REPLAY_STRIDE_INFER_IDX_ALL_STOPPED,
    REPLAY_STRIDE_INFER_IDX_K_SUBSAMPLE,
    REPLAY_STRIDE_INFER_IDX_SECOND_PASS,
    SCORE_ENCODING_ADVANTAGE,
    SCORE_ENCODING_RAW_Q,
    STRIDE_INFER_META_DIM,
    STRIDE_REPLAY_CAPACITY,
    TERMINAL_BONUS,
    VERIFIER_BIAS,
)
from elastic.pi05.training.replay import NumpyReplayBuffer, save_replay_npz
from elastic.pi05.training.replay import stride_infer_meta_row_from_npz
from elastic.pi05.training.reward import TransitionAccumulator
from elastic.pi05.training.reward import _pad_flat_obs_to_replay
from elastic.pi05.sampling.selectors import delta_to_action


def q_scores_to_advantage(scores: np.ndarray) -> np.ndarray:
    """A(s,a_k) = Q(s,a_k) - mean_k Q(s,a_k), the mean over finite scores (unfinished ``-inf`` excluded)."""
    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = arr.copy()
    for b_idx in range(out.shape[0]):
        row = out[b_idx]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        baseline = float(np.mean(finite))
        out[b_idx] = np.where(np.isfinite(row), row - baseline, row)
    return out


def finite_q_mean(scores: np.ndarray) -> float:
    """Mean over finite Q values (ignores ``-inf`` unfinished candidates)."""
    q = np.asarray(scores, dtype=np.float32).reshape(-1)
    finite = q[np.isfinite(q)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def q_scores_to_advantage_with_baseline(scores: np.ndarray, baseline: float) -> np.ndarray:
    """A(s,a) = Q(s,a) - V(s) with an explicit scalar baseline (e.g. second-pass mean Q)."""
    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    b = float(baseline)
    return np.where(np.isfinite(arr), arr - b, arr)


def _subsample_obs_tail_indices(sample_dim: int) -> tuple[int, int]:
    """Return ``(active_frac_idx, stopped_idx)`` for ``sample_obs`` layout."""
    # Minimal obs: [..., t_norm, active_frac, stopped] (3 tail dims).
    # Extended obs: [..., t_norm, active_frac, stopped, score_mag, x0_dev] (5 tail dims).
    if sample_dim >= 5:
        return -4, -3
    return -2, -1


def build_replay_from_record_dir(
    record_dir: str | Path | Sequence[str | Path],
    *,
    output_path: str | Path,
    num_samples: int,
    capacity: int = STRIDE_REPLAY_CAPACITY,
    terminal_bonus: float = TERMINAL_BONUS,
    latency_penalty: float = LATENCY_PENALTY,
    parallel_penalty: float = PARALLEL_PENALTY,
    no_finish_penalty: float = NO_FINISH_PENALTY,
    verifier_bias: float = VERIFIER_BIAS,
    add_k_subsample: bool = K_SUBSAMPLE_ADD_COUNTERFACTUALS,
    k_subsample_draws: int = K_SUBSAMPLE_DRAWS,
    add_all_stopped: bool = ALL_STOPPED_COUNTERFACTUALS,
    use_advantage: bool = True,
    max_records: int | None = None,
) -> NumpyReplayBuffer:
    """Convert per-inference STRIDE records into a replay buffer for offline pretraining."""
    record_dirs = [record_dir] if isinstance(record_dir, (str, Path)) else list(record_dir)
    paths = []
    for item in record_dirs:
        paths.extend(sorted(Path(item).glob("stride_step_*.npz")))
    paths = sorted(paths)
    if max_records is not None:
        paths = paths[: int(max_records)]
    if not paths:
        searched = [str(Path(item)) for item in record_dirs]
        raise ValueError(
            "No stride_step_*.npz files found. "
            f"Searched directories: {searched}. "
            "If running via docker compose, make sure the policy server writes replay files to a mounted path "
            "(for this repo, prefer /app/data/... inside the container)."
        )

    replay = None
    accumulator = TransitionAccumulator(
        n_envs=1,
        num_action_samples=num_samples,
        terminal_bonus=terminal_bonus,
        latency_penalty=latency_penalty,
        parallel_penalty=parallel_penalty,
        no_finish_penalty=no_finish_penalty,
        verifier_bias=verifier_bias,
    )
    for path in paths:
        data = dict(np.load(path, allow_pickle=True))
        transitions = _transitions_from_record(data)
        if replay is None:
            first = transitions[0]
            replay = NumpyReplayBuffer(
                capacity=capacity,
                action_shape=first["action"].shape[1:],
                state_dim=int(first["state_obs"].shape[-1]),
                sample_dim=int(first["sample_obs"].shape[-1]),
                num_samples=num_samples,
            )
        scores = np.asarray(data["verifier_scores"], dtype=np.float32).reshape(1, num_samples)
        if use_advantage:
            scores = q_scores_to_advantage(scores)
        human_es = float(
            np.asarray(data.get("robot_episode_success", np.array([1.0], dtype=np.float32))).reshape(-1)[0]
        )
        human_es = float(np.clip(human_es, 0.0, 1.0))
        ep_axis = int(np.asarray(data.get("robot_episode_idx", np.array([-1], dtype=np.int32))).reshape(-1)[0])
        infer_axis = int(
            np.asarray(data.get("stride_infer_idx_in_episode", np.array([-1], dtype=np.int32))).reshape(-1)[0]
        )
        meta_row = stride_infer_meta_row_from_npz(data)
        accumulator.add_to_replay(
            transitions,
            replay,
            scores,
            robot_episode_success_log=np.array([human_es], dtype=np.float32),
            robot_episode_idx=ep_axis,
            stride_infer_idx_in_episode=infer_axis,
            stride_infer_meta=meta_row,
        )
        if add_k_subsample:
            add_k_subsample_counterfactuals(
                transitions,
                replay,
                scores,
                terminal_bonus=terminal_bonus,
                latency_penalty=latency_penalty,
                parallel_penalty=parallel_penalty,
                no_finish_penalty=no_finish_penalty,
                verifier_bias=verifier_bias,
                num_draws=k_subsample_draws,
                robot_episode_success_log=float(human_es),
                robot_episode_idx=ep_axis,
                stride_infer_meta_row=meta_row,
                use_extended_obs=bool(int(first["sample_obs"].shape[-1]) >= 5),
            )
        if add_all_stopped:
            add_all_stopped_counterfactuals(
                transitions,
                replay,
                no_finish_penalty=no_finish_penalty,
                latency_penalty=latency_penalty,
                parallel_penalty=parallel_penalty,
                robot_episode_success_log=float(human_es),
                robot_episode_idx=ep_axis,
                stride_infer_meta_row=meta_row,
                use_extended_obs=bool(int(first["sample_obs"].shape[-1]) >= 5),
            )

    if replay is None:
        raise RuntimeError("Replay was not initialized")
    encoding = SCORE_ENCODING_ADVANTAGE if use_advantage else SCORE_ENCODING_RAW_Q
    save_replay_npz(
        replay,
        output_path,
        extra_metadata={
            "saved_score_encoding": np.asarray(encoding),
            "saved_terminal_bonus": np.asarray(float(terminal_bonus), dtype=np.float32),
            "saved_verifier_bias": np.asarray(float(verifier_bias), dtype=np.float32),
            "saved_latency_penalty": np.asarray(float(latency_penalty), dtype=np.float32),
            "saved_parallel_penalty": np.asarray(float(parallel_penalty), dtype=np.float32),
            "saved_no_finish_penalty": np.asarray(float(no_finish_penalty), dtype=np.float32),
        },
    )
    return replay


def add_k_subsample_counterfactuals(
    transitions: list[dict[str, np.ndarray]],
    replay: NumpyReplayBuffer,
    verifier_scores: np.ndarray,
    *,
    terminal_bonus: float,
    latency_penalty: float,
    parallel_penalty: float,
    no_finish_penalty: float = NO_FINISH_PENALTY,
    verifier_bias: float = VERIFIER_BIAS,
    num_draws: int = K_SUBSAMPLE_DRAWS,
    k_subsample_sizes: list[int] | None = None,
    rng: np.random.Generator | None = None,
    robot_episode_success_log: float = 1.0,
    robot_episode_idx: int = -1,
    stride_infer_meta_row: np.ndarray | None = None,
    use_extended_obs: bool | None = None,
) -> int:
    """Add K-subsampling counterfactual rows: a subset of the K scored candidates is kept, the rest stop.

    Terminal reward uses the subset's winner quality, latency and parallel cost.
    ``robot_episode_success_log`` only fills ``reward_components[:, 3]``.
    """
    if not transitions:
        return 0

    scores = np.asarray(verifier_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"Expected verifier_scores with shape (B, K), got {scores.shape}")
    b_count, k_count = scores.shape
    if k_count <= 1:
        return 0

    if k_subsample_sizes is None:
        k_subsample_sizes = sorted({2**idx for idx in range(int(np.log2(k_count)) + 1) if 2**idx <= k_count})
    k_subsample_sizes = [int(k) for k in k_subsample_sizes if 0 < int(k) < k_count]
    if not k_subsample_sizes:
        return 0

    rng = rng or np.random.default_rng()
    meta_cf = np.zeros((int(STRIDE_INFER_META_DIM),), dtype=np.float32)
    if stride_infer_meta_row is not None:
        mr = np.asarray(stride_infer_meta_row, dtype=np.float32).reshape(-1)
        n = min(int(mr.size), int(STRIDE_INFER_META_DIM))
        meta_cf[:n] = mr[:n]
    first = transitions[0]
    state = np.asarray(first["state_obs"], dtype=np.float32)
    sample = np.asarray(first["sample_obs"], dtype=np.float32)
    next_state = np.asarray(first["next_state_obs"], dtype=np.float32)
    next_sample = np.asarray(first["next_sample_obs"], dtype=np.float32)
    action = np.asarray(first["action"], dtype=np.float32)
    if use_extended_obs is None:
        use_extended_obs = int(sample.shape[-1]) >= 5
    if state.shape[:2] != (b_count, k_count):
        raise ValueError(f"Transition shape {state.shape[:2]} does not match verifier scores {(b_count, k_count)}")

    active = np.stack([transition["active"] for transition in transitions]).astype(np.float32)
    finished = np.stack([transition["done"] for transition in transitions]).astype(np.bool_)
    l_k = active.sum(axis=0)
    finished_any = np.any(finished, axis=0)
    stop_action = float(delta_to_action(np.array([0.0], dtype=np.float32), (0.0, 1.0))[0])
    added = 0

    for k_sub in k_subsample_sizes:
        for _ in range(max(1, int(num_draws))):
            kept = np.stack([rng.choice(k_count, size=k_sub, replace=False) for _ in range(b_count)])
            kept_mask = np.zeros((b_count, k_count), dtype=np.bool_)
            for b_idx in range(b_count):
                kept_mask[b_idx, kept[b_idx]] = True

            sample_cf = _patch_sample_obs_for_subsample(sample, kept_mask, use_extended_obs=use_extended_obs)
            next_sample_cf = _patch_sample_obs_for_subsample(next_sample, kept_mask, use_extended_obs=use_extended_obs)
            obs_cf = np.concatenate([state[:, 0, :], sample_cf.reshape(b_count, -1)], axis=-1)
            next_obs_cf = np.concatenate([next_state[:, 0, :], next_sample_cf.reshape(b_count, -1)], axis=-1)
            action_cf = np.full_like(action, stop_action, dtype=np.float32)
            action_cf[kept_mask] = action[kept_mask]

            for b_idx in range(b_count):
                kept_idx = kept[b_idx]
                kept_finished = finished_any[b_idx, kept_idx]
                if np.any(kept_finished):
                    best_score = float(np.max(scores[b_idx, kept_idx][kept_finished]))
                    best_quality = (best_score + float(verifier_bias)) * float(terminal_bonus)
                else:
                    best_quality = float(no_finish_penalty)
                kept_l = l_k[b_idx, kept_idx]
                latency_steps = float(np.max(kept_l)) if kept_l.size else 0.0
                safe_latency = max(latency_steps, 1.0)
                sum_l = float(np.sum(kept_l))
                norm_excess = max(sum_l - latency_steps, 0.0) / safe_latency
                reward = (
                    best_quality
                    - float(latency_penalty) * latency_steps
                    - float(parallel_penalty) * norm_excess
                )
                o_cf = _pad_flat_obs_to_replay(obs_cf[b_idx], replay.obs_dim)
                no_cf = _pad_flat_obs_to_replay(next_obs_cf[b_idx], replay.obs_dim)
                traj_cf = np.asarray(
                    [int(robot_episode_idx), int(REPLAY_STRIDE_INFER_IDX_K_SUBSAMPLE)],
                    dtype=np.int64,
                )
                replay.add(
                    obs=o_cf,
                    next_obs=no_cf,
                    action=action_cf[b_idx],
                    reward=np.asarray([reward], dtype=np.float32),
                    done=np.ones((1,), dtype=np.float32),
                    reward_components=np.asarray(
                        [best_quality, latency_steps, norm_excess, float(robot_episode_success_log)],
                        dtype=np.float32,
                    ),
                    traj_info=traj_cf,
                    lifecycle=-1,
                    stride_infer_meta=meta_cf,
                )
                added += 1
    return added


def add_all_stopped_counterfactuals(
    transitions: list[dict[str, np.ndarray]],
    replay: NumpyReplayBuffer,
    *,
    no_finish_penalty: float = NO_FINISH_PENALTY,
    latency_penalty: float = LATENCY_PENALTY,
    parallel_penalty: float = PARALLEL_PENALTY,
    robot_episode_success_log: float = 1.0,
    robot_episode_idx: int = -1,
    stride_infer_meta_row: np.ndarray | None = None,
    use_extended_obs: bool | None = None,
) -> int:
    """One row per env with all K samples paused, so the critic learns ``Q(s, all-stop) ≈ no_finish_penalty``."""
    if not transitions:
        return 0

    first = transitions[0]
    state = np.asarray(first["state_obs"], dtype=np.float32)
    sample = np.asarray(first["sample_obs"], dtype=np.float32)
    next_state = np.asarray(first["next_state_obs"], dtype=np.float32)
    next_sample = np.asarray(first["next_sample_obs"], dtype=np.float32)
    action = np.asarray(first["action"], dtype=np.float32)
    b_count, k_count = int(state.shape[0]), int(state.shape[1])
    if b_count == 0 or k_count == 0:
        return 0

    meta_cf = np.zeros((int(STRIDE_INFER_META_DIM),), dtype=np.float32)
    if stride_infer_meta_row is not None:
        mr = np.asarray(stride_infer_meta_row, dtype=np.float32).reshape(-1)
        n = min(int(mr.size), int(STRIDE_INFER_META_DIM))
        meta_cf[:n] = mr[:n]

    if use_extended_obs is None:
        use_extended_obs = int(sample.shape[-1]) >= 5

    kept_mask = np.zeros((b_count, k_count), dtype=np.bool_)
    stop_action = float(delta_to_action(np.array([0.0], dtype=np.float32), (0.0, 1.0))[0])
    sample_cf = _patch_sample_obs_for_subsample(sample, kept_mask, use_extended_obs=use_extended_obs)
    next_sample_cf = _patch_sample_obs_for_subsample(next_sample, kept_mask, use_extended_obs=use_extended_obs)
    obs_cf = np.concatenate([state[:, 0, :], sample_cf.reshape(b_count, -1)], axis=-1)
    next_obs_cf = np.concatenate([next_state[:, 0, :], next_sample_cf.reshape(b_count, -1)], axis=-1)
    action_cf = np.full_like(action, stop_action, dtype=np.float32)

    quality = float(no_finish_penalty)
    latency_steps = 0.0
    norm_excess = 0.0
    reward = quality - float(latency_penalty) * latency_steps - float(parallel_penalty) * norm_excess

    added = 0
    for b_idx in range(b_count):
        o_cf = _pad_flat_obs_to_replay(obs_cf[b_idx], replay.obs_dim)
        no_cf = _pad_flat_obs_to_replay(next_obs_cf[b_idx], replay.obs_dim)
        traj_cf = np.asarray(
            [int(robot_episode_idx), int(REPLAY_STRIDE_INFER_IDX_ALL_STOPPED)],
            dtype=np.int64,
        )
        replay.add(
            obs=o_cf,
            next_obs=no_cf,
            action=action_cf[b_idx],
            reward=np.asarray([reward], dtype=np.float32),
            done=np.ones((1,), dtype=np.float32),
            reward_components=np.asarray(
                [quality, latency_steps, norm_excess, float(robot_episode_success_log)],
                dtype=np.float32,
            ),
            traj_info=traj_cf,
            lifecycle=-1,
            stride_infer_meta=meta_cf,
        )
        added += 1
    return added


def add_second_pass_counterfactuals(
    cf_transitions: list[dict[str, np.ndarray]],
    replay: NumpyReplayBuffer,
    verifier_scores: np.ndarray,
    *,
    cf_stride_norm: float,
    terminal_bonus: float,
    latency_penalty: float,
    parallel_penalty: float,
    no_finish_penalty: float = NO_FINISH_PENALTY,
    verifier_bias: float = VERIFIER_BIAS,
    robot_episode_success_log: float = 1.0,
    robot_episode_idx: int = -1,
    stride_infer_meta_row: np.ndarray | None = None,
    use_extended_obs: bool | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Second-pass fixed-stride counterfactual rows (online only).

    Subsets of size 2^i up to K, with subset quality, CF-stride latency and full-K parallel baseline.
    """
    if not cf_transitions:
        return 0

    scores = np.asarray(verifier_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"Expected verifier_scores with shape (B, K), got {scores.shape}")
    b_count, k_count = scores.shape
    if k_count <= 1:
        return 0

    cf_lat_steps = max(1.0, 1.0 / max(float(cf_stride_norm), 1e-6))
    cf_lat_cost = float(latency_penalty) * cf_lat_steps
    cf_active_action = float(delta_to_action(np.array([float(cf_stride_norm)], dtype=np.float32), (0.0, 1.0))[0])
    stop_action = float(delta_to_action(np.array([0.0], dtype=np.float32), (0.0, 1.0))[0])

    k_sub_sizes = [2**i for i in range(int(np.log2(max(k_count, 1))) + 1) if 2**i <= k_count]
    rng = rng or np.random.default_rng()
    meta_cf = np.zeros((int(STRIDE_INFER_META_DIM),), dtype=np.float32)
    if stride_infer_meta_row is not None:
        mr = np.asarray(stride_infer_meta_row, dtype=np.float32).reshape(-1)
        n = min(int(mr.size), int(STRIDE_INFER_META_DIM))
        meta_cf[:n] = mr[:n]

    if use_extended_obs is None and cf_transitions:
        sample_dim = int(np.asarray(cf_transitions[0]["sample_obs"]).shape[-1])
        use_extended_obs = sample_dim >= 5

    added = 0
    for cf_tr in cf_transitions:
        state = np.asarray(cf_tr["state_obs"], dtype=np.float32)
        sample = np.asarray(cf_tr["sample_obs"], dtype=np.float32)
        next_state = np.asarray(cf_tr["next_state_obs"], dtype=np.float32)
        next_sample = np.asarray(cf_tr["next_sample_obs"], dtype=np.float32)
        action = np.asarray(cf_tr["action"], dtype=np.float32)

        for k_sub in k_sub_sizes:
            for b_idx in range(b_count):
                if k_sub == k_count:
                    kept_idx = np.arange(k_count)
                else:
                    kept_idx = rng.choice(k_count, size=k_sub, replace=False)
                kept_mask = np.zeros((b_count, k_count), dtype=np.bool_)
                kept_mask[b_idx, kept_idx] = True

                sample_cf = _patch_sample_obs_for_subsample(sample, kept_mask, use_extended_obs=use_extended_obs)
                next_sample_cf = _patch_sample_obs_for_subsample(
                    next_sample, kept_mask, use_extended_obs=use_extended_obs
                )
                obs_cf = np.concatenate([state[:, 0, :], sample_cf.reshape(b_count, -1)], axis=-1)
                next_obs_cf = np.concatenate([next_state[:, 0, :], next_sample_cf.reshape(b_count, -1)], axis=-1)
                action_cf = np.full_like(action, stop_action, dtype=np.float32)
                action_cf[kept_mask] = cf_active_action

                best_score = float(np.max(scores[b_idx, kept_idx]))
                quality_term = (best_score + float(verifier_bias)) * float(terminal_bonus)
                par_excess = max(int(k_sub) - 1, 0) * cf_lat_steps
                par_cost = float(parallel_penalty) * par_excess / cf_lat_steps
                reward = quality_term - cf_lat_cost - par_cost

                o_cf = _pad_flat_obs_to_replay(obs_cf[b_idx], replay.obs_dim)
                no_cf = _pad_flat_obs_to_replay(next_obs_cf[b_idx], replay.obs_dim)
                traj_cf = np.asarray(
                    [int(robot_episode_idx), int(REPLAY_STRIDE_INFER_IDX_SECOND_PASS)],
                    dtype=np.int64,
                )
                replay.add(
                    obs=o_cf,
                    next_obs=no_cf,
                    action=action_cf[b_idx],
                    reward=np.asarray([reward], dtype=np.float32),
                    done=np.ones((1,), dtype=np.float32),
                    reward_components=np.asarray(
                        [quality_term, cf_lat_steps, par_excess / cf_lat_steps, float(robot_episode_success_log)],
                        dtype=np.float32,
                    ),
                    traj_info=traj_cf,
                    lifecycle=-1,
                    stride_infer_meta=meta_cf,
                )
                added += 1
    return added


def _patch_sample_obs_for_subsample(
    sample_obs: np.ndarray,
    kept_mask: np.ndarray,
    *,
    use_extended_obs: bool = False,
) -> np.ndarray:
    patched = np.asarray(sample_obs, dtype=np.float32).copy()
    active_idx, stopped_idx = _subsample_obs_tail_indices(int(patched.shape[-1]))
    if use_extended_obs or int(patched.shape[-1]) >= 5:
        active_idx, stopped_idx = -4, -3
    else:
        active_idx, stopped_idx = -2, -1
    patched[:, :, active_idx] = kept_mask.sum(axis=1, keepdims=True).astype(np.float32) / max(
        kept_mask.shape[1], 1
    )
    patched[:, :, stopped_idx] = (~kept_mask).astype(np.float32)
    return patched


def _transitions_from_record(data: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    keys = [key.removeprefix("transition_") for key in data if key.startswith("transition_")]
    if not keys:
        raise ValueError("Record does not contain transition_* arrays")
    length = data[f"transition_{keys[0]}"].shape[0]
    transitions = []
    for idx in range(length):
        transitions.append({key: data[f"transition_{key}"][idx] for key in keys})
    return transitions
