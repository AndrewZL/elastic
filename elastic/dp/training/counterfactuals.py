"""Counterfactual replay rows: what the team reward would have been with fewer samples.

A row keeps a subset of the S scored candidates (the rest take the stop action) and is
rewarded with the best kept score and the kept subset's compute.
Sample-obs layout: [action_flat | t | n_active_frac | stopped].
"""
import numpy as np

SOBS_N_ACTIVE_FRAC = -2
SOBS_STOPPED = -1


def patch_sample_obs(sample_obs_B_S_D, kept_mask_B_S):
    """Copy of sample_obs as seen in a world where only the kept samples ran."""
    out = sample_obs_B_S_D.copy()
    S = out.shape[1]
    k_sub_B = kept_mask_B_S.sum(axis=1, keepdims=True)
    out[:, :, SOBS_N_ACTIVE_FRAC] = k_sub_B.astype(np.float32) / max(S, 1)
    out[:, :, SOBS_STOPPED] = (~kept_mask_B_S).astype(np.float32)
    return out


def default_subsample_sizes(S):
    """Powers of two up to S, plus S itself."""
    return sorted(set([2**i for i in range(int(np.log2(max(S, 1))) + 1) if 2**i <= S] + [S]))


def add_fixed_stride_subsample_rows(
    buffer,
    transitions,
    scores_B_S,
    *,
    stride,
    t_max,
    stop_action,
    active_action,
    terminal_bonus,
    latency_penalty,
    parallel_penalty,
    verifier_bias,
    episode_idx,
    traj_step,
    k_subsample_sizes,
    n_draws=2,
):
    """Offline rows from one fixed-stride chunk: ``n_draws`` random subsets per env for each size k < S."""
    S = scores_B_S.shape[1]
    tr0 = transitions[0]
    B = tr0["state_obs"].shape[0]
    state_col = tr0["state_obs"][:, 0, :]
    next_state_col = tr0["next_state_obs"][:, 0, :]
    latency_steps = max(1.0, float(t_max) / max(float(stride), 1.0))
    added = 0
    for k_sub in k_subsample_sizes:
        if k_sub >= S:
            continue  # the real full-S chunk is already in the buffer
        for _ in range(n_draws):
            kept = np.stack([np.random.choice(S, size=k_sub, replace=False) for _ in range(B)])
            kept_mask = np.zeros((B, S), dtype=bool)
            for b in range(B):
                kept_mask[b, kept[b]] = True

            kept_scores = scores_B_S[np.arange(B)[:, None], kept]
            best_quality_B = np.max(kept_scores, axis=1)
            quality_term = (best_quality_B + verifier_bias) * terminal_bonus
            lat_cost = latency_penalty * latency_steps
            excess = max(k_sub - 1, 0) * latency_steps
            par_cost = parallel_penalty * excess / max(latency_steps, 1.0)
            reward_B = (quality_term - lat_cost - par_cost).astype(np.float32)

            obs_np = np.concatenate(
                [state_col, patch_sample_obs(tr0["sample_obs"], kept_mask).reshape(B, -1)], axis=-1
            )
            next_obs_np = np.concatenate(
                [next_state_col, patch_sample_obs(tr0["next_sample_obs"], kept_mask).reshape(B, -1)],
                axis=-1,
            )
            action_cf = np.full((B, S, 1), stop_action, dtype=np.float32)
            for b in range(B):
                action_cf[b, kept[b], 0] = active_action

            max_adv = np.max(kept_scores, axis=1).astype(np.float32)
            L_B = np.full(B, latency_steps, dtype=np.float32)
            norm_excess = np.full(B, float(max(k_sub - 1, 0)), dtype=np.float32)
            for b in range(B):
                buffer.add(
                    obs_np[b : b + 1],
                    next_obs_np[b : b + 1],
                    action_cf[b : b + 1],
                    reward_B[b : b + 1, None],
                )
                slot = (buffer.pos - 1) % buffer.buffer_size
                if hasattr(buffer, "reward_components"):
                    buffer.reward_components[slot] = [
                        float(max_adv[b]), 1.0, float(L_B[b]), float(norm_excess[b])
                    ]
                if hasattr(buffer, "traj_info"):
                    buffer.traj_info[slot] = [episode_idx, traj_step]
            added += B
    return added


def add_second_pass_rows(
    buffer,
    cf_transitions,
    cf_scores_B_S,
    *,
    cf_stride,
    t_max,
    stop_action,
    active_action,
    terminal_bonus,
    latency_penalty,
    parallel_penalty,
    verifier_bias,
):
    """Online rows from an env-free fixed-stride pass: one random subset per env, step and power-of-two size.

    Rows carry no reward components.
    """
    B, S = cf_scores_B_S.shape
    lat_steps = max(1.0, float(t_max) / float(cf_stride))
    lat_cost = latency_penalty * lat_steps
    k_subs = [2**i for i in range(int(np.log2(max(S, 1))) + 1) if 2**i <= S]
    for tr in cf_transitions:
        state_col = tr["state_obs"][:, 0, :]
        sample_obs = tr["sample_obs"]
        next_state_col = tr["next_state_obs"][:, 0, :]
        next_sample_obs = tr["next_sample_obs"]
        for k_sub in k_subs:
            for b in range(B):
                kept_idx = np.arange(S) if k_sub == S else np.random.choice(S, size=k_sub, replace=False)
                kept_mask = np.zeros((B, S), dtype=bool)
                kept_mask[b, kept_idx] = True

                samp = sample_obs.copy()
                samp[:, :, SOBS_N_ACTIVE_FRAC] = float(k_sub) / S
                samp[:, :, SOBS_STOPPED] = (~kept_mask).astype(np.float32)
                next_samp = next_sample_obs.copy()
                next_samp[:, :, SOBS_N_ACTIVE_FRAC] = float(k_sub) / S
                next_samp[:, :, SOBS_STOPPED] = (~kept_mask).astype(np.float32)
                obs_np = np.concatenate([state_col, samp.reshape(B, -1)], axis=-1)
                next_obs_np = np.concatenate([next_state_col, next_samp.reshape(B, -1)], axis=-1)

                best_quality = float(np.max(cf_scores_B_S[b, kept_idx]))
                quality_term = (best_quality + verifier_bias) * terminal_bonus
                par_excess = max(k_sub - 1, 0) * lat_steps
                par_cost = parallel_penalty * par_excess / lat_steps
                reward = np.array([[quality_term - lat_cost - par_cost]], dtype=np.float32)
                action = np.full((1, S, 1), stop_action, dtype=np.float32)
                action[0, kept_idx, 0] = active_action
                buffer.add(
                    obs_np[b : b + 1],
                    next_obs_np[b : b + 1],
                    action,
                    reward,
                )
