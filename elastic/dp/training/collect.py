"""Demo collection for the Diffusion Policy backend.

Two products, one rollout loop (``collect_episodes``):

  buffer    Offline SAC replay for ``elastic.dp.training.train``: per-chunk random strides,
            then fixed strides with S-subsample counterfactual rows.
              python -m elastic.dp.training.collect buffer --config configs/jobs/pusht/pen_0p2_0p1.yaml
  verifier  Verifier training data (one candidate, env 0 only): fixed strides
            plus random strides, recorded as (obs, action, next_obs) episodes.
              python -m elastic.dp.training.collect verifier --checkpoint ckpts/pusht_lowdim.ckpt --output_path ...
"""
import argparse
import pathlib
import pickle
import sys
from pathlib import Path
from typing import Any, cast

import dill
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import open_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from elastic.core.action import delta_to_action  # noqa: E402
from elastic.dp.training.counterfactuals import add_fixed_stride_subsample_rows, default_subsample_sizes  # noqa: E402
from elastic.dp.runner import ensure_diffusion_policy_importable  # noqa: E402
from elastic.dp.training.reward import TransitionAccumulator  # noqa: E402
from elastic.dp.sampling.selectors import (  # noqa: E402
    FixedStrideSelector,
    PerChunkRandomStrideSelector,
    RandomStrideSelector,
)

# Recipe for the shipped buffers/*.pkl.
BUFFER_STRIDE_WEIGHTS = [0.30, 0.30, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02]
BUFFER_SUBSAMPLE_STRIDES = [5, 10, 20, 30, 50]
BUFFER_EPISODES = 10


def _copy_obs(obs):
    if isinstance(obs, dict):
        return {k: np.array(v, copy=True) for k, v in obs.items()}
    return np.array(obs, copy=True)


def _env0(obs):
    return {k: v[0] for k, v in obs.items()} if isinstance(obs, dict) else obs[0]


class VerifierRecorder:
    """Env-0 (obs, action, next_obs) episodes for verifier training."""

    def __init__(self, record_step_coverage=False):
        self.record_step_coverage = record_step_coverage
        self.episodes: list[list[dict]] = []
        self.episode_buffer: list[dict] = []

    def record(self, obs, action, next_obs, reward):
        row = {
            "obs": _copy_obs(_env0(obs)),
            "next_obs": _copy_obs(_env0(next_obs)),
            "clean_action": np.array(action, copy=True),
            "is_last": 0.0,
        }
        if self.record_step_coverage:
            arr = np.asarray(0.0 if reward is None else reward, dtype=np.float64).reshape(-1)
            row["step_coverage"] = float(arr[0]) if arr.size else 0.0
        self.episode_buffer.append(row)

    def finish_episode(self, episode_max_coverage):
        n = len(self.episode_buffer)
        if n == 0:
            return
        if all("step_coverage" in e for e in self.episode_buffer):
            covs = np.array([float(e["step_coverage"]) for e in self.episode_buffer], dtype=np.float64)
            r = np.zeros(n, dtype=np.float64)
            r[0] = max(0.0, float(covs[0]))
            for t in range(1, n):
                r[t] = max(0.0, float(covs[t] - covs[t - 1]))
            return_to_go = np.cumsum(r[::-1])[::-1]
            for i, entry in enumerate(self.episode_buffer):
                entry["episode_max_coverage"] = float(episode_max_coverage)
                entry["coverage_return_to_go"] = float(return_to_go[i])
                entry["is_last"] = float(i == (n - 1))
        else:
            for i, entry in enumerate(self.episode_buffer):
                entry["episode_max_coverage"] = float(episode_max_coverage)
                entry["is_last"] = float(i == (n - 1))
        self.episodes.append(self.episode_buffer)
        self.episode_buffer = []

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_trans = sum(len(ep) for ep in self.episodes)
        print(f"Saving {len(self.episodes)} episodes ({n_trans} transitions) to {path}")
        with open(path, "wb") as f:
            pickle.dump({"verifier_episodes": self.episodes}, f)


class MockBuffer:
    """Ring-buffer write pointer only, for verifier collection (no SAC rows kept)."""

    def __init__(self, buffer_size: int = 1_000_000):
        self.buffer_size = int(buffer_size)
        self.pos = 0

    def add(self, *args, **kwargs):
        self.pos = (self.pos + 1) % self.buffer_size


def collect_episodes(
    runner,
    selector,
    episodes,
    *,
    accumulator,
    buffer,
    t_max,
    device,
    verifier=None,
    recorder=None,
    after_chunk=None,
    desc="Collect",
):
    """Roll out ``episodes`` env batches, writing team-reward rows to ``buffer``; returns mean max coverage.

    ``after_chunk(transitions, exec_scores, episode_idx, traj_step)`` runs after each chunk's rows are written.
    """
    obs = runner.reset()
    collected = 0
    total_coverage = 0.0
    pbar = tqdm.tqdm(total=episodes, desc=desc)
    while collected < episodes:
        obs_robot = runner.prepare_obs(obs)
        selector.reset_episode()
        selector.set_current_max_coverage(accumulator.max_coverage)

        scoring_fn = None
        if verifier is not None:
            obs_for_scoring = obs

            def scoring_fn(_obs, action_B_S):
                return runner.verifier_advantage(verifier, obs_for_scoring, action_B_S, device)

        (next_obs, reward, done, _info), action_chunk, _, exec_scores = runner.step(
            obs_robot,
            scoring_fn,
            timestep_selector=selector,
            timestep_selector_mode="delta",
            max_denoise_steps=t_max,
        )
        if recorder is not None:
            recorder.record(obs, action_chunk[0], next_obs, reward)

        transitions = selector.pop_transitions()
        episode_idx = getattr(accumulator, "_episode_idx", 0)
        traj_step = accumulator._traj_step
        accumulator.compute_rewards(transitions, buffer, advantage_override=exec_scores, max_coverage=reward)
        if after_chunk is not None and transitions:
            after_chunk(transitions, exec_scores, episode_idx, traj_step)

        obs = next_obs
        if np.all(done):
            coverage, _, _, _ = accumulator.reset_accumulator()
            if recorder is not None:
                recorder.finish_episode(coverage)
            total_coverage += coverage
            collected += 1
            pbar.update(1)
            obs = runner.reset()
            selector.reset_episode()
    pbar.close()
    return total_coverage / max(collected, 1)


def _accumulator(runner, reward_cfg):
    return TransitionAccumulator(
        runner.n_envs,
        runner.num_action_samples,
        reward_cfg["terminal_bonus"],
        reward_cfg["latency_penalty"],
        reward_cfg["parallel_penalty"],
        gamma=1.0,
        no_finish_penalty=reward_cfg["no_finish_penalty"],
        store_raw_components=True,
        verifier_bias=reward_cfg["verifier_bias"],
    )


def _selector_kwargs(runner, policy, t_max, action_range):
    return dict(
        samples_per_env=runner.num_action_samples,
        t_max=t_max,
        action_dim=int(policy.action_dim),
        action_range=action_range,
    )


def collect_offline_buffer(
    buffer,
    runner,
    policy,
    verifier,
    device,
    reward_cfg,
    *,
    stride_buckets,
    episodes=BUFFER_EPISODES,
    subsample_strides=BUFFER_SUBSAMPLE_STRIDES,
    weights=BUFFER_STRIDE_WEIGHTS,
):
    """Fill ``buffer`` with the offline recipe: per-chunk random, then S-subsampled fixed strides."""
    t_max = int(policy.noise_scheduler.config.num_train_timesteps)
    action_range = (0, float(t_max))
    sel_kw = _selector_kwargs(runner, policy, t_max, action_range)
    loop_kw = dict(t_max=t_max, device=device, verifier=verifier, buffer=buffer)

    selector = PerChunkRandomStrideSelector(
        stride_options=stride_buckets, device=device, n_actions=1, weights=weights, **sel_kw
    )
    cov = collect_episodes(
        runner, selector, episodes, accumulator=_accumulator(runner, reward_cfg),
        desc="Per-chunk random stride", **loop_kw,
    )
    tqdm.tqdm.write(f"Per-chunk random stride: coverage={cov:.4f}, buffer={buffer.size()}")

    S = runner.num_action_samples
    sizes = default_subsample_sizes(S)
    stop_action = float(delta_to_action(np.array([0.0], dtype=np.float32), action_range=action_range)[0])
    for stride in subsample_strides:
        active_action = float(
            delta_to_action(np.array([float(stride)], dtype=np.float32), action_range=action_range)[0]
        )

        def add_rows(transitions, scores, episode_idx, traj_step, stride=stride, active_action=active_action):
            scores = np.asarray(scores)
            if scores.ndim == 2 and scores.shape == (runner.n_envs, S):
                add_fixed_stride_subsample_rows(
                    buffer, transitions, scores, stride=stride, t_max=t_max,
                    stop_action=stop_action, active_action=active_action,
                    terminal_bonus=reward_cfg["terminal_bonus"],
                    latency_penalty=reward_cfg["latency_penalty"],
                    parallel_penalty=reward_cfg["parallel_penalty"],
                    verifier_bias=reward_cfg["verifier_bias"],
                    episode_idx=episode_idx, traj_step=traj_step, k_subsample_sizes=sizes,
                )

        selector = FixedStrideSelector(stride=stride, device=device, n_actions=1, **sel_kw)
        cov = collect_episodes(
            runner, selector, episodes, accumulator=_accumulator(runner, reward_cfg),
            after_chunk=add_rows, desc=f"S-subsample stride={stride}", **loop_kw,
        )
        tqdm.tqdm.write(f"S-subsample stride={stride}: coverage={cov:.4f}, buffer={buffer.size()}")


def save_buffer(buffer, path, reward_cfg):
    """Pickle the filled part of a split-obs ReplayBuffer in the format ``elastic.dp.training.train`` loads."""
    n = buffer.size()
    obs = np.concatenate([buffer.state_obs[:n], buffer.sample_obs[:n]], axis=-1)[:, None, :]
    payload = {
        "observations": obs,
        "next_observations": None,
        "actions": buffer.actions[:n].copy(),
        "rewards": buffer.rewards[:n].copy(),
        "dones": np.zeros((n, 1), dtype=np.float32),
        "actives": np.ones((n, 1), dtype=np.float32),
        "pos": buffer.pos,
        "full": buffer.full,
        "n_envs": buffer.n_envs,
        "buffer_capacity": buffer.buffer_size,
        "obs_shape": buffer.obs_shape,
        "action_dim": buffer.action_dim,
        "optimize_memory_usage": True,
        "handle_timeout_termination": False,
        "reward_components": buffer.reward_components[:n].copy(),
        "traj_info": buffer.traj_info[:n].copy(),
        "saved_latency_penalty": reward_cfg["latency_penalty"],
        "saved_parallel_penalty": reward_cfg["parallel_penalty"],
        "saved_terminal_bonus": reward_cfg["terminal_bonus"],
        "saved_verifier_bias": reward_cfg["verifier_bias"],
        "saved_no_finish_penalty": reward_cfg["no_finish_penalty"],
        "saved_advantage_scale": reward_cfg.get("advantage_scale", 1.0),
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {n} transitions to {path}")


def collect_verifier_data(runner, policy, device, recorder, *, strides, episodes_per_stride, random_episodes):
    """Fixed-stride episodes for each stride, then random-stride episodes, into ``recorder``."""
    t_max = int(policy.noise_scheduler.config.num_train_timesteps)
    action_range = (1.0, float(t_max))
    buckets = [5, 10, 20, 50, 75, 100]
    sel_kw = dict(_selector_kwargs(runner, policy, t_max, action_range), n_actions=len(buckets))
    zero = dict(terminal_bonus=0.0, latency_penalty=0.0, parallel_penalty=0.0,
                no_finish_penalty=-30.0, verifier_bias=0.9)
    loop_kw = dict(buffer=MockBuffer(), t_max=t_max, device=device, recorder=recorder)
    for s in strides:
        selector = FixedStrideSelector(stride=s, device=device, stride_buckets=buckets, **sel_kw)
        collect_episodes(runner, selector, episodes_per_stride, accumulator=_accumulator(runner, zero),
                         desc=f"Stride {s}", **loop_kw)
    if random_episodes > 0:
        selector = RandomStrideSelector(device=device, **sel_kw)
        collect_episodes(runner, selector, random_episodes, accumulator=_accumulator(runner, zero),
                         desc="Random stride", **loop_kw)


def _build_verifier_runner(args):
    """One env, one candidate. PushT keeps the checkpoint's runner config (obs layout)."""
    if args.robomimic_dataset:
        from elastic.dp.runner import build_policy_and_runner

        args.samples = 1
        args.n_envs = 1
        args.max_steps = 400
        return build_policy_and_runner(args, output_dir="collect_demos_output", gamma=1.0)

    from elastic.dp.sampling.policies import adopt_policy

    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    runner_cfg = cfg.task.env_runner
    runner_cfg["_target_"] = "envs.pusht.PushTKeypointsResetVectorRunner"
    runner_cfg.update(n_train=0, n_train_vis=0, n_test=1, n_test_vis=0, n_envs=1, max_steps=300,
                      test_start_seed=args.seed)
    with open_dict(runner_cfg):
        runner_cfg.update(render_size=96, num_action_samples=1, include_obstacle=args.obstacle,
                          shape_type="tee")
    cfg.policy.noise_scheduler["_target_"] = "diffusers.schedulers.scheduling_ddim.DDIMScheduler"
    with open_dict(cfg.policy.noise_scheduler):
        cfg.policy.noise_scheduler.pop("variance_type", None)
    cfg.policy.num_inference_steps = 0

    workspace = cast(Any, hydra.utils.get_class(cfg._target_)(cfg, output_dir="collect_demos_output"))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    adopt_policy(policy)
    policy.to(torch.device("cuda:0"))
    policy.eval()
    runner = hydra.utils.instantiate(runner_cfg, output_dir="collect_demos_output")
    runner.set_policy(policy)
    return policy, runner


def _main_verifier(argv):
    p = argparse.ArgumentParser(prog="elastic.dp.training.collect verifier")
    p.add_argument("--output_path", default=None, help="Mixed-stride dataset")
    p.add_argument("--strides", default="5,10,20,50,75,100")
    p.add_argument("--episodes_per_stride", type=int, default=200)
    p.add_argument("--random_episodes", type=int, default=5)
    p.add_argument("--ref_output_path", default=None, help="Reference-policy dataset (advantage anchor)")
    p.add_argument("--ref_stride", type=int, default=10)
    p.add_argument("--ref_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obstacle", action="store_true")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--robomimic_dataset", default=None)
    p.add_argument("--robomimic_obs_keys", default=None)
    args = p.parse_args(argv)
    if args.output_path is None and args.ref_output_path is None:
        raise SystemExit("Nothing to collect: provide --output_path and/or --ref_output_path.")
    policy, runner = _build_verifier_runner(args)
    device = torch.device("cuda:0")
    step_coverage = not args.robomimic_dataset  # only PushT has a dense per-step reward
    if args.output_path is not None:
        rec = VerifierRecorder(step_coverage)
        collect_verifier_data(runner, policy, device, rec,
                              strides=[int(s) for s in args.strides.split(",")],
                              episodes_per_stride=args.episodes_per_stride,
                              random_episodes=args.random_episodes)
        rec.save(args.output_path)
    if args.ref_output_path is not None:
        rec = VerifierRecorder(step_coverage)
        collect_verifier_data(runner, policy, device, rec, strides=[args.ref_stride],
                              episodes_per_stride=args.ref_episodes, random_episodes=0)
        rec.save(args.ref_output_path)


def _main_buffer(argv):
    from elastic.dp.training.train import build_training_setup, load_job

    p = argparse.ArgumentParser(prog="elastic.dp.training.collect buffer")
    p.add_argument("--config", required=True, help="Train job YAML (reward coefficients, verifier, env)")
    p.add_argument("--output_path", default=None, help="Default: the job's buffer_path")
    p.add_argument("--episodes", type=int, default=BUFFER_EPISODES)
    p.add_argument("--set", action="append", default=[])
    cli = p.parse_args(argv)
    args = load_job(cli.config, cli.set)
    setup = build_training_setup(args, seed_rng=True)
    collect_offline_buffer(
        setup.buffer, setup.runner, setup.policy, setup.verifier, setup.device, setup.reward_cfg,
        stride_buckets=setup.stride_buckets, episodes=cli.episodes,
    )
    save_buffer(setup.buffer, cli.output_path or args.buffer_path, setup.reward_cfg)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("buffer", "verifier"):
        raise SystemExit(__doc__)
    ensure_diffusion_policy_importable()
    (_main_buffer if argv[0] == "buffer" else _main_verifier)(argv[1:])


if __name__ == "__main__":
    main()
