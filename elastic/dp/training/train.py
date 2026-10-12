"""Train an ELASTIC stride meta-policy with SAC on a frozen Diffusion Policy.

  python -m elastic.dp.training.train --config configs/jobs/pusht_obs/pen_0p2_0p1.yaml
"""
import argparse
import dataclasses
import gc
import os
import pathlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import tqdm

from elastic.core.action import delta_to_action
from elastic.core.actor import StrideActorCritic
from elastic.dp.training.replay import SplitReplayBuffer, load_offline_buffers
from elastic.dp.eval.checkpoint_select import is_better_eval_checkpoint, score_from_eval_metrics
from elastic.config import DATA_PATH_KEYS, apply_set, dump_yaml, load_yaml, resolve_data_path, to_namespace
from elastic.dp.training.counterfactuals import add_second_pass_rows
from elastic.dp.runner import (
    auto_calibrate_reward_coefficients,
    build_eval_runner,
    build_policy_and_runner,
    resolve_output_dir,
)
from elastic.dp.eval.episode import DEFAULT_SUCCESS_THRESHOLD, env_finished, flatten_env_vec, update_max_coverage
from elastic.dp.training.learner import DiffusionPolicySAC
from elastic.dp.eval.evaluate import run_heldout_eval
from elastic.dp.training.reward import TransitionAccumulator
from elastic.dp.sampling.selectors import DenoiseStrideSelector, FixedStrideSelector
from elastic.dp.training.stats import EpisodeStats
from verifiers.advantage import AdvantageVerifier

HIDDEN = 512
ALPHA_LR = 3e-4
TAU = 0.005
GAMMA = 1.0
STRIDE_BUCKETS = [5, 10, 20, 30, 40, 50, 75, 100]
TARGET_ENTROPY = -4.0
INIT_LOG_ALPHA = -4.0
MAX_LOG_ALPHA = 3.0
ONLINE_BUFFER_SIZE = int(5e6)
OFFLINE_BUFFER_SIZE = int(2e6)
PER_ALPHA, PER_BETA_START, PER_BETA_END, PER_EPS = 0.6, 0.4, 1.0, 1e-4
SECOND_PASS_EVERY = 4
SECOND_PASS_STRIDE_RANGE = (0.5, 1.5)  # x the policy's average stride last episode
INITIAL_AVG_STRIDE = 10.0
LOG_EVERY = 50


def train_defaults() -> dict:
    return {
        "output_dir": None,
        "samples": 1,
        "latency": 0.0,
        "parallel_penalty": 0.0,
        "terminal_bonus": 10.0,
        "no_finish_penalty": -10.0,
        "verifier_bias": 0.9,
        "verifier_advantage_scale": 1.0,
        "q_path": None,
        "verifier_noise_std": 0.0,
        "total_steps": 20100,
        "pretrain_steps": 5000,
        "num_updates": 16,
        "eval_every": 1000,
        "eval_episodes": 50,
        "eval_n_envs": 16,
        "eval_seed": 10000,
        "eval_ckpt_min_step": 0,
        "success_threshold": DEFAULT_SUCCESS_THRESHOLD,
        "ckpt_every": 500,
        "no_wandb": False,
        "wandb_project": None,
        "wandb_group": None,
        "wandb_run_name": None,
        "wandb_entity": None,
        "buffer_path": None,
        "critic_lr": 3e-4,
        "actor_lr": 1e-4,
        "batch_size": 512,
        "offline_batch_frac": 0.2,
        "offline_decay_steps": 5000,
        "checkpoint": None,
        "obstacle": False,
        "seed": 0,
        "n_envs": 32,
        "max_steps": 400,
        "robomimic_dataset": None,
        "robomimic_obs_keys": "object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos",
        "abs_action": None,
        "auto_calibrate_penalties": True,
        "calibrated_penalty_dir": "configs/calibration",
    }


def load_job(config_path, overrides=(), output_dir=None):
    """Job YAML + ``key=value`` overrides -> args, with data paths resolved and penalties calibrated."""
    cfg = apply_set(load_yaml(config_path), list(overrides))
    if output_dir:
        cfg["output_dir"] = output_dir
    for key in DATA_PATH_KEYS:
        if cfg.get(key):
            cfg[key] = resolve_data_path(cfg[key])
    args = to_namespace(cfg, train_defaults())
    if not args.q_path:
        raise SystemExit("config must set q_path")
    args.effective_utd = float(args.num_updates) / float(max(int(args.n_envs), 1))
    if float(args.verifier_advantage_scale) <= 0:
        raise SystemExit("verifier_advantage_scale must be positive")
    calib = auto_calibrate_reward_coefficients(args)
    if calib is not None:
        if calib.get("status") == "applied":
            tqdm.tqdm.write(
                f"Calibrated reward coefficients from Can_paired: task={calib['task']} "
                f"scale={calib['scale']:.6f} -> latency={args.latency:.6g} parallel={args.parallel_penalty:.6g} "
                f"terminal_bonus={args.terminal_bonus:.6g} verifier_bias={args.verifier_bias:.6g} "
                f"no_finish_penalty={args.no_finish_penalty:.6g}"
            )
        else:
            tqdm.tqdm.write(f"Reward coefficients unchanged (calibration status={calib.get('status')})")
    return args


def reward_cfg_from_args(args) -> dict:
    return {
        "terminal_bonus": float(args.terminal_bonus),
        "latency_penalty": float(args.latency),
        "parallel_penalty": float(args.parallel_penalty),
        "verifier_bias": float(args.verifier_bias),
        "no_finish_penalty": float(args.no_finish_penalty),
        "advantage_scale": float(args.verifier_advantage_scale),
    }


@dataclasses.dataclass
class TrainingSetup:
    args: Any
    device: torch.device
    policy: Any
    runner: Any
    t_max: int
    action_range: tuple
    stride_buckets: list
    obs_features_dim: int
    sample_obs_dim: int
    obs_dim: int
    stride_policy: StrideActorCritic
    stride_target: StrideActorCritic
    optimizer_actor: Any
    optimizer_critic: Any
    log_alpha: torch.Tensor
    alpha_optimizer: Any
    verifier: AdvantageVerifier
    buffer: Any
    offline_buffer: Any
    reward_cfg: dict

    @property
    def num_samples(self):
        return int(self.args.samples)


def build_training_setup(args, output_dir=None, seed_rng=False) -> TrainingSetup:
    """Frozen DP policy and runner, stride actor-critic, verifier and empty buffers."""
    if seed_rng:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
    device = torch.device("cuda:0")
    output_dir = output_dir or resolve_output_dir(args)
    policy, runner = build_policy_and_runner(args, output_dir, gamma=GAMMA)
    S = int(args.samples)
    t_max = int(policy.noise_scheduler.config.num_train_timesteps)

    obs_prepared = runner.prepare_obs(runner.reset())
    with torch.no_grad():
        np_obs = runner._build_obs_dict(obs_prepared)
        probe = policy.predict_action(
            {k: torch.from_numpy(v).to(device=device) for k, v in np_obs.items()}, num_inference_steps=t_max
        )
    obs_features_dim = int(np.prod(probe["obs_features"].shape[1:]))
    sample_obs_dim = int(np.prod(probe["action_pred"].shape[1:])) + 3  # + t, n_active_frac, stopped
    obs_dim = obs_features_dim + sample_obs_dim * S

    def make_actor():
        return StrideActorCritic(
            state_dim=obs_features_dim,
            sample_dim=sample_obs_dim,
            num_samples=S,
            hidden=HIDDEN,
        ).to(device)

    stride_policy = make_actor()
    stride_target = make_actor()
    stride_target.load_state_dict(stride_policy.state_dict())
    optimizer_critic = torch.optim.Adam(stride_policy.critic_parameters(), lr=float(args.critic_lr), eps=1e-4)
    optimizer_actor = torch.optim.Adam(stride_policy.actor_parameters(), lr=float(args.actor_lr), eps=1e-4)
    log_alpha = torch.tensor([INIT_LOG_ALPHA], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=ALPHA_LR, eps=1e-4)
    with torch.no_grad():
        log_alpha.clamp_(max=MAX_LOG_ALPHA)

    verifier = AdvantageVerifier(args.q_path, device, noise_std=float(args.verifier_noise_std or 0.0))
    buffer = SplitReplayBuffer(ONLINE_BUFFER_SIZE, obs_dim, S, obs_features_dim, device)
    offline_buffer = SplitReplayBuffer(OFFLINE_BUFFER_SIZE, obs_dim, S, obs_features_dim, device)
    return TrainingSetup(
        args=args, device=device, policy=policy, runner=runner, t_max=t_max, action_range=(0, float(t_max)),
        stride_buckets=STRIDE_BUCKETS, obs_features_dim=obs_features_dim, sample_obs_dim=sample_obs_dim,
        obs_dim=obs_dim, stride_policy=stride_policy, stride_target=stride_target,
        optimizer_actor=optimizer_actor, optimizer_critic=optimizer_critic, log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer, verifier=verifier, buffer=buffer, offline_buffer=offline_buffer,
        reward_cfg=reward_cfg_from_args(args),
    )


def _last_n_indices(buffer, n_added):
    """Ring-buffer slots of the last ``n_added`` writes."""
    if n_added <= 0:
        return np.empty((0,), dtype=np.int64)
    n = int(min(n_added, buffer.buffer_size))
    end = int(buffer.pos)
    start = (end - n) % buffer.buffer_size
    if start < end:
        return np.arange(start, end, dtype=np.int64)
    return np.concatenate([np.arange(start, buffer.buffer_size, dtype=np.int64), np.arange(0, end, dtype=np.int64)])


class Trainer:
    def __init__(self, setup: TrainingSetup, checkpoint_dir: pathlib.Path, wandb_run=None):
        s = self.s = setup
        self.args = setup.args
        self.wandb = wandb_run
        self.checkpoint_dir = checkpoint_dir
        self.runner = setup.runner
        self.device = setup.device
        self.online_priority = np.ones((ONLINE_BUFFER_SIZE,), dtype=np.float32)
        self.selector = DenoiseStrideSelector(
            policy=s.stride_policy,
            device=s.device,
            n_actions=len(s.stride_buckets),
            t_max=s.t_max,
            action_dim=int(s.policy.action_dim),
            samples_per_env=s.num_samples,
            stride_buckets=s.stride_buckets,
            action_range=s.action_range,
        )
        self.stop_action = float(delta_to_action(np.array([0.0], dtype=np.float32), action_range=s.action_range)[0])
        self.cf_avg_stride = INITIAL_AVG_STRIDE
        self.total_steps = int(self.args.total_steps)
        self.pretrain_steps = int(self.args.pretrain_steps)
        self.num_updates = int(self.args.num_updates)
        self.wandb_init()
        self.learner = DiffusionPolicySAC(
            policy=s.stride_policy,
            target=s.stride_target,
            optimizer_actor=s.optimizer_actor,
            optimizer_critic=s.optimizer_critic,
            log_alpha=s.log_alpha,
            alpha_optimizer=s.alpha_optimizer,
            buffer=s.buffer,
            offline_buffer=s.offline_buffer,
            online_priority=self.online_priority,
            pause_action_threshold=self.selector.pause_action_threshold,
            device=s.device,
            batch_size=int(self.args.batch_size),
            num_samples=s.num_samples,
            obs_features_dim=s.obs_features_dim,
            total_steps=self.total_steps,
            offline_batch_frac=float(self.args.offline_batch_frac),
            offline_decay_steps=int(self.args.offline_decay_steps),
            per_alpha=PER_ALPHA,
            per_beta_start=PER_BETA_START,
            per_beta_end=PER_BETA_END,
            per_eps=PER_EPS,
            target_entropy=TARGET_ENTROPY,
            tau=TAU,
        )
        self.eval_runner = None
        if int(self.args.eval_every) > 0 and int(self.args.eval_episodes) > 0:
            self.eval_runner = build_eval_runner(self.args, self.runner, s.policy, GAMMA, s.device)
        self.best_eval = {"score": -float("inf"), "latency": float("inf"), "step": -1}
        self.best_train_objective = -float("inf")

    def wandb_init(self):
        if self.wandb is None:
            return
        a = self.args
        out_path = pathlib.Path(self.runner.output_dir)
        kwargs = {
            "project": a.wandb_project or os.environ.get("WANDB_PROJECT") or "elastic",
            "name": a.wandb_run_name or out_path.name,
            "group": a.wandb_group or out_path.parent.name,
            "dir": str(out_path),
            "reinit": True,
        }
        if a.wandb_entity:
            kwargs["entity"] = a.wandb_entity
        self.wandb.init(**kwargs)
        self.wandb.config.update({**vars(a), "t_max": self.s.t_max, "S": self.s.num_samples})

    def log(self, data, step):
        if self.wandb is not None:
            self.wandb.log(data, step=self.pretrain_steps + step)

    def checkpoint_payload(self, extra=None):
        s = self.s
        payload = {
            "policy_state_dict": s.stride_policy.state_dict(),
            "policy_target_state_dict": s.stride_target.state_dict(),
            "optimizer_critic_state_dict": s.optimizer_critic.state_dict(),
            "optimizer_actor_state_dict": s.optimizer_actor.state_dict(),
            "log_alpha": s.log_alpha,
            "obs_dim": s.obs_dim,
            "hidden": HIDDEN,
            "t_max": s.t_max,
            "n_actions": len(s.stride_buckets),
            "num_action_samples": s.num_samples,
            "sample_obs_dim": s.sample_obs_dim,
            "obs_features_dim": s.obs_features_dim,
            "verifier_advantage_scale": self.args.verifier_advantage_scale,
            "verifier_q_path": str(self.args.q_path),
        }
        payload.update(extra or {})
        return payload

    def save(self, name, extra=None):
        torch.save(self.checkpoint_payload(extra), str(self.checkpoint_dir / name))

    def evaluate(self, outer_step, env_steps):
        """Held-out eval; keeps the checkpoint with the best penalized score."""
        s, a = self.s, self.args
        runner = self.eval_runner

        def scoring_fn(obs, action_B_S):
            return runner.verifier_advantage(s.verifier, obs, action_B_S, s.device)

        m = run_heldout_eval(
            runner,
            s.stride_policy,
            t_max=s.t_max,
            n_actions=len(s.stride_buckets),
            action_range=s.action_range,
            samples=s.num_samples,
            episodes=int(a.eval_episodes),
            eval_seed=int(a.eval_seed),
            device=s.device,
            scoring_fn=scoring_fn,
            success_threshold=float(a.success_threshold),
            stride_buckets=s.stride_buckets,
            max_denoise_steps=s.t_max,
        )
        tqdm.tqdm.write(
            f"[eval step={outer_step} env_steps={env_steps}] success={m['success_rate']:.3f} "
            f"return={m['mean_coverage']:.3f} L(seq)={m['mean_latency']:.2f}±{m['std_latency']:.2f} "
            f"P(par)={m['mean_sample_usage']:.2f}±{m['std_sample_usage']:.2f} "
            f"(n={m['n_episodes']} eps, {m['n_chunks']} chunks)"
        )
        eval_keys = (
            "mean_coverage", "success_rate", "mean_latency", "std_latency", "mean_sample_usage",
            "std_sample_usage", "mean_latency_to_success", "mean_sample_usage_to_success",
            "n_episodes", "n_chunks",
        )
        self.log({**{f"eval/{k}": float(m.get(k, np.nan)) for k in eval_keys},
                  "env_steps": float(env_steps), "outer_step": float(outer_step)}, outer_step)

        cov, lat, samp = float(m["mean_coverage"]), float(m["mean_latency"]), float(m["mean_sample_usage"])
        rc = s.reward_cfg
        score = score_from_eval_metrics(
            m, latency_penalty=rc["latency_penalty"], parallel_penalty=rc["parallel_penalty"],
            terminal_bonus=rc["terminal_bonus"],
        )
        tqdm.tqdm.write(f"[eval score] T*cov-aL-b(P-1)={score:.4f}")
        best = self.best_eval
        if is_better_eval_checkpoint(score, lat, best["score"], best["latency"], outer_step=int(outer_step),
                                     min_step=int(a.eval_ckpt_min_step or 0)):
            best.update(score=score, latency=lat, step=int(outer_step))
            extra = {
                "env_steps": int(env_steps),
                "outer_step": int(outer_step),
                "eval_mean_coverage": cov,
                "eval_mean_latency": lat,
                "eval_mean_sample_usage": samp,
                "eval_penalized_score": float(score),
                "eval_terminal_bonus": rc["terminal_bonus"],
                "eval_latency_penalty": rc["latency_penalty"],
                "eval_parallel_penalty": rc["parallel_penalty"],
            }
            payload = self.checkpoint_payload(extra)
            torch.save(payload, str(self.checkpoint_dir / "best_eval_stride_policy.pt"))
            torch.save(payload, str(self.checkpoint_dir / "best_penalized_stride_policy.pt"))
            tqdm.tqdm.write(
                f"New best eval score {score:.4f} (cov={cov:.3f} L={lat:.2f} P={samp:.2f}) "
                f"at step={outer_step} env_steps={env_steps}! Saved best_eval_stride_policy.pt"
            )
            self.log({"eval/best_penalized_score": score, "eval/best_coverage": cov,
                      "eval/best_latency": lat, "eval/best_step": float(outer_step)}, outer_step)

    def pretrain(self):
        """Critic-only updates on the offline buffer."""
        tqdm.tqdm.write(f"Offline pretraining for {self.pretrain_steps} steps")
        metrics = None
        for i in tqdm.tqdm(range(self.pretrain_steps), desc="Pretraining"):
            metrics = self.learner.train_step(i, update_actor=False, online_step=0)
            if self.wandb is not None and i % 200 == 0:
                self.wandb.log({f"pretrain/{k}": v for k, v in metrics.items()}, step=i)
        if metrics:
            tqdm.tqdm.write(f"Pretraining done. critic_loss={metrics['critic_loss']:.4f}")

    def second_pass(self, obs_prepared, raw_obs):
        """Env-free fixed-stride pass near the policy's current stride -> S-subsample rows."""
        s = self.s
        lo, hi = SECOND_PASS_STRIDE_RANGE
        stride = max(1, int(round(np.random.uniform(self.cf_avg_stride * lo, self.cf_avg_stride * hi))))
        active_action = float(
            delta_to_action(np.array([float(stride)], dtype=np.float32), action_range=s.action_range)[0]
        )
        selector = FixedStrideSelector(
            stride=stride,
            device=s.device,
            samples_per_env=s.num_samples,
            n_actions=len(s.stride_buckets),
            t_max=s.t_max,
            action_dim=int(s.policy.action_dim),
            action_range=s.action_range,
        )
        selector.reset_episode()
        with torch.no_grad():
            action_B_S, _ = self.runner.sim_step(
                obs_prepared, timestep_selector=selector, timestep_selector_mode="delta", max_denoise_steps=s.t_max
            )
        transitions = selector.pop_transitions()
        if transitions and action_B_S is not None:
            scores = self.runner.verifier_advantage(s.verifier, raw_obs, action_B_S, s.device)
            rc = s.reward_cfg
            add_second_pass_rows(
                s.buffer, transitions, scores, cf_stride=stride, t_max=s.t_max, stop_action=self.stop_action,
                active_action=active_action, terminal_bonus=rc["terminal_bonus"],
                latency_penalty=rc["latency_penalty"], parallel_penalty=rc["parallel_penalty"],
                verifier_bias=rc["verifier_bias"],
            )

    def end_episode(self, step, total_env_steps, accumulator, stats):
        coverage, scaled_return, lat_cost, par_cost = accumulator.reset_accumulator()
        self.episode_idx += 1
        summary = stats.finish()
        if summary is not None:
            self.cf_avg_stride = summary["episode_avg_stride"]
            objective = float(coverage) * self.s.reward_cfg["terminal_bonus"] * 38.0 - (lat_cost + par_cost)
            logs = {
                "train/episode_idx": float(self.episode_idx),
                "env_steps": float(total_env_steps),
                "outer_step": float(step),
                "episode_scaled_return_mean": float(scaled_return / self.runner.n_envs / self.s.num_samples),
                "episode_max_coverage_mean": float(coverage),
                **summary,
                "train/buffer_size": float(self.s.buffer.size()),
                "episode_true_objective": objective,
                "episode_latency_cost": float(lat_cost),
                "episode_parallel_cost": float(par_cost),
            }
            self.pbar.set_postfix(logs)
            if self.wandb is not None:
                strides = stats.last_strides
                if len(strides) > 1000:
                    strides = np.random.choice(strides, size=1000, replace=False)
                logs["actions/strides_hist"] = self.wandb.Histogram(strides.tolist())
                self.log(logs, step)
            if objective > self.best_train_objective:
                self.best_train_objective = objective
                self.save(f"best_train_stride_policy_{self.episode_idx}.pt")
        stats.reset()

    def run(self):
        s, a = self.s, self.args
        runner, buffer = self.runner, s.buffer
        self.pretrain()

        rc = s.reward_cfg
        accumulator = TransitionAccumulator(
            runner.n_envs,
            s.num_samples,
            terminal_bonus=rc["terminal_bonus"],
            latency_penalty=rc["latency_penalty"],
            parallel_penalty=rc["parallel_penalty"],
            gamma=GAMMA,
            no_finish_penalty=rc["no_finish_penalty"],
            verifier_bias=rc["verifier_bias"],
            advantage_scale=rc["advantage_scale"],
            store_raw_components=True,
        )
        # Online episode indices continue after the offline ones.
        valid = s.offline_buffer.traj_info[: s.offline_buffer.size(), 0]
        valid = valid[valid >= 0]
        accumulator._episode_idx = int(valid.max()) + 1 if len(valid) else 0

        stats = EpisodeStats(s.action_range, self.selector.pause_action_threshold)
        self.episode_idx = 0
        ever_done = np.zeros((runner.n_envs,), dtype=bool)
        threshold = float(a.success_threshold)
        obs = runner.reset()

        self.pbar = tqdm.tqdm(range(self.total_steps), desc="ELASTIC training")
        for step in self.pbar:
            want_log = self.wandb is not None and step % LOG_EVERY == 0
            obs_prepared = runner.prepare_obs(obs)
            self.selector.reset_episode()
            raw_obs = obs

            def scoring_fn(_obs, action_B_S):
                return runner.verifier_advantage(s.verifier, raw_obs, action_B_S, s.device)

            (next_obs, reward, done, _), _, _, exec_scores = runner.step(
                obs_prepared, scoring_fn, timestep_selector=self.selector,
                timestep_selector_mode="delta", max_denoise_steps=s.t_max,
            )
            step_log = {}
            if want_log:
                step_log = {"verifier/exec_scores_mean": float(np.mean(exec_scores)),
                            "verifier/exec_scores_std": float(np.std(exec_scores)),
                            "verifier/exec_scores_max": float(np.max(exec_scores))}
            transitions = self.selector.pop_transitions()

            # Coverage updates every step, even when no sample moved this chunk.
            update_max_coverage(accumulator.max_coverage, reward, runner.n_envs)
            alive = ~ever_done
            finished = env_finished(flatten_env_vec(done, runner.n_envs).astype(bool),
                                    accumulator.max_coverage, success_threshold=threshold)
            if not transitions:
                tqdm.tqdm.write(f"[WARNING step={step}] no sample moved this chunk; nothing added to replay")
            else:
                n_new = accumulator.compute_rewards(
                    transitions, buffer, advantage_override=exec_scores, max_coverage=reward, env_mask=alive
                )
                if n_new > 0:
                    size = buffer.size()
                    cur_max = float(np.max(self.online_priority[:size])) if size > 0 else 1.0
                    self.online_priority[_last_n_indices(buffer, n_new)] = max(cur_max, 1.0)
                if s.num_samples >= 2 and step % SECOND_PASS_EVERY == 0:
                    self.second_pass(obs_prepared, raw_obs)
                stats.add_chunk(transitions, alive)

            total_env_steps = (step + 1) * runner.n_envs
            obs = next_obs
            ever_done |= finished
            if np.all(ever_done):
                self.end_episode(step, total_env_steps, accumulator, stats)
                obs = runner.reset()
                ever_done[:] = False

            for i in range(self.num_updates):
                metrics = self.learner.train_step(step * self.num_updates + i, update_actor=True, online_step=step)
            if want_log:
                self.log({**step_log, **metrics, "env_steps": float(total_env_steps),
                          "outer_step": float(step)}, step)

            if int(a.ckpt_every) > 0 and step > 0 and step % int(a.ckpt_every) == 0:
                self.save(f"stride_policy_iter{step}.pt", {
                    "env_steps": int(total_env_steps), "outer_step": int(step), "n_envs": int(runner.n_envs),
                    "num_updates": self.num_updates, "effective_utd": float(a.effective_utd),
                })
                gc.collect()
            if self.eval_runner is not None and step > 0 and step % int(a.eval_every) == 0:
                self.evaluate(step, total_env_steps)

        if self.eval_runner is not None and (self.total_steps - 1) % int(a.eval_every) != 0:
            self.evaluate(self.total_steps - 1, self.total_steps * runner.n_envs)


def main(argv=None):
    p = argparse.ArgumentParser(description="Train an ELASTIC stride meta-policy from a job YAML.")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default=None, help="Override output_dir in the config")
    p.add_argument("--set", action="append", default=[], help="key=value override (YAML scalar). Repeatable.")
    cli = p.parse_args(argv)
    args = load_job(cli.config, cli.set, cli.output_dir)
    output_dir = resolve_output_dir(args)
    dump_yaml(pathlib.Path(output_dir) / "job.yaml", {
        k: v for k, v in vars(args).items()
        if not k.startswith("_") and isinstance(v, (type(None), bool, int, float, str, list, dict))
    })

    wandb_run = None
    if not args.no_wandb and os.environ.get("WANDB_MODE") != "disabled":
        import wandb as wandb_run

    setup = build_training_setup(args, output_dir)
    checkpoint_dir = pathlib.Path(setup.runner.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not args.buffer_path or not pathlib.Path(args.buffer_path).exists():
        raise SystemExit(
            f"offline buffer {args.buffer_path!r} not found; collect it with "
            f"python -m elastic.dp.training.collect buffer --config {cli.config}"
        )
    tqdm.tqdm.write(f"Loading replay buffer from {args.buffer_path}")
    load_offline_buffers(args.buffer_path, setup.buffer, setup.offline_buffer, setup.obs_features_dim,
                         setup.reward_cfg, log=tqdm.tqdm.write)
    Trainer(setup, checkpoint_dir, wandb_run).run()


if __name__ == "__main__":
    main()
