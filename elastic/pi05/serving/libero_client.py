"""LIBERO-10-R rollouts against an ELASTIC / pi0.5 policy server (runs in the LIBERO client image)."""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
import statistics
import sys
import time
from typing import Any, List, Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _parse_task_ids(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _policy_timing_total_ms(policy_timing: dict[str, Any]) -> float:
    """Server-side sampler + verifier time; sums infer and verifier ms when ``total_ms`` is absent."""
    if policy_timing.get("total_ms") is not None:
        return float(policy_timing["total_ms"])
    return float(policy_timing.get("infer_ms", 0.0)) + float(policy_timing.get("verifier_ms", 0.0))


def _is_transient_policy_websocket_error(exc: BaseException) -> bool:
    """True when the policy websocket likely dropped mid-rollout (common with long STRIDE infer)."""
    msg = str(exc).lower()
    needles = (
        "close frame",
        "connection closed",
        "connection reset",
        "broken pipe",
        "eof",
        "timed out",
        "timeout",
    )
    return any(needle in msg for needle in needles)


class _ElasticClient(_websocket_client_policy.WebsocketClientPolicy):
    """Upstream client without keepalive pings (one ELASTIC infer can run long), retrying
    while the server flushes training state, with an explicit ``close()`` so the server
    sees each disconnect."""

    def _wait_for_server(self):
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                try:
                    conn = websockets.sync.client.connect(
                        self._uri, compression=None, max_size=None, ping_interval=None, ping_timeout=None
                    )
                except TypeError:  # some websockets builds forward ping_* to socket.create_connection
                    conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                return conn, msgpack_numpy.unpackb(conn.recv())
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logging.info("Still waiting for server (%s)...", exc)
                time.sleep(5)

    def close(self) -> None:
        if getattr(self, "_ws", None) is not None:
            try:
                self._ws.close()
            except Exception:
                logging.exception("Error closing policy websocket")
            self._ws = None


class _ReconnectingWebsocketClientPolicy:
    """Websocket client that reconnects once on transient disconnects during LIBERO rollouts."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)
        self._client = _ElasticClient(host, self._port)

    def infer(self, obs: dict) -> dict:
        try:
            return self._client.infer(obs)
        except Exception as exc:
            if not _is_transient_policy_websocket_error(exc):
                raise
            logging.warning(
                "Policy websocket error (%s); reconnecting to ws://%s:%s",
                exc,
                self._host,
                self._port,
            )
            self._reconnect()
            return self._client.infer(obs)

    def _reconnect(self) -> None:
        try:
            self._client.close()
        except Exception:
            logging.exception("Error closing policy websocket before reconnect")
        self._client = _ElasticClient(self._host, self._port)

    def close(self) -> None:
        self._client.close()


def _policy_infer_postfix(out: dict):
    """Compact metrics from an infer() payload for the tqdm postfix."""
    extra = {}
    qvs = out.get("q_values")
    if qvs is not None:
        qvs = np.asarray(qvs).reshape(-1)
        if qvs.size > 1:
            extra["q_max"] = round(float(qvs.max()), 2)
            extra["q_min"] = round(float(qvs.min()), 2)
            extra["q_avg"] = round(float(qvs.mean()), 2)
    if "q_value" in out:
        extra["q_sel"] = round(float(out["q_value"]), 2)
    pt = out.get("policy_timing") or {}
    if "infer_ms" in pt:
        extra["model_ms"] = int(round(float(pt["infer_ms"])))
    if "flow_ms" in pt:
        extra["flow_ms"] = int(round(float(pt["flow_ms"])))
    if "verifier_ms" in pt:
        extra["ver_ms"] = int(round(float(pt["verifier_ms"])))
    if pt.get("total_ms") is not None:
        extra["total_ms"] = int(round(float(pt["total_ms"])))
    si = out.get("stride_info") or {}
    if si.get("selector"):
        sel = str(si["selector"])
        extra["stride"] = sel if len(sel) <= 16 else f"{sel[:15]}.."
    if si.get("selector_backend"):
        extra["sel"] = str(si["selector_backend"])
    if si.get("adaptive_steps") is not None:
        extra["steps"] = int(si["adaptive_steps"])
    ac = si.get("avg_active_samples")
    if ac is not None:
        extra["active"] = round(float(ac), 2)
    return extra


def _mean(xs: list[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def _pstdev(xs: list[float]) -> Optional[float]:
    return float(statistics.pstdev(xs)) if len(xs) >= 2 else None


def _compact_episode_records_for_summary(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-episode success and per-query denoising compute for the eval summary."""
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        rows.append(
            {
                "task_id": ep.get("task_id"),
                "trial_idx": ep.get("trial_idx"),
                "success": ep.get("success"),
                "task_description": ep.get("task_description"),
                "denoise_steps_per_query": list(ep.get("denoise_steps_per_query") or []),
                "avg_active_samples_per_query": list(ep.get("avg_active_samples_per_query") or []),
            }
        )
    return rows


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _aggregate_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Success, compute (denoise steps, active samples) and latency per policy query."""
    n = len(records)
    successes = sum(1 for r in records if r["success"])
    queries = sum(r["num_queries"] for r in records)
    steps = [float(x) for r in records for x in r["denoise_steps_per_query"] if x is not None]
    active = [float(x) for r in records for x in r["avg_active_samples_per_query"] if x is not None]
    return {
        "num_episodes": n,
        "num_successes": successes,
        "success_rate": successes / n if n else None,
        "episode_wall_s_mean": _mean([r["episode_wall_s"] for r in records]),
        "env_steps_after_stabilize_mean": _mean([float(r["env_steps_after_stabilize"]) for r in records]),
        "total_policy_queries": queries,
        "infer_ms_per_query_mean_globally": sum(r["infer_ms_total"] for r in records) / queries if queries else None,
        "total_ms_per_query_mean_globally": sum(r["total_ms_total"] for r in records) / queries if queries else None,
        "denoise_steps_per_query_mean_globally": _mean(steps),
        "denoise_steps_per_query_std_across_queries": _pstdev(steps),
        "avg_active_samples_per_query_mean_globally": _mean(active),
        "avg_active_samples_per_query_std_across_queries": _pstdev(active),
    }


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = (
        "libero_spatial"  # libero_spatial, libero_object, libero_goal, libero_10 or libero_90
    )
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    task_ids: Optional[str] = None  # Comma-separated; overrides the range args.
    task_start_index: int = 0
    # -1 runs to the end of the suite. LIBERO-10-R maps ``libero_10`` to 43 tasks; 10 gives the classic subset.
    num_tasks: int = -1

    video_out_path: str = "data/libero/videos"

    seed: int = 7

    eval_summary_json: Optional[str] = None


def _eval_summary_payload(episode_records_all: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    """Single JSON blob: ``overall`` stats plus one entry per LIBERO task id in ``tasks``."""
    overall = _aggregate_episodes(episode_records_all)
    by_tid: dict[int, list[dict[str, Any]]] = {}
    for r in episode_records_all:
        by_tid.setdefault(int(r["task_id"]), []).append(r)
    tasks: dict[str, dict[str, Any]] = {}
    for tid in sorted(by_tid.keys()):
        recs = by_tid[tid]
        tasks[str(tid)] = {
            "task_id": tid,
            "task_description": str(recs[0]["task_description"]),
            **_aggregate_episodes(recs),
        }
    return {
        "task_suite_name": args.task_suite_name,
        "overall": overall,
        "tasks": tasks,
        "episodes": _compact_episode_records_for_summary(episode_records_all),
        "replan_steps": args.replan_steps,
        "num_steps_wait": args.num_steps_wait,
        "host": args.host,
        "port": args.port,
    }


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    if args.task_ids:
        task_ids = _parse_task_ids(args.task_ids)
        if not task_ids:
            raise ValueError("--task_ids was provided but no valid ids were parsed.")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f"--task_ids contains duplicates: {task_ids}")
        for task_id in task_ids:
            if not 0 <= task_id < num_tasks_in_suite:
                raise ValueError(
                    f"task_id={task_id} out of bounds for suite with {num_tasks_in_suite} tasks "
                    f"(expected 0 <= task_id < {num_tasks_in_suite})."
                )
        logging.info(f"Evaluating explicit task ids: {task_ids}")
    else:
        task_end_index = (
            num_tasks_in_suite if args.num_tasks < 0 else min(num_tasks_in_suite, args.task_start_index + args.num_tasks)
        )
        if not 0 <= args.task_start_index < task_end_index <= num_tasks_in_suite:
            raise ValueError(
                f"Invalid task range [{args.task_start_index}, {task_end_index}) for suite with {num_tasks_in_suite} tasks."
            )
        task_ids = list(range(args.task_start_index, task_end_index))
        logging.info(f"Evaluating task ids [{args.task_start_index}, {task_end_index})")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Margin over the longest training demo of each suite.
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _ReconnectingWebsocketClientPolicy(args.host, args.port)
    try:
        _run_eval_libero_body(
            args=args,
            client=client,
            task_suite=task_suite,
            task_ids=task_ids,
            max_steps=max_steps,
        )
    finally:
        # Clean disconnect so the server flushes training state.
        client.close()


def _run_eval_libero_body(
    *,
    args,
    client: _ReconnectingWebsocketClientPolicy,
    task_suite,
    task_ids: list[int],
    max_steps: int,
) -> None:
    eval_summary_path = pathlib.Path(args.eval_summary_json) if args.eval_summary_json else None
    collect_episode_records = eval_summary_path is not None
    episode_records_all: list[dict[str, Any]] = []

    total_episodes, total_successes = 0, 0
    num_tasks_this_run = len(task_ids)
    # An outer bar of length 1 spams "0% 0/1" while the trial bar runs, so only nest for > 1 task.
    outer_pbar = None
    if num_tasks_this_run > 1:
        outer_pbar = tqdm.tqdm(task_ids, desc="LIBERO tasks", dynamic_ncols=True)

    def _iter_task_ids():
        if outer_pbar is not None:
            yield from outer_pbar
        else:
            yield from task_ids

    for task_id in _iter_task_ids():
        task = task_suite.get_task(task_id)

        initial_states = task_suite.get_task_init_states(task_id)
        n_init_states = int(initial_states.shape[0]) if hasattr(initial_states, "shape") else len(initial_states)
        if n_init_states <= 0:
            raise ValueError(f"No initial states for task {task_id}")
        if args.num_trials_per_task > n_init_states:
            logging.warning(
                "num_trials_per_task=%d exceeds LIBERO init states for this task (%d); indices will cycle.",
                args.num_trials_per_task,
                n_init_states,
            )

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        trial_desc = f"Task {task_id} trials" if num_tasks_this_run <= 1 else f"Task {task_id} trials (of {num_tasks_this_run})"
        tqdm.tqdm.write(
            f"[LIBERO] task={task_id} ({args.num_trials_per_task} trials): {task_description}"
        )
        # Manual update(1) per trial: iterating the bar prints a line per replan in non-TTY Docker logs.
        trial_pbar = tqdm.tqdm(
            total=args.num_trials_per_task,
            desc=trial_desc,
            dynamic_ncols=True,
            leave=num_tasks_this_run <= 1,
            position=0 if outer_pbar is None else 1,
            file=sys.stderr,
            mininterval=0.5,
        )
        for episode_idx in range(args.num_trials_per_task):
            logging.info("Trial %s/%s", episode_idx + 1, args.num_trials_per_task)

            env.reset()
            action_plan = collections.deque()

            obs = env.set_init_state(initial_states[episode_idx % n_init_states])

            t = 0
            replay_images = []
            last_policy_pf = {}
            episode_start = time.monotonic()
            query_count = 0
            query_infer_ms_total = 0.0
            query_verifier_ms_total = 0.0
            query_total_ms_total = 0.0
            query_denoise_steps: list[Optional[int]] = []
            query_active_samples: list[Optional[float]] = []
            done = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects dropped by the simulator to settle.
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Rotate 180 degrees to match training preprocessing.
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        infer_out = client.infer(element)
                        action_chunk = infer_out["actions"]
                        last_policy_pf = _policy_infer_postfix(infer_out)
                        policy_timing = infer_out.get("policy_timing") or {}
                        query_count += 1
                        query_infer_ms_total += float(policy_timing.get("infer_ms", 0.0))
                        query_verifier_ms_total += float(policy_timing.get("verifier_ms", 0.0))
                        query_total_ms_total += _policy_timing_total_ms(policy_timing)
                        stride_info = infer_out.get("stride_info") or {}
                        steps_raw = stride_info.get("adaptive_steps")
                        active_raw = stride_info.get("avg_active_samples")
                        query_denoise_steps.append(int(steps_raw) if steps_raw is not None else None)
                        query_active_samples.append(float(active_raw) if active_raw is not None else None)
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            episode_wall_s = time.monotonic() - episode_start
            env_steps_after_wait = max(0, int(t - args.num_steps_wait))

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = (
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_trial{episode_idx:04d}.mp4"
            )
            if replay_images:
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
            else:
                logging.warning("No frames collected for episode %s; skipping video write.", episode_idx + 1)

            logging.info(f"Success: {done}")
            logging.info(
                "libero_episode_perf task_id=%d trial=%d env_steps=%d wall_s=%.2f queries=%d "
                "infer_ms_total=%.1f ver_ms_total=%.1f total_ms_total=%.1f total_ms_mean=%.1f",
                task_id,
                episode_idx + 1,
                t,
                episode_wall_s,
                query_count,
                query_infer_ms_total,
                query_verifier_ms_total,
                query_total_ms_total,
                (query_total_ms_total / query_count) if query_count else 0.0,
            )
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            if collect_episode_records:
                episode_records_all.append(
                    {
                        "task_id": int(task_id),
                        "task_description": str(task_description),
                        "trial_idx": int(episode_idx),
                        "success": bool(done),
                        "env_steps_after_stabilize": env_steps_after_wait,
                        "episode_wall_s": float(episode_wall_s),
                        "num_queries": int(query_count),
                        "infer_ms_total": float(query_infer_ms_total),
                        "total_ms_total": float(query_total_ms_total),
                        "denoise_steps_per_query": query_denoise_steps,
                        "avg_active_samples_per_query": query_active_samples,
                    }
                )

            trial_pbar.set_postfix(
                **{
                    **last_policy_pf,
                    "wall_s": int(round(episode_wall_s)),
                    "q": query_count,
                    "success": bool(done),
                    "task_success": f"{task_successes}/{task_episodes}",
                    "total_success": f"{total_successes}/{total_episodes}",
                }
            )
            trial_pbar.update(1)

        trial_pbar.close()
        if outer_pbar is not None:
            outer_pbar.set_postfix(last_task=task_id, total_success=f"{total_successes}/{total_episodes}")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    if eval_summary_path is not None and episode_records_all:
        _write_json(eval_summary_path, _eval_summary_payload(episode_records_all, args))
        logging.info("Wrote eval summary to %s", eval_summary_path)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # seed affects object positions even with a fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Quaternion (x, y, z, w) to axis-angle, copied from robosuite ``transform_utils.quat2axisangle``."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
