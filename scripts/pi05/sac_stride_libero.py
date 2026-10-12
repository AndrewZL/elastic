#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable
import datetime
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect as _sync_websocket_connect

from elastic.pi05.constants import (
    DIFFUSION_T_MAX,
    LATENCY_PENALTY,
    NO_FINISH_PENALTY,
    OFFLINE_PRETRAIN_STEPS,
    ONLINE_MIN_TRANSITIONS,
    ONLINE_UPDATES_PER_STEP,
    PARALLEL_PENALTY,
    STRIDE_BUCKETS,
    STRIDE_BUCKET_WEIGHTS,
    TERMINAL_BONUS,
    VERIFIER_BIAS,
)
from elastic.pi05.training.reward import relabel_rewards_full
from elastic.pi05.training.counterfactuals import build_replay_from_record_dir
from launch import COMPOSE_FILE, ROOT, load_recipe, runs_host_dir, to_container_path

# Fixed strides (of DIFFUSION_T_MAX) for the S-subsample demo phase.
K_SUBSAMPLE_STRIDES = [5, 10, 20, 30, 50]
DEFAULT_VGPS_CHECKPOINT = load_recipe(ROOT / "configs/libero.yaml")["vgps_checkpoint"]
LIBERO_COMPOSE_FILE = str(COMPOSE_FILE)
# The server loads pi0.5, the offline replay and V-GPS before accepting clients.
SERVER_READY_TIMEOUT_S = 900.0
# A post-chunk flush of a large replay can block the server for minutes.
PROBE_HANDSHAKE_TIMEOUT_S = 120.0
WAIT_LOG_INTERVAL_S = 60.0


def parse_task_ids(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def normalize_suite_name(raw: str) -> str:
    aliases = {
        "libero_10_r": "libero_10",
        "libero10r": "libero_10",
        "libero-10-r": "libero_10",
    }
    return aliases.get(str(raw).strip(), str(raw).strip())










def _to_scalar(value):
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def _collection_hyperparams_from_replay(state: dict) -> tuple[float, float, float, float, float]:
    """Hyperparams used when the replay was written (npz metadata or LIBERO demo defaults)."""
    saved_terminal = _to_scalar(state.get("saved_terminal_bonus"))
    saved_bias = _to_scalar(state.get("saved_verifier_bias"))
    saved_latency = _to_scalar(state.get("saved_latency_penalty"))
    saved_parallel = _to_scalar(state.get("saved_parallel_penalty"))
    saved_nf = _to_scalar(state.get("saved_no_finish_penalty"))
    if saved_terminal is None:
        saved_terminal, saved_bias, saved_latency, saved_parallel = 0.1, 0.0, 0.0, 0.0
    if saved_bias is None:
        saved_bias = 0.0
    if saved_latency is None:
        saved_latency = 0.0
    if saved_parallel is None:
        saved_parallel = 0.0
    if saved_nf is None:
        saved_nf = float(NO_FINISH_PENALTY)
    return float(saved_terminal), float(saved_bias), float(saved_latency), float(saved_parallel), float(saved_nf)


def sanitize_compose_project_name(name: str) -> str:
    """Return a docker-compose-safe project name (lowercase alnum, hyphens, underscores)."""
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")
    if not cleaned:
        cleaned = "stride-run"
    if not cleaned[0].isalnum():
        cleaned = f"stride-{cleaned.lstrip('-_')}" or "stride-run"
    return cleaned[:63]


def resolve_gpu_device_id(*, cuda_device: str | None) -> str:
    """Pick the host GPU index for docker compose (GPU_DEVICE_ID / device_ids)."""
    if cuda_device is not None:
        return str(cuda_device)
    if gpu := os.environ.get("GPU_DEVICE_ID"):
        return gpu
    if gpu := os.environ.get("NVIDIA_VISIBLE_DEVICES"):
        return gpu
    if gpu := os.environ.get("CUDA_VISIBLE_DEVICES"):
        return gpu
    return "0"


def prepare_offline_replay_for_run(
    *,
    source_path: Path,
    output_dir: Path,
    terminal_bonus: float,
    verifier_bias: float,
    latency_penalty: float,
    parallel_penalty: float,
    no_finish_penalty: float,
) -> Path:
    """Copy/relabel ``source_path`` into ``output_dir/offline_replay.npz`` for this run."""
    source_path = source_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "offline_replay.npz"

    state = dict(np.load(source_path, allow_pickle=True))
    if "reward_components" not in state:
        print(f"[WARN] Replay has no reward_components; copying as-is: {source_path}", flush=True)
        if source_path != target_path:
            shutil.copyfile(source_path, target_path)
        return target_path if target_path.exists() else source_path

    saved_terminal, saved_bias, saved_latency, saved_parallel, saved_nf = _collection_hyperparams_from_replay(state)
    reward_components = np.asarray(state["reward_components"], dtype=np.float32)
    if reward_components.ndim != 2 or reward_components.shape[1] < 3:
        raise ValueError(f"Unexpected reward_components shape {reward_components.shape} in {source_path}")
    if saved_terminal <= 0 or terminal_bonus <= 0:
        raise ValueError("saved_terminal_bonus and terminal_bonus must be positive")

    relabeled_rewards, best_new = relabel_rewards_full(
        reward_components,
        terminal_bonus=float(terminal_bonus),
        verifier_bias=float(verifier_bias),
        latency_penalty=float(latency_penalty),
        parallel_penalty=float(parallel_penalty),
        no_finish_penalty=float(no_finish_penalty),
        saved_terminal_bonus=saved_terminal,
        saved_verifier_bias=saved_bias,
        saved_no_finish_penalty=saved_nf,
    )
    n = relabeled_rewards.shape[0]
    state["rewards"] = np.asarray(state["rewards"], dtype=np.float32)
    state["rewards"][:n, 0] = relabeled_rewards
    rc_out = np.asarray(state["reward_components"], dtype=np.float32)
    rc_out[:n, 0] = best_new
    state["reward_components"] = rc_out
    state["saved_terminal_bonus"] = np.asarray(float(terminal_bonus), dtype=np.float32)
    state["saved_verifier_bias"] = np.asarray(float(verifier_bias), dtype=np.float32)
    state["saved_latency_penalty"] = np.asarray(float(latency_penalty), dtype=np.float32)
    state["saved_parallel_penalty"] = np.asarray(float(parallel_penalty), dtype=np.float32)
    state["saved_no_finish_penalty"] = np.asarray(float(no_finish_penalty), dtype=np.float32)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npz", dir=str(output_dir), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, **state)
        tmp_path.replace(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    print(
        "[INFO] Prepared offline replay for this run:\n"
        f"  source: {source_path}\n"
        f"  saved:  {target_path}\n"
        f"  (term={saved_terminal}, bias={saved_bias}, lat={saved_latency}, par={saved_parallel}) -> "
        f"(term={terminal_bonus}, bias={verifier_bias}, lat={latency_penalty}, par={parallel_penalty})\n"
        f"  reward mean={float(relabeled_rewards.mean()):.4f} std={float(relabeled_rewards.std()):.4f}",
        flush=True,
    )
    return target_path


def _pids_listening_on_tcp_port(port: int) -> set[int]:
    """Return host PIDs listening on ``port`` (empty if nothing bound)."""
    proc = subprocess.run(
        ["fuser", f"{int(port)}/tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: set[int] = set()
    for token in (proc.stdout + " " + proc.stderr).replace("/", " ").split():
        if token.isdigit():
            pids.add(int(token))
    return pids


def _openpi_server_container_ids_for_pids(pids: set[int]) -> list[tuple[str, str]]:
    """Map host PIDs to ``(container_id, name)`` for ``openpi_server`` containers."""
    if not pids:
        return []
    proc = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    matches: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        cid, name = line.split("\t", 1)
        if "-server-" not in name:
            continue
        top = subprocess.run(
            ["docker", "top", cid, "-eo", "pid"],
            capture_output=True,
            text=True,
            check=False,
        )
        for row in top.stdout.splitlines()[1:]:
            row = row.strip()
            if not row:
                continue
            try:
                pid = int(row.split()[0])
            except (ValueError, IndexError):
                continue
            if pid in pids:
                matches.append((cid, name))
                break
    return matches


def _stop_openpi_servers_on_port(port: int) -> list[str]:
    """Stop ``openpi_server`` containers bound to ``port`` (parallel runs use other ports)."""
    stopped: list[str] = []
    for cid, name in _openpi_server_container_ids_for_pids(_pids_listening_on_tcp_port(port)):
        print(f"[INFO] Stopping openpi_server on port {port}: {name}", flush=True)
        subprocess.run(["docker", "stop", cid], check=False)
        stopped.append(name)
    if stopped:
        time.sleep(2.0)
    return stopped


def _tcp_port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except OSError:
        return False


def _probe_server_stride_output_dir_name(
    host: str,
    port: int,
    *,
    handshake_timeout_s: float = PROBE_HANDSHAKE_TIMEOUT_S,
) -> str | None:
    """Read STRIDE handshake metadata from the server on ``port`` (or None if unavailable)."""
    uri = f"ws://{host}:{int(port)}"
    try:
        conn = _sync_websocket_connect(uri)
    except OSError:
        return None
    try:
        sock = getattr(conn, "socket", None)
        if sock is not None:
            sock.settimeout(float(handshake_timeout_s))
        meta = msgpack_numpy.unpackb(conn.recv())
    except (OSError, TimeoutError):
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not isinstance(meta, dict):
        return None
    name = meta.get("stride_output_dir_name")
    return str(name) if name is not None else None


def _wait_policy_server_for_run(
    host: str,
    port: int,
    *,
    expected_output_dir_name: str,
    timeout_s: float,
    on_stale_stop: Callable[[], None] | None = None,
    wait_context: str = "",
) -> None:
    """Wait until ``port`` serves this run (not a leftover server from another output_dir).

    When ``on_stale_stop`` is set (rotate eval between chunks), a stale server on this port is stopped
    and the callback is invoked to start a fresh ``openpi_server`` for the same run.
    """
    deadline = time.monotonic() + float(timeout_s)
    last_seen: str | None = None
    next_log_at = time.monotonic()
    ctx = f" ({wait_context})" if wait_context else ""
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_log_at:
            port_open = _tcp_port_is_open(host, port)
            print(
                f"[INFO] Waiting for policy server on {host}:{port}{ctx}: "
                f"port_open={port_open} last_stride_output_dir_name={last_seen!r} "
                f"expected={expected_output_dir_name!r}",
                flush=True,
            )
            next_log_at = now + WAIT_LOG_INTERVAL_S

        seen = _probe_server_stride_output_dir_name(host, port)
        if seen == expected_output_dir_name:
            print(
                f"[INFO] Policy server on {host}:{port} confirmed for output_dir={expected_output_dir_name!r}",
                flush=True,
            )
            return
        if seen is not None:
            last_seen = seen
            print(
                f"[WARN] Port {port} is serving stale run {seen!r} (expected {expected_output_dir_name!r}); "
                f"stopping openpi_server on port {port}...",
                flush=True,
            )
            _stop_openpi_servers_on_port(port)
            if on_stale_stop is not None:
                on_stale_stop()
            else:
                time.sleep(2.0)
            continue
        if _tcp_port_is_open(host, port):
            # TCP is up but the handshake is not ready yet (startup or post-chunk flush).
            time.sleep(2.0)
            continue
        time.sleep(0.5)
    msg = (
        f"Policy server on {host}:{port} did not confirm stride_output_dir_name="
        f"{expected_output_dir_name!r} within {timeout_s}s"
    )
    if last_seen is not None:
        msg += f" (last saw {last_seen!r})"
    elif _tcp_port_is_open(host, port):
        msg += (
            " (port accepts TCP but websocket handshake never returned metadata; "
            "check openpi_server logs — server may be hung flushing a large online replay or stuck after a dropped client)"
        )
    else:
        msg += " (port was not accepting connections; openpi_server may have exited — check `docker ps` / server logs)"
    raise TimeoutError(msg)


def _docker_compose_down(*, root: Path, env_base: dict[str, str]) -> None:
    subprocess.run(
        ["docker", "compose", "-f", LIBERO_COMPOSE_FILE, "down"],
        cwd=root,
        env=env_base,
        check=False,
    )


def _docker_start_openpi_server(
    *,
    root: Path,
    env_base: dict[str, str],
    server_args: str,
    policy_port: int,
    expected_output_dir_name: str,
) -> None:
    """Start ``openpi_server`` detached and wait until it serves ``expected_output_dir_name``."""
    _stop_openpi_servers_on_port(policy_port)
    env = env_base.copy()
    env["SERVER_ARGS"] = f"{server_args} --port {policy_port}"
    subprocess.run(
        ["docker", "compose", "-f", LIBERO_COMPOSE_FILE, "up", "-d", "server"],
        cwd=root,
        env=env,
        check=True,
    )
    _wait_policy_server_for_run(
        "127.0.0.1",
        policy_port,
        expected_output_dir_name=expected_output_dir_name,
        timeout_s=SERVER_READY_TIMEOUT_S,
    )


def _build_libero_client_args(
    *,
    policy_port: int,
    suite: str,
    trials: int,
    task_ids: list[int],
    out_dir: Path,
    run_name: str,
    path_suffix: str,
) -> str:
    task_ids_csv = ",".join(str(tid) for tid in task_ids)
    video_tag = f"videos_{out_dir.name}_{run_name}_tasks_{task_ids_csv.replace(',', '-')}{path_suffix}"
    return (
        f"--args.host 127.0.0.1 "
        f"--args.port {policy_port} "
        f"--args.task-suite-name {suite} "
        f"--args.num-trials-per-task {trials} "
        f"--args.task-ids {task_ids_csv} "
        f"--args.video-out-path /runs/videos/{video_tag}"
    )


def _docker_run_libero_client(
    *,
    root: Path,
    env_base: dict[str, str],
    client_args: str,
    log_path: Path,
) -> int:
    """Run the LIBERO ``runtime`` container once (``openpi_server`` must already be up)."""
    env = env_base.copy()
    env["CLIENT_ARGS"] = client_args
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                "docker",
                "compose",
                "-f",
                LIBERO_COMPOSE_FILE,
                "run",
                "--rm",
                "--no-deps",
                "runtime",
            ],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
        return int(proc.wait())


def _docker_eval_task_list_invocation(
    *,
    root: Path,
    env_base: dict[str, str],
    run_name: str,
    server_args: str,
    policy_port: int,
    task_ids: list[int],
    trials: int,
    suite: str,
    out_dir: Path,
    path_suffix: str,
    server_already_running: bool = False,
    expected_output_dir_name: str | None = None,
) -> None:
    """Run ``trials`` episodes per task; with ``server_already_running`` only the client starts."""
    if not task_ids:
        raise ValueError("task_ids must be non-empty")
    task_ids_csv = ",".join(str(tid) for tid in task_ids)
    client_args = _build_libero_client_args(
        policy_port=policy_port,
        suite=suite,
        trials=trials,
        task_ids=task_ids,
        out_dir=out_dir,
        run_name=run_name,
        path_suffix=path_suffix,
    )
    log_path = out_dir / f"{run_name}_tasks_{task_ids_csv.replace(',', '-')}{path_suffix}.log"

    if server_already_running:
        if expected_output_dir_name:
            restart_server = lambda: _docker_start_openpi_server(
                root=root,
                env_base=env_base,
                server_args=server_args,
                policy_port=policy_port,
                expected_output_dir_name=expected_output_dir_name,
            )
            _wait_policy_server_for_run(
                "127.0.0.1",
                policy_port,
                expected_output_dir_name=expected_output_dir_name,
                timeout_s=SERVER_READY_TIMEOUT_S,
                on_stale_stop=restart_server,
                wait_context=f"before {run_name} chunk log={log_path.name}",
            )
        exit_code = _docker_run_libero_client(
            root=root,
            env_base=env_base,
            client_args=client_args,
            log_path=log_path,
        )
    else:
        _docker_compose_down(root=root, env_base=env_base)
        _stop_openpi_servers_on_port(policy_port)
        env = env_base.copy()
        env["SERVER_ARGS"] = f"{server_args} --port {policy_port}"
        env["CLIENT_ARGS"] = client_args
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                [
                    "docker",
                    "compose",
                    "-f",
                    LIBERO_COMPOSE_FILE,
                    "up",
                    "--abort-on-container-exit",
                ],
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if proc.stdout is not None:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log.write(line)
            exit_code = int(proc.wait())
        _docker_compose_down(root=root, env_base=env_base)

    if exit_code != 0:
        raise RuntimeError(
            f"{run_name} task_ids={task_ids_csv} failed with exit code {exit_code}; see {log_path}"
        )


def docker_eval(
    *,
    root: Path,
    run_name: str,
    server_args: str,
    policy_port: int,
    compose_project_name: str,
    gpu_device_id: str,
    task_ids: list[int],
    trials: int,
    suite: str,
    out_dir: Path,
    rotate_every = None,
) -> None:
    """Run LIBERO eval in Docker, ``trials`` episodes per task.

    With ``rotate_every``, tasks are cycled in chunks of at most that many trials while one
    ``openpi_server`` stays up, so online training continues across chunks.
    """
    env_base = os.environ.copy()
    # compose.yml resolves mounts relative to PWD, which may differ from the repo root.
    env_base["PWD"] = str(root.resolve())
    env_base.setdefault("ELASTIC_RUNS", str(runs_host_dir()))
    env_base["COMPOSE_PROJECT_NAME"] = compose_project_name
    env_base["GPU_DEVICE_ID"] = gpu_device_id

    if rotate_every is None:
        ids_sorted = sorted(task_ids)
        print(
            f"\n=== {run_name}: task_ids={ids_sorted} trials_per_task={trials} ===",
            flush=True,
        )
        _docker_eval_task_list_invocation(
            root=root,
            env_base=env_base,
            run_name=run_name,
            server_args=server_args,
            policy_port=policy_port,
            task_ids=ids_sorted,
            trials=trials,
            suite=suite,
            out_dir=out_dir,
            path_suffix="",
        )
        return

    chunk = int(rotate_every)
    if chunk <= 0:
        raise ValueError("--online_rotate_every must be a positive integer")
    remaining = {tid: int(trials) for tid in task_ids}
    inv = 0
    _docker_compose_down(root=root, env_base=env_base)
    try:
        print(
            f"\n=== {run_name}: starting single openpi_server for multitask rotation "
            f"(chunk size {chunk}, {trials} trials/task total) ===",
            flush=True,
        )
        _docker_start_openpi_server(
            root=root,
            env_base=env_base,
            server_args=server_args,
            policy_port=policy_port,
            expected_output_dir_name=out_dir.name,
        )
        while any(r > 0 for r in remaining.values()):
            active_ids = [tid for tid in sorted(task_ids) if remaining[tid] > 0]
            n = min(chunk, min(remaining[tid] for tid in active_ids))
            suffix = f"_inv{inv:05d}"
            before = ",".join(f"{tid}:{remaining[tid]}" for tid in active_ids)
            after = ",".join(f"{tid}:{remaining[tid] - n}" for tid in active_ids)
            print(
                f"\n=== {run_name}: task_ids={active_ids} "
                f"trials_chunk={n} remaining_before=[{before}] remaining_after=[{after}] ===",
                flush=True,
            )
            _docker_eval_task_list_invocation(
                root=root,
                env_base=env_base,
                run_name=run_name,
                server_args=server_args,
                policy_port=policy_port,
                task_ids=active_ids,
                trials=n,
                suite=suite,
                out_dir=out_dir,
                path_suffix=suffix,
                server_already_running=True,
                expected_output_dir_name=out_dir.name,
            )
            for tid in active_ids:
                remaining[tid] -= n
            inv += 1
    finally:
        _docker_compose_down(root=root, env_base=env_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline-to-online STRIDE SAC for pi0.5 on LIBERO.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Run directory (checkpoints, logs, replay), under ELASTIC_RUNS (default data/libero) "
        "or the repo. Default: ELASTIC_RUNS/runs/<stamp>.",
    )
    parser.add_argument("--collect_demos", action="store_true")
    parser.add_argument("--task_ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--demo_trials", type=int, default=10)
    parser.add_argument("--online_trials", type=int, default=20)
    parser.add_argument(
        "--online_rotate_every",
        type=int,
        default=None,
        help="Optional. Interleave LIBERO eval in chunks of this many trials per task (multitask schedule) "
        "while keeping one openpi_server / STRIDE training session for the whole online phase. "
        "Each task still receives --online_trials episodes in total.",
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--baseline_num_steps", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--k_subsample_trials", type=int, default=None)
    parser.add_argument("--vgps_checkpoint", default=DEFAULT_VGPS_CHECKPOINT)
    parser.add_argument(
        "--offline_replay",
        default=None,
        help="Host path to offline_replay.npz. When --output_dir is omitted, a new "
        "ELASTIC_RUNS/runs/<stamp> directory is created; this path only selects which "
        "buffer to load.",
    )
    parser.add_argument("--latency_penalty", type=float, default=LATENCY_PENALTY)
    parser.add_argument("--parallel_penalty", type=float, default=PARALLEL_PENALTY)
    parser.add_argument("--terminal_bonus", type=float, default=TERMINAL_BONUS)
    parser.add_argument("--no_finish_penalty", type=float, default=NO_FINISH_PENALTY)
    parser.add_argument("--verifier_bias", type=float, default=VERIFIER_BIAS)
    parser.add_argument("--k_subsample_counterfactuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k_subsample_draws", type=int, default=2)
    parser.add_argument("--second_pass_counterfactuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--second_pass_every", type=int, default=4)
    parser.add_argument(
        "--all_stopped_counterfactuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add one synthetic replay row per infer where all K samples are paused (reward=no_finish_penalty).",
    )
    parser.add_argument("--offline_pretrain_steps", type=int, default=OFFLINE_PRETRAIN_STEPS)
    parser.add_argument("--online_updates_per_step", type=int, default=ONLINE_UPDATES_PER_STEP)
    parser.add_argument("--min_online_transitions", type=int, default=ONLINE_MIN_TRANSITIONS)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_project", default="sac_stride_openpi")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument(
        "--policy-port",
        type=int,
        default=8000,
        help=(
            "TCP port for serve_policy (--port) and the LIBERO client (--args.port). "
            "Use a different port per parallel run (e.g. 8000 and 8001)."
        ),
    )
    parser.add_argument(
        "--compose-project-name",
        default=None,
        help=(
            "Docker Compose project name (COMPOSE_PROJECT_NAME). Defaults to a sanitized --output_dir "
            "basename so parallel runs do not tear each other down via ``docker compose down``."
        ),
    )
    parser.add_argument(
        "--cuda-device",
        default=None,
        metavar="ID",
        help=(
            "Physical GPU index for Docker (sets compose GPU_DEVICE_ID / device_ids). "
            "Use 1 for the second GPU when running a second instance in parallel. "
            "If omitted, uses GPU_DEVICE_ID, NVIDIA_VISIBLE_DEVICES, or CUDA_VISIBLE_DEVICES "
            "from the environment, else 0."
        ),
    )
    args = parser.parse_args()

    root = ROOT
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = (runs_host_dir() / "runs" / stamp).resolve()
    container_run_dir = to_container_path(out_dir, writable=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    compose_project_name = args.compose_project_name or sanitize_compose_project_name(out_dir.name)
    policy_port = int(args.policy_port)
    gpu_device_id = resolve_gpu_device_id(cuda_device=args.cuda_device)
    print(
        f"[INFO] Docker compose project={compose_project_name!r} policy_port={policy_port} "
        f"gpu_device_id={gpu_device_id!r} output_dir={out_dir}",
        flush=True,
    )
    task_ids = parse_task_ids(args.task_ids)
    runtime_suite = normalize_suite_name(args.suite)
    docker_eval_kwargs = {
        "root": root,
        "policy_port": policy_port,
        "compose_project_name": compose_project_name,
        "gpu_device_id": gpu_device_id,
        "task_ids": task_ids,
        "suite": runtime_suite,
        "out_dir": out_dir,
    }
    baseline_num_steps = args.baseline_num_steps if args.num_steps is None else int(args.num_steps)
    stride_options = ",".join(str(item) for item in STRIDE_BUCKETS)
    stride_weights = ",".join(str(item) for item in STRIDE_BUCKET_WEIGHTS)
    offline_replay = (
        Path(args.offline_replay).expanduser().resolve() if args.offline_replay else out_dir / "offline_replay.npz"
    )
    if args.offline_replay:
        offline_replay = prepare_offline_replay_for_run(
            source_path=offline_replay,
            output_dir=out_dir,
            terminal_bonus=args.terminal_bonus,
            verifier_bias=args.verifier_bias,
            latency_penalty=args.latency_penalty,
            parallel_penalty=args.parallel_penalty,
            no_finish_penalty=args.no_finish_penalty,
        )
    container_offline_replay = to_container_path(offline_replay)

    if args.collect_demos:
        record_dirs = []
        record_dir = out_dir / "demo_records_random"
        record_dirs.append(record_dir)
        server_args = (
            f"--eval-type stride --num-samples {args.num_samples} --num-steps {baseline_num_steps} "
            "--stride-selector per_chunk_random "
            f"--stride-replay-dir {container_run_dir}/demo_records_random "
            f"--stride-random-step-options {stride_options} "
            f"--stride-random-step-weights {stride_weights} "
            f"--vgps-checkpoint {args.vgps_checkpoint}"
        )
        docker_eval(
            **docker_eval_kwargs,
            run_name="collect_demos",
            server_args=server_args,
            trials=args.demo_trials,
        )

        k_subsample_trials = args.demo_trials if args.k_subsample_trials is None else args.k_subsample_trials
        for stride_bucket in K_SUBSAMPLE_STRIDES:
            fixed_dir = out_dir / f"demo_records_fixed_stride{stride_bucket:03d}"
            record_dirs.append(fixed_dir)
            fixed_stride = float(stride_bucket) / DIFFUSION_T_MAX
            fixed_server_args = (
                f"--eval-type stride --num-samples {args.num_samples} --num-steps {baseline_num_steps} "
                f"--stride-selector fixed --stride-fixed-stride {fixed_stride} "
                f"--stride-replay-dir {container_run_dir}/demo_records_fixed_stride{stride_bucket:03d} "
                f"--vgps-checkpoint {args.vgps_checkpoint}"
            )
            docker_eval(
                **docker_eval_kwargs,
                run_name=f"collect_fixed_stride{stride_bucket:03d}",
                server_args=fixed_server_args,
                trials=k_subsample_trials,
            )

        build_replay_from_record_dir(
            record_dirs,
            output_path=offline_replay,
            num_samples=args.num_samples,
            terminal_bonus=args.terminal_bonus,
            latency_penalty=args.latency_penalty,
            parallel_penalty=args.parallel_penalty,
            no_finish_penalty=args.no_finish_penalty,
            verifier_bias=args.verifier_bias,
            add_k_subsample=args.k_subsample_counterfactuals,
            k_subsample_draws=args.k_subsample_draws,
            add_all_stopped=args.all_stopped_counterfactuals,
        )
        print(f"Saved offline replay to {offline_replay}")

    if not offline_replay.exists():
        raise FileNotFoundError(
            f"Offline replay not found: {offline_replay}. Run with --collect_demos first or pass --offline_replay."
        )

    ckpt_dir = out_dir / "checkpoints"
    wandb_name = args.wandb_name or out_dir.name
    k_subsample_flag = (
        "--stride-k-subsample-counterfactuals"
        if args.k_subsample_counterfactuals
        else "--no-stride-k-subsample-counterfactuals"
    )
    second_pass_flag = (
        "--stride-second-pass-counterfactuals"
        if args.second_pass_counterfactuals
        else "--no-stride-second-pass-counterfactuals"
    )
    all_stopped_flag = (
        "--stride-all-stopped-counterfactuals"
        if args.all_stopped_counterfactuals
        else "--no-stride-all-stopped-counterfactuals"
    )
    wandb_args = ""
    if args.wandb and args.wandb_project:
        wandb_args = (
            f"--stride-wandb-project {args.wandb_project} "
            f"--stride-wandb-name {wandb_name} "
            f"--stride-wandb-dir {container_run_dir} "
        )
    server_args = (
        f"--eval-type stride --num-samples {args.num_samples} --num-steps {baseline_num_steps} "
        "--stride-selector learned "
        f"--stride-offline-replay {container_offline_replay} "
        f"--stride-checkpoint-dir {container_run_dir}/checkpoints "
        f"--stride-latency-penalty {args.latency_penalty} "
        f"--stride-parallel-penalty {args.parallel_penalty} "
        f"--stride-terminal-bonus {args.terminal_bonus} "
        f"--stride-no-finish-penalty {args.no_finish_penalty} "
        f"--stride-verifier-bias {args.verifier_bias} "
        f"{k_subsample_flag} "
        f"--stride-k-subsample-draws {args.k_subsample_draws} "
        f"{second_pass_flag} "
        f"--stride-second-pass-every {args.second_pass_every} "
        f"{all_stopped_flag} "
        f"--stride-offline-pretrain-steps {args.offline_pretrain_steps} "
        f"--stride-online-updates-per-step {args.online_updates_per_step} "
        f"--stride-min-online-transitions {args.min_online_transitions} "
        f"{wandb_args}"
        f"--vgps-checkpoint {args.vgps_checkpoint}"
    )
    docker_eval(
        **docker_eval_kwargs,
        run_name="online_train",
        server_args=server_args,
        trials=args.online_trials,
        rotate_every=args.online_rotate_every,
    )
    print(f"STRIDE run complete. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
