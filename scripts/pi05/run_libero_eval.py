#!/usr/bin/env python3
"""Run LIBERO-10-R evals through docker/pi05/compose.yml, one server/client pair per run.

Runs, in order: ``--normal-baseline``, ``--parallel-baseline``, ``--fixed-baseline``, ``--stride-run NAME=CKPT``.
Checkpoints are container paths (/runs, /data, /elastic); existing host paths under those roots are translated.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch import ROOT, compose_cmd, compose_env, load_recipe, runs_host_dir, to_container_path  # noqa: E402

DEFAULT_VGPS_CHECKPOINT = load_recipe(ROOT / "configs/libero.yaml")["vgps_checkpoint"]

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TASK_RE = re.compile(r"\[LIBERO\]\s*task=(\d+)\s*\([^)]*\):\s*(.+)$")
SUCCESS_RE = re.compile(r"Success:\s*(True|False)")
TOTAL_RE = re.compile(r"Total success rate:\s*([0-9.]+)")


def normal_server_args(*, num_steps: int) -> str:
    return f"--env LIBERO --eval-type normal --num-steps {int(num_steps)}"


def parallel_server_args(*, vgps_checkpoint: str, num_samples: int, num_steps: int, action_temp: float) -> str:
    return (
        "--env LIBERO --eval-type parallel "
        f"--num-samples {int(num_samples)} --num-steps {int(num_steps)} "
        f"--action-temp {float(action_temp)} --vgps-checkpoint {vgps_checkpoint}"
    )


def stride_server_args(
    *, checkpoint: str, vgps_checkpoint: str, num_samples: int, num_steps: int, action_temp: float
) -> str:
    return (
        "--env LIBERO --eval-type stride --stride-selector learned "
        f"--stride-checkpoint {checkpoint} "
        f"--num-samples {int(num_samples)} --num-steps {int(num_steps)} --action-temp {float(action_temp)} "
        f"--vgps-checkpoint {vgps_checkpoint}"
    )


def server_path(raw: str) -> str:
    """Host path (must exist) -> container path; anything else is taken as a container path already.

    Existence decides, not the prefix: a host path like /data/<user>/... looks like the container's /data.
    """
    if Path(raw).expanduser().exists():
        return str(to_container_path(Path(raw)))
    if raw.startswith(("/runs/", "/data/", "/elastic/")):
        return raw
    raise SystemExit(f"{raw}: not found on the host, and not a container path (/runs, /data, /elastic)")


def clean(line: str) -> str:
    line = ANSI.sub("", line).replace("\r", "\n")
    if "|" in line:
        line = line.split("|", 1)[1]
    return line.strip()


def parse_log(path: Path) -> dict:
    stats: dict[str, dict[str, int]] = {}
    current = None
    overall = None
    for raw in path.read_text(errors="replace").splitlines():
        line = clean(raw)
        if m := TASK_RE.search(line):
            current = m.group(1)
            stats.setdefault(current, {"successes": 0, "episodes": 0})
            continue
        if (m := SUCCESS_RE.search(line)) and current:
            stats[current]["episodes"] += 1
            stats[current]["successes"] += m.group(1) == "True"
        if m := TOTAL_RE.search(line):
            overall = float(m.group(1))
    successes = sum(v["successes"] for v in stats.values())
    episodes = sum(v["episodes"] for v in stats.values())
    return {
        "overall_success_rate": overall if overall is not None else (successes / episodes if episodes else 0.0),
        "total_successes": successes,
        "total_episodes": episodes,
        "tasks": stats,
    }


def write_summary(run_dir: Path, name: str, parsed: dict) -> None:
    (run_dir / f"{name}.json").write_text(json.dumps(parsed, indent=2) + "\n")
    lines = ["task_id,successes,episodes,success_rate"]
    for task_id, s in sorted(parsed["tasks"].items(), key=lambda kv: int(kv[0])):
        rate = s["successes"] / s["episodes"] if s["episodes"] else 0.0
        lines.append(f'"{task_id}",{s["successes"]},{s["episodes"]},{rate:.4f}')
    lines.append(f'"OVERALL",{parsed["total_successes"]},{parsed["total_episodes"]},{parsed["overall_success_rate"]:.4f}')
    (run_dir / f"{name}.csv").write_text("\n".join(lines) + "\n")


def stride_summary_compute(summary_path: Path) -> tuple[float, float]:
    """(avg denoise steps, avg active samples) per query from an ELASTIC eval summary."""
    overall = json.loads(summary_path.read_text()).get("overall") or {}
    steps = overall.get("denoise_steps_per_query_mean_globally")
    samples = overall.get("avg_active_samples_per_query_mean_globally")
    if steps is None or samples is None:
        raise ValueError(f"{summary_path} has no overall.*_mean_globally steps/samples")
    return float(steps), float(samples)


def server_eval_metadata(server_args: str) -> dict:
    meta = {"eval_type": "normal", "num_samples": None, "num_steps": None, "vgps_checkpoint": None}
    parts = shlex.split(server_args)
    for flag, value in zip(parts, parts[1:]):
        if flag == "--eval-type":
            meta["eval_type"] = value
        elif flag == "--num-samples":
            meta["num_samples"] = int(value)
        elif flag == "--num-steps":
            meta["num_steps"] = int(value)
        elif flag == "--vgps-checkpoint":
            meta["vgps_checkpoint"] = value
    meta["server_args"] = server_args
    return meta


def write_run_meta(eval_summary_path: Path, meta: dict) -> Path | None:
    """Sidecar next to the (container-written, often root-owned) eval summary."""
    if not eval_summary_path.exists():
        return None
    sidecar = eval_summary_path.with_name(f"{eval_summary_path.stem}_run_meta.json")
    sidecar.write_text(json.dumps({"eval_summary_json": eval_summary_path.name, **meta}, indent=2) + "\n")
    return sidecar


def build_runs(args: argparse.Namespace) -> list[tuple[str, str]]:
    vgps = server_path(args.vgps_checkpoint)
    runs: list[tuple[str, str]] = []
    if args.normal_baseline:
        runs.append((args.normal_run_name, normal_server_args(num_steps=args.normal_num_steps)))
    if args.parallel_baseline:
        runs.append(
            (
                args.parallel_run_name,
                parallel_server_args(
                    vgps_checkpoint=vgps,
                    num_samples=args.parallel_num_samples,
                    num_steps=args.parallel_num_steps,
                    action_temp=args.parallel_action_temp,
                ),
            )
        )
    if args.fixed_baseline:
        runs.append(
            (
                args.fixed_run_name,
                parallel_server_args(
                    vgps_checkpoint=vgps,
                    num_samples=args.fixed_num_samples,
                    num_steps=args.fixed_num_steps,
                    action_temp=args.fixed_action_temp,
                ),
            )
        )
    for item in args.stride_run:
        name, sep, checkpoint = item.partition("=")
        if not sep or not name.strip() or not checkpoint.strip():
            raise SystemExit(f"--stride-run must be NAME=CHECKPOINT, got: {item}")
        runs.append(
            (
                name.strip(),
                stride_server_args(
                    checkpoint=server_path(checkpoint.strip()),
                    vgps_checkpoint=vgps,
                    num_samples=args.stride_num_samples,
                    num_steps=args.stride_num_steps,
                    action_temp=args.stride_action_temp,
                ),
            )
        )
    return runs


def client_args(args: argparse.Namespace, *, video_dir: Path, eval_summary: Path) -> str:
    parts = [
        "--args.host 127.0.0.1",
        f"--args.port {args.policy_port}",
        f"--args.task-suite-name {args.suite}",
        f"--args.num-trials-per-task {args.trials}",
        f"--args.seed {args.seed}",
        f"--args.video-out-path {to_container_path(video_dir, writable=True)}",
        f"--args.eval-summary-json {to_container_path(eval_summary, writable=True)}",
    ]
    if args.only_task is not None:
        parts += [f"--args.task-start-index {args.only_task}", "--args.num-tasks 1"]
    elif args.num_tasks is not None:
        parts.append(f"--args.num-tasks {args.num_tasks}")
    return " ".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20, help="Trials per task.")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="First N tasks of the suite. LIBERO-10-R's libero_10 has 43 tasks; the paper evaluates the first 10. -1 for all.",
    )
    ap.add_argument("--only-task", type=int, default=None, help="Evaluate a single task id.")
    ap.add_argument("--seed", type=int, default=7, help="LIBERO client seed.")
    ap.add_argument("--policy-seed", type=int, default=0, help="Seed for the policy's sampling noise (server jax.random.key).")
    ap.add_argument("--out", type=Path, default=None, help="Run directory. Default: ELASTIC_RUNS/eval/<stamp>.")
    ap.add_argument("--policy-port", type=int, default=8001)
    ap.add_argument(
        "--vgps-checkpoint",
        "--default-vgps-checkpoint",
        dest="vgps_checkpoint",
        default=DEFAULT_VGPS_CHECKPOINT,
        help="V-GPS checkpoint for parallel, fixed and stride runs. Default: configs/libero.yaml.",
    )

    ap.add_argument("--normal-baseline", action="store_true")
    ap.add_argument("--normal-run-name", default="normal_10step")
    ap.add_argument("--normal-num-steps", type=int, default=10)

    ap.add_argument("--parallel-baseline", action="store_true")
    ap.add_argument("--parallel-run-name", default="parallel_10x10")
    ap.add_argument("--parallel-num-samples", type=int, default=10)
    ap.add_argument("--parallel-num-steps", type=int, default=10)
    ap.add_argument("--parallel-action-temp", type=float, default=1.0)

    ap.add_argument("--fixed-baseline", action="store_true")
    ap.add_argument("--fixed-run-name", default="fixed")
    ap.add_argument("--fixed-num-samples", type=int, default=10)
    ap.add_argument("--fixed-num-steps", type=int, default=10)
    ap.add_argument("--fixed-action-temp", type=float, default=1.0)
    ap.add_argument(
        "--match-stride-summary",
        type=Path,
        default=None,
        help="ELASTIC *_eval_summary.json; sets the fixed baseline to round(avg samples) x round(avg steps).",
    )
    ap.add_argument("--matched-avg-steps", type=float, default=None)
    ap.add_argument("--matched-avg-samples", type=float, default=None)

    ap.add_argument("--stride-run", action="append", default=[], metavar="NAME=CHECKPOINT")
    ap.add_argument("--stride-num-samples", type=int, default=10)
    ap.add_argument("--stride-num-steps", type=int, default=10)
    ap.add_argument("--stride-action-temp", type=float, default=1.0)
    args = ap.parse_args(argv)

    if args.match_stride_summary is not None:
        args.matched_avg_steps, args.matched_avg_samples = stride_summary_compute(args.match_stride_summary)
    if (args.matched_avg_steps is None) != (args.matched_avg_samples is None):
        ap.error("--matched-avg-steps and --matched-avg-samples go together")
    if args.matched_avg_steps is not None:
        args.fixed_num_steps = max(1, round(args.matched_avg_steps))
        args.fixed_num_samples = max(1, round(args.matched_avg_samples))
        args.fixed_baseline = True
        print(
            f"Matched fixed baseline: {args.fixed_num_samples}x{args.fixed_num_steps} "
            f"(avg samples {args.matched_avg_samples:.3f}, avg steps {args.matched_avg_steps:.3f})",
            flush=True,
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = build_runs(args)
    if not runs:
        print("No runs configured: pass --normal-baseline, --parallel-baseline, --fixed-baseline or --stride-run.")
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.out or runs_host_dir() / "eval" / stamp).expanduser().resolve()
    to_container_path(run_dir, writable=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    env_base = compose_env()
    project = f"elastic_eval_{stamp}"

    results = {}
    for name, server_args in runs:
        print(f"\n=== Running {name} ===", flush=True)
        subprocess.run(compose_cmd("down", project=project), cwd=ROOT, env=env_base)
        eval_summary = run_dir / f"{name}_eval_summary.json"
        env = dict(env_base)
        env["SERVER_ARGS"] = f"{server_args} --port {args.policy_port} --seed {args.policy_seed}"
        env["CLIENT_ARGS"] = client_args(args, video_dir=run_dir / f"videos_{name}", eval_summary=eval_summary)

        log_path = run_dir / f"{name}.log"
        with log_path.open("w") as log:
            p = subprocess.Popen(
                compose_cmd("up", "--abort-on-container-exit", project=project),
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert p.stdout is not None
            for line in p.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = p.wait()
        if rc == 0 and "Error in inference server" in log_path.read_text(errors="replace"):
            print(f"{name}: the policy server raised during inference; see {log_path}", flush=True)
            rc = 1
        subprocess.run(compose_cmd("down", project=project), cwd=ROOT, env=env_base)

        meta = server_eval_metadata(server_args)
        run_meta = write_run_meta(eval_summary, meta)
        parsed = parse_log(log_path)
        parsed.update(
            exit_code=rc,
            eval_summary_path=str(eval_summary),
            eval_summary_run_meta_path=str(run_meta) if run_meta else None,
            **{k: meta[k] for k in ("eval_type", "num_samples", "num_steps", "vgps_checkpoint")},
        )
        results[name] = parsed
        write_summary(run_dir, name, parsed)
        print(
            f"=== {name}: {parsed['total_successes']}/{parsed['total_episodes']} "
            f"= {parsed['overall_success_rate']:.3f} ===",
            flush=True,
        )

    md = ["# LIBERO-10-R eval", "", "| run | successes | episodes | success rate | exit |", "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        md.append(
            f"| {name} | {r['total_successes']} | {r['total_episodes']} | {r['overall_success_rate']:.3f} | {r['exit_code']} |"
        )
    (run_dir / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote summaries to: {run_dir}")
    return 0 if all(r["exit_code"] == 0 for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
