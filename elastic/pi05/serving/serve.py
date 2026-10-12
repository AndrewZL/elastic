"""ELASTIC pi0.5 policy server on top of unmodified openpi.

Serves ``normal`` (single-sample flow matching), ``parallel`` (best-of-S with V-GPS) or
``stride`` (ELASTIC adaptive strides, optionally trained online with SAC).
"""
import asyncio
import atexit
import dataclasses
import enum
import logging
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax  # noqa: E402
import tyro  # noqa: E402
import websockets  # noqa: E402
import websockets.asyncio.server as _server  # noqa: E402
import websockets.frames  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402

from elastic.pi05.serving.baselines import NormalPolicy, ParallelPolicy  # noqa: E402
from elastic.pi05.config import StrideConfig  # noqa: E402
from elastic.pi05.sampling.model import adopt_model  # noqa: E402
from elastic.pi05.serving.policy import StrideAdaptivePolicy  # noqa: E402


class EnvMode(enum.Enum):
    LIBERO = "libero"
    DROID = "droid"


DEFAULT_CHECKPOINT = {
    EnvMode.LIBERO: ("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
    EnvMode.DROID: ("pi05_droid", "gs://openpi-assets/checkpoints/pi05_droid"),
}


@dataclasses.dataclass
class Args(StrideConfig):
    env: EnvMode = EnvMode.LIBERO
    # Default: the env's released pi0.5 config and checkpoint.
    config: str | None = None
    dir: str | None = None
    default_prompt: str | None = None
    port: int = 8000
    seed: int = 0
    eval_type: Literal["normal", "parallel", "stride"] = "normal"
    num_samples: int = 10
    num_steps: int = 10
    vgps_checkpoint: str | None = None
    vgps_config_path: str | None = None
    # Softmax temperature over V-GPS scores; <= 0 for argmax.
    action_temp: float = 1.0
    # Must match the V-GPS verifier's training resolution.
    vgps_image_size: int = 256
    include_vgps_candidate_chunks: bool = False


def create_policy(args: Args):
    config_name, ckpt_dir = DEFAULT_CHECKPOINT[args.env]
    base = _policy_config.create_trained_policy(
        _config.get_config(args.config or config_name),
        args.dir or ckpt_dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_steps} if args.eval_type == "normal" else None,
    )
    if args.eval_type == "normal":
        base.__class__ = NormalPolicy
        base._rng = jax.random.key(args.seed)
        return base
    if not args.vgps_checkpoint:
        raise SystemExit("--vgps-checkpoint is required for parallel and stride eval")
    model = adopt_model(base._model)
    common = dict(
        rng=jax.random.key(args.seed),
        transforms=[base._input_transform],
        output_transforms=[base._output_transform],
        metadata=base._metadata,
    )
    if args.eval_type == "parallel":
        return ParallelPolicy(
            model,
            sample_kwargs=dict(
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                vgps_checkpoint=args.vgps_checkpoint,
                vgps_config_path=args.vgps_config_path,
                action_temp=args.action_temp,
                include_vgps_candidate_chunks=args.include_vgps_candidate_chunks,
                vgps_image_size=args.vgps_image_size,
            ),
            **common,
        )
    return StrideAdaptivePolicy(
        model,
        args,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        vgps_checkpoint=args.vgps_checkpoint,
        vgps_config_path=args.vgps_config_path,
        action_temp=args.action_temp,
        vgps_image_size=args.vgps_image_size,
        include_vgps_candidate_chunks=args.include_vgps_candidate_chunks,
        **common,
    )


def _flush_training_state(policy) -> None:
    flush = getattr(policy, "flush_stride_training_state", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            logging.exception("flush_stride_training_state failed")


class ElasticPolicyServer(websocket_policy_server.WebsocketPolicyServer):
    """Upstream server with keepalive pings off and ``infer`` in a worker thread (one infer can be long).

    Training state is flushed on every client disconnect.
    """

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=websocket_policy_server._health_check,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total_time = None
        try:
            while True:
                try:
                    start_time = time.monotonic()
                    obs = msgpack_numpy.unpackb(await websocket.recv())
                    infer_start = time.monotonic()
                    action = await asyncio.to_thread(self._policy.infer, obs)
                    action["server_timing"] = {"infer_ms": (time.monotonic() - infer_start) * 1000}
                    if prev_total_time is not None:
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time
                except websockets.ConnectionClosed:
                    logging.info(f"Connection from {websocket.remote_address} closed")
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            # Off the event loop: a flush can write a large replay file.
            await asyncio.to_thread(_flush_training_state, self._policy)


def _configure_jax_compilation_cache() -> None:
    import jax

    cache_dir = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", Path.home() / ".cache" / "jax"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))


def _install_sigterm_flush(policy) -> None:
    def handler(signum, frame):
        logging.info("Received signal %s; flushing ELASTIC training state.", signum)
        _flush_training_state(policy)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def main(args: Args) -> None:
    _configure_jax_compilation_cache()
    policy = create_policy(args)
    if args.eval_type == "stride" and args.stride_checkpoint_dir:
        atexit.register(_flush_training_state, policy)
        _install_sigterm_flush(policy)
    logging.info("Serving on %s:%d", socket.gethostname(), args.port)
    ElasticPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
