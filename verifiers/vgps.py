from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared import image_tools


def _jaxrl_m_repo_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` containing ``jaxrl_m/agents``, or None."""
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(48):
        if (cur / "jaxrl_m" / "agents").is_dir():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


def _resolve_vgps_repo_root(
    checkpoint: str | None,
    config_path: str | None,
    *,
    bundled_root: Path,
) -> Path:
    """Directory to prepend to ``sys.path`` so ``import jaxrl_m`` works."""
    env = os.environ.get("VGPS_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "jaxrl_m" / "agents").is_dir():
            return root
        raise FileNotFoundError(
            f"VGPS_ROOT={env!r} is not a V-GPS checkout root (missing jaxrl_m/agents/)."
        )

    for raw in (checkpoint, config_path):
        if not raw:
            continue
        found = _jaxrl_m_repo_root(Path(raw))
        if found is not None:
            return found

    if (bundled_root / "jaxrl_m" / "agents").is_dir():
        return bundled_root

    raise ModuleNotFoundError(
        "jaxrl_m (V-GPS) not found: init the verifiers/v-gps submodule or set VGPS_ROOT to a V-GPS checkout."
    )


def load_vgps_agent_and_text_processor(
    checkpoint: str | Path | None,
    config_path: str | Path | None,
    *,
    bundled_root: Path | None = None,
) -> tuple[Any, Any]:
    """Load jaxrl_m V-GPS agent and text processor from checkpoint + YAML config."""
    import yaml
    from flax.training import checkpoints

    bundled = bundled_root if bundled_root is not None else Path(__file__).resolve().parent / "v-gps"
    ck_arg = str(checkpoint) if checkpoint is not None else None
    cfg_arg = str(config_path) if config_path is not None else None
    vgps_root = _resolve_vgps_repo_root(ck_arg, cfg_arg, bundled_root=bundled)
    root_s = str(vgps_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    logging.info("V-GPS repo root on sys.path: %s", vgps_root)

    from jaxrl_m.agents import agents
    from jaxrl_m.data.text_processing import text_processors
    from jaxrl_m.vision import encoders

    checkpoint_path = Path(checkpoint) if checkpoint is not None else vgps_root / "checkpoint_500000"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"V-GPS checkpoint path does not exist: {checkpoint_path}")

    config_file = Path(config_path) if config_path is not None else vgps_root / "experiments/configs/pretrained_checkpoint.yaml"
    with config_file.open("r") as f:
        config = yaml.safe_load(f)

    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
    example_actions = np.zeros((1, 7), dtype=np.float32)
    example_obs = {"image": np.zeros((1, 256, 256, 3), dtype=np.uint8)}
    example_goals = {"language": np.zeros((1, 512), dtype=np.float32)}
    agent = agents[config["agent"]].create(
        rng=jax.random.PRNGKey(0),
        observations=example_obs,
        goals=example_goals,
        actions=example_actions,
        encoder_def=encoder_def,
        **config["agent_kwargs"],
    )
    logging.info("Loading V-GPS checkpoint from %s", checkpoint_path)
    restored = checkpoints.restore_checkpoint(str(checkpoint_path), agent)
    # Must match V-GPS training: omitting ``text_processor_kwargs`` changes the
    # instruction embeddings and Q(s, a).
    tp_name = config["text_processor"]
    tp_kwargs = dict(config.get("text_processor_kwargs") or {})
    text_processor = text_processors[tp_name](**tp_kwargs)
    return restored, text_processor


def extract_vgps_image(
    obs: dict,
    *,
    convert_bgr_to_rgb: bool = False,
    vgps_image_size: int = 256,
) -> np.ndarray:
    """The camera frame V-GPS scores, resized (with padding) to the checkpoint's ``vgps_image_size``.

    LIBERO clients send ``observation/image``; DROID-style clients send the left exterior camera.
    ``convert_bgr_to_rgb`` is only for checkpoints trained on BGR-converted frames.
    """
    if "observation/exterior_image_1_left" in obs:
        image = obs["observation/exterior_image_1_left"]
    elif "observation/image" in obs:
        image = obs["observation/image"]
    elif "image" in obs and isinstance(obs["image"], np.ndarray):
        image = obs["image"]
    elif "image" in obs and isinstance(obs["image"], dict):
        image = next(iter(obs["image"].values()))
    else:
        raise ValueError(
            "V-GPS scoring requires an image in obs['observation/exterior_image_1_left'], "
            "obs['observation/image'], or obs['image']."
        )

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    if convert_bgr_to_rgb:
        image = np.ascontiguousarray(image[..., ::-1])
    sz = int(vgps_image_size)
    if sz <= 0:
        raise ValueError(f"vgps_image_size must be positive, got {vgps_image_size!r}")
    out = np.asarray(image_tools.resize_with_pad(image, sz, sz))
    return out


def extract_vgps_prompt(obs: dict) -> str:
    prompt = obs.get("prompt")
    if prompt is None:
        raise ValueError("V-GPS scoring requires obs['prompt'].")
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    if isinstance(prompt, np.ndarray):
        prompt = prompt.item()
        if isinstance(prompt, bytes):
            return prompt.decode("utf-8")
    return str(prompt)


class VGPSActionVerifier:
    """V-GPS Q-values for a set of candidate action chunks; selects one by softmax (or argmax) over Q."""

    def __init__(
        self,
        *,
        checkpoint: str | None = None,
        config_path: str | None = None,
        action_temp: float = 1.0,
        rng: jax.Array | None = None,
        convert_bgr_to_rgb: bool = False,
        vgps_image_size: int = 256,
    ):
        self._action_temp = float(action_temp)
        self._rng = jax.random.key(0) if rng is None else rng
        self._convert_bgr_to_rgb = bool(convert_bgr_to_rgb)
        self._vgps_image_size = int(vgps_image_size)
        self._encoded_prompt_cache: dict[str, np.ndarray] = {}
        self._init_vgps(checkpoint, config_path)

    def score_actions(self, obs: dict, actions: np.ndarray) -> np.ndarray:
        image = extract_vgps_image(
            obs,
            convert_bgr_to_rgb=self._convert_bgr_to_rgb,
            vgps_image_size=self._vgps_image_size,
        )
        prompt = extract_vgps_prompt(obs)
        action_count = actions.shape[0]
        values = self._agent.get_q_values(
            observations={"image": np.repeat(image[None, ...], action_count, axis=0)},
            goals={"language": np.repeat(self._encode_prompt(prompt)[None, ...], action_count, axis=0)},
            actions=actions[:, 0, :7],
        )
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape != (action_count,):
            raise ValueError(f"Expected V-GPS values shape {(action_count,)}, got {values.shape}")
        logging.debug("V-GPS values: max=%.2f min=%.2f mean=%.2f", values.max(), values.min(), values.mean())
        return values

    def select(
        self,
        obs: dict,
        actions: np.ndarray,
        *,
        eligible_mask: np.ndarray | None = None,
    ) -> tuple[int, float, np.ndarray]:
        action_count = actions.shape[0]
        eligible = None
        if eligible_mask is not None:
            eligible = np.asarray(eligible_mask, dtype=np.bool_).reshape(-1)
            if eligible.shape != (action_count,):
                raise ValueError(f"Expected eligible_mask shape {(action_count,)}, got {eligible.shape}")
            if not np.any(eligible):
                logging.warning(
                    "No STRIDE candidates finished denoising; V-GPS will score every candidate including "
                    "partially denoised chunks (often unstable early in training). Increase denoise budget / "
                    "num_steps or relax STRIDE compute targets so at least one sample reaches t<=0."
                )
                eligible = None

        if eligible is None:
            values = self.score_actions(obs, actions)
            selection_values = values
        else:
            values = np.full((action_count,), -np.inf, dtype=np.float32)
            eligible_indices = np.flatnonzero(eligible)
            eligible_values = self.score_actions(obs, actions[eligible_indices])
            values[eligible_indices] = eligible_values
            selection_values = values

        finite = values[np.isfinite(values)]
        if finite.size:
            logging.debug("V-GPS values: max=%.2f min=%.2f mean=%.2f", finite.max(), finite.min(), finite.mean())
        if self._action_temp > 0:
            self._rng, action_rng = jax.random.split(self._rng)
            index = int(jax.random.categorical(action_rng, jnp.asarray(selection_values) / self._action_temp))
        else:
            index = int(np.argmax(selection_values))
        return index, float(values[index]), values

    def _encode_prompt(self, prompt: str) -> np.ndarray:
        cached = self._encoded_prompt_cache.get(prompt)
        if cached is not None:
            return cached
        encoded = np.asarray(self._text_processor.encode([prompt])[0], dtype=np.float32)
        self._encoded_prompt_cache[prompt] = encoded
        return encoded

    def _init_vgps(self, checkpoint: str | None, config_path: str | None) -> None:
        bundled_root = Path(__file__).resolve().parent / "v-gps"
        self._agent, self._text_processor = load_vgps_agent_and_text_processor(
            checkpoint,
            config_path,
            bundled_root=bundled_root,
        )
