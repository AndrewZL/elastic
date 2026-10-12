"""Frozen DuelingVerifier used to score candidate action chunks (Diffusion Policy tasks)."""
import torch
import tqdm

from verifiers.dueling import DuelingVerifier, dueling_uses_layer_norm


class AdvantageVerifier:
    """Loads a lowdim ``DuelingVerifier`` checkpoint and scores (obs, action chunk) pairs.

    ``noise_std`` adds Gaussian noise to advantage and Q outputs (verifier-noise ablation).
    """

    def __init__(self, q_path, device, noise_std: float = 0.0):
        self.device = device
        self.noise_std = float(noise_std)
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")

        state = torch.load(q_path, map_location=device)
        if not (isinstance(state, dict) and "value_stream.0.weight" in state and "advantage_stream.0.weight" in state):
            keys = list(state)[:12] if isinstance(state, dict) else type(state)
            raise RuntimeError(f"Expected a DuelingVerifier checkpoint, got keys={keys} for {q_path}")
        obs_w = state["obs_enc.0.weight"]
        if obs_w.ndim == 4:
            raise RuntimeError(f"Image-mode verifier checkpoints are not supported: {q_path}")

        self.expected_action_flat_dim = int(state["action_mean"].numel())
        self.expected_obs_dim = int(obs_w.shape[1])
        use_ln = dueling_uses_layer_norm(state)
        self.net = DuelingVerifier(
            obs_shape=self.expected_obs_dim,
            action_dim=self.expected_action_flat_dim,
            hidden_dim=int(state["action_enc.0.weight"].shape[0]),
            use_layer_norm=use_ln,
        ).to(device)
        self.net.load_state_dict(state)
        self.net.eval()
        tqdm.tqdm.write(f"AdvantageVerifier: loaded {q_path} (layer_norm={use_ln})")

    def _maybe_add_noise(self, values: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0:
            return values
        return values + self.noise_std * torch.randn_like(values)

    def _flat_inputs(self, obs, action):
        """Flatten; an action chunk longer than the verifier's training length is truncated."""
        B = obs.shape[0]
        flat_action = action.reshape(B, -1)
        expected = self.expected_action_flat_dim
        if flat_action.shape[1] > expected:
            flat_action = flat_action[:, :expected]
        elif flat_action.shape[1] < expected:
            raise RuntimeError(f"Verifier action width {flat_action.shape[1]} < expected {expected}")
        return obs.reshape(B, -1), flat_action

    def compute_advantage(self, obs, action):
        with torch.no_grad():
            _, _, adv = self.net(*self._flat_inputs(obs, action))
            return self._maybe_add_noise(adv)

    def compute_v(self, obs, action):
        with torch.no_grad():
            _, v, _ = self.net(*self._flat_inputs(obs, action))
            return v
