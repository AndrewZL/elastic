import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.clip_grad import clip_grad_norm_
import pathlib
import pickle
import numpy as np
import tqdm
import argparse
from typing import List, Optional, cast


def _set_mc_targets(episode_chunk: List[dict]) -> None:
    """Set undiscounted Monte Carlo ``target`` on each step."""
    if len(episode_chunk) == 0:
        return
    if all(step.get("target_is_explicit", 0.0) for step in episode_chunk):
        for step in episode_chunk:
            step["target"] = float(step["target"])
        return
    if all("coverage_return_to_go" in step for step in episode_chunk):
        for step in episode_chunk:
            step["target"] = float(step["coverage_return_to_go"])
        return
    if all("episode_max_coverage" in step for step in episode_chunk):
        for step in episode_chunk:
            step["target"] = float(step["episode_max_coverage"])
        return
    last_step = episode_chunk[-1]
    if "target" not in last_step:
        raise KeyError("Final episode step is missing both 'target' and 'episode_max_coverage'")
    terminal_reward = float(last_step["target"])
    for step in episode_chunk:
        step["target"] = terminal_reward


def _load_verifier_episodes(blob: object) -> List[List[dict]]:
    """Episodes from a ``{"verifier_episodes": list[list[dict]]}`` pickle."""
    if not isinstance(blob, dict) or "verifier_episodes" not in blob:
        raise ValueError('Verifier dataset must be {"verifier_episodes": list[list[dict]]}.')
    episodes = blob["verifier_episodes"]
    if not isinstance(episodes, list):
        raise ValueError("verifier_episodes must be a list")
    for i, steps in enumerate(episodes):
        if not isinstance(steps, list):
            raise ValueError(f"verifier_episodes[{i}] must be a list, got {type(steps).__name__}")
        if len(steps) == 0:
            raise ValueError(f"verifier_episodes[{i}] is empty")
        if not all(isinstance(step, dict) for step in steps):
            raise ValueError(f"verifier_episodes[{i}] must contain dict transitions only.")
    return episodes


class CleanVerifierDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            blob = pickle.load(f)

        episodes = _load_verifier_episodes(blob)
        raw_data: List[dict] = []
        for ep in episodes:
            _set_mc_targets(ep)
            raw_data.extend(ep)

        for step in raw_data:
            for req in ("obs", "next_obs", "clean_action", "target", "is_last"):
                if req not in step:
                    raise KeyError(f"Transition missing {req!r} after load.")

        self.raw_data = raw_data
        print(f"Loaded verifier dataset: {len(episodes)} episodes, {len(raw_data)} transitions.")

    def _to_lowdim_obs(self, obs):
        if isinstance(obs, dict):
            parts = [
                obs[key].reshape(obs[key].shape[0], -1)
                for key in sorted(obs.keys())
                if isinstance(obs[key], np.ndarray)
            ]
            if not parts:
                raise RuntimeError("No numpy arrays found in dict obs for lowdim mode")
            return np.concatenate(parts, axis=1).astype(np.float32)
        return np.asarray(obs, dtype=np.float32)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        obs = item["obs"]
        next_obs = item.get("next_obs", obs)
        obs_processed = self._to_lowdim_obs(obs)
        next_obs_processed = self._to_lowdim_obs(next_obs)
        target = item.get("target")
        if target is None:
            raise KeyError("Sample is missing 'target' key after processing.")
        is_last = float(item.get("is_last", 0.0))

        return {
            "obs": torch.from_numpy(obs_processed).float(),
            "next_obs": torch.from_numpy(next_obs_processed).float(),
            "action": torch.from_numpy(np.asarray(item["clean_action"], dtype=np.float32)).float(),
            "target": torch.tensor(target).float(),
            "is_last": torch.tensor(is_last).float(),
        }


def _compute_normalization_stats(train_loader, max_batches=200):
    obs_sum, obs_sq_sum, obs_count = None, None, 0
    action_sum, action_sq_sum, action_count = None, None, 0

    for i, batch in enumerate(tqdm.tqdm(train_loader, desc="Computing Stats", leave=False)):
        if i >= max_batches:
            break

        action = batch["action"].double().view(batch["action"].shape[0], -1)
        action_sum = action.sum(dim=0) if action_sum is None else action_sum + action.sum(dim=0)
        action_sq_sum = (
            (action**2).sum(dim=0)
            if action_sq_sum is None
            else action_sq_sum + (action**2).sum(dim=0)
        )
        action_count += action.shape[0]

        obs = batch["obs"].double()
        obs = obs.view(obs.shape[0], -1)
        obs_sum = obs.sum(dim=0) if obs_sum is None else obs_sum + obs.sum(dim=0)
        obs_sq_sum = (obs**2).sum(dim=0) if obs_sq_sum is None else obs_sq_sum + (obs**2).sum(dim=0)
        obs_count += obs.shape[0]

    if obs_count == 0 or action_count == 0:
        raise RuntimeError("No samples for stats")

    obs_mean = obs_sum / obs_count
    obs_std = torch.sqrt(torch.clamp(obs_sq_sum / obs_count - obs_mean**2, min=1e-8))
    obs_std[obs_std < 1e-4] = 1.0

    action_mean = action_sum / action_count
    action_std = torch.sqrt(torch.clamp(action_sq_sum / action_count - action_mean**2, min=1e-8))
    action_std[action_std < 1e-4] = 1.0

    obs_mean = obs_mean.float()
    obs_std = obs_std.float()

    return obs_mean, obs_std, action_mean.float(), action_std.float()


def dueling_uses_layer_norm(state: dict) -> bool:
    """True if the checkpoint has a LayerNorm after each encoder."""
    w = state.get("obs_enc.4.weight")
    return isinstance(w, torch.Tensor) and int(w.ndim) == 1


class DuelingVerifier(torch.nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim, use_layer_norm: bool = False):
        super().__init__()
        obs_dim = obs_shape
        self.use_layer_norm = bool(use_layer_norm)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        obs_layers = [
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
        if self.use_layer_norm:
            obs_layers.append(nn.LayerNorm(hidden_dim))
        self.obs_enc = nn.Sequential(*obs_layers)

        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        action_layers = [
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
        ]
        if self.use_layer_norm:
            action_layers.append(nn.LayerNorm(hidden_dim))
        self.action_enc = nn.Sequential(*action_layers)

        value_layers = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        ]
        if self.use_layer_norm:
            value_layers.append(nn.LayerNorm(hidden_dim // 2))
        value_layers.append(nn.Linear(hidden_dim // 2, 1))
        self.value_stream = nn.Sequential(*value_layers)

        adv_layers = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        ]
        if self.use_layer_norm:
            adv_layers.append(nn.LayerNorm(hidden_dim))
        adv_layers.append(nn.Linear(hidden_dim, 1))
        self.advantage_stream = nn.Sequential(*adv_layers)

    def set_normalization(self, obs_mean, obs_std, action_mean, action_std):
        obs_mean_buf = cast(torch.Tensor, self.obs_mean)
        obs_std_buf = cast(torch.Tensor, self.obs_std)
        action_mean_buf = cast(torch.Tensor, self.action_mean)
        action_std_buf = cast(torch.Tensor, self.action_std)
        obs_mean_buf.data.copy_(obs_mean)
        obs_std_buf.data.copy_(obs_std)
        action_mean_buf.data.copy_(action_mean)
        action_std_buf.data.copy_(action_std)

    def forward(self, obs, action):
        obs_mean_buf = cast(torch.Tensor, self.obs_mean)
        obs_std_buf = cast(torch.Tensor, self.obs_std)
        action_mean_buf = cast(torch.Tensor, self.action_mean)
        action_std_buf = cast(torch.Tensor, self.action_std)

        obs = (obs - obs_mean_buf) / (obs_std_buf + 1e-8)
        action = (action - action_mean_buf) / (action_std_buf + 1e-8)

        obs_feat = self.obs_enc(obs)
        act_feat = self.action_enc(action)

        V = self.value_stream(obs_feat)

        adv_input = torch.cat([obs_feat.detach(), act_feat], dim=-1)
        A = self.advantage_stream(adv_input)

        Q = V + A
        return Q.squeeze(), V.squeeze(), A.squeeze()


def compute_td_lambda_targets(v_next, targets, is_last, lmbda):
    v_next = v_next.detach()
    targets = targets.detach()
    is_last = is_last.detach()
    td0_target = (1.0 - is_last) * v_next + is_last * targets
    return (1.0 - float(lmbda)) * td0_target + float(lmbda) * targets


def _cql_logsumexp_penalty(
    verifier: DuelingVerifier,
    obs_in: torch.Tensor,
    action_in: torch.Tensor,
    Q_data: torch.Tensor,
    n_random_actions: int,
    cql_temp: float,
    *,
    n_policy_actions: int = 0,
    mc_returns: Optional[torch.Tensor] = None,
    use_calql: bool = False,
    return_weighted: bool = False,
) -> torch.Tensor:
    """CQL logsumexp penalty over random and policy-sampled actions (optionally Cal-QL floored)."""
    n_rand = int(n_random_actions)
    n_pi = int(n_policy_actions)
    if n_rand <= 0 and n_pi <= 0:
        return torch.tensor(0.0, device=obs_in.device, dtype=obs_in.dtype)
    B = obs_in.shape[0]
    action_dim_in = action_in.shape[1]
    am = cast(torch.Tensor, verifier.action_mean).detach().to(device=obs_in.device, dtype=obs_in.dtype)
    astd = cast(torch.Tensor, verifier.action_std).detach().to(device=obs_in.device, dtype=obs_in.dtype)
    astd = torch.clamp(astd, min=1e-6)

    q_parts = [Q_data.view(B, 1)]

    def _score_actions(actions: torch.Tensor) -> torch.Tensor:
        n_act = int(actions.shape[1])
        obs_rep = obs_in.unsqueeze(1).expand(-1, n_act, -1).reshape(B * n_act, -1)
        q_val, _, _ = verifier(obs_rep, actions.reshape(B * n_act, -1))
        return q_val.view(B, n_act)

    if n_rand > 0:
        u = torch.empty(
            B, n_rand, action_dim_in, device=obs_in.device, dtype=obs_in.dtype
        ).uniform_(-1.0, 1.0)
        q_parts.append(_score_actions(am + 2.5 * astd * u))

    if n_pi > 0:
        a_pi = am + astd * torch.randn(
            B, n_pi, action_dim_in, device=obs_in.device, dtype=obs_in.dtype
        )
        q_pi = _score_actions(a_pi)
        if use_calql:
            if mc_returns is None:
                raise ValueError("use_calql requires mc_returns (per-step Monte Carlo targets).")
            floor = mc_returns.view(B, 1).to(device=obs_in.device, dtype=obs_in.dtype)
            if return_weighted:
                floor = floor * (floor > 0.0).to(dtype=floor.dtype)
            q_pi = torch.maximum(q_pi, floor)
        q_parts.append(q_pi)

    Q_stack = torch.cat(q_parts, dim=1)
    temp = float(cql_temp)
    if temp <= 0.0:
        raise ValueError("cql_temp must be positive")
    lse = temp * torch.logsumexp(Q_stack / temp, dim=1)
    cql_diff = lse - Q_data.view(-1)
    if return_weighted:
        if mc_returns is None:
            raise ValueError("return_weighted CQL requires mc_returns.")
        weight = mc_returns.view(-1).to(device=cql_diff.device, dtype=cql_diff.dtype).clamp(0.0, 1.0)
        denom = weight.sum().clamp_min(1.0)
        return (cql_diff * weight).sum() / denom
    return cql_diff.mean()


def train_dueling_verifier(
    verifier,
    train_loader,
    val_loader,
    epochs,
    lr,
    save_path,
    lmbda=0.7,
    ref_loader=None,
    anchor_weight=0.1,
    cql_alpha: float = 0.0,
    cql_n_random_actions: int = 10,
    cql_n_policy_actions: int = 0,
    cql_temp: float = 1.0,
    use_calql: bool = False,
    cql_return_weighted: bool = False,
):
    optimizer = torch.optim.Adam(verifier.parameters(), lr=lr, weight_decay=1e-4)
    mse_fn = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verifier.to(device)
    if cql_alpha > 0.0:
        print(
            f"CQL: alpha={cql_alpha:g} n_rand={cql_n_random_actions} n_pi={cql_n_policy_actions} "
            f"calql={use_calql} return_weighted={cql_return_weighted}"
        )
    best_val_score = float("inf")
    save_path_obj = pathlib.Path(save_path)
    latest_path = save_path_obj.with_name(f"{save_path_obj.stem}_latest{save_path_obj.suffix}")

    def _inf_iter(loader):
        while True:
            yield from loader

    ref_iter = _inf_iter(ref_loader) if ref_loader is not None else None

    for epoch in range(epochs):
        verifier.train()
        total_q_loss = 0.0
        total_anchor_loss = 0.0
        total_cql_loss = 0.0
        total_loss = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            obs = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)
            action = batch["action"].to(device)
            targets = batch["target"].to(device)
            is_last = batch["is_last"].to(device)

            obs_in = obs.view(obs.shape[0], -1)
            next_obs_in = next_obs.view(next_obs.shape[0], -1)
            action_in = action.view(action.shape[0], -1)
            targets = targets.view(-1)
            is_last = is_last.view(-1)

            Q, _, A = verifier(obs_in, action_in)
            Q = Q.view(-1)

            with torch.no_grad():
                _, V_next, _ = verifier(next_obs_in, action_in)
                V_next = V_next.view(-1)
                td_lambda_target = compute_td_lambda_targets(V_next, targets, is_last, lmbda)

            loss_q = mse_fn(Q, td_lambda_target)

            loss_anchor = torch.tensor(0.0, device=device)
            if ref_iter is not None:
                ref_batch = next(ref_iter)
                ref_obs = ref_batch["obs"].to(device)
                ref_action = ref_batch["action"].to(device)
                ref_obs_in = ref_obs.view(ref_obs.shape[0], -1)
                ref_action_in = ref_action.view(ref_action.shape[0], -1)
                _, _, A_ref = verifier(ref_obs_in, ref_action_in)
                loss_anchor = anchor_weight * A_ref.mean() ** 2

            loss_cql = torch.tensor(0.0, device=device)
            if cql_alpha > 0.0:
                cql_core = _cql_logsumexp_penalty(
                    verifier,
                    obs_in,
                    action_in,
                    Q,
                    cql_n_random_actions,
                    cql_temp,
                    n_policy_actions=cql_n_policy_actions,
                    mc_returns=targets,
                    use_calql=use_calql,
                    return_weighted=cql_return_weighted,
                )
                loss_cql = float(cql_alpha) * cql_core

            loss = loss_q + loss_anchor + loss_cql

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(verifier.parameters(), max_norm=1.0)
            optimizer.step()

            total_q_loss += loss_q.item()
            total_anchor_loss += loss_anchor.item()
            total_cql_loss += loss_cql.item()
            total_loss += loss.item()
            postfix = {
                "Q_loss": loss_q.item(),
                "anchor": loss_anchor.item(),
                "Total": loss.item(),
            }
            if cql_alpha > 0.0:
                postfix["CQL"] = loss_cql.item()
            pbar.set_postfix(postfix)

        verifier.eval()
        val_q_loss = 0.0
        val_cql_metric = 0.0
        val_cql_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                next_obs = batch["next_obs"].to(device)
                action = batch["action"].to(device)
                targets = batch["target"].to(device)
                is_last = batch["is_last"].to(device)

                obs_in = obs.view(obs.shape[0], -1)
                next_obs_in = next_obs.view(next_obs.shape[0], -1)
                action_in = action.view(action.shape[0], -1)
                targets = targets.view(-1)
                is_last = is_last.view(-1)

                Q, _, A = verifier(obs_in, action_in)
                Q = Q.view(-1)

                _, V_next, _ = verifier(next_obs_in, action_in)
                V_next = V_next.view(-1)
                td_lambda_target = compute_td_lambda_targets(V_next, targets, is_last, lmbda)

                lq = mse_fn(Q, td_lambda_target)
                val_q_loss += lq.item()
                if cql_alpha > 0.0 and (cql_n_random_actions > 0 or cql_n_policy_actions > 0):
                    val_cql_metric += _cql_logsumexp_penalty(
                        verifier,
                        obs_in,
                        action_in,
                        Q,
                        cql_n_random_actions,
                        cql_temp,
                        n_policy_actions=cql_n_policy_actions,
                        mc_returns=targets,
                        use_calql=use_calql,
                        return_weighted=cql_return_weighted,
                    ).item()
                    val_cql_batches += 1

        avg_train_q = total_q_loss / len(train_loader)
        avg_train_anchor = total_anchor_loss / len(train_loader)
        avg_train_cql = total_cql_loss / len(train_loader)
        avg_train_total = total_loss / len(train_loader)
        avg_val_q = val_q_loss / len(val_loader)
        avg_val_cql = val_cql_metric / max(val_cql_batches, 1) if cql_alpha > 0.0 else 0.0
        val_score = avg_val_q + float(cql_alpha) * avg_val_cql if cql_alpha > 0.0 else avg_val_q

        torch.save(verifier.state_dict(), str(latest_path))

        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(verifier.state_dict(), save_path)
            if cql_alpha > 0.0:
                print(
                    f"** New best model saved to {save_path} "
                    f"(val_score=val_Q + α·val_CQLcore={val_score:.4f}; "
                    f"val_Q={avg_val_q:.4f}, val_CQLcore={avg_val_cql:.4f}) **"
                )
            else:
                print(
                    f"** New best model saved to {save_path} with Val Q loss: {avg_val_q:.4f} **"
                )

        if cql_alpha > 0.0:
            print(
                f"Epoch {epoch+1}: "
                f"Train {avg_train_total:.4f} (Q {avg_train_q:.4f}, anchor {avg_train_anchor:.4f}, "
                f"CQL {avg_train_cql:.4f}), Val Q {avg_val_q:.4f}, Val CQL-core {avg_val_cql:.4f}"
            )
        else:
            print(
                f"Epoch {epoch+1}: "
                f"Train {avg_train_total:.4f} (Q {avg_train_q:.4f}, anchor {avg_train_anchor:.4f}), "
                f"Val Q {avg_val_q:.4f} [latest→{latest_path.name}]"
            )


def train_from_cfg(cfg: dict) -> None:
    """Train from a verifier YAML recipe (paths may be relative to data_root)."""
    from elastic.config import resolve_data_path

    dataset = resolve_data_path(cfg["dataset"])
    output = resolve_data_path(cfg["output"])
    ref_dataset = cfg.get("ref_dataset")
    if ref_dataset:
        ref_dataset = resolve_data_path(ref_dataset)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        init_checkpoint = resolve_data_path(init_checkpoint)

    hidden_dim = int(cfg.get("hidden_dim", 512))
    batch_size = int(cfg.get("batch_size", 256))
    epochs = int(cfg.get("epochs", 50))
    lr = float(cfg.get("lr", 1e-4))
    lmbda = float(cfg.get("td_lambda", cfg.get("lmbda", 0.7)))
    anchor_weight = float(cfg.get("anchor_weight", 0.1))
    seed = int(cfg.get("seed", 42))
    stats_max_batches = int(cfg.get("stats_max_batches", 200))
    use_calql = bool(cfg.get("use_calql", False))
    cql_return_weighted = bool(cfg.get("cql_return_weighted", False))
    cql_alpha = float(cfg.get("cql_alpha", 0.0))
    cql_n_random_actions = int(cfg.get("cql_n_random_actions", 10))
    cql_n_policy_actions = int(cfg.get("cql_n_policy_actions", 0))
    cql_temp = float(cfg.get("cql_temp", 1.0))
    if use_calql:
        if cql_alpha <= 0.0:
            cql_alpha = 5.0
            print("use_calql: cql_alpha was 0, setting to 5.0 (V-GPS default).")
        if cql_n_policy_actions <= 0:
            cql_n_policy_actions = max(1, cql_n_random_actions)
            print(f"use_calql: cql_n_policy_actions was 0, setting to {cql_n_policy_actions}.")

    print(f"Loading dataset from {dataset}")
    full_dataset = CleanVerifierDataset(dataset)

    ref_loader_base = None
    if ref_dataset is not None:
        print(f"Loading reference dataset from {ref_dataset}")
        ref_ds = CleanVerifierDataset(ref_dataset)
        ref_loader_base = DataLoader(
            ref_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )
        print(f"Reference dataset: {len(ref_ds)} samples")

    _tmp_loader = DataLoader(full_dataset, batch_size=min(256, len(full_dataset)), shuffle=False)
    _tmp_batch = next(iter(_tmp_loader))
    obs_shape = _tmp_batch["obs"].view(_tmp_batch["obs"].shape[0], -1).shape[1]
    action_dim = _tmp_batch["action"].view(_tmp_batch["action"].shape[0], -1).shape[1]
    print(f"Obs Shape: {obs_shape}, Action Dim: {action_dim}")
    del _tmp_loader, _tmp_batch

    rng = np.random.default_rng(seed)
    all_indices = rng.permutation(len(full_dataset))
    train_count = int(0.8 * len(full_dataset))
    train_dataset = Subset(full_dataset, all_indices[:train_count].tolist())
    val_dataset = Subset(full_dataset, all_indices[train_count:].tolist())
    print(f"Train transitions: {len(train_dataset)}, Val transitions: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    obs_mean, obs_std, action_mean, action_std = _compute_normalization_stats(
        train_loader, max_batches=stats_max_batches
    )

    verifier = DuelingVerifier(
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    verifier.set_normalization(obs_mean, obs_std, action_mean, action_std)
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location="cpu")
        verifier.load_state_dict(state)
        print(f"Initialized verifier from {init_checkpoint}")

    train_dueling_verifier(
        verifier,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        save_path=str(pathlib.Path(output)),
        lmbda=lmbda,
        ref_loader=ref_loader_base,
        anchor_weight=anchor_weight,
        cql_alpha=cql_alpha,
        cql_n_random_actions=cql_n_random_actions,
        cql_n_policy_actions=cql_n_policy_actions,
        cql_temp=cql_temp,
        use_calql=use_calql,
        cql_return_weighted=cql_return_weighted,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Dueling Q/V Verifier from YAML.")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/verifiers/{task}.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Overlay key=value (YAML scalar). Repeatable.",
    )
    cli = parser.parse_args()
    from elastic.config import apply_set, load_yaml

    cfg = apply_set(load_yaml(cli.config), cli.set)
    train_from_cfg(cfg)
