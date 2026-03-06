"""PyTorch-backed actor-critic used for BC and PPO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quake_ai.actions import ACTION_HEADS
from quake_ai.utils.device import configure_torch_runtime, resolve_torch_device

HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "move": 1.5,
    "strafe": 1.0,
    "turn": 1.25,
    "use": 2.0,
}


class _ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, trunk_hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, trunk_hidden),
            nn.Tanh(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.Tanh(),
        )
        self.policy_heads = nn.ModuleDict({head: nn.Linear(trunk_hidden, size) for head, size in ACTION_HEADS.items()})
        self.value_head = nn.Linear(trunk_hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features = self.trunk(obs)
        logits = {head: layer(features) for head, layer in self.policy_heads.items()}
        values = self.value_head(features).squeeze(-1)
        return features, logits, values


class MLPGRUPolicy:
    """Legacy name retained for compatibility with the v0 training code."""

    def __init__(
        self,
        obs_dim: int,
        trunk_hidden: int = 128,
        gru_hidden: int = 0,
        use_gru: bool = False,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        del gru_hidden
        del use_gru

        self.obs_dim = obs_dim
        self.trunk_hidden = trunk_hidden
        self.gru_hidden = 0
        self.use_gru = False
        self.seed = seed
        self.device_spec = resolve_torch_device(device)
        configure_torch_runtime(self.device_spec)
        self.device = self.device_spec.device

        torch.manual_seed(seed)
        self.model = _ActorCriticNet(obs_dim=obs_dim, trunk_hidden=trunk_hidden).to(self.device)
        self.model.train()

        self._optimizers: Dict[str, torch.optim.Optimizer] = {}

    @property
    def w1(self) -> np.ndarray:
        layer = self.model.trunk[0]
        assert isinstance(layer, nn.Linear)
        return layer.weight.detach().cpu().numpy().T.copy()

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def zero_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, 0), dtype=np.float32)

    def _tensor(self, value: np.ndarray | torch.Tensor | Iterable[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _forward_tensors(self, obs: np.ndarray | torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        obs_tensor = self._tensor(obs, dtype=torch.float32)
        return self.model(obs_tensor)

    def _policy_parameters(self) -> list[nn.Parameter]:
        return list(self.model.trunk.parameters()) + list(self.model.policy_heads.parameters())

    def _value_parameters(self) -> list[nn.Parameter]:
        return list(self.model.trunk.parameters()) + list(self.model.value_head.parameters())

    def _optimizer(self, name: str, params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        optimizer = self._optimizers.get(name)
        if optimizer is None:
            optimizer = torch.optim.Adam(list(params), lr=lr)
            self._optimizers[name] = optimizer
        for group in optimizer.param_groups:
            group["lr"] = lr
        return optimizer

    def encode(self, obs: np.ndarray, hidden: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        del hidden
        self.model.eval()
        with torch.no_grad():
            features, _, _ = self._forward_tensors(obs)
        self.model.train()
        return features.detach().cpu().numpy().astype(np.float32), self.zero_hidden(obs.shape[0])

    def forward(
        self,
        obs: np.ndarray,
        hidden: np.ndarray | None = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        del hidden
        self.model.eval()
        with torch.no_grad():
            features, logits_t, values_t = self._forward_tensors(obs)
        self.model.train()
        logits = {head: tensor.detach().cpu().numpy().astype(np.float32) for head, tensor in logits_t.items()}
        values = values_t.detach().cpu().numpy().astype(np.float32)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        return logits, values, self.zero_hidden(obs.shape[0]), features_np

    def log_prob(self, logits: Mapping[str, np.ndarray], actions: Mapping[str, np.ndarray]) -> np.ndarray:
        logp = np.zeros(actions[next(iter(actions))].shape[0], dtype=np.float32)
        for head in ACTION_HEADS:
            probs = self._softmax(logits[head])
            idx = actions[head].astype(np.int64)
            chosen = np.clip(probs[np.arange(len(idx)), idx], 1e-8, 1.0)
            logp += np.log(chosen)
        return logp

    def greedy_actions(self, logits: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {head: np.argmax(logits[head], axis=1).astype(np.int64) for head in ACTION_HEADS}

    def sample_actions(
        self,
        logits: Mapping[str, np.ndarray],
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
        actions: Dict[str, np.ndarray] = {}
        total_logp = None
        total_entropy = 0.0

        for head in ACTION_HEADS:
            probs = self._softmax(logits[head])
            batch = probs.shape[0]
            samples = np.array([rng.choice(probs.shape[1], p=probs[i]) for i in range(batch)], dtype=np.int64)
            actions[head] = samples
            chosen = np.clip(probs[np.arange(batch), samples], 1e-8, 1.0)
            logp = np.log(chosen)
            total_logp = logp if total_logp is None else total_logp + logp
            entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)
            total_entropy += float(np.mean(entropy))

        assert total_logp is not None
        return actions, total_logp.astype(np.float32), total_entropy / len(ACTION_HEADS)

    def supervised_step(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor],
        lr: float,
    ) -> Dict[str, float]:
        self.model.train()
        optimizer = self._optimizer("bc", self.model.parameters(), lr)
        optimizer.zero_grad()

        _, logits, _ = self._forward_tensors(obs)

        losses = []
        accuracies = []
        for head in ACTION_HEADS:
            target = self._tensor(actions[head], dtype=torch.long)
            weights = self._tensor(class_weights[head], dtype=torch.float32)
            head_loss = F.cross_entropy(logits[head], target, weight=weights, reduction="mean")
            head_loss = head_loss * HEAD_LOSS_WEIGHTS.get(head, 1.0)
            losses.append(head_loss)
            pred = torch.argmax(logits[head], dim=1)
            accuracies.append(float((pred == target).float().mean().item()))

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "accuracy": float(np.mean(accuracies) if accuracies else 0.0),
        }

    def evaluate_supervised(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            _, logits, _ = self._forward_tensors(obs)
            losses = []
            accuracies = []
            for head in ACTION_HEADS:
                target = self._tensor(actions[head], dtype=torch.long)
                head_loss = F.cross_entropy(logits[head], target, reduction="mean")
                losses.append(float(head_loss.item()))
                pred = torch.argmax(logits[head], dim=1)
                accuracies.append(float((pred == target).float().mean().item()))
        self.model.train()
        return {
            "loss": float(np.mean(losses) if losses else 0.0),
            "accuracy": float(np.mean(accuracies) if accuracies else 0.0),
        }

    def ppo_policy_step(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        old_log_probs: np.ndarray | torch.Tensor,
        advantages: np.ndarray | torch.Tensor,
        clip_ratio: float,
        lr: float,
    ) -> Dict[str, float]:
        self.model.train()
        optimizer = self._optimizer("ppo_policy", self._policy_parameters(), lr)
        optimizer.zero_grad()

        _, logits, _ = self._forward_tensors(obs)
        old_log_probs_t = self._tensor(old_log_probs, dtype=torch.float32)
        advantages_t = self._tensor(advantages, dtype=torch.float32)

        log_probs = []
        for head in ACTION_HEADS:
            target = self._tensor(actions[head], dtype=torch.long)
            log_probs.append(torch.distributions.Categorical(logits=logits[head]).log_prob(target))
        new_log_probs = torch.stack(log_probs, dim=0).sum(dim=0)

        ratios = torch.exp(new_log_probs - old_log_probs_t)
        clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surrogate = torch.minimum(ratios * advantages_t, clipped * advantages_t)
        loss = -torch.mean(surrogate)
        loss.backward()
        optimizer.step()

        approx_kl = torch.mean(old_log_probs_t - new_log_probs).item()
        clip_fraction = torch.mean(((ratios > 1.0 + clip_ratio) | (ratios < 1.0 - clip_ratio)).float()).item()
        return {
            "policy_loss": float(loss.item()),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
        }

    def value_step(self, obs: np.ndarray | torch.Tensor, returns: np.ndarray | torch.Tensor, lr: float) -> Dict[str, float]:
        self.model.train()
        optimizer = self._optimizer("ppo_value", self._value_parameters(), lr)
        optimizer.zero_grad()

        _, _, values = self._forward_tensors(obs)
        returns_t = self._tensor(returns, dtype=torch.float32)
        loss = F.mse_loss(values, returns_t)
        loss.backward()
        optimizer.step()
        return {"value_loss": float(loss.item())}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "obs_dim": self.obs_dim,
            "trunk_hidden": self.trunk_hidden,
            "gru_hidden": self.gru_hidden,
            "use_gru": self.use_gru,
            "model_version": 3,
            "backend": "pytorch",
            "requested_device": self.device_spec.requested,
            "resolved_device": self.device_spec.resolved,
            "accelerator_backend": self.device_spec.backend,
        }
        payload = {
            "meta": meta,
            "state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
        }
        torch.save(payload, target)
        target.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def _load_legacy_npz(cls, source: Path, device: str = "auto") -> "MLPGRUPolicy":
        meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        model = cls(
            obs_dim=int(meta["obs_dim"]),
            trunk_hidden=int(meta["trunk_hidden"]),
            seed=0,
            device=device,
        )
        payload = np.load(source)

        first = model.model.trunk[0]
        second = model.model.trunk[2]
        assert isinstance(first, nn.Linear)
        assert isinstance(second, nn.Linear)
        first.weight.data.copy_(torch.from_numpy(payload["w1"].T))
        first.bias.data.copy_(torch.from_numpy(payload["b1"]))
        second.weight.data.copy_(torch.from_numpy(payload["w2"].T))
        second.bias.data.copy_(torch.from_numpy(payload["b2"]))

        model.model.value_head.weight.data.copy_(torch.from_numpy(payload["value_w"][None, :]))
        model.model.value_head.bias.data.fill_(float(payload["value_b"][0]))

        for head in ACTION_HEADS:
            head_layer = model.model.policy_heads[head]
            assert isinstance(head_layer, nn.Linear)
            head_layer.weight.data.copy_(torch.from_numpy(payload[f"head_w_{head}"].T))
            head_layer.bias.data.copy_(torch.from_numpy(payload[f"head_b_{head}"]))

        return model

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "MLPGRUPolicy":
        source = Path(path)
        try:
            payload = torch.load(source, map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload and "meta" in payload:
                meta = dict(payload["meta"])
                model = cls(
                    obs_dim=int(meta["obs_dim"]),
                    trunk_hidden=int(meta["trunk_hidden"]),
                    seed=0,
                    device=device,
                )
                model.model.load_state_dict(payload["state_dict"])
                model.model.to(model.device)
                return model
        except Exception:
            pass
        return cls._load_legacy_npz(source, device=device)
