"""PyTorch-backed actor-critic used for BC and PPO."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qnn.actions import (
    ACTION_HEADS,
    CONTINUOUS_ACTION_HEADS,
    DISCRETE_ACTION_HEADS,
)
from qnn.model.transformer import TransformerTrunk
from qnn.utils.device import configure_torch_runtime, resolve_torch_device
from qnn.utils.io import trusted_torch_load

HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "move": 1.0,
    "look": 1.0,
    "fire": 1.0,
    "switch": 0.0,
    "recall_0": 0.0,
    "recall_1": 0.0,
    "recall_2": 0.0,
    "recall_3": 0.0,
}
_CONTINUOUS_HEAD_STD_INIT = -1.0

# Sparse binary heads: skip true-negative ticks in BC loss.
# Only train on ticks where the demonstrator acted or the model predicted action.
_SPARSE_BINARY_HEADS = frozenset({"fire", "switch"})


def _continuous_mean(head: str, logits: torch.Tensor, *, look_cosine: bool = False) -> torch.Tensor:
    """Map raw logits to continuous action prediction.

    When *look_cosine* is True, the look head is L2-normalized to the unit
    sphere (preserves linear obs→action path).  When False, tanh is used
    for all continuous heads (legacy behaviour).
    Move head always uses tanh squash to [-1, 1].
    """
    if head == "look" and look_cosine:
        return F.normalize(logits, dim=-1)
    return torch.tanh(logits)


def _look_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    turn_alpha: float = 0.0,
) -> torch.Tensor:
    """Cosine loss for unit-vector look targets with optional turn magnitude weighting."""
    cos_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1)
    if turn_alpha > 0:
        turn_mag = torch.sqrt(target[:, 1] ** 2 + target[:, 2] ** 2)
        weights = 1.0 + turn_alpha * turn_mag
        return (cos_loss * weights).mean()
    return cos_loss.mean()


def _focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy with optional focal modulation.

    When *gamma* is 0 this is equivalent to ``F.cross_entropy``.
    When *gamma* > 0 each sample's loss is scaled by ``(1 - p_t)^gamma``,
    down-weighting easy/confident examples so rare-class samples dominate.
    """
    if gamma <= 0.0:
        return F.cross_entropy(logits, target, weight=weight, reduction="mean")
    # Per-sample CE (unreduced) with optional class weights.
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    p_t = torch.exp(-ce)  # p_t = softmax probability of the true class
    focal = ((1.0 - p_t) ** gamma) * ce
    return focal.mean()


@dataclass(slots=True)
class PolicyActionBatch:
    actions: Dict[str, np.ndarray]
    log_probs: torch.Tensor
    values: torch.Tensor
    entropies: Dict[str, torch.Tensor]
    next_hidden: torch.Tensor


def _normal_from_mean_log_std(mean: torch.Tensor, log_std: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Independent:
    safe_temperature = max(float(temperature), 1e-3)
    std = torch.exp(log_std).unsqueeze(0).expand_as(mean) * safe_temperature
    return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)


def _continuous_head_metrics(
    head: str,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute L1 metrics for continuous heads. Returns 0-d GPU tensors — caller
    syncs once at report/epoch-end, not per step."""
    l1 = torch.abs(pred.detach() - target.detach())
    n = l1.shape[0]
    device = l1.device
    n_t = torch.tensor(float(n), device=device)

    if head == "move":
        s0, s1, s2, st = l1[:, 0].sum(), l1[:, 1].sum(), l1[:, 2].sum(), l1.sum()
        return {
            "n_move": n_t,
            "l1_sum_move_forward": s0, "l1_sum_move_strafe": s1,
            "l1_sum_move_up": s2, "l1_sum_move": st,
            "mae_move_forward": s0 / n_t, "mae_move_strafe": s1 / n_t,
            "mae_move_up": s2 / n_t, "mae_move": st / n_t,
        }

    if head == "look":
        p = pred.detach()
        t = target.detach()
        s0, s1, s2, st = l1[:, 0].sum(), l1[:, 1].sum(), l1[:, 2].sum(), l1.sum()
        cos_sim = F.cosine_similarity(p, t, dim=-1).clamp(-1.0, 1.0)
        angle_deg = torch.acos(cos_sim) * (180.0 / torch.pi)
        turn_mag = torch.sqrt(t[:, 1] ** 2 + t[:, 2] ** 2).clamp(0.0, 1.0)
        target_turn_deg = torch.asin(turn_mag) * (180.0 / torch.pi)
        result: Dict[str, torch.Tensor] = {
            "n_look": n_t,
            "l1_sum_look_x": s0, "l1_sum_look_y": s1, "l1_sum_look_z": s2, "l1_sum_look": st,
            "mae_look_x": s0 / n_t, "mae_look_y": s1 / n_t, "mae_look_z": s2 / n_t, "mae_look": st / n_t,
            "mae_look_angle_deg": angle_deg.mean(),
        }
        _bins = [("0_1", 0.0, 1.0), ("1_5", 1.0, 5.0), ("5_15", 5.0, 15.0), ("15p", 15.0, 180.0)]
        for tag, lo, hi in _bins:
            mask = (target_turn_deg >= lo) & (target_turn_deg < hi)
            mask_f = mask.to(angle_deg.dtype)
            cnt = mask_f.sum()
            angle_sum = (angle_deg * mask_f).sum()
            safe_cnt = torch.clamp(cnt, min=1.0)
            result[f"n_look_{tag}"] = cnt
            result[f"mae_look_{tag}_deg"] = angle_sum / safe_cnt
        return result

    raise ValueError(f"Unsupported continuous head for metrics: {head}")


class _ActorCriticNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        trunk_hidden: int,
        gru_hidden: int,
        use_gru: bool,
        d_model: int | None = None,
        n_heads: int = 1,
        n_layers: int = 2,
        ffn_dim: int = 256,
        attn_dropout: float = 0.0,
        readout: str = "self",
        action_history_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = int(d_model if d_model is not None else trunk_hidden)
        self.trunk_hidden = self.d_model
        self.use_gru = bool(use_gru and gru_hidden > 0)
        self.gru_hidden = int(gru_hidden if self.use_gru else 0)
        self.readout = str(readout) if readout else "cls"
        self.trunk = TransformerTrunk(
            obs_dim=obs_dim,
            d_model=self.d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=attn_dropout,
            readout=self.readout,
            action_history_tokens=action_history_tokens,
        )
        if self.use_gru:
            self.gru = nn.GRU(input_size=self.d_model, hidden_size=self.gru_hidden, num_layers=1)
            head_in = self.d_model + self.gru_hidden
        else:
            self.gru = None
            head_in = self.d_model
        self.head_hidden = head_in
        self.policy_heads = nn.ModuleDict({head: nn.Linear(head_in, size) for head, size in ACTION_HEADS.items()})
        self.continuous_log_std = nn.ParameterDict(
            {
                head: nn.Parameter(torch.full((ACTION_HEADS[head],), _CONTINUOUS_HEAD_STD_INIT))
                for head in CONTINUOUS_ACTION_HEADS
            }
        )
        self.value_head = nn.Linear(head_in, 1)

    def _run_gru(
        self,
        trunk_features: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gru is None:
            raise RuntimeError("GRU requested for a feed-forward model")
        current_hidden = hidden.unsqueeze(0)
        seq_len = int(trunk_features.shape[0])
        batch_size = int(trunk_features.shape[1])
        if masks is None:
            outputs, next_hidden = self.gru(trunk_features, current_hidden)
            return outputs, next_hidden.squeeze(0)

        if masks.ndim != 2 or tuple(masks.shape) != (seq_len, batch_size):
            raise ValueError(f"Expected masks with shape ({seq_len}, {batch_size})")

        mask_values = masks.to(dtype=trunk_features.dtype)
        reset_rows = torch.any(mask_values != 1, dim=1)
        if not bool(reset_rows.any()):
            outputs, next_hidden = self.gru(trunk_features, current_hidden)
            return outputs, next_hidden.squeeze(0)

        outputs = []
        reset_indices = reset_rows.nonzero(as_tuple=False).flatten().tolist()
        start = 0
        for idx, reset_t in enumerate(reset_indices):
            if reset_t > start:
                span_out, current_hidden = self.gru(trunk_features[start:reset_t], current_hidden)
                outputs.append(span_out)

            next_reset = reset_indices[idx + 1] if idx + 1 < len(reset_indices) else seq_len
            current_hidden = current_hidden * mask_values[reset_t].view(1, batch_size, 1)
            span_out, current_hidden = self.gru(trunk_features[reset_t:next_reset], current_hidden)
            outputs.append(span_out)
            start = next_reset

        return torch.cat(outputs, dim=0), current_hidden.squeeze(0)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        hidden: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        sample = obs["self_scalars"]
        input_is_sequence = sample.ndim == 3
        if input_is_sequence:
            seq_len = int(sample.shape[0])
            batch_size = int(sample.shape[1])
            flat_obs = {
                key: value.reshape(seq_len * batch_size, *value.shape[2:])
                for key, value in obs.items()
            }
            trunk_features = self.trunk(flat_obs).reshape(seq_len, batch_size, self.d_model)
        else:
            batch_size = int(sample.shape[0])
            trunk_features = self.trunk(obs)

        if self.use_gru:
            hidden_t = hidden if hidden is not None else torch.zeros(
                (batch_size, self.gru_hidden),
                dtype=sample.dtype,
                device=sample.device,
            )
            if input_is_sequence:
                gru_features, next_hidden = self._run_gru(trunk_features, hidden_t, masks=masks)
                features = torch.cat([trunk_features, gru_features], dim=-1)
            else:
                seq_masks = None
                if masks is not None:
                    seq_masks = masks.reshape(1, batch_size)
                seq_features, next_hidden = self._run_gru(trunk_features.unsqueeze(0), hidden_t, masks=seq_masks)
                features = torch.cat([trunk_features, seq_features.squeeze(0)], dim=-1)
        else:
            features = trunk_features
            next_hidden = torch.zeros((batch_size, 0), dtype=sample.dtype, device=sample.device)

        if input_is_sequence:
            flat_features = features.reshape(seq_len * batch_size, -1)
            logits = {
                head: layer(flat_features).reshape(seq_len, batch_size, size)
                for head, (layer, size) in zip(
                    ACTION_HEADS,
                    ((self.policy_heads[head], ACTION_HEADS[head]) for head in ACTION_HEADS),
                )
            }
            values = self.value_head(flat_features).reshape(seq_len, batch_size)
        else:
            logits = {head: layer(features) for head, layer in self.policy_heads.items()}
            values = self.value_head(features).squeeze(-1)
        return features, logits, values, next_hidden


class QNNPolicy:
    """Actor-critic policy: transformer encoder + GRU temporal core."""

    def __init__(
        self,
        obs_dim: int,
        trunk_hidden: int = 64,
        gru_hidden: int = 64,
        use_gru: bool = True,
        seed: int = 0,
        device: str = "auto",
        d_model: int | None = None,
        n_heads: int = 1,
        n_layers: int = 2,
        ffn_dim: int = 256,
        attn_dropout: float = 0.0,
        readout: str = "self",
        action_history_tokens: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.look_cosine = False
        self.d_model = int(d_model if d_model is not None else trunk_hidden)
        self.trunk_hidden = self.d_model
        self.use_gru = bool(use_gru and gru_hidden > 0)
        self.gru_hidden = int(gru_hidden if self.use_gru else 0)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ffn_dim = int(ffn_dim)
        self.attn_dropout = float(attn_dropout)
        self.readout = str(readout) if readout else "cls"
        self.action_history_tokens = int(action_history_tokens)
        self.head_hidden = self.d_model + self.gru_hidden if self.use_gru else self.d_model
        self.seed = seed
        self.device_spec = resolve_torch_device(device)
        configure_torch_runtime(self.device_spec)
        self.device = self.device_spec.device
        self._rocm_inference_pad_batch = 0
        if self.device_spec.backend == "rocm" and not self.use_gru:
            raw_pad_batch = os.environ.get("QNN_ROCM_INFERENCE_PAD_BATCH", "32").strip()
            try:
                self._rocm_inference_pad_batch = max(int(raw_pad_batch), 0)
            except ValueError:
                self._rocm_inference_pad_batch = 32

        torch.manual_seed(seed)
        self.model = _ActorCriticNet(
            obs_dim=obs_dim,
            trunk_hidden=self.trunk_hidden,
            gru_hidden=self.gru_hidden,
            use_gru=self.use_gru,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ffn_dim=self.ffn_dim,
            attn_dropout=self.attn_dropout,
            readout=self.readout,
            action_history_tokens=self.action_history_tokens,
        ).to(self.device)
        self.model.train()
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}

    @staticmethod
    def _sample_from_probs(
        probs: torch.Tensor,
        generator: torch.Generator | None = None,
        row_generators: Sequence[torch.Generator | None] | None = None,
    ) -> torch.Tensor:
        if row_generators is not None:
            if len(row_generators) != probs.shape[0]:
                raise ValueError("row_generators must match the batch size")
            samples = []
            for row_idx, row_generator in enumerate(row_generators):
                row_probs = probs[row_idx]
                if row_generator is not None and row_generator.device.type != row_probs.device.type:
                    sample = torch.multinomial(row_probs.detach().cpu(), 1, replacement=True, generator=row_generator)
                    samples.append(sample.to(device=probs.device, dtype=torch.long).squeeze(0))
                else:
                    samples.append(torch.multinomial(row_probs, 1, replacement=True, generator=row_generator).squeeze(0))
            return torch.stack(samples, dim=0)

        if generator is not None and generator.device.type != probs.device.type:
            return (
                torch.multinomial(probs.detach().cpu(), 1, replacement=True, generator=generator)
                .squeeze(1)
                .to(device=probs.device, dtype=torch.long)
            )

        return torch.multinomial(probs, 1, replacement=True, generator=generator).squeeze(1)

    @staticmethod
    def _sample_normal(
        mean: torch.Tensor,
        std: torch.Tensor,
        generator: torch.Generator | None = None,
        row_generators: Sequence[torch.Generator | None] | None = None,
    ) -> torch.Tensor:
        if row_generators is not None:
            if len(row_generators) != mean.shape[0]:
                raise ValueError("row_generators must match the batch size")
            samples = []
            for row_idx, row_generator in enumerate(row_generators):
                row_mean = mean[row_idx]
                row_std = std[row_idx]
                noise = torch.randn(
                    row_mean.shape,
                    generator=row_generator,
                    device=row_mean.device if row_generator is None or row_generator.device.type == row_mean.device.type else "cpu",
                    dtype=row_mean.dtype,
                )
                if noise.device != row_mean.device:
                    noise = noise.to(device=row_mean.device, dtype=row_mean.dtype)
                samples.append(row_mean + (row_std * noise))
            return torch.stack(samples, dim=0)

        if generator is not None and generator.device.type != mean.device.type:
            noise = torch.randn(mean.shape, generator=generator, device="cpu", dtype=mean.dtype).to(device=mean.device)
        else:
            noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return mean + (std * noise)

    @staticmethod
    def _log_prob_entropy_from_logits(logits: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=1)
        return chosen_log_probs, entropy, probs

    def zero_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.gru_hidden), dtype=np.float32)

    def _tensor(self, value: np.ndarray | torch.Tensor | Iterable[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device == self.device and value.dtype == dtype:
                return value
            non_blocking = value.device.type == "cpu" and (
                not isinstance(self.device, torch.device) or self.device.type != "cpu"
            )
            return value.to(device=self.device, dtype=dtype, non_blocking=non_blocking)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _hidden_tensor(self, hidden: np.ndarray | torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if not self.use_gru:
            return torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)
        if hidden is None:
            return torch.zeros((batch_size, self.gru_hidden), dtype=torch.float32, device=self.device)
        hidden_t = self._tensor(hidden, dtype=torch.float32)
        if hidden_t.ndim == 3 and int(hidden_t.shape[0]) == 1:
            hidden_t = hidden_t.squeeze(0)
        if hidden_t.ndim != 2 or int(hidden_t.shape[0]) != batch_size or int(hidden_t.shape[1]) != self.gru_hidden:
            raise ValueError(f"Expected hidden state with shape ({batch_size}, {self.gru_hidden})")
        return hidden_t

    def _maybe_pad_obs_batch(
        self,
        obs_dict: Dict[str, torch.Tensor],
        hidden_tensor: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        sample = obs_dict["self_scalars"]
        if self.use_gru or sample.ndim != 2:
            return obs_dict, hidden_tensor, 0
        if self._rocm_inference_pad_batch <= 0:
            return obs_dict, hidden_tensor, 0

        batch_size = int(sample.shape[0])
        target_batch = self._rocm_inference_pad_batch
        if batch_size == 0 or batch_size >= target_batch:
            return obs_dict, hidden_tensor, 0

        pad_rows = target_batch - batch_size
        padded_obs: Dict[str, torch.Tensor] = {}
        for key, value in obs_dict.items():
            pad_shape = (pad_rows, *value.shape[1:])
            pad_value = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            padded_obs[key] = torch.cat([value, pad_value], dim=0)
        if hidden_tensor.numel() == 0:
            return padded_obs, hidden_tensor, pad_rows
        padded_hidden = F.pad(hidden_tensor, (0, 0, 0, pad_rows))
        return padded_obs, padded_hidden, pad_rows

    @staticmethod
    def _sampling_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        safe_temperature = max(float(temperature), 1e-3)
        if abs(safe_temperature - 1.0) < 1e-6:
            return logits
        return logits / safe_temperature

    @staticmethod
    def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3:
            return logits.reshape(-1, logits.shape[-1])
        return logits

    @staticmethod
    def _flatten_targets(target: torch.Tensor) -> torch.Tensor:
        if target.ndim > 1:
            return target.reshape(-1)
        return target

    def _action_targets_for_head(
        self,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        head: str,
        head_logits: torch.Tensor,
    ) -> torch.Tensor:
        source = actions.get(head)
        if source is None:
            return torch.zeros(
                (self._flatten_logits(head_logits).shape[0],),
                dtype=torch.long,
                device=self.device,
            )
        return self._flatten_targets(self._tensor(source, dtype=torch.long))

    def _continuous_targets_for_head(
        self,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        head: str,
        head_logits: torch.Tensor,
    ) -> torch.Tensor:
        source = actions.get(head)
        flat_shape = self._flatten_logits(head_logits).shape
        if source is None:
            return torch.zeros(flat_shape, dtype=torch.float32, device=self.device)
        target = self._tensor(source, dtype=torch.float32)
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[-1])
        elif target.ndim == 2:
            target = target.reshape(-1, target.shape[-1])
        else:
            raise ValueError(f"Continuous head {head} expects rank-2 or rank-3 targets")
        if target.shape != flat_shape:
            raise ValueError(f"Expected continuous target shape {flat_shape} for {head}, got {target.shape}")
        return target

    def _autocast(self):
        """Mixed-precision context. Controlled by env var QNN_AUTOCAST_DTYPE
        (one of: fp32, bf16, fp16). Defaults to fp32 (no autocast)."""
        dtype_name = os.environ.get("QNN_AUTOCAST_DTYPE", "fp32").lower()
        if dtype_name == "fp32" or self.device.type != "cuda":
            return torch.amp.autocast(device_type=self.device.type, enabled=False)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_name)
        if dtype is None:
            return torch.amp.autocast(device_type=self.device.type, enabled=False)
        return torch.amp.autocast(device_type=self.device.type, dtype=dtype, enabled=True)

    def _class_weights_for_head(
        self,
        class_weights: Mapping[str, np.ndarray | torch.Tensor],
        head: str,
        size: int,
    ) -> torch.Tensor:
        source = class_weights.get(head)
        if source is None:
            return torch.ones((size,), dtype=torch.float32, device=self.device)
        # Cache: source ndarrays are built once per run and passed unchanged
        # every step. Keep a per-instance id→tensor map so we skip the
        # numpy→GPU transfer on the hot path.
        cache = getattr(self, "_class_weights_cache", None)
        if cache is None:
            cache = {}
            self._class_weights_cache = cache
        key = (head, id(source))
        cached = cache.get(key)
        if cached is not None:
            return cached
        tensor = self._tensor(source, dtype=torch.float32)
        cache[key] = tensor
        return tensor

    def _forward_tensors(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if not isinstance(obs, dict):
            raise ValueError("Token policy expects dict observations")
        obs_tensors: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            dtype = torch.float32
            if key.endswith("_id") or key.endswith("_ids"):
                dtype = torch.long
            elif key.endswith("_mask"):
                dtype = torch.bool
            obs_tensors[key] = self._tensor(value, dtype=dtype)

        sample = obs_tensors["self_scalars"]
        if sample.ndim == 2:
            batch_size = int(sample.shape[0])
            hidden_t = self._hidden_tensor(hidden, batch_size)
            padded_obs, padded_hidden, pad_rows = self._maybe_pad_obs_batch(obs_tensors, hidden_t)
            features, logits, values, next_hidden = self.model(padded_obs, hidden=padded_hidden)
            if pad_rows == 0:
                return features, logits, values, next_hidden

            return (
                features[:batch_size],
                {head: tensor[:batch_size] for head, tensor in logits.items()},
                values[:batch_size],
                next_hidden[:batch_size],
            )

        if sample.ndim != 3:
            raise ValueError("obs must be rank-2 or rank-3")
        seq_len, batch_size, _ = sample.shape
        hidden_t = self._hidden_tensor(hidden, int(batch_size))
        mask_t = None
        if masks is not None:
            mask_t = self._tensor(masks, dtype=torch.float32).reshape(seq_len, batch_size)
        return self.model(obs_tensors, hidden=hidden_t, masks=mask_t)

    def _policy_parameters(self) -> list[nn.Parameter]:
        params = list(self.model.trunk.parameters())
        if self.model.gru is not None:
            params.extend(self.model.gru.parameters())
        params.extend(self.model.policy_heads.parameters())
        params.extend(self.model.continuous_log_std.parameters())
        return params

    def _value_parameters(self) -> list[nn.Parameter]:
        return list(self.model.value_head.parameters())

    def bc_zero_grad(self) -> None:
        """Zero gradients on the BC optimizer (for gradient accumulation)."""
        opt = self._optimizers.get("bc")
        if opt is not None:
            opt.zero_grad()

    def bc_step(self) -> None:
        """Step the BC optimizer (after accumulating gradients)."""
        opt = self._optimizers.get("bc")
        if opt is not None:
            opt.step()

    def _optimizer(self, name: str, params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        optimizer = self._optimizers.get(name)
        if optimizer is None:
            optimizer = torch.optim.Adam(list(params), lr=lr, fused=True)
            self._optimizers[name] = optimizer
        for group in optimizer.param_groups:
            group["lr"] = lr
        return optimizer

    def _ppo_optimizer(self, policy_lr: float, value_lr: float) -> torch.optim.Optimizer:
        optimizer = self._optimizers.get("ppo")
        shared_params = list(self.model.trunk.parameters())
        if self.model.gru is not None:
            shared_params.extend(self.model.gru.parameters())
        if optimizer is None:
            optimizer = torch.optim.Adam(
                [
                    {"params": shared_params, "lr": policy_lr},
                    {
                        "params": list(self.model.policy_heads.parameters()) + list(self.model.continuous_log_std.parameters()),
                        "lr": policy_lr,
                    },
                    {"params": list(self.model.value_head.parameters()), "lr": value_lr},
                ]
            )
            self._optimizers["ppo"] = optimizer

        lrs = (policy_lr, policy_lr, value_lr)
        if len(optimizer.param_groups) != len(lrs):
            raise RuntimeError("Unexpected PPO optimizer parameter groups")
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr
        return optimizer

    def encode(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        hidden: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            features, _, _, next_hidden = self._forward_tensors(obs, hidden=hidden)
        return (
            features.detach().cpu().numpy().astype(np.float32),
            next_hidden.detach().cpu().numpy().astype(np.float32),
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        hidden: np.ndarray | None = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode():
            features, logits_t, values_t, next_hidden = self._forward_tensors(obs, hidden=hidden)
        logits = {head: tensor.detach().cpu().numpy().astype(np.float32) for head, tensor in logits_t.items()}
        values = values_t.detach().cpu().numpy().astype(np.float32)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        return logits, values, next_hidden.detach().cpu().numpy().astype(np.float32), features_np

    def act(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        *,
        mode: str,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: np.ndarray | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        row_generators: Sequence[torch.Generator | None] | None = None,
        sample_temperatures: Mapping[str, float] | None = None,
    ) -> PolicyActionBatch:
        if generator is not None and row_generators is not None:
            raise ValueError("Specify either generator or row_generators, not both")

        with torch.inference_mode():
            _, logits, values, next_hidden = self._forward_tensors(obs, hidden=hidden, masks=masks)
            action_tensors: Dict[str, torch.Tensor] = {}
            log_probs = []
            entropies: Dict[str, torch.Tensor] = {}

            for head in ACTION_HEADS:
                head_logits = logits[head]
                if head in CONTINUOUS_ACTION_HEADS:
                    head_mean = _continuous_mean(head, head_logits, look_cosine=self.look_cosine)
                    temperature = float(sample_temperatures.get(head, 1.0)) if sample_temperatures else 1.0
                    dist = _normal_from_mean_log_std(head_mean, self.model.continuous_log_std[head], temperature=temperature)
                    if mode == "greedy":
                        action_tensor = head_mean
                    elif mode == "sampled":
                        std = torch.exp(self.model.continuous_log_std[head]).unsqueeze(0).expand_as(head_mean) * max(temperature, 1e-3)
                        action_tensor = self._sample_normal(
                            head_mean,
                            std,
                            generator=generator,
                            row_generators=row_generators,
                        )
                        action_tensor = torch.clamp(action_tensor, min=-1.0, max=1.0)
                    else:
                        raise ValueError(f"Unsupported policy mode {mode}")
                    action_tensors[head] = action_tensor
                    log_probs.append(dist.log_prob(action_tensor))
                    entropies[head] = dist.entropy()
                    continue

                if mode == "sampled":
                    head_logits = self._sampling_logits(head_logits, float(sample_temperatures.get(head, 1.0)) if sample_temperatures else 1.0)
                if mode == "greedy":
                    action_tensor = torch.argmax(head_logits, dim=1)
                elif mode == "sampled":
                    head_probs = torch.softmax(head_logits, dim=1)
                    action_tensor = self._sample_from_probs(
                        head_probs,
                        generator=generator,
                        row_generators=row_generators,
                    )
                else:
                    raise ValueError(f"Unsupported policy mode {mode}")

                action_tensor = action_tensor.to(dtype=torch.long)
                action_tensors[head] = action_tensor
                head_log_probs, head_entropy, _ = self._log_prob_entropy_from_logits(head_logits, action_tensor)
                log_probs.append(head_log_probs)
                entropies[head] = head_entropy

        return PolicyActionBatch(
            actions={
                head: (
                    tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                    if head in CONTINUOUS_ACTION_HEADS
                    else tensor.detach().cpu().numpy().astype(np.int64, copy=False)
                )
                for head, tensor in action_tensors.items()
            },
            log_probs=torch.stack(log_probs, dim=0).sum(dim=0),
            values=values,
            entropies=entropies,
            next_hidden=next_hidden,
        )

    def _compute_head_losses_and_metrics(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        head_loss_weights: Mapping[str, float] | None = None,
        focal_gamma: float = 0.0,
        sparse_discrete: bool = True,
        look_deadzone: float = 0.0,
        look_turn_alpha: float = 0.0,
        compute_metrics: bool = True,
    ) -> tuple[list[torch.Tensor], list[bool], Dict[str, torch.Tensor | int | float]]:
        """Shared per-head loss + metrics for both train and eval.

        Returns (losses, loss_is_real_flags, metrics_dict).  Metrics values
        stay as 0-d GPU tensors — caller accumulates on device and syncs only
        at report/epoch boundaries.  loss_is_real_flags[i] is False for the
        sparse-binary placeholder (no positive samples), True otherwise.  When
        compute_metrics is False, metrics_dict only contains "loss" and
        "accuracy" (required by callers); all MAE/stat computation is skipped.
        """
        weights_map = head_loss_weights or HEAD_LOSS_WEIGHTS
        losses: list[torch.Tensor] = []
        loss_is_real: list[bool] = []
        accuracy_components: list[torch.Tensor] = []
        metrics: Dict[str, torch.Tensor | int | float] = {}

        for head in ACTION_HEADS:
            head_logits = self._flatten_logits(logits[head])
            is_real = True
            if head in CONTINUOUS_ACTION_HEADS:
                pred = _continuous_mean(head, head_logits, look_cosine=self.look_cosine)
                target = self._continuous_targets_for_head(actions, head, head_logits)
                if head == "look" and self.look_cosine:
                    if look_deadzone > 0:
                        raw_target = target
                        turn_mag = torch.sqrt(target[:, 1] ** 2 + target[:, 2] ** 2)
                        mask = turn_mag < look_deadzone
                        target = target.clone()
                        target[mask, 0] = 1.0
                        target[mask, 1] = 0.0
                        target[mask, 2] = 0.0
                        head_loss = _look_cosine_loss(pred, target, look_turn_alpha)
                        if compute_metrics:
                            metrics.update(_continuous_head_metrics(head, pred, raw_target))
                    else:
                        head_loss = _look_cosine_loss(pred, target, look_turn_alpha)
                        if compute_metrics:
                            metrics.update(_continuous_head_metrics(head, pred, target))
                else:
                    head_loss = F.smooth_l1_loss(pred, target)
                    if compute_metrics:
                        metrics.update(_continuous_head_metrics(head, pred, target))
            else:
                target = self._action_targets_for_head(actions, head, head_logits)
                if sparse_discrete and head in _SPARSE_BINARY_HEADS:
                    pred = torch.argmax(head_logits, dim=1)
                    sparse_mask = (target != 0) | (pred != 0)
                    if sparse_mask.any():
                        masked_logits = head_logits[sparse_mask]
                        masked_target = target[sparse_mask]
                        head_loss = F.cross_entropy(masked_logits, masked_target)
                    else:
                        head_loss = torch.zeros((), device=head_logits.device)
                        is_real = False
                else:
                    head_loss = _focal_cross_entropy(head_logits, target, focal_gamma, weight=(
                        self._class_weights_for_head(class_weights, head, ACTION_HEADS[head])
                        if class_weights is not None else None
                    ))
                    pred = torch.argmax(head_logits, dim=1) if compute_metrics else None

                if compute_metrics:
                    match = pred == target
                    n_t = torch.tensor(float(pred.shape[0]), device=pred.device)
                    correct = match.float().sum()
                    pos_pred = (pred != 0)
                    pos_target = (target != 0)
                    tp = (pos_pred & match).float().sum()
                    fp = (pos_pred & ~match).float().sum()
                    fn = (pos_target & ~match).float().sum()
                    target_pos = pos_target.float().sum()
                    pred_pos = pos_pred.float().sum()
                    safe_n = torch.clamp(n_t, min=1.0)
                    metrics[f"n_{head}"] = n_t
                    metrics[f"correct_{head}"] = correct
                    metrics[f"acc_{head}"] = correct / safe_n
                    metrics[f"tp_{head}"] = tp
                    metrics[f"fp_{head}"] = fp
                    metrics[f"fn_{head}"] = fn
                    metrics[f"target_pos_{head}"] = target_pos
                    metrics[f"pred_pos_{head}"] = pred_pos
                    accuracy_components.append(correct / safe_n)
            head_loss = head_loss * weights_map.get(head, 1.0)
            losses.append(head_loss)
            loss_is_real.append(is_real)

        if compute_metrics:
            if accuracy_components:
                metrics["accuracy"] = torch.stack(accuracy_components).mean()
            else:
                metrics["accuracy"] = torch.zeros((), device=losses[0].device if losses else torch.device("cpu"))
        else:
            # Placeholder so downstream code has a tensor to reference.
            metrics["accuracy"] = torch.zeros((), device=losses[0].device if losses else torch.device("cpu"))

        return losses, loss_is_real, metrics

    def supervised_step(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor],
        lr: float,
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: np.ndarray | torch.Tensor | None = None,
        accumulate_only: bool = False,
        head_loss_weights: Mapping[str, float] | None = None,
        focal_gamma: float = 0.0,
        sparse_discrete: bool = True,
        look_deadzone: float = 0.0,
        look_turn_alpha: float = 0.0,
        loss_scale: float = 1.0,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        optimizer = self._optimizer("bc", self.model.parameters(), lr)
        if not accumulate_only:
            optimizer.zero_grad()

        with self._autocast():
            _, logits, _, next_hidden = self._forward_tensors(obs, hidden=hidden, masks=masks)
            losses, loss_is_real, metrics = self._compute_head_losses_and_metrics(
                logits, actions, class_weights=class_weights, head_loss_weights=head_loss_weights,
                focal_gamma=focal_gamma, sparse_discrete=sparse_discrete,
                look_deadzone=look_deadzone, look_turn_alpha=look_turn_alpha,
                compute_metrics=compute_metrics,
            )
            real = [l for l, r in zip(losses, loss_is_real) if r]
            loss = torch.stack(real).mean() if real else torch.zeros((), device=losses[0].device)
        (loss * float(loss_scale)).backward()
        if not accumulate_only:
            optimizer.step()

        metrics["loss"] = loss.detach()
        metrics["_next_hidden"] = next_hidden.detach()
        return metrics

    def evaluate_supervised(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: np.ndarray | torch.Tensor | None = None,
        head_loss_weights: Mapping[str, float] | None = None,
        focal_gamma: float = 0.0,
        sparse_discrete: bool = True,
        look_deadzone: float = 0.0,
        look_turn_alpha: float = 0.0,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        with torch.inference_mode(), self._autocast():
            _, logits, _, next_hidden = self._forward_tensors(obs, hidden=hidden, masks=masks)
            losses, loss_is_real, metrics = self._compute_head_losses_and_metrics(
                logits, actions, head_loss_weights=head_loss_weights,
                focal_gamma=focal_gamma, sparse_discrete=sparse_discrete,
                look_deadzone=look_deadzone, look_turn_alpha=look_turn_alpha,
                compute_metrics=compute_metrics,
            )

        real = [l for l, r in zip(losses, loss_is_real) if r]
        loss = torch.stack(real).mean() if real else torch.zeros((), device=losses[0].device if losses else torch.device("cpu"))
        metrics["loss"] = loss
        metrics["_next_hidden"] = next_hidden.detach()
        return metrics

    def ppo_step(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        old_log_probs: np.ndarray | torch.Tensor,
        advantages: np.ndarray | torch.Tensor,
        returns: np.ndarray | torch.Tensor,
        clip_ratio: float,
        policy_lr: float,
        value_lr: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float | None,
        reference_policy: "QNNPolicy | None" = None,
        reference_kl_coef: float = 0.0,
        sample_temperatures: Mapping[str, float] | None = None,
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: np.ndarray | torch.Tensor | None = None,
    ) -> Dict[str, float]:
        self.model.train()
        optimizer = self._ppo_optimizer(policy_lr=policy_lr, value_lr=value_lr)
        optimizer.zero_grad()

        _, logits, values, _ = self._forward_tensors(obs, hidden=hidden, masks=masks)
        old_log_probs_t = self._tensor(old_log_probs, dtype=torch.float32).reshape(-1)
        advantages_t = self._tensor(advantages, dtype=torch.float32).reshape(-1)
        returns_t = self._tensor(returns, dtype=torch.float32).reshape(-1)
        values_flat = values.reshape(-1)

        log_probs = []
        entropies = []
        for head in ACTION_HEADS:
            head_logits = self._flatten_logits(logits[head])
            if head in CONTINUOUS_ACTION_HEADS:
                target = self._continuous_targets_for_head(actions, head, head_logits)
                mean = _continuous_mean(head, head_logits, look_cosine=self.look_cosine)
                temperature = float(sample_temperatures.get(head, 1.0)) if sample_temperatures else 1.0
                dist = _normal_from_mean_log_std(mean, self.model.continuous_log_std[head], temperature=temperature)
                log_probs.append(dist.log_prob(target))
                entropies.append(dist.entropy())
                continue

            head_logits = self._sampling_logits(
                head_logits,
                float(sample_temperatures.get(head, 1.0)) if sample_temperatures else 1.0,
            )
            target = self._action_targets_for_head(actions, head, head_logits)
            chosen_log_probs, head_entropy, _ = self._log_prob_entropy_from_logits(head_logits, target)
            log_probs.append(chosen_log_probs)
            entropies.append(head_entropy)
        new_log_probs = torch.stack(log_probs, dim=0).sum(dim=0)
        entropy = torch.stack(entropies, dim=0).mean(dim=0).mean()

        ratios = torch.exp(new_log_probs - old_log_probs_t)
        clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surrogate = torch.minimum(ratios * advantages_t, clipped * advantages_t)
        policy_loss = -torch.mean(surrogate)
        value_loss = F.mse_loss(values_flat, returns_t)
        reference_kl = torch.zeros((), dtype=torch.float32, device=self.device)
        if reference_policy is not None and reference_kl_coef > 0.0:
            with torch.no_grad():
                _, reference_logits, _, _ = reference_policy._forward_tensors(
                    obs,
                    hidden=hidden if reference_policy.use_gru else None,
                    masks=masks,
                )
            per_head_kl = []
            for head in ACTION_HEADS:
                if head in CONTINUOUS_ACTION_HEADS:
                    temperature = float(sample_temperatures.get(head, 1.0)) if sample_temperatures else 1.0
                    reference_mean = _continuous_mean(head, reference_policy._flatten_logits(reference_logits[head]), look_cosine=reference_policy.look_cosine)
                    current_mean = _continuous_mean(head, self._flatten_logits(logits[head]), look_cosine=self.look_cosine)
                    reference_dist = _normal_from_mean_log_std(
                        reference_mean,
                        reference_policy.model.continuous_log_std[head],
                        temperature=temperature,
                    )
                    current_dist = _normal_from_mean_log_std(
                        current_mean,
                        self.model.continuous_log_std[head],
                        temperature=temperature,
                    )
                    per_head_kl.append(torch.distributions.kl_divergence(reference_dist, current_dist))
                    continue

                temperature = float(sample_temperatures.get(head, 1.0)) if sample_temperatures else 1.0
                reference_head_logits = self._sampling_logits(reference_policy._flatten_logits(reference_logits[head]), temperature)
                current_head_logits = self._sampling_logits(self._flatten_logits(logits[head]), temperature)
                ref_log_probs = torch.log_softmax(reference_head_logits, dim=1)
                current_log_probs = torch.log_softmax(current_head_logits, dim=1)
                ref_probs = torch.exp(ref_log_probs)
                per_head_kl.append(torch.sum(ref_probs * (ref_log_probs - current_log_probs), dim=1))
            reference_kl = torch.stack(per_head_kl, dim=0).mean(dim=0).mean()

        loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy) + (reference_kl_coef * reference_kl)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        optimizer.step()

        approx_kl = torch.mean(old_log_probs_t - new_log_probs).item()
        clip_fraction = torch.mean(((ratios > 1.0 + clip_ratio) | (ratios < 1.0 - clip_ratio)).float()).item()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(loss.item()),
            "entropy": float(entropy.item()),
            "reference_kl": float(reference_kl.item()),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
        }

    def value_step(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        returns: np.ndarray | torch.Tensor,
        lr: float,
    ) -> Dict[str, float]:
        self.model.train()
        optimizer = self._optimizer("ppo_value", self._value_parameters(), lr)
        optimizer.zero_grad()

        _, _, values, _ = self._forward_tensors(obs)
        returns_t = self._tensor(returns, dtype=torch.float32).reshape(-1)
        loss = F.mse_loss(values.reshape(-1), returns_t)
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
            "d_model": self.d_model,
            "head_hidden": self.head_hidden,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ffn_dim": self.ffn_dim,
            "attn_dropout": self.attn_dropout,
            "readout": self.readout,
            "action_history_tokens": self.action_history_tokens,
            "model_version": 9,
            "action_schema_version": 1,
            "backend": "pytorch",
            "requested_device": self.device_spec.requested,
            "resolved_device": self.device_spec.resolved,
            "accelerator_backend": self.device_spec.backend,
        }
        # Strip _orig_mod. prefix from torch.compile so checkpoints are
        # always in uncompiled format (loadable with or without compile).
        raw_sd = self.model.state_dict()
        clean_sd = {
            key.replace("_orig_mod.", ""): value.detach().cpu()
            for key, value in raw_sd.items()
        }
        payload = {
            "meta": meta,
            "state_dict": clean_sd,
        }
        torch.save(payload, target)
        target.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "QNNPolicy":
        source = Path(path)
        payload = trusted_torch_load(source, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload or "meta" not in payload:
            raise ValueError(f"Unrecognised checkpoint format: {source}")
        meta = dict(payload["meta"])
        model = cls(
            obs_dim=int(meta.get("obs_dim", 0)),
            trunk_hidden=int(meta.get("d_model", meta["trunk_hidden"])),
            gru_hidden=int(meta.get("gru_hidden", 0)),
            use_gru=bool(meta.get("use_gru", False)),
            seed=0,
            device=device,
            d_model=int(meta.get("d_model", meta["trunk_hidden"])),
            n_heads=int(meta.get("n_heads", 2)),
            n_layers=int(meta.get("n_layers", 2)),
            ffn_dim=int(meta.get("ffn_dim", 256)),
            attn_dropout=float(meta.get("attn_dropout", 0.0)),
            readout=str(meta.get("readout", "cls")),
            action_history_tokens=int(meta.get("action_history_tokens", 0)),
        )
        from qnn.utils.checkpoint_converter import migrate_modality_embed
        migrate_modality_embed(payload["state_dict"])
        try:
            model.model.load_state_dict(payload["state_dict"])
        except RuntimeError as exc:
            version = meta.get("model_version", "unknown")
            raise ValueError(
                f"Incompatible checkpoint architecture for {source} (model_version={version}). "
                "This code expects the transformer-summary + fused-GRU policy layout; "
                "retrain or re-export the checkpoint for the current model."
            ) from exc
        model.model.to(model.device)
        return model

    @classmethod
    def load_for_finetune(
        cls,
        path: str | Path,
        *,
        use_gru: bool,
        gru_hidden: int,
        device: str = "auto",
    ) -> "QNNPolicy":
        loaded = cls.load(path, device=device)
        target_use_gru = bool(use_gru and gru_hidden > 0)
        if loaded.use_gru == target_use_gru and loaded.gru_hidden == (gru_hidden if target_use_gru else 0):
            return loaded
        # Copy shared weights (trunk, policy heads, value head) into new model
        source_state = loaded.model.state_dict()
        target = cls(
            obs_dim=loaded.obs_dim,
            trunk_hidden=loaded.trunk_hidden,
            gru_hidden=gru_hidden,
            use_gru=target_use_gru,
            seed=0,
            device=device,
            d_model=loaded.d_model,
            n_heads=loaded.n_heads,
            n_layers=loaded.n_layers,
            ffn_dim=loaded.ffn_dim,
            attn_dropout=loaded.attn_dropout,
            readout=loaded.readout,
        )
        target_state = target.model.state_dict()
        for key, target_value in target_state.items():
            source_value = source_state.get(key)
            if source_value is not None and source_value.shape == target_value.shape:
                target_value.copy_(source_value)
        target.model.load_state_dict(target_state)
        return target
