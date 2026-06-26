"""Training-time wrapper around the model.

The compute graph (``Network``) lives in :mod:`qnn.model.network`. This
module owns the surrounding machinery: optimizers, loss shaping,
sampling, hidden-state lifecycle, and checkpoint I/O.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qnn.actions import (
    MOVE_AXES,
    MOVE_AXIS_NAMES,
    MOVE_AXIS_CLASSES,
    MOVE_CLASS_NEG,
    MOVE_CLASS_NONE,
    MOVE_CLASS_POS,
)
from qnn.model.network import (
    ATTACK_HEAD,
    ATTACK_HEAD_SIZE,
    LOOK_HEAD,
    LOOK_HEAD_SIZE,
    MOVE_HEAD,
    MOVE_HEAD_SIZE,
    ModelConfig,
    Network,
    WEAPON_HEAD,
)
from qnn.model.weapon_head import weapon_index_from_id
from qnn.schema import WEAPON_HEAD_SIZE
from qnn.utils.device import configure_torch_runtime, resolve_torch_device
from qnn.utils.io import trusted_torch_load
from qnn.vocab import TOKEN_ACTOR


HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "target": 1.0,
    "move": 1.0,
    "look": 1.0,
    "attack": 1.0,
}


WEAPON_HEAD_CLASS_NAMES: Tuple[Tuple[int, str], ...] = (
    (0, "axe"),
    (1, "shotgun"),
    (2, "super_shotgun"),
    (3, "nailgun"),
    (4, "super_nailgun"),
    (5, "grenade_launcher"),
    (6, "rocket_launcher"),
    (7, "thunderbolt"),
)


@dataclass(slots=True)
class PolicyActionBatch:
    actions: Dict[str, np.ndarray]
    log_probs: torch.Tensor
    values: torch.Tensor
    entropies: Dict[str, torch.Tensor]
    next_hidden: torch.Tensor


def _null_side_channel_scope(*_args, **_kwargs):
    """Default side-channel provider — a no-op scope for the canonical model.

    Bench probes inject ``qnn.model.bench.side_channels.bench_side_channel_scope``
    via ``QNNPolicy(side_channel_provider=...)`` to enter the label-derived
    engagement_ema / target-supervision contexts. The canonical
    model needs none of those, so it uses this.
    """
    import contextlib
    return contextlib.nullcontext()


class QNNPolicy:
    """Feed-forward combat-objective model for BC."""

    def __init__(
        self,
        *,
        obs_dim: int,
        model: ModelConfig | None,
        jump_pos_weight: float,
        attack_focal_gamma: float,
        attack_focal_alpha: float,
        attack_distance_sigma: float,
        jump_distance_sigma: float,
        look_label_smoothing_sigma: float = 0.0,
        seed: int,
        device: str,
        model_factory: Callable[[int, ModelConfig], nn.Module] | None = None,
        side_channel_provider: Callable[..., Any] | None = None,
        graph: Any | None = None,
    ) -> None:
        """Construct a BC policy.

        ``graph`` (a :class:`qnn.model.graph.GraphSpec`) is the declarative
        assembly path: the inner module is built by
        ``qnn.model.graph.build_network`` and the spec is persisted in
        checkpoint meta (``model_graph``) so the checkpoint is fully
        self-describing — no factory rehydration on load. Mutually
        exclusive with ``model_factory``. ``model`` may be ``None`` when
        ``graph`` is given (derived via ``model_config_from_graph``).

        ``side_channel_provider(actions, masks) -> context manager`` is an
        optional hook the trainer enters around each forward pass. The
        canonical model passes ``None`` (no extra context). Bench probes pass
        ``qnn.model.bench.side_channels.bench_side_channel_scope`` to enter the
        label-derived engagement_ema / target-supervision contexts.
        Keeping it a plain callable keeps this class free of any bench import.

        ``model_factory`` is an optional override for the inner ``nn.Module``:
        when ``None`` (the default, used by all production BC training) the
        canonical ``Network`` is built from ``model``. Ablation
        runners (e.g. ``qnn.model.bench``) pass a factory that builds an
        alternate module — typically one that drops the encoder or GRU — but
        the factory must respect ``Network``'s forward contract
        so the canonical BC supervised loop can drive it unchanged.

        The injected module's flags should still be consistent with
        ``model`` (use_gru / use_weapon_head / etc.) since QNNPolicy's
        policy-layer logic — hidden-state shaping, weapon-switch heuristics,
        head-loss gating — reads from ``model``, not from the module.
        """
        if graph is not None:
            if model_factory is not None:
                raise ValueError("pass either graph or model_factory, not both")
            from qnn.model.graph import build_network, model_config_from_graph

            if model is None:
                model = model_config_from_graph(graph)
            model_factory = lambda _obs_dim, _cfg: build_network(_obs_dim, graph)  # noqa: E731
        self.graph = graph
        self.obs_dim = int(obs_dim)
        self._side_channel_provider = side_channel_provider or _null_side_channel_scope
        self.config = model
        self.d_model = int(model.d_model)
        self.use_gru = bool(model.use_gru and model.d_gru > 0)
        self.d_gru = int(model.d_gru) if self.use_gru else 0
        self.use_weapon_head = bool(model.use_weapon_head)
        self.look_bypass_gru = bool(model.look_bypass_gru and self.use_gru)
        self.weapon_switch_confidence = float(model.weapon_switch_confidence)
        self.weapon_switch_margin = float(model.weapon_switch_margin)
        self.weapon_context_from_obs = bool(model.weapon_context_from_obs)
        self.d_target = int(model.d_target)
        self.d_move = int(model.d_move)
        self.d_look = int(model.d_look)
        self.d_attack = int(model.d_attack)
        self.d_weapon = int(model.d_weapon)
        self.head_activation = model.head_activation
        # jump_pos_weight > 1.0 upweights the POS class on the move ud-axis CE
        # — direct imbalance fix for the rare jump-positive case (~4% pos rate).
        # Inverse-frequency reference: ~24× for 4% positive rate.
        self.jump_pos_weight = float(jump_pos_weight)
        # attack_focal_gamma > 0 swaps the attack-head BCE for focal BCE
        # (Lin et al. 2017): each frame's BCE is multiplied by
        # (1 - p_t)^gamma so easy examples contribute less gradient and
        # capacity flows to the borderline ready-frame "fire or wait?"
        # decisions. 0 = standard BCE.
        self.attack_focal_gamma = float(attack_focal_gamma)
        # attack_focal_alpha is Lin's per-class prefactor on the focal weight:
        # alpha_t = alpha on positives, (1 - alpha) on negatives. Active
        # only when attack_focal_gamma > 0. Default 0.5 is neutral (both
        # classes weighted equally up to a global scale). To run the Lin
        # recipe end-to-end set attack_pos_weight_override=1.0 alongside —
        # otherwise pos_weight stacks multiplicatively on the positive
        # branch and alpha loses its canonical class-fraction meaning.
        self.attack_focal_alpha = float(attack_focal_alpha)
        # input_mask is a training-time attribute (NOT a ModelConfig
        # field — checkpoint meta stays clean and the same ckpt can be
        # retrained either way). Trainer sets this to True after
        # construction when train.json.input_mask is true. Read by
        # ``_compute_head_losses_and_metrics`` to swap each head's
        # supervisory label from the raw demo button (usercmd) to the
        # engine outcome (act = max(usercmd − infeasibility_mask, 0));
        # for fire this collapses to "label = op_input bit 3".
        self.input_mask: bool = False
        # attack_label_shift is another training-time attribute (NOT a
        # ModelConfig field — keeps checkpoint meta clean and a single
        # ckpt can be retrained either way). Trainer sets this to True
        # after construction when train.json.attack_label_shift is true.
        # When on, ``_compute_head_losses_and_metrics`` swaps the attack
        # BCE *target* from ``actions["attack"]`` to
        # ``actions["attack_shifted"]`` (per-episode +1 op-frame OR). The
        # val precision/recall/F1 metrics still use the original
        # ``attack`` label so they remain comparable to the baseline.
        self.attack_label_shift: bool = False
        # attack_op_only — training-time toggle for the attack head's
        # noop-frame handling. When True (and input_mask is on), op=0
        # frames are MASKED OUT of the BCE loss (no gradient), pos_weight
        # is computed from op=1 counts only, and val precision/recall/F1
        # are scored on op=1 frames only. When False (default — preserves
        # historical behavior), op=0 frames stay in: their label is
        # forced to 0 (feasibility AND demo_press), they contribute
        # gradient pulling the model toward predicting 0 there, and they
        # show up in the metric as TN / occasional FP. Trainer sets this
        # after construction from train.json.attack_op_only.
        self.attack_op_only: bool = False
        # attack_distance_sigma > 0 enables Gaussian-shouldered BCE on the
        # attack head: per-frame BCE is multiplied by 1 at positives and by
        # 1 - exp(-d^2/(2*sigma^2)) at negatives, where d is distance (in
        # frames) to the nearest positive. Adjacent-to-press FPs cost
        # near-zero loss; far-from-press FPs cost full loss. Inference is
        # unchanged. See src/qnn/bc/heads/loss_shaping.py. 0 = standard BCE.
        self.attack_distance_sigma = float(attack_distance_sigma)
        # Same shoulder applied to the move ud-axis (jump) CE. Tuned
        # independently of fire because jump-press timing noise has a
        # different scale.
        self.jump_distance_sigma = float(jump_distance_sigma)
        # >0 swaps the binned look head's hard-argmin cross-entropy for a
        # distance-aware Gaussian soft-target CE (σ in radians). Smooths the
        # foveated sub-degree center bins whose hard distinctions aren't
        # learnable; 0 = standard one-hot CE. See qnn.model.look_bins.
        self.look_label_smoothing_sigma = float(look_label_smoothing_sigma)
        self.n_heads = int(model.n_heads)
        self.n_layers = int(model.n_layers)
        self.d_ffn = int(model.d_ffn)
        self.attn_dropout = float(model.attn_dropout)
        self.head_hidden = (self.d_gru + self.d_model) if self.use_gru else (2 * self.d_model)
        self.seed = int(seed)
        self.device_spec = resolve_torch_device(device)
        configure_torch_runtime(self.device_spec)
        self.device = self.device_spec.device
        self._rocm_inference_pad_batch = 0
        if self.device_spec.backend == "rocm":
            raw_pad_batch = os.environ.get("QNN_ROCM_INFERENCE_PAD_BATCH", "32").strip()
            try:
                self._rocm_inference_pad_batch = max(int(raw_pad_batch), 0)
            except ValueError:
                self._rocm_inference_pad_batch = 32

        torch.manual_seed(self.seed)
        if model_factory is None:
            self.model = Network(obs_dim=self.obs_dim, model=model).to(self.device)
        else:
            built = model_factory(self.obs_dim, model)
            if not isinstance(built, nn.Module):
                raise TypeError(
                    f"model_factory must return nn.Module, got {type(built).__name__}"
                )
            self.model = built.to(self.device)
        self.model.train()
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}

    def zero_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.d_gru), dtype=np.float32)

    def _tensor(self, value: np.ndarray | torch.Tensor | Iterable[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device == self.device and value.dtype == dtype:
                return value
            non_blocking = value.device.type == "cpu" and (
                not isinstance(self.device, torch.device) or self.device.type != "cpu"
            )
            return value.to(device=self.device, dtype=dtype, non_blocking=non_blocking)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _autocast(self):
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

    def _maybe_pad_obs_batch(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], int]:
        # `vel` matches the old `self_scalars` ndim semantics: (B, 3)
        # flat, (B, T, 3) sequence. Native obs replaced self_scalars
        # as a single-field key with per-field arrays.
        sample = obs_dict.get("vel")
        if sample is None:
            sample = obs_dict["self_scalars"]  # legacy fallback
        if sample.ndim != 2 or self._rocm_inference_pad_batch <= 0:
            return obs_dict, 0

        batch_size = int(sample.shape[0])
        target_batch = self._rocm_inference_pad_batch
        if batch_size == 0 or batch_size >= target_batch:
            return obs_dict, 0

        pad_rows = target_batch - batch_size
        padded_obs: Dict[str, torch.Tensor] = {}
        for key, value in obs_dict.items():
            pad_shape = (pad_rows, *value.shape[1:])
            pad_value = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            padded_obs[key] = torch.cat([value, pad_value], dim=0)
        return padded_obs, pad_rows

    @staticmethod
    def _pad_companion(
        tensor: torch.Tensor | None, pad_rows: int,
    ) -> torch.Tensor | None:
        """Zero-pad a companion tensor along dim 0 to match obs padding.

        Used after ``_maybe_pad_obs_batch`` to extend hidden state so
        callers that pass it on ROCm with small batches don't hit a B
        mismatch. Pass-through when the tensor is None or no padding
        was applied.
        """
        if tensor is None or pad_rows <= 0:
            return tensor
        pad_shape = (pad_rows, *tensor.shape[1:])
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=0)

    def _inference_dequant(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the native→model-facing dequant chain (Self/Spatial/Entity) for
        inference. Idempotent (each stage short-circuits if its output key is
        present), so it's a no-op on already-dequantized obs. Stateless modules,
        instantiated once and cached."""
        chain = getattr(self, "_dequant_chain", None)
        if chain is None:
            from qnn.model.dequant import (
                SelfDequantizer, SpatialDequantizer, EntityDequantizer,
            )
            chain = (SelfDequantizer().eval(), SpatialDequantizer().eval(),
                     EntityDequantizer().eval())
            self._dequant_chain = chain
        self_dq, spatial_dq, entity_dq = chain
        return entity_dq(spatial_dq(self_dq(obs)))

    def _forward_tensors(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Native-obs dequant for inference. Live obs (bridge / engine) arrives
        # at native widths; BC training pre-dequantizes at preload, and the
        # canonical Network carries its own dequant adapters, but bench-built
        # Networks (e.g. full_4head) do not — so dequant here so act()/encode()
        # work uniformly. The three dequantizers are idempotent (each short-
        # circuits when its output key is already present), so this is a no-op
        # on already-dequantized obs. Inference-only: training drives
        # self.model directly, not _forward_tensors.
        obs_tensors = self._inference_dequant(obs_tensors)

        hidden_tensor: torch.Tensor | None = None
        if self.use_gru and hidden is not None:
            hidden_tensor = self._tensor(hidden, dtype=torch.float32)

        # Use `vel` to detect flat-batch (B, 3) vs sequence (B, T, 3).
        # The legacy obs carried `self_scalars` (B, 17) here; the native
        # obs has per-field arrays, with vel matching the same ndim
        # semantics (2D flat, 3D sequence).
        sample = obs_tensors.get("vel")
        if sample is None:
            sample = obs_tensors["self_scalars"]  # legacy fallback
        if sample.ndim == 2:
            batch_size = int(sample.shape[0])
            padded_obs, pad_rows = self._maybe_pad_obs_batch(obs_tensors)
            padded_hidden = (
                self._pad_companion(hidden_tensor, pad_rows)
                if self.use_gru else hidden_tensor
            )
            features, logits, values, next_hidden, target_logits = self.model(
                padded_obs,
                padded_hidden,
            )
            if pad_rows == 0:
                return features, logits, values, next_hidden, target_logits
            return (
                features[:batch_size],
                {head: tensor[:batch_size] for head, tensor in logits.items()},
                values[:batch_size],
                next_hidden[:batch_size],
                target_logits[:batch_size],
            )

        if sample.ndim != 3:
            raise ValueError("obs must be rank-2 or rank-3")
        reset_mask_tensor = None
        if isinstance(masks, Mapping) and "reset_mask" in masks:
            reset_mask_tensor = self._tensor(masks["reset_mask"], dtype=torch.bool)
        return self.model(
            obs_tensors,
            hidden_tensor,
            reset_mask=reset_mask_tensor,
        )

    def _optimizer(self, name: str, params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        optimizer = self._optimizers.get(name)
        if optimizer is None:
            optimizer = torch.optim.Adam(list(params), lr=lr, fused=True)
            self._optimizers[name] = optimizer
        for group in optimizer.param_groups:
            group["lr"] = lr
        return optimizer

    def bc_zero_grad(self) -> None:
        opt = self._optimizers.get("bc")
        if opt is not None:
            opt.zero_grad()

    def bc_step(self) -> None:
        opt = self._optimizers.get("bc")
        if opt is not None:
            opt.step()

    def encode(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        hidden: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            features, _, _, next_hidden, _ = self._forward_tensors(obs, hidden=hidden)
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
            features, logits_t, values_t, next_hidden, _ = self._forward_tensors(obs, hidden=hidden)
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
        row_generators: Any | None = None,
        sample_temperatures: Mapping[str, float] | None = None,
        diag_log_path: str | Path | None = None,
    ) -> PolicyActionBatch:
        """Emit engine actions from a forward pass.

        Output dict shape matches the engine's action contract (see
        qnn.actions.ActionLabels):
          move   : (B, 3) float in [-1, 1] — view-relative wishvel/maxspeed.
                   Categorical mode argmaxes/samples each axis to {-1, 0, +1};
                   continuous mode passes the regression output through clamp.
                   Up axis is 0 (no jump head).
          look   : (B, 3) float — look_predict unit vector from the head.
          fire   : (B,)   int   — 0/1 from sigmoid(logit) threshold or bernoulli.
          weapon : (B,)   int   — engine weapon byte 1..8 (or 0 = no switch).

        log_probs / values / entropies are placeholders for shape compatibility
        with the action batch consumer; greedy/sampled eval doesn't read them.

        When *diag_log_path* is set, append one JSONL record per call with
        target/look/move/fire internals — for distribution-shift debugging in
        live eval.
        """
        del masks, generator
        with torch.inference_mode():
            _, logits, _, next_hidden, target_logits = self._forward_tensors(obs, hidden=hidden)

        sample_mode = str(mode).lower()
        if sample_mode not in ("greedy", "sampled"):
            raise ValueError(f"Unsupported policy mode: {mode}")
        temps = dict(sample_temperatures or {})

        # ---- move ----
        # 3 categorical axes (fb, lr, ud), each a 3-class softmax over
        # {neg, none, pos}.  Greedy = argmax per axis; sampled = categorical
        # per axis.  Decoded engine value per axis = class - 1, i.e. {-1, 0, +1}.
        move_logits = logits[MOVE_HEAD].reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        n_rows = int(move_logits.shape[0])
        if sample_mode == "greedy":
            move_classes = torch.argmax(move_logits, dim=-1)               # (n_rows, 3)
        else:
            t_move = float(temps.get("move", 1.0))
            move_probs = F.softmax(move_logits / max(t_move, 1e-6), dim=-1)  # (n_rows, 3, 3)
            # Sample one class per axis.  Flatten axes into the batch dim
            # so _categorical_sample can run row-wise; reshape back after.
            flat_probs = move_probs.reshape(-1, MOVE_AXIS_CLASSES)         # (n_rows*3, 3)
            if row_generators is None:
                flat_classes = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            else:
                flat_classes = torch.empty(flat_probs.shape[0], dtype=torch.long, device=flat_probs.device)
                for axis_idx in range(flat_probs.shape[0]):
                    row_idx = axis_idx // MOVE_AXES
                    gen = row_generators[row_idx]
                    flat_classes[axis_idx] = torch.multinomial(
                        flat_probs[axis_idx:axis_idx + 1], num_samples=1, generator=gen,
                    ).squeeze()
            move_classes = flat_classes.reshape(n_rows, MOVE_AXES)
        move = (move_classes.float() - float(MOVE_CLASS_NONE))             # {-1, 0, +1} per axis

        # ---- look ----
        # Greedy: look_predict is the head's deterministic readout (already
        # unit-normalized inside the model; clamp guards fp noise).  For the
        # polar look head this is the argmax reconstruction — which collapses
        # the hold-dominated turn distribution to a near-frozen view.  Sampled
        # (HYBRID decode — must mirror tools/export_onnx.ExportWrapper so offline
        # eval matches the deployed ONNX graph): sample the MAGNITUDE bin (human
        # turn-size distribution) but take DIRECTION as the continuous circular
        # mean of the direction softmax, NOT a sampled dir bin.  Per-frame
        # direction sampling + 16-bin quantization break the learned heading-hold
        # and cause the live "spin"; the circular mean restores near-human
        # heading persistence with the magnitude distribution untouched.  See
        # src/docs/look-head.md §3 and qnn.model.look_bins.
        if sample_mode == "sampled" and "_look_mag_logits" in logits:
            from qnn.model.look_bins import (
                DIR_CENTERS, MAG_CENTERS, N_DIR, N_MAG, tangent_expmap,
            )
            t_look = max(float(temps.get("look", 1.0)), 1e-6)
            mag_logits = logits["_look_mag_logits"].reshape(-1, N_MAG + 1)
            dir_logits = logits["_look_dir_logits"].reshape(-1, N_DIR)
            mag_bin = self._categorical_sample(
                F.softmax(mag_logits / t_look, dim=-1), row_generators)
            theta = MAG_CENTERS.to(mag_logits.device)[mag_bin]               # (R,)
            dir_centers = DIR_CENTERS.to(dir_logits.device)
            p_dir = F.softmax(dir_logits, dim=-1)                            # (R, N_DIR)
            phi = torch.atan2((p_dir * dir_centers.sin()).sum(dim=-1),
                              (p_dir * dir_centers.cos()).sum(dim=-1))       # (R,)
            z = torch.stack([theta * phi.cos(), theta * phi.sin()], dim=-1)  # (R, 2)
            look = tangent_expmap(z).reshape(-1, LOOK_HEAD_SIZE)
        else:
            look = torch.clamp(logits[LOOK_HEAD].reshape(-1, LOOK_HEAD_SIZE), -1.0, 1.0)

        # ---- fire ----
        attack_logit = logits[ATTACK_HEAD].reshape(-1)
        fire_prob = torch.sigmoid(attack_logit)
        if sample_mode == "greedy":
            fire = (fire_prob > 0.5).long()
        else:
            t_fire = float(temps.get("attack", 1.0))
            # Temperature-modulate the logit: prob(class=1) = sigmoid(logit/T).
            fire_prob_t = torch.sigmoid(attack_logit / max(t_fire, 1e-6))
            fire = self._bernoulli_sample(fire_prob_t, row_generators)

        # ---- weapon ----
        # Sticky weapon controller: pick the desired weapon class when
        # the head is both confident and separated from the runner-up;
        # otherwise emit the currently-held weapon so the resulting
        # impulse is a no-op.  The action dict carries the engine
        # impulse byte 1..8 directly (class index + 1); the bridge
        # writes it to the binary wire unchanged.
        weapon_impulse = torch.ones(int(move.shape[0]), dtype=torch.long, device=move.device)
        if self.use_weapon_head and WEAPON_HEAD in logits and isinstance(obs, Mapping) and "self_weapon_id" in obs:
            weapon_logits = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
            weapon_probs = F.softmax(weapon_logits, dim=-1)
            top2 = torch.topk(weapon_probs, k=2, dim=-1)
            desired_class = top2.indices[:, 0]
            confidence = top2.values[:, 0]
            margin = top2.values[:, 0] - top2.values[:, 1]
            current_ids = self._tensor(obs["self_weapon_id"], dtype=torch.long).reshape(-1)
            current_class = weapon_index_from_id(current_ids)
            should_switch = (
                (desired_class != current_class)
                & (confidence >= self.weapon_switch_confidence)
                & (margin >= self.weapon_switch_margin)
            )
            chosen_class = torch.where(should_switch, desired_class, current_class)
            weapon_impulse = chosen_class + 1   # class 0..7 → impulse 1..8

        actions = {
            "move":   move.detach().cpu().numpy().astype(np.float32),
            "look":   look.detach().cpu().numpy().astype(np.float32),
            "attack":   fire.detach().cpu().numpy().astype(np.int64),
            "weapon": weapon_impulse.detach().cpu().numpy().astype(np.int64),
        }

        if diag_log_path is not None:
            self._append_act_diagnostics(
                diag_log_path, obs, logits, target_logits, attack_logit, fire_prob,
                actions,
            )

        zero = torch.zeros(int(move.shape[0]), dtype=torch.float32, device=move.device)
        entropies = {
            "move":   zero.clone(),
            "look":   zero.clone(),
            "attack":   zero.clone(),
            "weapon": zero.clone(),
        }
        return PolicyActionBatch(
            actions=actions,
            log_probs=zero.clone(),
            values=zero.clone(),
            entropies=entropies,
            next_hidden=next_hidden.detach(),
        )

    @staticmethod
    def _append_act_diagnostics(
        path: str | Path,
        obs: Any,
        logits: Dict[str, torch.Tensor],
        target_logits: torch.Tensor,
        attack_logit: torch.Tensor,
        fire_prob: torch.Tensor,
        actions: Dict[str, np.ndarray],
    ) -> None:
        """Append per-row JSONL records with target/look/move/fire internals."""
        import json as _json
        from qnn.vocab import TOKEN_ACTOR  # local import to keep top-level clean

        def _np(x: torch.Tensor) -> np.ndarray:
            return x.detach().cpu().numpy()

        # Mask invalid target indices (TargetPointer pre-masks with -1e9 — pick
        # them up so we can tell "no actor present" from "low confidence".
        tl = _np(target_logits.reshape(target_logits.shape[0], -1))  # (B, N)
        # entity_types lets us see how many actor indices actually contain a bot
        et = obs.get("entity_types") if isinstance(obs, dict) else None
        actor_counts = None
        if et is not None:
            et_np = np.asarray(et)
            actor_counts = (et_np == TOKEN_ACTOR).sum(axis=-1).reshape(-1).tolist()

        # Soft attention probs (with masked indices ~0 thanks to -1e9 logit)
        tl_t = target_logits.reshape(target_logits.shape[0], -1)
        probs = torch.softmax(tl_t, dim=-1).detach().cpu().numpy()
        argmax_idx = probs.argmax(axis=-1).tolist()
        max_prob = probs.max(axis=-1).tolist()
        # Entropy in nats
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)

        look_prior = _np(logits["_look_prior"]).reshape(-1, 3) if "_look_prior" in logits else None
        look_delta = _np(logits["_look_delta"]).reshape(-1, 3) if "_look_delta" in logits else None
        look_predict = _np(logits[LOOK_HEAD]).reshape(-1, 3)

        # Alignment scalar that feeds the attack head
        if look_prior is not None:
            align = (look_predict * look_prior).sum(axis=-1)
        else:
            align = np.full(look_predict.shape[0], np.nan, dtype=np.float32)

        move_logits_np = (
            _np(logits[MOVE_HEAD]).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES).tolist()
            if MOVE_HEAD in logits else None
        )
        move_prob_np = (
            _np(F.softmax(logits[MOVE_HEAD], dim=-1)).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES).tolist()
            if MOVE_HEAD in logits else None
        )
        attack_logit_np = _np(attack_logit).reshape(-1).tolist()
        fire_prob_np = _np(fire_prob).reshape(-1).tolist()

        # Weapon internals — to diagnose closed-loop switching (echo-lock): the
        # head's DESIRED weapon (argmax) + confidence vs the HELD weapon, and the
        # decided impulse. If desired == held nearly always in closed loop, the
        # head is echoing its held-weapon input (never wants to switch).
        weapon_logits_np = weapon_desired = weapon_conf = held_class = None
        if WEAPON_HEAD in logits and isinstance(obs, dict) and "self_weapon_id" in obs:
            wl = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
            wp = torch.softmax(wl, dim=-1)
            weapon_logits_np = _np(wl).tolist()
            weapon_desired = wp.argmax(dim=-1).detach().cpu().numpy().reshape(-1).tolist()
            weapon_conf = wp.max(dim=-1).values.detach().cpu().numpy().reshape(-1).tolist()
            held = weapon_index_from_id(
                torch.as_tensor(np.asarray(obs["self_weapon_id"])).long().reshape(-1))
            held_class = held.detach().cpu().numpy().reshape(-1).tolist()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for i in range(look_predict.shape[0]):
                rec = {
                    "row": i,
                    "actor_count": (actor_counts[i] if actor_counts is not None else None),
                    "target": {
                        "argmax_idx": int(argmax_idx[i]),
                        "max_prob": float(max_prob[i]),
                        "entropy_nats": float(ent[i]),
                        "logits": tl[i].tolist(),
                    },
                    "look": {
                        "pred": look_predict[i].tolist(),
                        "base": (look_prior[i].tolist() if look_prior is not None else None),
                        "base_mag": (
                            float(np.linalg.norm(look_prior[i])) if look_prior is not None else None
                        ),
                        "delta": (look_delta[i].tolist() if look_delta is not None else None),
                        "delta_mag": (
                            float(np.linalg.norm(look_delta[i])) if look_delta is not None else None
                        ),
                        "alignment": float(align[i]),
                    },
                    "move": {
                        "axes":   list(MOVE_AXIS_NAMES),
                        "logits": (move_logits_np[i] if move_logits_np is not None else None),
                        "prob":   (move_prob_np[i]   if move_prob_np   is not None else None),
                        "action": actions["move"][i].tolist(),
                    },
                    "attack": {
                        "logit": float(attack_logit_np[i]),
                        "prob":  float(fire_prob_np[i]),
                        "action": int(actions["attack"][i]),
                    },
                    "weapon": ({
                        "desired": int(weapon_desired[i]),     # argmax class 0..7
                        "held":    int(held_class[i]),         # held weapon class 0..7
                        "conf":    float(weapon_conf[i]),      # softmax prob of desired
                        "decided": int(actions["weapon"][i]),  # impulse 1..8 (or 0)
                        "logits":  weapon_logits_np[i],
                    } if weapon_desired is not None else None),
                }
                f.write(_json.dumps(rec) + "\n")

    @staticmethod
    def _categorical_sample(
        probs: torch.Tensor,
        row_generators: Any | None,
    ) -> torch.Tensor:
        """Sample one class index per row from probs (B, K).

        When row_generators is None, uses default RNG. When provided, draws
        one row at a time so each episode's RNG advances independently — the
        eval pipeline relies on this for reproducibility across batched envs.
        """
        if row_generators is None:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        out = torch.empty(probs.shape[0], dtype=torch.long, device=probs.device)
        for idx, gen in enumerate(row_generators):
            row_p = probs[idx:idx + 1]
            out[idx] = torch.multinomial(row_p, num_samples=1, generator=gen).squeeze()
        return out

    @staticmethod
    def _emit_look_tangent_sums(
        metrics: dict,
        pred_u: torch.Tensor,
        tgt_u: torch.Tensor,
    ) -> None:
        """Emit additive tangent-space sufficient stats for look_r2 / look_ewa_deg.

        See qnn.bc.look_metrics. Log-map both unit vectors to the 2D rotation
        vector at (1,0,0) (||z|| = turn angle), then emit raw sums combined at
        epoch end. Emitted under the ``looksum_`` raw-sum prefix + n_look_valid.
        """
        def _logmap(u: torch.Tensor) -> torch.Tensor:
            theta = torch.arccos(u[:, 0].clamp(-1.0, 1.0))                # (n,)
            yz = u[:, 1:3]                                                # (n,2)
            yz_norm = torch.linalg.vector_norm(yz, dim=-1)               # (n,)
            scale = torch.where(yz_norm > 1e-8, theta / yz_norm.clamp(min=1e-8),
                                torch.zeros_like(theta))
            return yz * scale[:, None]

        z = _logmap(tgt_u)
        zh = _logmap(pred_u)
        diff = z - zh
        theta = torch.linalg.vector_norm(z, dim=-1)                      # target turn
        cos = (pred_u * tgt_u).sum(dim=-1).clamp(-1.0, 1.0)
        dtheta = torch.arccos(cos)                                       # angular err
        metrics["n_look_valid"] = torch.tensor(float(z.shape[0]), device=z.device)
        metrics["looksum_ssres"] = (diff * diff).sum().detach()
        metrics["looksum_z0"] = z[:, 0].sum().detach()
        metrics["looksum_z1"] = z[:, 1].sum().detach()
        metrics["looksum_zz"] = (z * z).sum().detach()
        metrics["looksum_ewnum"] = (theta * dtheta).sum().detach()
        metrics["looksum_ewden"] = theta.sum().detach()

    @staticmethod
    def _bernoulli_sample(
        prob: torch.Tensor,
        row_generators: Any | None,
    ) -> torch.Tensor:
        if row_generators is None:
            return torch.bernoulli(prob).long()
        out = torch.empty(prob.shape[0], dtype=torch.long, device=prob.device)
        for idx, gen in enumerate(row_generators):
            out[idx] = torch.bernoulli(prob[idx:idx + 1], generator=gen).long()
        return out

    def _compute_head_losses_and_metrics(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        head_loss_weights: Mapping[str, float] | None = None,
        compute_metrics: bool = True,
        target_logits: torch.Tensor | None = None,
        obs: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[bool], Dict[str, torch.Tensor | int | float]]:
        weights_map = head_loss_weights or HEAD_LOSS_WEIGHTS
        losses: list[torch.Tensor] = []
        loss_is_real: list[bool] = []
        metrics: Dict[str, torch.Tensor | int | float] = {}
        accuracy_components: list[torch.Tensor] = []
        valid_flat = valid_mask.reshape(-1).bool() if valid_mask is not None else None

        # input_mask: pure per-axis feasibility — "would the engine
        # accept this axis press right now?".  When true, each head's
        # label is the engine OUTCOME = feasibility AND demo press;
        # the trainer recombines them per axis (see the per-head label
        # rewrite blocks below).  Bit layout of actions["input_mask"]
        # (packed by QNN_PackInputMask in the collector):
        #
        #   bit 0 = attack feasibility   (W_Attack would fire if button0=1)
        #   bit 1 = forward neg feasibility
        #   bit 2 = forward pos feasibility   (pmove always processes
        #                                      fmove → bits 1-2 both 1
        #                                      whenever alive)
        #   bit 3 = side neg feasibility
        #   bit 4 = side pos feasibility
        #   bit 5 = up neg feasibility   (swim down, water only)
        #   bit 6 = up pos feasibility   (swim up,   water only)
        #   bit 7 = jump feasibility     (ground-jump would fire if
        #                                 button2=1; depends on onground
        #                                 + anti-pogo + alive)
        #
        # Requires the recollected corpus that carries
        # act_input_mask.npy — hard-fails if missing.
        input_mask_on = bool(self.input_mask)
        if input_mask_on and "input_mask" not in actions:
            raise RuntimeError(
                "input_mask=True but actions['input_mask'] is absent. "
                "Recollect the corpus on a post-input_mask branch — the "
                "engine emits act_input_mask.npy as part of every shard."
            )
        input_mask_flat: torch.Tensor | None = None
        if input_mask_on:
            input_mask_flat = self._tensor(
                actions["input_mask"], dtype=torch.long).reshape(-1)

        target_loss_weight = float(weights_map.get("target", 1.0))
        if (
            target_logits is not None
            and "target_probs" in actions
            and target_loss_weight != 0.0
        ):
            target_flat = self._flatten_logits(target_logits)
            dist_t = self._tensor(actions["target_probs"], dtype=torch.float32)
            if dist_t.ndim == 3:
                dist_t = dist_t.reshape(-1, dist_t.shape[-1])
            # dist_t[:, 0] = NO_TARGET; dist_t[:, 1:] = idx probabilities.
            present = (1.0 - dist_t[:, 0]).clamp(min=0.0)
            idx_dist = dist_t[:, 1:]
            # No in-policy gate. Engagement filtering is the caller's job
            # via segment_mask (e.g. `{"act.target": {"$ne": 0}}`). Frames
            # with present=0 are already dropped at the dataset level; what
            # remains contributes to target loss and metrics in proportion
            # to its present value, with the clamp(min=1e-6) in the
            # renormalize keeping the divide numerically safe.
            valid = (
                valid_flat
                if valid_flat is not None
                else torch.ones_like(present, dtype=torch.bool)
            )
            aux_is_real = bool(valid.any().item())
            if aux_is_real:
                # Present-weighted soft CE: -sum_s p_idx * log_softmax(logits).
                # idx_dist sums to `present`; renormalize so each frame's
                # target probability mass sums to 1 before computing CE, then
                # weight the per-frame term by `present`.
                log_probs = F.log_softmax(target_flat[valid], dim=-1)
                present_v = present[valid]
                idx_target = idx_dist[valid] / present_v.clamp(min=1e-6).unsqueeze(-1)
                per_frame_ce = -(idx_target * log_probs).sum(dim=-1)
                aux_ce = (present_v * per_frame_ce).sum() / present_v.sum().clamp(min=1e-6)
            else:
                aux_ce = torch.zeros((), dtype=target_flat.dtype, device=target_flat.device)
            losses.append(aux_ce * target_loss_weight)
            loss_is_real.append(aux_is_real)
            if compute_metrics:
                metrics["loss_target"] = aux_ce.detach()
                metrics["target_present_mean"] = present.mean().detach()
                if aux_is_real:
                    # Aggregate KL — primary selection metric for target heads.
                    # KL(label || model) = present-weighted-NLL - entropy(label).
                    ent_per_frame = -(idx_target.clamp(min=1e-8) * idx_target.clamp(min=1e-8).log()).sum(dim=-1)
                    target_entropy = (present_v * ent_per_frame).sum() / present_v.sum().clamp(min=1e-6)
                    metrics["target_kl"] = (aux_ce - target_entropy).detach()
                    metrics["n_target_valid"] = torch.as_tensor(
                        float(present_v.numel()), dtype=target_flat.dtype, device=target_flat.device,
                    )

                    # target_skill sufficient stats (head-first; the proper
                    # scoring rule used for selection — see
                    # src/docs/head-metrics.md). Present-weighted CE sum +
                    # present-weighted marginal label mass per entity slot;
                    # supervised_loop turns these into target_dll / target_skill
                    # (gain over predicting the marginal target distribution,
                    # normalised by its entropy). Distinct reference from
                    # target_kl (distance to the per-frame soft label).
                    present_sum = present_v.sum()
                    metrics["targetdist_n"] = present_sum.detach()
                    metrics["targetdist_ce_sum"] = (aux_ce.detach() * present_sum)
                    target_marg = (present_v.unsqueeze(-1) * idx_target).sum(dim=0)
                    for _k in range(target_marg.shape[0]):
                        metrics[f"targetdist_marg_{_k}"] = target_marg[_k].detach()

                    # Multi-candidate KL — KL restricted to frames where the
                    # head genuinely had a choice (#live enemies in obs > 1).
                    # Single-candidate frames are trivial wins for any head
                    # that points at the only available actor; this metric
                    # isolates real discrimination ability.  Requires
                    # entity_types + entity_ids[..., 2] (player_id) in obs.
                    if (
                        obs is not None
                        and isinstance(obs, dict)
                        and "entity_types" in obs
                        and "entity_ids" in obs
                    ):
                        types_t = self._tensor(obs["entity_types"], dtype=torch.long)
                        eids_t = self._tensor(obs["entity_ids"], dtype=torch.long)
                        if types_t.dim() >= 2 and eids_t.dim() >= 3:
                            types_flat = types_t.reshape(-1, types_t.shape[-1])
                            pids_flat = eids_t.reshape(-1, eids_t.shape[-2], eids_t.shape[-1])[..., 2]
                            live_actor = (types_flat == TOKEN_ACTOR) & (pids_flat > 0)
                            n_live = live_actor.sum(dim=-1)
                            multi_mask = (n_live > 1)
                            multi_valid = valid & multi_mask
                            if bool(multi_valid.any().item()):
                                log_probs_m = F.log_softmax(target_flat[multi_valid], dim=-1)
                                present_m = present[multi_valid]
                                idx_target_m = (
                                    idx_dist[multi_valid]
                                    / present_m.clamp(min=1e-6).unsqueeze(-1)
                                )
                                ce_m = -(idx_target_m * log_probs_m).sum(dim=-1)
                                ent_m = -(idx_target_m.clamp(min=1e-8) * idx_target_m.clamp(min=1e-8).log()).sum(dim=-1)
                                kl_m = (
                                    (present_m * (ce_m - ent_m)).sum()
                                    / present_m.sum().clamp(min=1e-6)
                                )
                                metrics["target_kl_multi"] = kl_m.detach()
                                metrics["n_target_valid_multi"] = torch.as_tensor(
                                    float(present_m.numel()),
                                    dtype=target_flat.dtype,
                                    device=target_flat.device,
                                )

        weapon_loss_fn = getattr(getattr(self.model, "weapon_head", None), "weapon_loss", None)
        if weapon_loss_fn is not None and WEAPON_HEAD in logits and WEAPON_HEAD in actions:
            # A head may own its loss (mirrors look_head.look_loss) by
            # exposing a ``weapon_loss`` method. Production heads lack the
            # method, so they fall through to the canonical CE below.
            weapon_loss, _wm = weapon_loss_fn(logits, actions, valid_flat, compute_metrics)
            losses.append(weapon_loss * weights_map.get(WEAPON_HEAD, 1.0))
            loss_is_real.append(True)
            if compute_metrics and _wm:
                metrics.update(_wm)
        elif WEAPON_HEAD in logits and WEAPON_HEAD in actions:
            weapon_logits = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
            weapon_target = self._weapon_target_from_actions(actions)
            # No-weapon frames carry target=-100; F.cross_entropy with
            # ignore_index=-100 skips them on-GPU. Avoid the
            # ``valid.any().item()`` host sync that used to gate the call —
            # syncing per microbatch stalled the ROCm dispatch queue and
            # cost ~10ms per step on the head-probe loop.
            if valid_flat is not None:
                weapon_target = torch.where(
                    valid_flat, weapon_target, torch.full_like(weapon_target, -100)
                )
            valid_weapon = weapon_target >= 0
            weapon_loss = F.cross_entropy(
                weapon_logits, weapon_target, ignore_index=-100, reduction="mean",
            )
            losses.append(weapon_loss * weights_map.get(WEAPON_HEAD, 1.0))
            # Engaged training always has at least one valid weapon frame
            # per microbatch — skip the per-step host sync that previously
            # checked `valid.any().item()`. If you ever train on a corpus
            # where a microbatch could be all-no-weapon, restore the sync
            # or switch to a reduction='sum' / clamped-divisor scheme to
            # avoid the 0/0 → NaN in F.cross_entropy(reduction='mean').
            loss_is_real.append(True)
            if compute_metrics:
                metrics["loss_weapon"] = weapon_loss.detach()
                with torch.no_grad():
                    # Vectorized 8-class confusion matrix: 1 scatter_add
                    # instead of an 8-iteration Python loop with ~10 tensor
                    # ops per iteration. Cuts per-batch weapon-metric kernel
                    # count from ~80 to ~5 — measured ~5-8s/epoch saved at
                    # bs=4096 on this head-probe loop.
                    weapon_probs = F.softmax(weapon_logits, dim=-1)
                    weapon_pred = torch.argmax(weapon_probs, dim=-1)
                    # Map invalid frames to a sentinel out-of-range index
                    # so they don't land in any of the WEAPON_HEAD_SIZE rows.
                    safe_target = torch.where(
                        valid_weapon, weapon_target,
                        torch.full_like(weapon_target, WEAPON_HEAD_SIZE),
                    )
                    safe_pred = torch.where(
                        valid_weapon, weapon_pred,
                        torch.full_like(weapon_pred, WEAPON_HEAD_SIZE),
                    )
                    # Confusion matrix: rows=pred, cols=target, size (K+1)^2.
                    # Last row/col is the "invalid" bucket and is discarded.
                    K = WEAPON_HEAD_SIZE
                    flat_idx = (safe_pred * (K + 1) + safe_target).long()
                    conf = torch.zeros(
                        (K + 1) * (K + 1), dtype=torch.float32, device=weapon_logits.device,
                    )
                    conf.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
                    conf = conf.view(K + 1, K + 1)[:K, :K]  # (K, K), drop invalid bucket
                    # Per-class tp/fp/fn: tp = diag; row sum - tp = fp; col sum - tp = fn.
                    tp_all = conf.diagonal()
                    fp_all = conf.sum(dim=1) - tp_all
                    fn_all = conf.sum(dim=0) - tp_all
                    valid_count = conf.sum()
                    metrics["n_weapon_valid"] = valid_count.detach().to(weapon_logits.dtype)
                    metrics["acc_weapon"] = (tp_all.sum() / valid_count.clamp(min=1.0)).detach()
                    metrics["confidence_weapon"] = weapon_probs.max(dim=-1).values.mean().detach()
                    # Per-class precision / recall / F1 + base rate so the
                    # rare classes (axe / GL / NG / SNG together <10% of
                    # frames) don't disappear into the headline number.
                    class_f1s = []
                    for cls_idx, cls_name in WEAPON_HEAD_CLASS_NAMES:
                        tp = tp_all[cls_idx]
                        fp = fp_all[cls_idx]
                        fn = fn_all[cls_idx]
                        metrics[f"tp_weapon_{cls_name}"] = tp.detach()
                        metrics[f"fp_weapon_{cls_name}"] = fp.detach()
                        metrics[f"fn_weapon_{cls_name}"] = fn.detach()
                        prec = tp / (tp + fp).clamp(min=1.0)
                        rec = tp / (tp + fn).clamp(min=1.0)
                        f1 = 2.0 * prec * rec / (prec + rec).clamp(min=1e-6)
                        metrics[f"precision_weapon_{cls_name}"] = prec.detach()
                        metrics[f"recall_weapon_{cls_name}"] = rec.detach()
                        metrics[f"f1_weapon_{cls_name}"] = f1.detach()
                        metrics[f"pos_rate_weapon_{cls_name}"] = (
                            (tp + fn) / valid_count.clamp(min=1.0)
                        ).detach()
                        class_f1s.append(f1)
                    metrics["f1_weapon"] = torch.stack(class_f1s).mean().detach()

                    # weapon_skill sufficient stats (head-first; the proper
                    # scoring rule used for selection — see
                    # src/docs/head-metrics.md). Clean CE sum + true-class
                    # histogram → marginal entropy; supervised_loop derives
                    # weapon_dll / weapon_skill (fraction of the 8-class
                    # marginal entropy the head eliminates). loss_weapon is the
                    # clean per-frame mean CE, so ce_sum = loss_weapon × n.
                    metrics["weapondist_n"] = valid_count.detach().to(weapon_logits.dtype)
                    metrics["weapondist_ce_sum"] = (
                        weapon_loss.detach() * valid_count
                    ).to(weapon_logits.dtype)
                    for cls_idx, _cls_name in WEAPON_HEAD_CLASS_NAMES:
                        metrics[f"weapondist_h_{cls_idx}"] = (
                            tp_all[cls_idx] + fn_all[cls_idx]
                        ).detach().to(weapon_logits.dtype)

        if MOVE_HEAD in logits and MOVE_HEAD in actions:
            # Move = 3 categorical axes (fb, lr, ud) × 3 classes {neg, none,
            # pos}. Labels are uint8[T, 3] axis class indices from the
            # corpus loader.
            move_logits = logits[MOVE_HEAD]
            move_pred = move_logits.reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
            move_target_t = self._tensor(actions[MOVE_HEAD], dtype=torch.long)
            # Distance-weighted shoulder on the ud (jump) axis. Only sensible
            # when targets arrive in (T, B, 3) form so the conv sees a real
            # time axis; single-step inference falls back to plain CE.
            jump_dist_weight_flat: torch.Tensor | None = None
            if self.jump_distance_sigma > 0.0:
                ud_idx = MOVE_AXIS_NAMES.index("ud")
                if move_target_t.ndim == 3:
                    from qnn.bc.loss_shaping import distance_weighted_neg_weights
                    jump_pos_2d = (move_target_t[..., ud_idx] == MOVE_CLASS_POS).to(torch.float32)
                    valid_2d = valid_mask.bool() if valid_mask is not None else None
                    w_2d = distance_weighted_neg_weights(
                        jump_pos_2d, valid_2d, self.jump_distance_sigma,
                    )
                    jump_dist_weight_flat = w_2d.reshape(-1)
                elif move_target_t.ndim == 2 and "jump_distance_to_pos" in actions:
                    # Flat batch (frame-shuffled SGD). The jump-positive mask
                    # is derived from move[..., ud_idx] == MOVE_CLASS_POS;
                    # the per-frame distance was precomputed at preload time.
                    from qnn.bc.loss_shaping import flat_distance_weight
                    jump_pos_1d = (move_target_t[..., ud_idx] == MOVE_CLASS_POS).to(torch.float32)
                    jump_d = self._tensor(actions["jump_distance_to_pos"], dtype=torch.float32).reshape(-1)
                    jump_dist_weight_flat = flat_distance_weight(
                        jump_d, jump_pos_1d, self.jump_distance_sigma,
                    )
            move_target = move_target_t.reshape(-1, MOVE_AXES)
            base_move_valid = valid_flat if valid_flat is not None else torch.ones(
                (move_target.shape[0],), dtype=torch.bool, device=move_target.device,
            )
            # Per-axis label rewrite. When input_mask is off, every axis
            # uses the raw demo button (usercmd) as the label. When on,
            # axis i's label is the engine OUTCOME = (demo intent) AND
            # (per-direction feasibility from input_mask bits). Under
            # pure-feasibility semantics from the C side:
            #   fb (axis 0): feasibility bits 1-2 are always 1 when alive
            #                (pmove always processes fmove). Label =
            #                demo intent unchanged.
            #   lr (axis 1): same — feasibility bits 3-4 always 1.
            #   ud (axis 2): direction-specific. POS feasibility is bit 7
            #                (jump on ground) OR bit 6 (swim up in water).
            #                NEG feasibility is bit 5 (swim down in water).
            #                Demo intent in MOVE_CLASS_POS that's not
            #                feasible (e.g. air-jump press with no
            #                ground) is rewritten to NONE — the engine
            #                couldn't have honoured that press.
            # No frames dropped; the model trains on every frame against
            # the engine-outcome label.
            move_valid_per_axis: list[torch.Tensor] = [base_move_valid] * MOVE_AXES
            if input_mask_on and input_mask_flat is not None:
                none_t = torch.full_like(move_target[:, 0], MOVE_CLASS_NONE)
                rewritten = move_target.clone()
                # fb / lr: feasibility is always 1 (alive frames) — no
                # rewrite needed; demo intent IS the engine outcome.
                # ud: gate the demo intent through per-direction
                # feasibility.
                up_neg_feas = ((input_mask_flat >> 5) & 1) != 0  # swim down
                up_pos_feas = ((input_mask_flat >> 6) & 1) != 0  # swim up
                jump_feas   = ((input_mask_flat >> 7) & 1) != 0  # ground jump
                ud_pos_feas = jump_feas | up_pos_feas
                ud_intent = move_target[:, 2]
                # POS intent: keep only if pos feasible, else NONE.
                # NEG intent: keep only if neg feasible (swim down), else
                # NONE. NONE intent stays NONE.
                pos_mask = (ud_intent == MOVE_CLASS_POS) & ud_pos_feas
                neg_mask = (ud_intent == MOVE_CLASS_NEG) & up_neg_feas
                rewritten[:, 2] = torch.where(
                    pos_mask,
                    torch.full_like(ud_intent, MOVE_CLASS_POS),
                    torch.where(
                        neg_mask,
                        torch.full_like(ud_intent, MOVE_CLASS_NEG),
                        none_t,
                    ),
                )
                move_target = rewritten
            move_is_real = bool(base_move_valid.any().item())
            # ud (jump) axis is heavily imbalanced (~4% pos rate); upweight
            # the POS class via jump_pos_weight when set above 1.0.  fb/lr
            # are balanced enough that plain CE works.
            ud_class_weight = None
            if self.jump_pos_weight != 1.0:
                ud_class_weight = torch.tensor(
                    [1.0, 1.0, float(self.jump_pos_weight)],
                    dtype=move_pred.dtype, device=move_pred.device,
                )
            ce_per_axis = []
            for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                axis_valid = move_valid_per_axis[axis_i]
                axis_pred = move_pred[axis_valid, axis_i, :]
                axis_target = move_target[axis_valid, axis_i]
                axis_is_real = axis_pred.shape[0] > 0
                if not axis_is_real:
                    ce_axis = torch.zeros((), dtype=move_pred.dtype, device=move_pred.device)
                elif axis_name == "ud":
                    if jump_dist_weight_flat is not None:
                        # Per-frame CE then multiplicative distance weight,
                        # matching the attack-head .mean() reduction so both
                        # heads' loss magnitudes scale the same way.
                        ce_pf = F.cross_entropy(
                            axis_pred, axis_target,
                            weight=ud_class_weight, reduction="none",
                        )
                        ce_axis = (ce_pf * jump_dist_weight_flat[axis_valid]).mean()
                    else:
                        ce_axis = F.cross_entropy(
                            axis_pred, axis_target, weight=ud_class_weight, reduction="mean",
                        )
                else:
                    ce_axis = F.cross_entropy(axis_pred, axis_target, reduction="mean")
                ce_per_axis.append(ce_axis)
            move_loss = torch.stack(ce_per_axis).mean()  # equal-weight axes
            losses.append(move_loss * weights_map.get(MOVE_HEAD, 1.0))
            loss_is_real.append(move_is_real)
            if compute_metrics:
                metrics["loss_move"] = move_loss.detach()
                if move_is_real:
                    with torch.no_grad():
                        # Per-axis argmax computed once; per-axis indexing
                        # below selects each axis's valid frames separately
                        # so op_input-masked axes drop their stale frames.
                        move_argmax_all = torch.argmax(move_pred, dim=-1)  # (B, 3)
                        per_axis_acc: list[torch.Tensor] = []
                        per_axis_macro_f1 = []
                        for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                            axis_valid = move_valid_per_axis[axis_i]
                            metrics[f"loss_move_{axis_name}"] = ce_per_axis[axis_i].detach()
                            pred_axis = move_argmax_all[axis_valid, axis_i]
                            true_axis = move_target[axis_valid, axis_i]
                            if pred_axis.numel() > 0:
                                axis_acc = (pred_axis == true_axis).float().mean()
                            else:
                                axis_acc = torch.zeros((), dtype=move_pred.dtype, device=move_pred.device)
                            metrics[f"acc_move_{axis_name}"] = axis_acc.detach()
                            per_axis_acc.append(axis_acc)
                            # Per-class precision / recall / F1 across all three
                            # classes (neg/none/pos).  Macro-F1 per axis is the
                            # honest single-axis summary that doesn't hide the
                            # rare-class failure modes (jump under ud, backpedal
                            # under fb) behind the dominant "none" class.
                            class_f1s = []
                            for cls_idx, cls_name in ((MOVE_CLASS_NEG, "neg"),
                                                      (MOVE_CLASS_NONE, "none"),
                                                      (MOVE_CLASS_POS, "pos")):
                                pred_cls = pred_axis == cls_idx
                                true_cls = true_axis == cls_idx
                                tp = (pred_cls & true_cls).sum().float()
                                fp = (pred_cls & ~true_cls).sum().float()
                                fn = (~pred_cls & true_cls).sum().float()
                                prec = tp / (tp + fp).clamp(min=1.0)
                                rec = tp / (tp + fn).clamp(min=1.0)
                                f1 = 2.0 * prec * rec / (prec + rec).clamp(min=1e-6)
                                metrics[f"precision_move_{axis_name}_{cls_name}"] = prec.detach()
                                metrics[f"recall_move_{axis_name}_{cls_name}"] = rec.detach()
                                metrics[f"f1_move_{axis_name}_{cls_name}"] = f1.detach()
                                if true_cls.numel() > 0:
                                    metrics[f"pos_rate_move_{axis_name}_{cls_name}"] = true_cls.float().mean().detach()
                                else:
                                    metrics[f"pos_rate_move_{axis_name}_{cls_name}"] = torch.zeros(
                                        (), dtype=move_pred.dtype, device=move_pred.device,
                                    )
                                class_f1s.append(f1)
                            macro = torch.stack(class_f1s).mean()
                            metrics[f"f1_move_{axis_name}"] = macro.detach()
                            per_axis_macro_f1.append(macro)
                        # Equal-axes overall acc/F1 — the mean-of-per-axis
                        # form is identical to the original ``argmax ==
                        # target`` mean when all axes share one valid mask
                        # (i.e. input_mask off), and remains
                        # well-defined when per-axis valids differ.
                        metrics["acc_move"] = torch.stack(per_axis_acc).mean().detach()
                        metrics["f1_move"] = torch.stack(per_axis_macro_f1).mean().detach()
                        # Distributional human-likeness sufficient statistics
                        # (additive; combined at epoch end into move_dll /
                        # move_kl_joint / move_kl_marg / jump calibration — see
                        # qnn.bc.supervised_loop._move_distribution_metrics_from_sums).
                        # f1_move / acc_move reward argmax point-accuracy and are
                        # blind to whether the model reproduces (a) the human
                        # per-axis class marginals, (b) the joint fb/lr/ud combo
                        # histogram, and (c) the ~4% jump rate. These score the
                        # predicted DISTRIBUTION instead. Emitted as scalars (the
                        # metric flush stacks raw-sum metrics into one tensor, so
                        # per-class / per-combo vectors are unrolled); prefix
                        # movedist_ is registered raw-sum in supervised_loop.
                        # The joint uses the all-axes-shared valid mask
                        # (base_move_valid) so the 27-combo histogram is well
                        # defined; move_target is the engine-outcome label (ud
                        # rewritten under input_mask), matching the loss.
                        jv = base_move_valid
                        njv = int(jv.sum().item())
                        if njv > 0:
                            pf = torch.softmax(move_pred[jv], dim=-1)      # (V,3,3)
                            logpf = torch.log_softmax(move_pred[jv], dim=-1)
                            tj = move_target[jv]                            # (V,3)
                            ar = torch.arange(njv, device=move_pred.device)
                            metrics["movedist_n"] = move_pred.new_tensor(float(njv))
                            for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                                # per-axis model NLL sum (for move_dll) + human
                                # class counts + summed model probs (model marginal).
                                metrics[f"movedist_ce_{axis_name}"] = (
                                    -logpf[ar, axis_i, tj[:, axis_i]].sum()
                                ).detach()
                                hist_a = torch.bincount(
                                    tj[:, axis_i], minlength=MOVE_AXIS_CLASSES,
                                ).to(move_pred.dtype)
                                pred_a = pf[:, axis_i, :].sum(0)            # (3,)
                                for c in range(MOVE_AXIS_CLASSES):
                                    metrics[f"movedist_h_{axis_name}_{c}"] = hist_a[c].detach()
                                    metrics[f"movedist_p_{axis_name}_{c}"] = pred_a[c].detach()
                            # Joint combo histogram, combo = fb + 3*lr + 9*ud
                            # (27 bins). Human = counts; model = expected combo
                            # mass = sum_frames outer(P_fb, P_lr, P_ud) — the
                            # model is per-frame axis-independent, so this is its
                            # implied joint aggregated over the feature stream.
                            combo_idx = tj[:, 0] + 3 * tj[:, 1] + 9 * tj[:, 2]
                            jh = torch.bincount(combo_idx, minlength=27).to(move_pred.dtype)
                            # einsum gives E[i,j,k]=[fb,lr,ud]; a plain reshape
                            # would be C-order (fb*9+lr*3+ud), but jh is binned
                            # fb+3*lr+9*ud. permute(2,1,0)→[ud,lr,fb] so the
                            # C-order reshape lands on the SAME combo index as
                            # jh (else move_kl_joint compares scrambled bins).
                            jp = torch.einsum(
                                "vi,vj,vk->ijk", pf[:, 0, :], pf[:, 1, :], pf[:, 2, :],
                            ).permute(2, 1, 0).reshape(27)
                            for m in range(27):
                                metrics[f"movedist_jh_{m}"] = jh[m].detach()
                                metrics[f"movedist_jp_{m}"] = jp[m].detach()
                            # ud argmax-pos count: the "jump collapse" number. The
                            # expected pos prob (movedist_p_ud_2 / n) is the
                            # calibration counterpart — a calibrated jump_pos_weight
                            # =1.0 head has expected≈human while argmax≈0.
                            ud_i = MOVE_AXIS_NAMES.index("ud")
                            am_pos = (
                                move_pred[jv][:, ud_i, :].argmax(-1) == MOVE_CLASS_POS
                            ).sum().to(move_pred.dtype)
                            metrics["movedist_ampos_ud"] = am_pos.detach()

        if ATTACK_HEAD in logits and ATTACK_HEAD in actions:
            attack_logits = logits[ATTACK_HEAD]
            attack_target_t = self._tensor(actions[ATTACK_HEAD], dtype=torch.float32)
            # Two distance-shoulder paths exist:
            #
            # 1. Sequence path (ndim==2, lane-packed pipeline): compute
            #    weights via Conv1d on the (T, B) target stream so each
            #    frame sees its time-axis neighbors.
            # 2. Flat path (ndim==1, GPU-resident frame-shuffled SGD):
            #    no time axis exists in the batch, so we use a
            #    per-frame "distance to nearest positive in same episode"
            #    that was precomputed at preload time and shipped via
            #    actions["attack_distance_to_pos"].
            #
            # Both produce the same loss semantics; only the
            # convolution/precompute boundary moves.
            distance_weight_flat: torch.Tensor | None = None
            if self.attack_distance_sigma > 0.0:
                if attack_target_t.ndim == 2:
                    from qnn.bc.loss_shaping import distance_weighted_neg_weights
                    valid_2d = valid_mask.bool() if valid_mask is not None else None
                    w_2d = distance_weighted_neg_weights(
                        attack_target_t, valid_2d, self.attack_distance_sigma,
                    )
                    distance_weight_flat = w_2d.reshape(-1)
                elif attack_target_t.ndim == 1 and "attack_distance_to_pos" in actions:
                    from qnn.bc.loss_shaping import flat_distance_weight
                    attack_d = self._tensor(actions["attack_distance_to_pos"], dtype=torch.float32)
                    distance_weight_flat = flat_distance_weight(
                        attack_d.reshape(-1), attack_target_t.reshape(-1),
                        self.attack_distance_sigma,
                    )

            attack_pred_full = attack_logits.reshape(-1)
            attack_target_full = attack_target_t.reshape(-1)
            attack_dw_full = distance_weight_flat
            # Optional +1 op-frame shifted loss target. Built per-episode at
            # preload by qnn.bc.supervised_loop._compute_attack_shifted and
            # surfaced as actions["attack_shifted"]. Only the BCE *target*
            # shifts; metrics below stay on the original attack label.
            attack_loss_target_full: torch.Tensor | None = None
            if (
                self.attack_label_shift
                and "attack_shifted" in actions
            ):
                attack_loss_target_full = self._tensor(
                    actions["attack_shifted"], dtype=torch.float32,
                ).reshape(-1)
            if valid_flat is not None:
                attack_pred_full = attack_pred_full[valid_flat]
                attack_target_full = attack_target_full[valid_flat]
                if attack_dw_full is not None:
                    attack_dw_full = attack_dw_full[valid_flat]
                if attack_loss_target_full is not None:
                    attack_loss_target_full = attack_loss_target_full[valid_flat]
            # Label rewrite under input_mask. Off: label is the raw demo
            # button (usercmd, move byte bit 6). On: label becomes the
            # engine OUTCOME = pure feasibility (input_mask bit 0) AND
            # the demo's actual press (current attack_target_full, the
            # usercmd attack bit). Feasibility is "would W_Attack fire
            # if button0=1 right now"; AND with demo press recovers
            # "did W_Attack actually fire this tick".
            if input_mask_on and input_mask_flat is not None:
                input_mask_full = input_mask_flat
                if valid_flat is not None:
                    input_mask_full = input_mask_full[valid_flat]
                feasibility = (input_mask_full & 1).to(attack_target_full.dtype)
                demo_press  = attack_target_full
                attack_target_full = feasibility * demo_press
                if attack_loss_target_full is not None:
                    # Apply the same feasibility AND to the shifted target
                    # so the loss label remains an engine-outcome bit when
                    # input_mask is on.
                    attack_loss_target_full = feasibility * attack_loss_target_full
            attack_pred = attack_pred_full
            attack_target = attack_target_full
            # attack_loss_target = label fed to BCE; defaults to the metric
            # target (current attack label) so behavior is bit-identical
            # when attack_label_shift is off.
            attack_loss_target = (
                attack_loss_target_full
                if attack_loss_target_full is not None
                else attack_target_full
            )
            attack_dw = attack_dw_full
            attack_is_real = attack_target.numel() > 0
            # pos_weight conventionally lives in class_weights[ATTACK_HEAD] (set
            # at training startup from corpus statistics: neg_count/pos_count).
            pos_weight: torch.Tensor | None = None
            if class_weights is not None and ATTACK_HEAD in class_weights:
                cw = class_weights[ATTACK_HEAD]
                pos_weight = cw if isinstance(cw, torch.Tensor) else torch.as_tensor(cw, device=attack_pred.device)
            if attack_is_real:
                # Unified path: per-frame BCE, weighted by (op? * focal? *
                # distance?), then weighted-mean reduction. When
                # attack_op_only is on, the op mask (from input_mask bit 0)
                # zeroes gradient on no-op frames — those rows can't
                # actuate at inference (engine ignores fire during
                # cooldown). When attack_op_only is off (default), op=0
                # frames stay in: their label is feasibility AND demo_press
                # (forced to 0), the BCE pulls predictions toward 0 there,
                # and pos_weight is computed across the full corpus.
                bce = F.binary_cross_entropy_with_logits(
                    attack_pred, attack_loss_target,
                    pos_weight=pos_weight, reduction="none",
                )
                weight = torch.ones_like(bce)
                op_full: torch.Tensor | None = None
                if input_mask_on and input_mask_flat is not None:
                    op_full = (input_mask_flat & 1).to(bce.dtype)
                    if valid_flat is not None:
                        op_full = op_full[valid_flat]
                if self.attack_op_only and op_full is not None:
                    weight = weight * op_full
                if self.attack_focal_gamma > 0.0:
                    p = torch.sigmoid(attack_pred)
                    pt = torch.where(attack_loss_target > 0.5, p, 1.0 - p)
                    alpha_t = torch.where(
                        attack_loss_target > 0.5,
                        torch.full_like(p, self.attack_focal_alpha),
                        torch.full_like(p, 1.0 - self.attack_focal_alpha),
                    )
                    weight = weight * alpha_t * (1.0 - pt).clamp(min=1e-6) ** self.attack_focal_gamma
                if attack_dw is not None:
                    weight = weight * attack_dw
                attack_loss = (weight * bce).sum() / weight.sum().clamp(min=1.0)
            else:
                attack_loss = torch.zeros((), dtype=attack_logits.dtype, device=attack_logits.device)
            losses.append(attack_loss * weights_map.get(ATTACK_HEAD, 1.0))
            loss_is_real.append(attack_is_real)
            if compute_metrics:
                metrics["loss_attack"] = attack_loss.detach()
                # Single fire f1, computed against whatever label the
                # input_mask flag selected. Off → usercmd label; on →
                # engine-outcome label. No separate ``*_masked`` metric
                # — there's only one label per run now, so the metric
                # is unambiguous.
                if attack_target.numel() > 0:
                    with torch.no_grad():
                        pred_pos = (torch.sigmoid(attack_pred) > 0.5)
                        target_pos = attack_target > 0.5
                        # When training is op-only, the metric is also
                        # op-only: op=0 frames had no gradient and their
                        # model predictions are uncalibrated noise. Score
                        # only frames where the head's decision matters.
                        if (self.attack_op_only and input_mask_on
                                and input_mask_flat is not None):
                            op_full = (input_mask_flat & 1).to(torch.bool)
                            if valid_flat is not None:
                                op_full = op_full[valid_flat]
                            pred_pos = pred_pos & op_full
                            target_pos = target_pos & op_full
                            tp = (pred_pos & target_pos).sum()
                            fp = (pred_pos & ~target_pos & op_full).sum()
                            fn = (~pred_pos & target_pos & op_full).sum()
                            tn = (~pred_pos & ~target_pos & op_full).sum()
                        else:
                            tp = (pred_pos & target_pos).sum()
                            fp = (pred_pos & ~target_pos).sum()
                            fn = (~pred_pos & target_pos).sum()
                            tn = (~pred_pos & ~target_pos).sum()
                        metrics["tp_attack"] = tp.detach()
                        metrics["fp_attack"] = fp.detach()
                        metrics["fn_attack"] = fn.detach()
                        metrics["tn_attack"] = tn.detach()
                        n_total = tp + fp + fn + tn
                        metrics["acc_attack"] = ((tp + tn).float() / n_total.clamp(min=1)).detach()
                        prec_denom = (tp + fp).clamp(min=1)
                        rec_denom = (tp + fn).clamp(min=1)
                        prec = tp.float() / prec_denom
                        rec = tp.float() / rec_denom
                        f1_denom = (prec + rec).clamp(min=1e-6)
                        metrics["precision_attack"] = prec.detach()
                        metrics["recall_attack"] = rec.detach()
                        metrics["f1_attack"] = (2.0 * prec * rec / f1_denom).detach()

                        # attack_skill sufficient stats (head-first; the proper
                        # scoring rule used for selection — see
                        # src/docs/head-metrics.md). A CLEAN, unweighted BCE
                        # (no pos_weight / focal / distance weighting — those
                        # shape the gradient, not the likelihood) on the scored
                        # frames, plus the positive count for the base-rate
                        # binary entropy. supervised_loop derives attack_dll /
                        # attack_skill. Scored population mirrors the f1 above:
                        # op-only when attack_op_only + input_mask, else all
                        # valid frames.
                        clean_bce = F.binary_cross_entropy_with_logits(
                            attack_pred, attack_loss_target, reduction="none",
                        )
                        scored = torch.ones_like(attack_loss_target, dtype=torch.bool)
                        if (self.attack_op_only and input_mask_on
                                and input_mask_flat is not None):
                            _op_scored = (input_mask_flat & 1).to(torch.bool)
                            if valid_flat is not None:
                                _op_scored = _op_scored[valid_flat]
                            scored = _op_scored
                        metrics["attackdist_ce_sum"] = (clean_bce * scored).sum().detach()
                        metrics["attackdist_n"] = scored.sum().to(clean_bce.dtype).detach()
                        metrics["attackdist_pos"] = (
                            (attack_loss_target > 0.5) & scored
                        ).sum().to(clean_bce.dtype).detach()
                # Diagnostics for the prior-residual decomposition.
                # mean/std of the prior and delta logits across the same
                # frames the loss sees — answers "is the residual
                # actually doing anything?" Skipped when the prior is
                # off ("_attack_prior" absent / zeros).
                if "_attack_prior" in logits and "_attack_delta" in logits and attack_is_real:
                    with torch.no_grad():
                        prior_full = logits["_attack_prior"].reshape(-1)
                        delta_full = logits["_attack_delta"].reshape(-1)
                        if valid_flat is not None:
                            prior_full = prior_full[valid_flat]
                            delta_full = delta_full[valid_flat]
                        metrics["attack_prior_mean"] = prior_full.mean().detach()
                        metrics["attack_prior_std"] = prior_full.std().detach()
                        metrics["attack_delta_mean"] = delta_full.mean().detach()
                        metrics["attack_delta_std"] = delta_full.std().detach()

        if LOOK_HEAD in logits and LOOK_HEAD in actions:
            # Magnitude-sensitive supervision: regress the raw look_delta
            # output against the geometric residual (demo_unit - look_prior).
            # The residual has bounded magnitude (≤ 2 for unit vectors);
            # forces the head to "pay" the right magnitude for whatever
            # direction it expresses, instead of growing delta arbitrarily
            # large to override the prior.  Quality is tracked via look_r2 /
            # look_ewa_deg (see _emit_look_tangent_sums); the loss is unchanged.
            look_pred = logits[LOOK_HEAD].reshape(-1, LOOK_HEAD_SIZE)
            look_prior = logits["_look_prior"].reshape(-1, LOOK_HEAD_SIZE)
            look_delta = logits["_look_delta"].reshape(-1, LOOK_HEAD_SIZE)

            look_label_raw_t = self._tensor(actions[LOOK_HEAD], dtype=torch.float32)
            look_label_raw = look_label_raw_t.reshape(-1, look_label_raw_t.shape[-1])
            target_norm = torch.linalg.vector_norm(look_label_raw, dim=-1, keepdim=True)
            valid = target_norm.squeeze(-1) > 1e-6
            if valid_flat is not None:
                valid = valid & valid_flat
            aux_is_real = bool(valid.any().item())
            # Bench look heads may carry their own loss (binned/polar/vMF), so a new
            # loss form needs NO canonical change: the head reads its outputs from
            # `logits` (forwarded generically by Network) and returns (loss, metrics).
            look_loss_fn = getattr(getattr(self.model, "look_head", None), "look_loss", None)
            if aux_is_real:
                look_label = look_label_raw[valid] / target_norm[valid].clamp(min=1e-6)
                if look_loss_fn is not None:
                    look_loss, _look_metrics = look_loss_fn(
                        logits, look_label, valid, compute_metrics,
                    )
                    if compute_metrics:
                        metrics.update(_look_metrics)
                elif "_look_bins" in logits:
                    # Binned (classification) look head: per-axis cross-entropy
                    # over foveated tangent bins instead of smooth_l1 on the
                    # delta. look_predict (decoded direction) still drives the
                    # look_r2 metric. See qnn.model.look_bins.
                    from qnn.model.look_bins import (
                        N_BINS as _LOOK_N_BINS,
                        bin_targets as _look_bin_targets,
                        soft_bin_targets as _look_soft_targets,
                        tangent_logmap as _look_logmap,
                    )
                    bins = logits["_look_bins"].reshape(-1, 2, _LOOK_N_BINS)[valid]
                    z_look = _look_logmap(look_label)                      # (V, 2) tangent
                    tgt = _look_bin_targets(z_look)                        # (V, 2) long
                    if self.look_label_smoothing_sigma > 0.0:
                        # Distance-aware Gaussian soft-target CE: per axis,
                        # -(soft * log_softmax(logits)).sum over bins, mean over
                        # rows, summed over the 2 axes — same reduction as the
                        # hard two-CE path below.
                        soft = _look_soft_targets(z_look, self.look_label_smoothing_sigma)
                        logp = F.log_softmax(bins, dim=-1)                 # (V, 2, N_BINS)
                        look_loss = -(soft * logp).sum(dim=-1).mean(dim=0).sum()
                    else:
                        look_loss = (
                            F.cross_entropy(bins[:, 0, :], tgt[:, 0])
                            + F.cross_entropy(bins[:, 1, :], tgt[:, 1])
                        )
                    if compute_metrics:
                        # Human-likeness sufficient statistics (additive; combined at
                        # epoch end into look_dll / look_emd_deg — see
                        # qnn.bc.look_metrics.humanlike_from_sums). The binned head's
                        # PRODUCT is its per-axis bin DISTRIBUTION, so we score the
                        # distribution, not the decoded mean. look_r2 / look_ewa_deg
                        # (mean-fidelity proxies) are intentionally NOT emitted for the
                        # binned head — for human-likeness they reward mean-regression.
                        # Emit as SCALARS only — the metric flush stacks all raw-sum
                        # metrics into one tensor (qnn.bc.supervised_loop), so per-bin
                        # vectors must be unrolled. Combined into look_dll / look_emd_deg
                        # at epoch end (and these keys popped). Prefix: lookdist_.
                        with torch.no_grad():
                            logp = torch.log_softmax(bins, dim=-1)          # (V,2,N_BINS)
                            p = logp.exp()
                            V = bins.shape[0]
                            ar = torch.arange(V, device=bins.device)
                            metrics["lookdist_n"] = bins.new_tensor(float(V))
                            for a in (0, 1):
                                metrics[f"lookdist_ce_{a}"] = (-logp[ar, a, tgt[:, a]].sum()).detach()
                                hist_a = torch.bincount(tgt[:, a], minlength=_LOOK_N_BINS).to(bins.dtype)
                                pred_a = p[:, a, :].sum(0)                  # (N_BINS,)
                                for b in range(_LOOK_N_BINS):
                                    metrics[f"lookdist_h_{a}_{b}"] = hist_a[b].detach()
                                    metrics[f"lookdist_p_{a}_{b}"] = pred_a[b].detach()
                else:
                    # Target residual: what delta should be to make
                    # normalize(base + delta) = look_label.
                    look_residual = look_label - look_prior[valid]
                    look_loss = F.smooth_l1_loss(
                        look_delta[valid], look_residual, beta=0.05, reduction="mean",
                    )
            else:
                look_loss = torch.zeros((), dtype=look_pred.dtype, device=look_pred.device)
            losses.append(look_loss * weights_map.get(LOOK_HEAD, 1.0))
            loss_is_real.append(aux_is_real)
            if compute_metrics:
                metrics["loss_look"] = look_loss.detach()
                # Regression-only diagnostics (delta magnitude + tangent-space
                # look_r2 sums). Skipped when the head carries its own loss
                # (hook path) or is binned — those emit their own metrics.
                if aux_is_real and look_loss_fn is None and "_look_bins" not in logits:
                    with torch.no_grad():
                        # Track delta magnitude so we can confirm the head
                        # is no longer growing it unbounded.
                        metrics["mag_delta_look"] = (
                            torch.linalg.vector_norm(look_delta[valid], dim=-1).mean().detach()
                        )
                        # Smooth, non-saturated look metrics (look_r2,
                        # look_ewa_deg) via tangent-space sufficient statistics,
                        # combined at epoch end in supervised_loop. Plain
                        # cos_sim is NOT tracked: it saturates near 1.0 (no-turn
                        # = (1,0,0)) and drifts with label recollections, so it
                        # can't separate or compare models. See qnn.bc.look_metrics.
                        # Mean-fidelity sums (look_r2 / look_ewa_deg) only for the
                        # regression head; the binned head emits lookdist_* instead.
                        if "_look_bins" not in logits:
                            self._emit_look_tangent_sums(
                                metrics, look_pred[valid], look_label,
                            )

        if compute_metrics:
            metrics["accuracy"] = (
                torch.stack(accuracy_components).mean()
                if accuracy_components
                else torch.zeros((), device=self.device)
            )
        else:
            metrics["accuracy"] = torch.zeros((), device=self.device)

        return losses, loss_is_real, metrics

    def _weapon_target_from_actions(
        self,
        actions: Mapping[str, np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        """Return dense desired-weapon targets from collected BC labels.

        The collector stores `weapon` as the raw engine weapon byte:
          0 = no weapon held (pre-spawn / dead / transitional),
          1..8 = Quake weapon id in impulse order (axe..thunderbolt).
        The 8-class weapon head trains on weapons only; no-weapon frames
        map to -100 so F.cross_entropy(..., ignore_index=-100) skips
        them while their move/fire/look labels still train.
        """
        weapon = self._tensor(actions[WEAPON_HEAD], dtype=torch.long).reshape(-1)
        bad = (weapon < 0) | (weapon > WEAPON_HEAD_SIZE)
        if bool(bad.any().item()):
            sample = weapon[bad][:8].detach().cpu().tolist()
            raise ValueError(
                f"weapon bytes must be in 0..{WEAPON_HEAD_SIZE}, got {sample}"
            )
        # 1..8 → class 0..7; 0 (no weapon) → -100 ignore.
        target = weapon - 1
        target = target.masked_fill(weapon == 0, -100)
        return target

    def supervised_step(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor],
        lr: float,
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        accumulate_only: bool = False,
        head_loss_weights: Mapping[str, float] | None = None,
        loss_scale: float = 1.0,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        optimizer = self._optimizer("bc", self.model.parameters(), lr)
        if not accumulate_only:
            optimizer.zero_grad()

        # Bench side-channel contexts (no-op for the canonical model, which
        # passes no provider). The bench provider enters label-derived
        # engagement_ema / target-supervision scopes.
        with (
            self._autocast(),
            self._side_channel_provider(actions, masks),
        ):
            _, logits, _, next_hidden, target_logits = self._forward_tensors(
                obs,
                hidden=hidden,
                masks=masks,
            )
            valid_mask = (
                self._tensor(masks["valid_mask"], dtype=torch.bool)
                if isinstance(masks, Mapping) and "valid_mask" in masks
                else None
            )
            losses, loss_is_real, metrics = self._compute_head_losses_and_metrics(
                logits,
                actions,
                class_weights=class_weights,
                head_loss_weights=head_loss_weights,
                compute_metrics=compute_metrics,
                target_logits=target_logits,
                obs=obs,
                valid_mask=valid_mask,
            )
            real = [l for l, r in zip(losses, loss_is_real) if r]
            loss = torch.stack(real).mean() if real else torch.zeros((), device=self.device)
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
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        head_loss_weights: Mapping[str, float] | None = None,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        with (
            torch.inference_mode(),
            self._autocast(),
            self._side_channel_provider(actions, masks),
        ):
            _, logits, _, next_hidden, target_logits = self._forward_tensors(
                obs,
                hidden=hidden,
                masks=masks,
            )
            valid_mask = (
                self._tensor(masks["valid_mask"], dtype=torch.bool)
                if isinstance(masks, Mapping) and "valid_mask" in masks
                else None
            )
            losses, loss_is_real, metrics = self._compute_head_losses_and_metrics(
                logits,
                actions,
                head_loss_weights=head_loss_weights,
                compute_metrics=compute_metrics,
                target_logits=target_logits,
                obs=obs,
                valid_mask=valid_mask,
            )
        real = [l for l, r in zip(losses, loss_is_real) if r]
        metrics["loss"] = torch.stack(real).mean() if real else torch.zeros((), device=self.device)
        metrics["_next_hidden"] = next_hidden.detach()
        return metrics

    def ppo_step(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        del args, kwargs
        raise RuntimeError("Combat-objective phase 1 does not support PPO")

    def value_step(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        del args, kwargs
        raise RuntimeError("Combat-objective phase 1 has no value head")

    def save(self, path: str | Path) -> None:
        from qnn.contracts import current_contract

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        model_cfg = self.config.to_dict()
        meta = {
            "obs_dim": self.obs_dim,
            "model": model_cfg,
            # Declarative assembly spec (None for legacy flat/factory
            # checkpoints). When present, loaders rebuild the module via
            # qnn.model.graph.build_network — the checkpoint is fully
            # self-describing and no probe.json / factory rehydration runs.
            "model_graph": self.graph.to_dict() if self.graph is not None else None,
            # Self-versioning contract block (model↔engine). wire/semantics are
            # the LIVE engine_norm ids (what this code produces); arch is derived
            # from the ModelConfig. The checkpoint is the source of truth — the
            # exporter stamps these exact ids into the ONNX, nothing is inferred
            # from the graph or the filename. See qnn.contracts / src/docs/contracts.
            "contract": current_contract(model_cfg),
            "jump_pos_weight": self.jump_pos_weight,
            "attack_focal_gamma": self.attack_focal_gamma,
            "attack_focal_alpha": self.attack_focal_alpha,
            "attack_distance_sigma": self.attack_distance_sigma,
            "jump_distance_sigma": self.jump_distance_sigma,
            "look_label_smoothing_sigma": self.look_label_smoothing_sigma,
            # input_mask is a training-time label-rewrite toggle (not a
            # ModelConfig field) — persist it so downstream eval / final
            # val pass / PPO seed selection don't silently revert to
            # raw-demo-press labels when reloading a checkpoint that was
            # trained against the masked-outcome labels.
            "input_mask": bool(self.input_mask),
            "attack_op_only": bool(self.attack_op_only),
            "backend": "pytorch",
            "requested_device": self.device_spec.requested,
            "resolved_device": self.device_spec.resolved,
            "accelerator_backend": self.device_spec.backend,
        }
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
    def load(
        cls,
        path: str | Path,
        *,
        device: str,
        model_factory: Callable[[int, ModelConfig], nn.Module] | None = None,
    ) -> "QNNPolicy":
        """Load a saved checkpoint.

        ``model_factory`` mirrors the constructor hook: when None, the
        canonical ``Network`` is built from the saved
        ModelConfig and strict-loaded; when set, the factory builds the
        alternate module (e.g. a head-probe model) and the state_dict
        is loaded into it. The caller is responsible for passing the
        same factory used to train the checkpoint — checkpoints don't
        embed the factory identity.
        """
        source = Path(path)
        payload = trusted_torch_load(source, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload or "meta" not in payload:
            raise ValueError(f"Unrecognised checkpoint format: {source}")
        meta = dict(payload["meta"])
        if "model" not in meta:
            from qnn.utils.checkpoint_converter import migrate_legacy_flat_meta
            migrated = migrate_legacy_flat_meta(meta)
            if migrated is None:
                raise ValueError(
                    f"Checkpoint {source} is missing the 'model' arch block "
                    "and migrate_legacy_flat_meta did not recognize the schema."
                )
            meta = migrated
        # Graph-described checkpoints rebuild through the declarative
        # assembly path; flat/legacy checkpoints stay on the ModelConfig
        # path with the state-dict migrations below. An embedded graph
        # ALWAYS wins — even over a passed model_factory (a legacy
        # probe.json sitting next to a graph checkpoint must not demote
        # it back to factory-dependent), and the ModelConfig bridge is
        # re-derived from the graph so meta.model can never silently
        # diverge from the module actually built.
        graph = None
        if meta.get("model_graph") is not None:
            from qnn.model.graph import GraphSpec

            graph = GraphSpec.from_dict(meta["model_graph"])
            model_factory = None
        model_cfg = None if graph is not None else ModelConfig.from_dict(meta["model"])
        policy = cls(
            obs_dim=int(meta["obs_dim"]),
            model=model_cfg,
            graph=graph,
            jump_pos_weight=float(meta["jump_pos_weight"]),
            attack_focal_gamma=float(meta["attack_focal_gamma"]),
            attack_focal_alpha=float(meta["attack_focal_alpha"]),
            attack_distance_sigma=float(meta["attack_distance_sigma"]),
            jump_distance_sigma=float(meta["jump_distance_sigma"]),
            look_label_smoothing_sigma=float(meta.get("look_label_smoothing_sigma", 0.0)),
            seed=0,
            device=device,
            model_factory=model_factory,
        )
        # Restore the input_mask label-rewrite toggle (default False for
        # pre-fix checkpoints that didn't carry the field).
        policy.input_mask = bool(meta.get("input_mask", False))
        policy.attack_op_only = bool(meta.get("attack_op_only", False))
        if model_factory is None and graph is None:
            from qnn.utils.checkpoint_converter import (
                migrate_drop_action_history,
                migrate_drop_fire_align_scalar,
                migrate_drop_weapon_embed_self,
                migrate_entity_embed,
                migrate_hoist_encoder_obs_embedding,
                migrate_obs_embedding_self_token_builder,
                migrate_rename_fire_head_to_attack_head,
                migrate_rename_tokenizer_to_obs_embedding,
                migrate_rename_trunk_to_encoder,
                migrate_self_attack_finished_scalar,
                migrate_self_scalars,
                migrate_v17_move_heads,
                migrate_wrap_gru_in_temporal,
                migrate_wrap_heads_in_components,
            )

            migrate_rename_trunk_to_encoder(payload["state_dict"])
            migrate_rename_tokenizer_to_obs_embedding(payload["state_dict"])
            migrate_hoist_encoder_obs_embedding(payload["state_dict"])
            migrate_entity_embed(payload["state_dict"])
            migrate_self_scalars(payload["state_dict"])
            migrate_self_attack_finished_scalar(payload["state_dict"])
            migrate_obs_embedding_self_token_builder(payload["state_dict"])
            migrate_v17_move_heads(payload["state_dict"])
            migrate_drop_action_history(payload["state_dict"])
            # fire_head→attack_head runs BEFORE migrate_drop_fire_align_scalar
            # so the latter sees the new ``attack_head.*`` layout.
            migrate_rename_fire_head_to_attack_head(payload["state_dict"])
            migrate_drop_fire_align_scalar(payload["state_dict"])
            migrate_drop_weapon_embed_self(payload["state_dict"])
            # gru → temporal.gru and heads → component-wrapped layout.
            # Run LAST so all prior renames have settled.
            migrate_wrap_gru_in_temporal(payload["state_dict"])
            migrate_wrap_heads_in_components(payload["state_dict"])
        # When model_factory is set the saved state_dict is for a
        # probe-built module (e.g. a Network with slot overrides from
        # qnn.model.bench), not the canonical Network — the legacy
        # migrations don't apply and the strict-load below uses empty
        # allow-prefixes.
        try:
            # strict=False so v17 checkpoints still load:
            #  - migrate_v17_move_heads packs split fb/lr into the unified
            #    move_head and bias-locks the ud axis (no random init)
            #  - migrate_drop_action_history strips the pre-rip-out
            #    action_proj / action_pos_embed weights and truncates
            #    kind_embed from 4 -> 3 rows
            #  - migrate_drop_fire_align_scalar trims the trailing
            #    alignment-scalar column from v17/v20-era attack heads
            #    (settled-null in ablation; the column is dead weight)
            #  - weapon_head / weapon_embed start fresh on v17/v20-pre-v21
            #  - encoder.gru_input_proj weight (pre-v20 mean-actors pool) is
            #    silently dropped
            missing, unexpected = policy.model.load_state_dict(payload["state_dict"], strict=False)
            if model_factory is None and graph is None:
                allowed_missing_prefixes: tuple[str, ...] = (
                    # Pre-v21 checkpoints predate the weapon head; the
                    # whole WeaponHead component (mlp + embed) starts fresh.
                    "weapon_head.",
                    # Pre-MLP-pointer checkpoints carried target_pointer.query_proj
                    # instead of the current MLP scorer; the score module starts
                    # fresh (random-init). See migrate_legacy_flat_meta's d_target note.
                    "target_pointer.score.",
                )
                allowed_unexpected_prefixes: tuple[str, ...] = (
                    "encoder.gru_input_proj.",  # pre-v20: mean-actors pool projection
                    # Pre-refactor TransformerEncoder carried an internal
                    # TargetPointer that was dead weight whenever use_gru=True
                    # (Network's own target_pointer ran instead). Now that
                    # the encoder's internal pointer is removed entirely,
                    # those keys are unexpected on load — silently drop them.
                    "encoder.target_pointer.",
                    # Pre-MLP-pointer single-Linear query projection — superseded
                    # by target_pointer.score; dropped on load.
                    "target_pointer.query_proj.",
                )
            else:
                # No legacy allowances for factory- or graph-built modules —
                # they save and load their own exact state_dict shape.
                allowed_missing_prefixes = ()
                allowed_unexpected_prefixes = ()
            missing_keep = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
            unexpected_keep = [k for k in unexpected if not k.startswith(allowed_unexpected_prefixes)]
            if missing_keep or unexpected_keep:
                raise RuntimeError(
                    f"state_dict mismatch: missing={missing_keep}, unexpected={unexpected_keep}"
                )
        except RuntimeError as exc:
            raise ValueError(
                f"Incompatible checkpoint architecture for {source}. "
                "This code expects the combat-objective BC policy layout."
            ) from exc
        policy.model.to(policy.device)
        return policy

    @classmethod
    def load_for_finetune(
        cls,
        path: str | Path,
        *,
        use_gru: bool,
        d_gru: int,
        device: str,
    ) -> "QNNPolicy":
        del use_gru, d_gru
        return cls.load(path, device=device)
