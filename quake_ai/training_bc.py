"""Behavior cloning trainer for the v0 Quake policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from quake_ai.data.dataset import (
    batch_index_iter,
    build_samples,
    build_world_samples,
    class_weights,
    load_metadata_index,
    split_samples,
    stack_actions,
    stack_observations,
    success_proxy,
    write_split_manifest,
)
from quake_ai.models.competitive_encoder import CompetitiveObservationEncoder
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.utils.io import write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class BCConfig:
    map_id: str
    output_dir: str
    telemetry_path: str = ""
    map_features_path: str = ""
    observation_format: str = "symbolic"
    world_ticks_path: str = ""
    map_state_path: str = ""
    seed: int = 7
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    epochs: int = 40
    lr: float = 0.01
    patience: int = 5
    use_gru: bool = False
    gru_hidden: int = 0
    trunk_hidden: int = 128
    class_weight_power: float = 0.5
    class_weight_min: float = 0.5
    class_weight_max: float = 2.0
    device: str = "auto"
    metadata_path: str = ""
    mode_filter: List[str] = field(default_factory=list)


def _evaluate_supervised_split(
    model: MLPGRUPolicy,
    obs: np.ndarray | torch.Tensor | None,
    actions: Dict[str, np.ndarray | torch.Tensor] | None,
    batch_size: int,
) -> Dict[str, float]:
    if obs is None or actions is None or len(obs) == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    total_rows = 0
    total_loss = 0.0
    total_accuracy = 0.0
    for start in range(0, len(obs), batch_size):
        stop = min(start + batch_size, len(obs))
        metrics = model.evaluate_supervised(obs[start:stop], {head: values[start:stop] for head, values in actions.items()})
        rows = stop - start
        total_rows += rows
        total_loss += float(metrics["loss"]) * rows
        total_accuracy += float(metrics["accuracy"]) * rows

    return {
        "loss": total_loss / max(total_rows, 1),
        "accuracy": total_accuracy / max(total_rows, 1),
    }


def run_behavior_cloning(config: BCConfig) -> Dict[str, float]:
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metadata_index = load_metadata_index(config.metadata_path) if config.metadata_path else None
    allowed_modes = config.mode_filter if config.mode_filter else None

    if config.observation_format == "world_v2":
        if not config.world_ticks_path or not config.map_state_path:
            raise RuntimeError("world_v2 behavior cloning requires world_ticks_path and map_state_path")
        samples = build_world_samples(
            config.world_ticks_path,
            config.map_state_path,
            metadata_index=metadata_index,
            mode_filter=allowed_modes,
        )
    elif config.observation_format == "world_v2_competitive":
        if not config.world_ticks_path or not config.map_state_path:
            raise RuntimeError("world_v2_competitive behavior cloning requires world_ticks_path and map_state_path")
        encoder = CompetitiveObservationEncoder()
        samples = build_world_samples(
            config.world_ticks_path,
            config.map_state_path,
            encoder=encoder,
            metadata_index=metadata_index,
            mode_filter=allowed_modes,
        )
    elif config.observation_format == "symbolic":
        samples = build_samples(config.telemetry_path, config.map_features_path)
    else:
        raise ValueError(f"Unsupported observation_format {config.observation_format}")
    split = split_samples(samples, config.train_ratio, config.val_ratio, config.seed)
    write_split_manifest(output / "split_manifest.json", split)

    if not split.train:
        raise RuntimeError("No training samples available after split")

    obs_dim = split.train[0].obs.shape[0]
    model = MLPGRUPolicy(
        obs_dim=obs_dim,
        trunk_hidden=config.trunk_hidden,
        gru_hidden=config.gru_hidden,
        use_gru=config.use_gru,
        seed=config.seed,
        device=config.device,
    )

    weights = {
        head: torch.as_tensor(values, dtype=torch.float32, device=model.device)
        for head, values in class_weights(
            split.train,
            power=config.class_weight_power,
            min_weight=config.class_weight_min,
            max_weight=config.class_weight_max,
        ).items()
    }
    train_obs = torch.as_tensor(
        stack_observations(split.train).astype(np.float32, copy=False),
        dtype=torch.float32,
        device=model.device,
    )
    train_actions = {
        head: torch.as_tensor(values, dtype=torch.long, device=model.device)
        for head, values in stack_actions(split.train).items()
    }
    val_obs = None
    val_actions = None
    if split.val:
        val_obs = torch.as_tensor(
            stack_observations(split.val).astype(np.float32, copy=False),
            dtype=torch.float32,
            device=model.device,
        )
        val_actions = {
            head: torch.as_tensor(values, dtype=torch.long, device=model.device)
            for head, values in stack_actions(split.val).items()
        }
    test_obs = None
    test_actions = None
    if split.test:
        test_obs = torch.as_tensor(
            stack_observations(split.test).astype(np.float32, copy=False),
            dtype=torch.float32,
            device=model.device,
        )
        test_actions = {
            head: torch.as_tensor(values, dtype=torch.long, device=model.device)
            for head, values in stack_actions(split.test).items()
        }

    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        train_losses: List[float] = []
        train_accs: List[float] = []

        for batch_idx in batch_index_iter(len(split.train), config.batch_size, rng):
            batch_index = torch.as_tensor(batch_idx, dtype=torch.long, device=model.device)
            obs = train_obs.index_select(0, batch_index)
            actions = {head: values.index_select(0, batch_index) for head, values in train_actions.items()}
            metrics = model.supervised_step(obs, actions, weights, lr=config.lr)
            train_losses.append(metrics["loss"])
            train_accs.append(metrics["accuracy"])

        val_metrics = _evaluate_supervised_split(model, val_obs, val_actions, batch_size=config.batch_size)

        proxy = success_proxy(split.val if split.val else split.train)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses) if train_losses else 0.0),
            "train_accuracy": float(np.mean(train_accs) if train_accs else 0.0),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_success_proxy": float(proxy),
        }
        history.append(epoch_metrics)

        improved = val_metrics["accuracy"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save(output / "bc_best_model.npz")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_epoch < 0:
        model.save(output / "bc_best_model.npz")

    final_model = MLPGRUPolicy.load(output / "bc_best_model.npz", device=config.device)

    test_metrics = _evaluate_supervised_split(final_model, test_obs, test_actions, batch_size=config.batch_size)

    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "num_train_samples": len(split.train),
        "num_val_samples": len(split.val),
        "num_test_samples": len(split.test),
        "epochs_ran": len(history),
    }

    write_json(output / "bc_history.json", {"history": history})
    write_json(output / "bc_summary.json", summary)
    write_experiment_manifest(output / "bc_manifest.json", asdict(config), summary)

    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
