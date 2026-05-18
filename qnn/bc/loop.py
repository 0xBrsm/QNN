"""Single-epoch supervised training loop for BC.

Thin orchestration on top of :mod:`qnn.bc.supervised_loop`: pick a
``chunk_size`` from ``sequence_length`` / ``tbptt_limit``, then dispatch
to the chunked driver.  The dataclasses are re-exported here so existing
callers can keep importing them from ``qnn.bc.loop``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import numpy as np
import torch

from qnn.bc.supervised_loop import (
    MidEpochState,
    PrecomputedEpisode,
    _run_batched,
)
from qnn.model.policy import QNNPolicy

__all__ = ["MidEpochState", "PrecomputedEpisode", "run_epoch"]


def run_epoch(
    model: QNNPolicy,
    episodes: Sequence[PrecomputedEpisode],
    batch_size: int,
    sequence_length: int,
    *,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    rng: np.random.Generator | None = None,
    max_grad_norm: float = 1.0,
    tbptt_limit: int = 256,
    head_loss_weights: Mapping[str, float] | None = None,
    step_callback: Any | None = None,
    report_every: int = 0,
    report_interval_seconds: float = 0.0,
    pin_memory: bool = True,
    prefetch: int,
    microbatch_size: int = 0,
    save_state_callback: Any | None = None,
    snapshot_interval: int = 0,
    resume_state: MidEpochState | None = None,
) -> Dict[str, float]:
    """Run one epoch of supervised training or evaluation over all episodes."""
    use_full_episode = int(sequence_length) <= 0
    if use_full_episode:
        if tbptt_limit > 0:
            chunk_size = max(int(tbptt_limit), 1)
        elif episodes:
            chunk_size = max(ep.n_samples for ep in episodes)
        else:
            chunk_size = 64
    else:
        chunk_size = max(int(sequence_length), 1)

    return _run_batched(
        model,
        episodes,
        max(1, int(batch_size)),
        chunk_size,
        class_weights=class_weights,
        lr=lr,
        rng=rng,
        max_grad_norm=max_grad_norm,
        head_loss_weights=head_loss_weights,
        step_callback=step_callback,
        report_every=report_every,
        report_interval_seconds=report_interval_seconds,
        pin_memory=pin_memory,
        prefetch=prefetch,
        microbatch_size=microbatch_size,
        save_state_callback=save_state_callback,
        snapshot_interval=snapshot_interval,
        resume_state=resume_state,
    )
