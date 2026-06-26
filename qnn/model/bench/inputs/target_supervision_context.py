"""Forward-scoped target-supervision context for bench pointer ablations.

Mirrors :mod:`engagement_ema_context`: the
trainer derives privileged target supervision (``target_gt``,
``target_probs_idx``, ``prev_target_probs``) from BC labels and enters
this context before the model forward. Bench pointer modules that
consume any of these (``CanonicalTargetPointer`` hard / gt-dist / prev
modes, ``GTTargetPointer`` oracle) read from the contextvar instead of
receiving them via Network.forward.

The canonical model (``qnn.model.target.TargetPointer``) ignores this
context — these tensors only feed bench-resident variants. Keeping the
supervision flow out of Network.forward keeps the main-model forward
signature clean (no privileged-input plumbing) while preserving bench
ablation capability through the BC supervised loop.
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True, slots=True)
class TargetSupervisionContext:
    target_gt: torch.Tensor | None = None              # (B*,) long
    target_probs_idx: torch.Tensor | None = None       # (B*, N) float — renormalized GT idx distribution
    prev_target_probs: torch.Tensor | None = None      # (B*, N) float — previous-frame distribution


_CTX: "contextvars.ContextVar[TargetSupervisionContext | None]" = contextvars.ContextVar(
    "qnn_target_supervision_ctx", default=None,
)


@contextlib.contextmanager
def target_supervision_context(ctx: TargetSupervisionContext | None) -> Iterator[None]:
    """Enter / exit a target-supervision scope. ``None`` disables (pointers see nothing)."""
    token = _CTX.set(ctx)
    try:
        yield
    finally:
        _CTX.reset(token)


def current_target_supervision_context() -> TargetSupervisionContext | None:
    """Return the active context, or ``None`` if no scope is set."""
    return _CTX.get()
