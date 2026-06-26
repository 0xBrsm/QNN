"""Forward-scoped engagement_ema context.

Mirrors the sibling forward-scoped side-channel contexts: it carries the
per-frame engagement EMA scalar without extending the core
``AttackHeadInput`` dataclass. The trainer stashes the batch's
``engagement_ema`` tensor on entry to the forward pass, and consumers
(``ObsAccessor.aux("engagement")``) read it from the contextvar during
``forward``.

``engagement_ema`` is a per-frame scalar in [0, 1] tracking how committed
the demonstrator is to attacking, derived from the op-frame attack stream.
Built once at preload in ``qnn.bc.supervised_loop`` (resident +
streaming paths); the trainer enters the context before calling the
model so any consumer sees it without per-frame plumbing through
``Network.forward``.
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True, slots=True)
class EngagementEMAContext:
    engagement_ema: torch.Tensor  # (B*,) — flattened to match the head's batch dim


_CTX: "contextvars.ContextVar[EngagementEMAContext | None]" = contextvars.ContextVar(
    "qnn_engagement_ema_ctx", default=None,
)


@contextlib.contextmanager
def engagement_ema_context(ctx: EngagementEMAContext | None) -> Iterator[None]:
    """Enter / exit an engagement_ema scope. ``None`` disables (heads see nothing)."""
    token = _CTX.set(ctx)
    try:
        yield
    finally:
        _CTX.reset(token)


def current_engagement_ema_context() -> EngagementEMAContext:
    """Return the active context; raises if no scope is set."""
    ctx = _CTX.get()
    if ctx is None:
        raise RuntimeError(
            "EngagementEMAContext requested but no scope set — the trainer "
            "must wrap the forward pass via engagement_ema_context(...)."
        )
    return ctx
