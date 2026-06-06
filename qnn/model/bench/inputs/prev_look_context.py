"""Forward-scoped prev_look context for the temporal look-ablation probe.

Mirrors the ``WeaponAimContext`` pattern: the trainer stashes the
batch's ``prev_look`` tensor on entry to the forward pass, and any
bench head can read it from the contextvar during its
``forward(LookHeadInput)``. Keeps the canonical ``Network`` and
``LookHeadInput`` contracts untouched.

``prev_look`` is the demonstrator's look-direction vector at the
previous frame within the episode, zero-padded at episode starts.
Built once at preload in ``qnn.bc.supervised_loop._make_resident_source``;
the trainer enters the context before calling the model so any bench
look head consuming temporal signal sees it without per-frame
plumbing through Network.forward.
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True, slots=True)
class PrevLookContext:
    prev_look: torch.Tensor  # (B*, 3) — zero at episode starts


_CTX: "contextvars.ContextVar[PrevLookContext | None]" = contextvars.ContextVar(
    "qnn_prev_look_ctx", default=None,
)


@contextlib.contextmanager
def prev_look_context(ctx: PrevLookContext | None) -> Iterator[None]:
    """Enter / exit a prev_look scope. ``None`` disables (heads see nothing)."""
    token = _CTX.set(ctx)
    try:
        yield
    finally:
        _CTX.reset(token)


def current_prev_look_context() -> PrevLookContext:
    """Return the active context; raises if no scope is set."""
    ctx = _CTX.get()
    if ctx is None:
        raise RuntimeError(
            "PrevLookContext requested but no scope set — the trainer "
            "must wrap the forward pass via prev_look_context(...)."
        )
    return ctx
