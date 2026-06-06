"""Forward-scoped context for the weapon_aim bench.

``WeaponAimNetwork`` produces a ``WeaponAimContext`` from each batch's obs
and exposes it via a ``contextvars`` token for the duration of one forward
call. Bench heads (``WeaponAimLookHead``, ``WeaponAimAttackHead``) read
their geometric extras through :func:`current_weapon_aim_context` instead
of via module-attribute stashes — which had two failure modes the
contextvars pattern eliminates:

1. **Stale tensors across forward calls.** Module attributes survive the
   call that wrote them; any code path that runs a head outside the
   wrapper (ONNX export, partial-forward in tests, a future cached
   rollout) would silently reuse the previous batch's extras. A context
   token raises ``RuntimeError`` outside the wrapper's ``with`` block.

2. **Side-channel obscurity.** The wrapper-to-head contract was implicit
   (which attrs to set, on which head, in which order). One typed
   dataclass + one accessor function is explicit and discoverable.

The bench wrapper is only constructed for ``variant == "weapon_aim"`` —
the canonical variant uses the plain ``Network`` directly, so no context
is built or installed when it isn't needed.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class WeaponAimContext:
    entity_rel:    torch.Tensor   # (B*, N, 3)  view-frame XYZ / DIST_SCALE
    entity_vel:    torch.Tensor   # (B*, N, 3)  view-frame rel vel / VEL_SCALE
    weapon_id:     torch.Tensor   # (B*,)        impulse-indexed long, 0..8
    weapon_static: torch.Tensor   # (9, 7)       static weapon-trajectory table
    noop:          torch.Tensor   # (B*,)        approx engine noop bit


_current: contextvars.ContextVar[WeaponAimContext | None] = contextvars.ContextVar(
    "qnn_weapon_aim_context", default=None,
)


@contextlib.contextmanager
def weapon_aim_context(ctx: WeaponAimContext):
    """Install ``ctx`` for the duration of the ``with`` block."""
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def current_weapon_aim_context() -> WeaponAimContext:
    """Return the active context, or raise if no wrapper is installed."""
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError(
            "no WeaponAimContext is active — bench weapon_aim heads must "
            "run inside WeaponAimNetwork.forward (which installs the "
            "context for the duration of the call)."
        )
    return ctx
