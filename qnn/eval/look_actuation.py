"""Per-side LOOK actuators for h2h's decision/actuation decoupling
(agents/plans/look-actuation-decoupling.md, work item 2).

Work item 1 (qnn.eval.h2h._resolve_world_rate) lets a side decide (call its
model) only once every ``stride`` world ticks; on the other ``stride - 1``
ticks move/attack/jump/weapon HOLD (mirrors the engine's own input-hold) but
LOOK must still move every world tick — humans and the stock Quake client
execute continuously between decisions, and freezing the view for
``stride - 1`` ticks is exactly the "teleport then freeze" pathology the
parent plan exists to fix. An actuator turns ONE decision's decoded look
intent into ``stride`` per-world-tick unit-vector deltas.

Choices (``LOOK_ACTUATION_CHOICES``):
  step   — the decision's FULL decoded delta on the decision tick, identity
           (no turn) on the other ``stride - 1`` ticks. The Phase-2 baseline
           behavior (seg-vs-frame plan), now expressed explicitly in a world
           that runs faster than the decider — the "do nothing new" control.
  smear  — split the decision's decoded delta into ``stride`` EQUAL tangent-
           space increments (constant angular velocity across the window).
           Operates on the model's OWN decoded look vector — decode-
           mechanism-agnostic, no retraining, works on any side (rung 1 of
           the plan). The decomposition is the exact inverse of
           ``qnn.bc.resample._compose_look``'s composition law (sum of yaw/
           pitch, which is how the training corpus's own resampled look
           labels were built — see that module's docstring): dividing yaw
           and pitch by ``stride`` and reconstructing per increment makes
           the ``stride`` increments compose back to the original delta
           EXACTLY (see smear_increments), not just approximately.
  stroke — retime the look_seg COMMITMENT playout to world-tick rate: a
           stroke whose native duration is D DECISION ticks spans
           ``D * stride`` world ticks, and
           ``qnn.model.look_seg_decode.stroke_theta_schedule``'s corpus-fit
           velocity profile is resampled onto that many points directly (the
           SAME profile shape, just sampled more finely — no re-fit) so
           per-world-tick deltas integrate to the stroke's full amplitude.
           Requires the side to be playing the look_seg commitment decode
           (``model.look_commitment`` — shape-derived for a look_seg-only
           checkpoint, or forced onto a dual-head one via
           ``--force-look-commit``); there is no commitment state to retime
           on a side that never enters that decode.

KNOWN COMPROMISE (see the look-actuation-decoupling plan report): the stroke
executor plays back the RAW committed geometry (amplitude/direction/profile)
only — it does not re-apply whatever aim-prior blend or feet-aim pitch
correction ``decode_look_from_polar`` may have layered onto the decision
tick's own decoded look. Those blends are obs-driven (they need the CURRENT
target direction), and obs is frozen between decisions by construction (work
item 1) — there is no principled per-world-tick analog once decisions stop
firing every tick, on ANY actuator (smear/step have the identical limitation:
they only ever see the one decision's already-composed output). This is a
consequence of the decoupling itself, not a stroke-specific shortcut.

Stride-1 sides never reach this module: qnn.eval.h2h's decision loop takes
the decoded action verbatim every world tick, unchanged from pre-stride h2h
(the BYTE-IDENTICAL-at-stride-1 contract) — see run_h2h's ``stride[s] == 1``
branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from qnn.model.look_seg_decode import (
    LOOK_COMMIT_STATE_DIM, _DIR_CENTERS, amp_center_for_class,
    stroke_theta_schedule)

LOOK_ACTUATION_CHOICES = ("step", "smear", "stroke")

# The identity look delta (no turn) — the same convention h2h uses to seed a
# lane's action array before any decoded action overwrites it.
IDENTITY_LOOK = np.array([1.0, 0.0, 0.0], dtype=np.float32)


# ── smear (rung 1: polar decode-only, decode-agnostic) ───────────────────────

def _yaw_pitch(look: np.ndarray) -> "tuple[float, float]":
    """(x, y, z) unit look-delta -> (yaw, pitch), the INVERSE half of
    ``qnn.bc.resample._compose_look``'s reconstruction (same law, applied to
    one vector instead of composing several)."""
    x, y, z = (float(v) for v in look)
    yaw = float(np.arctan2(y, x))
    pitch = float(np.arctan2(z, np.hypot(x, y)))
    return yaw, pitch


def _from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """(yaw, pitch) -> (x, y, z) unit vector — ``_compose_look``'s own final
    reconstruction step, applied to a single (yaw, pitch) pair."""
    return np.array([np.cos(pitch) * np.cos(yaw),
                     np.cos(pitch) * np.sin(yaw),
                     np.sin(pitch)], dtype=np.float32)


def smear_increments(look_delta: "np.ndarray | Sequence[float]",
                     stride: int) -> np.ndarray:
    """``stride`` EQUAL tangent-space sub-rotations that compose back to
    ``look_delta`` under ``qnn.bc.resample._compose_look``'s own composition
    law (sum yaw, sum pitch, reconstruct once). Dividing yaw and pitch by
    ``stride`` up front and reconstructing per increment makes this exact
    (to float precision) for both single-axis and mixed-axis deltas — see
    tests/test_h2h_stride.py, which checks the round trip through
    ``_compose_look`` directly rather than re-deriving the identity here.
    Returns an ``(stride, 3)`` float32 array, one identical row per tick."""
    yaw, pitch = _yaw_pitch(np.asarray(look_delta, dtype=np.float64))
    inc = _from_yaw_pitch(yaw / stride, pitch / stride)
    return np.tile(inc, (stride, 1)).astype(np.float32, copy=False)


# ── stroke (rung 2: look_seg commitment, retimed to world-tick rate) ────────

@dataclass
class StrokeExecutor:
    """Per-lane world-tick retiming of one side's look_seg commitment
    playout.

    ``advance`` is called once per DECISION tick with that tick's post-act()
    ``look_commit`` state (``qnn.model.look_seg_decode.LOOK_COMMIT_STATE_DIM``
    lanes: ``[cls, rem, elapsed, dur_bucket, dir_bin]``). ``elapsed == 1`` is
    the tick an onset happened (see look_commit_step: a fresh commitment
    always sets elapsed 0 then increments once before returning, so no
    continuing commitment ever reads back 1) — on that tick this rebuilds the
    retimed schedule from the commitment's OWN parameters:

      * ``dur = elapsed + rem`` — invariant across a commitment's whole life
        (look_commit_step decrements rem and increments elapsed by the same
        amount every tick), so it recovers the native decision-tick length D
        even read mid-stroke.
      * ``amp = amp_center_for_class(cls, hz)`` (0.0 for a hold).
      * ``phi = _DIR_CENTERS[dir_bin]`` (irrelevant when amp is 0).
      * ``stroke_theta_schedule(bkt, dur * stride, hz)`` — the SAME corpus-
        fit profile shape (keyed by ``hz``, the side's own native rate),
        resampled onto ``dur * stride`` points instead of ``dur``: no re-fit,
        just finer sampling of the same continuous shape, still renormalized
        to sum to 1 over the full retimed span (mass-conserving).

    A genuinely new onset ALWAYS wins and rebuilds from ``_pos = 0`` — even
    if the previous schedule had ticks left un-consumed — which is exactly
    the plan's "new decision interrupts an in-flight stroke" override
    semantics. Under the intended cadence (this side's own model expiring
    its commitment exactly every ``dur`` decision ticks) the previous
    schedule is always fully consumed by then anyway; the override exists so
    a state discontinuity (episode reset — see qnn.eval.h2h's fresh
    ``look_commit`` init, whose ``rem == 0`` forces an onset) can never leave
    a stale tail playing into a new episode.
    """

    stride: int
    hz: int
    _sched: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    _phi: float = 0.0
    _pos: int = 0

    def advance(self, look_commit_state: "np.ndarray | Sequence[float]") -> list[np.ndarray]:
        vals = np.asarray(look_commit_state, dtype=np.float64).reshape(-1)
        if vals.size < LOOK_COMMIT_STATE_DIM:
            raise ValueError(
                f"look_commit_state has {vals.size} lanes, need "
                f"{LOOK_COMMIT_STATE_DIM} ([cls, rem, elapsed, dur_bucket, "
                "dir_bin]) — stroke actuation needs the FULL commitment state")
        cls, rem, elapsed, bkt, dirbin = (int(round(v)) for v in vals[:5])
        if elapsed == 1 or self._sched.size == 0:
            dur = elapsed + rem
            amp = amp_center_for_class(cls, self.hz)
            self._phi = float(_DIR_CENTERS[dirbin]) if amp != 0.0 else 0.0
            self._sched = stroke_theta_schedule(bkt, dur * self.stride, self.hz) * amp
            self._pos = 0
        out: list[np.ndarray] = []
        for _ in range(self.stride):
            theta = float(self._sched[self._pos])
            out.append(np.array(
                [np.cos(theta), np.sin(theta) * np.cos(self._phi),
                 np.sin(theta) * np.sin(self._phi)], dtype=np.float32))
            self._pos += 1
        return out


# ── the per-side dispatcher h2h's decision loop calls ────────────────────────

def actuate_window(choice: str, decoded_look: "np.ndarray | Sequence[float]",
                   stride: int, *,
                   stroke_exec: "StrokeExecutor | None" = None,
                   look_commit_state: "np.ndarray | None" = None,
                   ) -> list[np.ndarray]:
    """The ``stride`` per-world-tick look vectors for the window this
    decision just opened. ``decoded_look`` is the model's own decoded look
    field (used verbatim by step/smear; ignored by stroke, which retimes the
    commitment straight from ``look_commit_state`` — see StrokeExecutor)."""
    if choice == "step":
        return [np.asarray(decoded_look, dtype=np.float32).copy()] + \
            [IDENTITY_LOOK.copy() for _ in range(stride - 1)]
    if choice == "smear":
        return list(smear_increments(decoded_look, stride))
    if choice == "stroke":
        if stroke_exec is None or look_commit_state is None:
            raise ValueError(
                "stroke actuation needs both a StrokeExecutor and this "
                "tick's look_commit_state")
        return stroke_exec.advance(look_commit_state)
    raise ValueError(
        f"unknown look actuation {choice!r}; choices are {LOOK_ACTUATION_CHOICES}")
