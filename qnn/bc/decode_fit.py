"""Shared corpus-forward diagnostic helpers (policy load, operative-frame
mask, attack-with forward, committed-weapon stream).

The OFFLINE decode fit that lived here — the corpus-forward pins producing a
joint attack/weapon bias (+ the ``python -m`` CLI) — was
DELETED with the a26 live-pins redesign: offline-fit pins mis-sign when the
offline↔live regime gap is large (the a26 awposw fit put NG at -0.35 while
live held-NG waves needed >= +4.7 bias to discharge AT ALL). The attack
operating point is now measured closed-loop with fire-only
``attack.fire_bias_vec`` by ``qnn.decode_fit.live_pins``
(round 0, instrumental discharge floor) and trimmed to the human world-rate
by the free-play attack trim (``qnn.decode_fit.gates.attack_trim``).

What remains is the shared substrate other diagnostics import:
  * ``SEGMENT_MASK`` / ``_op_mask`` — the engaged-subset + operative-frame
    contracts (EVERY model↔human attack comparison rides ``_op_mask``);
  * ``_load_policy`` / ``_to_np`` — canonical checkpoint load (delegates to
    ``qnn.diag.loader``) + host-numpy copy;
  * ``_forward_attack_with_logits`` — per-episode 9-way attack-with forward;
  * ``_committed_stream`` — carry-forward committed-weapon stream (canonical
    single copy; re-exported by ``qnn.diag.weapon``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from qnn.diag.loader import load_policy as _diag_load_policy
from qnn.model.network import ATTACK_HEAD

SEGMENT_MASK = {"act.target": {"$ne": 0}}   # the training subset the heads were fit on


# ── policy load ──────────────────────────────────────────────────────────────
def _load_policy(run: Path):
    """Load a policy from a bench run directory.

    Delegates to :func:`qnn.diag.loader.load_policy` — the canonical
    single implementation shared with all analysis scripts.
    """
    return _diag_load_policy(run)


# ── shared helpers (op-filter / committed stream) ────────────────────────────
def _to_np(x) -> np.ndarray:
    """Tensor (possibly on GPU) or array-like → host numpy. The resident source
    loads obs/actions onto policy.device, so cuda tensors must be copied first."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _op_mask(source) -> np.ndarray:
    """Per-frame OPERATIVE flag (input_mask bit 0): the frames on which the engine
    HONORS the attack input. Training rewrites the attack label to
    ``feasibility * demo_press`` and scores BCE/precision/recall/F1 only where op=1
    (``qnn.model.policy._compute_head_losses_and_metrics``); op=0 frames are no-op
    holds — a trigger held through cooldown or an auto-attack weapon's continuous hold,
    which the engine ignores — and the model's predictions there are uncalibrated.

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  EVERY comparison of MODEL attack behaviour to the HUMAN/DEMO attack stream    ║
    ║  MUST be restricted to op=1 frames FIRST. Raw ``mean(attack)`` over all frames ║
    ║  over-counts the human attack rate by every held-trigger cooldown frame, so a  ║
    ║  calibration fit against it is garbage. The segment_mask (act.target!=0) is    ║
    ║  the ENGAGED mask, NOT this one — engaged-but-in-cooldown frames exist.        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    im = source.actions.get("input_mask")
    if im is None:
        raise ValueError("operative filter requires actions['input_mask'] — recollect "
                         "the corpus on a post-input_mask branch")
    return (_to_np(im).reshape(-1).astype(np.uint8) & 0x1).astype(bool)


def _committed_stream(probs: np.ndarray, seed: np.ndarray, offsets: np.ndarray,
                      C: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    """Carry-forward committed-weapon stream + per-frame 'committed' mask, under the
    sticky gate (commit argmax iff conf>=C and margin>=M, else hold the last commit).
    Each episode start is force-seeded from ``seed`` (impulses 1..8) so the carry never
    crosses episodes. This is the closed-loop proxy for the deployed gate: the engine's
    committed weapon converges on the bot's own last commit, so the committed stream — not
    the human's demo-equipped weapon — is what the dwell distribution should be fit on.
    Same construction as _weapon_switch_threshold_eval._committed_stream."""
    pred_imp = probs.argmax(1) + 1
    top2 = np.sort(probs, axis=1)[:, -2:]
    conf = top2[:, 1]
    margin = top2[:, 1] - top2[:, 0]
    n = len(seed)
    arange = np.arange(n)
    offs = np.asarray(offsets, dtype=np.int64)
    ep_start = np.zeros(n, dtype=bool)
    ep_start[offs[:-1][offs[:-1] < n]] = True
    above = (conf >= C) & (margin >= M)
    take = above | ep_start                                  # episode start seeds
    take_value = np.where(above, pred_imp, seed)             # start & !above -> seed init
    com = take_value[np.maximum.accumulate(np.where(take, arange, -1))]
    return com, above


# ── a25 9-way ATTACK-WITH forward (per-episode, deployment-faithful) ──────────
@torch.inference_mode()
def _forward_attack_with_logits(policy, source) -> np.ndarray:
    """Per-frame 9-way attack-with logits (N, 9). Same per-episode forward as
    diag's _forward_weapon_logits, but reads the full (T, ATTACK_WITH_SIZE)
    head (the 8-way reshape there would raise on this head). Forwarded WITHOUT
    the bench side-channel — the live policy.act path teacher-forces nothing."""
    from qnn.model.attack_with_head import ATTACK_WITH_SIZE
    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    out = np.empty((int(offsets[-1]), ATTACK_WITH_SIZE), np.float32)
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        T = e - s
        idx = torch.arange(s, e, dtype=torch.int64, device=policy.device)
        obs_seq = {k: v.index_select(0, idx).reshape((1, T) + tuple(v.shape[1:]))
                   for k, v in source.obs.items()}
        _f, logits, _v, _nh, _tl = policy.model(obs_seq, hidden=None, reset_mask=None)
        out[s:e] = logits[ATTACK_HEAD].reshape(T, ATTACK_WITH_SIZE).float().cpu().numpy()
    return out
