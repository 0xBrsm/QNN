"""Decode-fit v2 — per-checkpoint mechanical-aim calibration.

Replaces the v1 grid-median pipeline (`qnn.eval.decode_fit_pipeline` +
`qnn.eval.skill_vector`) with stochastic simulation calibration
(`agents/plans/decode-fit-v2.md`, approved 2026-07-16):

  * every operative discharge is a likelihood sample (`events`), not a cell
    median — the per-weapon gain/α/tremor responses are 3-parameter monotone
    saturating curves fit by MLE with CIs (`response`);
  * two budget-boxed adaptive rounds replace the five v1 subprocess sweep
    campaigns (`design`, `instruments`);
  * native / knee / inversion all derive from ONE fitted curve, so they can
    never disagree (the v1 SG knee-0.04-vs-p50-gain-0.443 class);
  * targets below the fitted achievable frontier are REFUSED — deploys ride
    the frontier (Brian 2026-07-16 decision 1);
  * stage 6 confirms on the instrument the fit ran on (CI overlap at
    percentile ±5, decision 3) and style-gates free play; free-play
    per-weapon hbw is a report card only (decision 4);
  * every fit artifact lives under ``runs/decode_fit/<model>/``; waves
    resume-skip off their content-hashed done dirs + the substrate/env
    staleness check — no global ``runs/head_probe`` namespace, no
    cwd-relative defaults, no delete-the-dir cache invalidation.

Entry point: ``python -m qnn.decode_fit --run-dir <bench run> [--skill …]``.
"""
from qnn.decode_fit.context import (  # noqa: F401
    FitContext,
    INTERCEPT_WEAPONS,
    WEAPON_IMPULSE,
    MODELNAME_TO_ABBR,
    ABBR_TO_MODELNAME,
    TRANSFER_ALIAS,
    CALIBRATION_FAMILIES,
    CALIBRATION_FAMILY_KEY,
    CALIBRATION_GROUPS,
    CALIBRATION_SOURCE,
    calibration_members,
    resolve_fit_context,
)
