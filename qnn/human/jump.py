"""JUMP domain — no per-collect baseline artifact yet.

Jump is contextual: sampled from the calibrated posterior and judged on PLACEMENT
(reliability curve + closed-loop ledge/gap/rocket-jump), never frequency-matched — so
there is no human "jump distribution" to create per collect the way look/attack have
(feedback_jump_no_rate_calibration). The only human jump number today is the diag
`human_jump_rate` (qnn.diag). This module reserves the domain slot; when a per-collect
jump reliability baseline is defined, its creator lands here and joins
qnn.human.ensure_from_collect.
"""
from __future__ import annotations
