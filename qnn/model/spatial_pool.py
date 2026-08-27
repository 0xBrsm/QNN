"""Pooled 9-sector spatial summary — the v1 capacity-class bench arm.

Reduces the finalized depth atlas (11 elevation bands × 24 yaw cells) to the
v1 9-sector geometry (``qnn_spatial.c`` on main, pre-atlas): seven
horizontal sectors at yaw offsets 0/±40/±90/±150° (spans 40/40/30°) read
from the 0° elevation band, plus ground/ceiling sectors read from the
∓75° bands — the closest atlas analog of v1's 128u vertical bbox probes
(the ±75° band limit 128/sin 75° ≈ 132.5u matches v1's vertical
contract).

Per sector, four depth statistics over the covered yaw cells — nearest,
mean, openness (mean over the sector's range cap), hit fraction — plus a
9-wide sector one-hot: 13 scalars per sector, mirroring v1's 13-scalar
token width. The channels v1 derived from extra engine traces
(clearance/traversable/dropoff, material fractions) have no atlas
source, so this arm tests the *capacity class* of the v1 representation
(its depth statistics at its angular resolution), not its exact feature
set.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import qnn.engine_norm as en

# (name, elevation-band index, yaw center°, yaw span°). Horizontal
# sectors read band 5 (0° elevation); ground/ceiling read the ∓75°
# bands over the full circle. Centers/spans are the v1 sector table
# (QNN_SpatialEmitTokens); positive yaw is counter-clockwise, matching
# both v1's center_deg and the atlas yaw-cell convention.
_SECTORS: tuple[tuple[str, int, float, float], ...] = (
    ("fov_center",  5,    0.0,  40.0),
    ("fov_left",    5,   40.0,  40.0),
    ("fov_right",   5,  -40.0,  40.0),
    ("flank_left",  5,   90.0,  40.0),
    ("flank_right", 5,  -90.0,  40.0),
    ("rear_left",   5,  150.0,  30.0),
    ("rear_right",  5, -150.0,  30.0),
    ("ground",      0,    0.0, 360.0),
    ("ceiling",    10,    0.0, 360.0),
)

_YAW_STEP = 360.0 / en.ATLAS_YAWS


def _sector_cells(band: int, center: float, span: float) -> list[int]:
    """Flat (band, yaw) cell indices whose yaw centers lie in the span."""
    if span >= 360.0:
        yaws = list(range(en.ATLAS_YAWS))
    else:
        yaws = [
            yaw for yaw in range(en.ATLAS_YAWS)
            if abs(((yaw * _YAW_STEP - center + 180.0) % 360.0) - 180.0)
            <= span / 2.0 + 1e-6
        ]
    return [band * en.ATLAS_YAWS + y for y in yaws]


class SectorPool9(nn.Module):
    """Atlas ``spatial_scalars`` (…, 11, 48) → 9-sector summary (…, 9, 13)."""

    N_SECTORS = len(_SECTORS)
    OUT_DIM = 4 + N_SECTORS  # nearest, mean, openness, hit_frac + one-hot

    def __init__(self) -> None:
        super().__init__()
        cells = [_sector_cells(b, c, s) for _, b, c, s in _SECTORS]
        width = max(len(c) for c in cells)
        index = torch.zeros(self.N_SECTORS, width, dtype=torch.long)
        valid = torch.zeros(self.N_SECTORS, width, dtype=torch.bool)
        for i, c in enumerate(cells):
            index[i, : len(c)] = torch.tensor(c, dtype=torch.long)
            valid[i, : len(c)] = True
        # Per-sector range cap in DIST_SCALE units: the band limit of the
        # sector's source band (1024u horizontal, ~132.5u for ∓75°).
        caps = torch.tensor(
            [en.ATLAS_BAND_LIMIT[b] / en.DIST_SCALE for _, b, _, _ in _SECTORS],
            dtype=torch.float32,
        )
        self.register_buffer("_index", index, persistent=False)
        self.register_buffer("_valid", valid, persistent=False)
        self.register_buffer("_caps", caps, persistent=False)
        self.register_buffer(
            "_one_hot", torch.eye(self.N_SECTORS), persistent=False,
        )

    def forward(self, spatial_scalars: torch.Tensor) -> torch.Tensor:
        lead = spatial_scalars.shape[:-2]
        # Per band the last dim is [depth(24) | hit(24)]; split, then
        # flatten bands so sector cell indices address (band*24 + yaw).
        depth = spatial_scalars[..., : en.ATLAS_YAWS].reshape(*lead, -1)
        hit = spatial_scalars[..., en.ATLAS_YAWS :].reshape(*lead, -1)

        idx = self._index.reshape(-1)
        d = depth[..., idx].reshape(*lead, self.N_SECTORS, -1)
        h = hit[..., idx].reshape(*lead, self.N_SECTORS, -1)
        valid = self._valid
        counts = valid.sum(dim=-1).to(d.dtype)

        nearest = d.masked_fill(~valid, torch.inf).amin(dim=-1)
        mean = (d * valid).sum(dim=-1) / counts
        openness = (mean / self._caps).clamp(0.0, 1.0)
        hit_frac = (h * valid).sum(dim=-1) / counts

        stats = torch.stack([nearest, mean, openness, hit_frac], dim=-1)
        one_hot = self._one_hot.to(stats.dtype).expand(
            *lead, self.N_SECTORS, self.N_SECTORS,
        )
        return torch.cat([stats, one_hot], dim=-1)
