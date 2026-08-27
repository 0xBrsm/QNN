"""Deterministic eight-match Quake arena used by the PPO arena backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .brush import MapFile, axis_aligned_box
from .textures import MAP_TEXTURES, materialize_texture_wad


ARENA_GRID_MAP_ID = "qnn_arena8"
ARENA_GRID_TIERED_MAP_ID = "qnn_arena8v"

# Tiered-cell geometry (Workstream 1 / E2, agents/plans/a26-superiority-
# decomposition.md): a floating deck at half the cell's ceiling height,
# reached by a staircase of <=16u risers (Quake autostep is 18u). Tread run
# is kept narrow (6u, well under the 16u rise) so the staircase's footprint
# clears the cell center: qbsp's hull-2 (large-hull) auto-fill seeds from
# the qnn_arena_match controller sitting at the cell center, and if the new
# geometry's footprint reaches under that entity's inflated hull-2 box
# (+-32u), qbsp emits "No entities in empty space -- no filling performed
# (hull 2)" and silently produces broken hull-2 collision (empirically
# bisected 2026-08-09/10: overlap of even a few units is sufficient).
_DECK_THICKNESS = 16
_STEP_RISE = 16
_STEP_RUN = 6
_MIN_OPEN_FLOOR = 128
_CENTER_CLEARANCE = 32  # matches hull-2's half-width; keeps geometry off the controller


@dataclass(frozen=True)
class ArenaGridSpec:
    """Geometry and identity contract for a grid of sealed 1v1 matches."""

    columns: int = 4
    rows: int = 2
    cell_size: int = 512
    wall: int = 64
    height: int = 256
    spawn_offset: int = 128
    tiered: bool = False

    @property
    def match_count(self) -> int:
        return self.columns * self.rows

    @property
    def deck_top(self) -> int:
        """Deck top z: half the playable ceiling height (256 -> 128)."""
        return self.height // 2

    @property
    def deck_bottom(self) -> int:
        return self.deck_top - _DECK_THICKNESS

    @property
    def deck_width(self) -> int:
        """Roughly one third of the cell floor, at the +X end."""
        return self.cell_size // 3

    @property
    def ramp_steps(self) -> int:
        """Number of <=16u-rise steps needed to reach the deck's underside."""
        return self.deck_bottom // _STEP_RISE

    @property
    def ramp_run(self) -> int:
        return self.ramp_steps * _STEP_RUN

    @property
    def open_floor_width(self) -> int:
        """Uncovered ground at the cell's far (-X) end, under the full ceiling."""
        return self.cell_size - self.deck_width - self.ramp_run

    @property
    def width(self) -> int:
        return self.columns * self.cell_size + (self.columns + 1) * self.wall

    @property
    def depth(self) -> int:
        return self.rows * self.cell_size + (self.rows + 1) * self.wall

    def cell_bounds(self, match_id: int) -> tuple[int, int, int, int]:
        if not 0 <= match_id < self.match_count:
            raise ValueError(f"match_id must be in [0, {self.match_count})")
        column = match_id % self.columns
        row = match_id // self.columns
        x0 = self.wall + column * (self.cell_size + self.wall)
        y0 = self.wall + row * (self.cell_size + self.wall)
        return x0, y0, x0 + self.cell_size, y0 + self.cell_size


def build_arena_grid(spec: ArenaGridSpec = ArenaGridSpec()) -> MapFile:
    """Build a sealed grid with explicit match/seat-tagged spawn entities."""
    if spec.match_count > 8:
        raise ValueError("Quake's 16 maxclients limit permits at most eight 1v1 matches")
    if spec.spawn_offset * 2 + 64 >= spec.cell_size:
        raise ValueError("spawn_offset leaves insufficient separation inside a cell")
    if spec.tiered:
        if spec.open_floor_width < _MIN_OPEN_FLOOR:
            raise ValueError(
                "tiered deck+ramp leaves less than 128u of open floor "
                f"({spec.open_floor_width}u) at the cell's far end"
            )
        center_clearance = spec.open_floor_width - spec.cell_size // 2
        if center_clearance < _CENTER_CLEARANCE:
            raise ValueError(
                "tiered deck+ramp footprint reaches within "
                f"{_CENTER_CLEARANCE}u of the cell center "
                f"(clearance={center_clearance}u) -- qbsp hull-2 auto-fill "
                "seeds from the qnn_arena_match controller there; this "
                "silently breaks hull-2 collision (see module docstring)"
            )
        deck_x0_local = spec.cell_size - spec.deck_width
        if spec.cell_size // 2 + spec.spawn_offset < deck_x0_local:
            raise ValueError("spawn_offset does not land the second spawn on the deck")

    arena = MapFile()
    arena.worldspawn.properties.update(
        {
            "message": (
                "QNN eight-match PPO arena — tiered"
                if spec.tiered
                else "QNN eight-match PPO arena"
            ),
            "qnn_arena_matches": str(spec.match_count),
        }
    )

    # One continuous floor and ceiling plus solid perimeter/internal walls.
    # Adjacent cells share wall brushes; there are no portals, so projectiles,
    # traces, PVS visibility, and collision remain match-local by construction.
    arena.add_brush(
        axis_aligned_box(0, 0, -spec.wall, spec.width, spec.depth, 0, MAP_TEXTURES.floor)
    )
    arena.add_brush(
        axis_aligned_box(
            0,
            0,
            spec.height,
            spec.width,
            spec.depth,
            spec.height + spec.wall,
            MAP_TEXTURES.ceiling,
        )
    )
    for boundary in range(spec.columns + 1):
        x0 = boundary * (spec.cell_size + spec.wall)
        arena.add_brush(
            axis_aligned_box(
                x0,
                0,
                0,
                x0 + spec.wall,
                spec.depth,
                spec.height,
                MAP_TEXTURES.shell,
            )
        )
    for boundary in range(spec.rows + 1):
        y0 = boundary * (spec.cell_size + spec.wall)
        arena.add_brush(
            axis_aligned_box(
                0,
                y0,
                0,
                spec.width,
                y0 + spec.wall,
                spec.height,
                MAP_TEXTURES.shell,
            )
        )

    for match_id in range(spec.match_count):
        x0, y0, x1, y1 = spec.cell_bounds(match_id)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        if spec.tiered:
            _add_tier(arena, x0, y0, x1, y1, spec)
            seat1_z = spec.deck_top + 24
        else:
            seat1_z = 24
        arena.add_entity(
            "info_player_deathmatch",
            (cx - spec.spawn_offset, cy, 24),
            angle="0",
            qnn_match_id=str(match_id),
            qnn_seat_id="0",
        )
        arena.add_entity(
            "info_player_deathmatch",
            (cx + spec.spawn_offset, cy, seat1_z),
            angle="180",
            qnn_match_id=str(match_id),
            qnn_seat_id="1",
        )
        arena.add_entity(
            "qnn_arena_match",
            (cx, cy, 32),
            qnn_match_id=str(match_id),
        )
        arena.add_light((cx, cy, spec.height - 32), brightness=450)

    return arena


def _add_tier(arena: MapFile, x0: int, y0: int, x1: int, y1: int, spec: ArenaGridSpec) -> None:
    """Add one cell's deck + staircase: elevation delta between the two spawns.

    Deck: a floating platform at the +X end, top z = spec.deck_top, thickness
    _DECK_THICKNESS, spanning the cell's full Y width. Hollow underneath (its
    bottom face sits above the floor) so ground fighting continues under it.
    Ramp: a staircase of <=16u-rise brushes (Quake autostep is 18u) climbing
    from the floor to the deck's underside; the deck's own top face supplies
    the final 16u riser onto the platform.
    """
    deck_x0 = x1 - spec.deck_width
    arena.add_brush(
        axis_aligned_box(deck_x0, y0, spec.deck_bottom, x1, y1, spec.deck_top, MAP_TEXTURES.fill)
    )

    ramp_x0 = deck_x0 - spec.ramp_run
    for step in range(spec.ramp_steps):
        step_x0 = ramp_x0 + step * _STEP_RUN
        step_x1 = step_x0 + _STEP_RUN
        step_top = (step + 1) * _STEP_RISE
        arena.add_brush(
            axis_aligned_box(step_x0, y0, 0, step_x1, y1, step_top, MAP_TEXTURES.fill)
        )


def write_arena_grid(
    output_dir: Path,
    *,
    spec: ArenaGridSpec = ArenaGridSpec(),
    map_id: str = ARENA_GRID_MAP_ID,
) -> Path:
    """Write the deterministic source map and texture WAD; return the map path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    materialize_texture_wad(output_dir)
    map_path = output_dir / f"{map_id}.map"
    with map_path.open("w", encoding="utf-8") as handle:
        build_arena_grid(spec).write(handle)
    return map_path
