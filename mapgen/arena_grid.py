"""Deterministic eight-match Quake arena used by the PPO arena backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .brush import MapFile, axis_aligned_box
from .textures import MAP_TEXTURES, materialize_texture_wad


ARENA_GRID_MAP_ID = "qnn_arena8"


@dataclass(frozen=True)
class ArenaGridSpec:
    """Geometry and identity contract for a grid of sealed 1v1 matches."""

    columns: int = 4
    rows: int = 2
    cell_size: int = 512
    wall: int = 64
    height: int = 256
    spawn_offset: int = 128

    @property
    def match_count(self) -> int:
        return self.columns * self.rows

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

    arena = MapFile()
    arena.worldspawn.properties.update(
        {
            "message": "QNN eight-match PPO arena",
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
        arena.add_entity(
            "info_player_deathmatch",
            (cx - spec.spawn_offset, cy, 24),
            angle="0",
            qnn_match_id=str(match_id),
            qnn_seat_id="0",
        )
        arena.add_entity(
            "info_player_deathmatch",
            (cx + spec.spawn_offset, cy, 24),
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
