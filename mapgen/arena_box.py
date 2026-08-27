"""Sealed single-cell 1v1 box arena (``arena_box``) and scaled variants.

The shipped ``assets/id1/maps/arena_box.bsp`` predates this module and has no
tracked source (src/docs/vendor.md); its geometry — interior 1024x1024x192,
walls ~16u, mirrored spawns at +-256 on x — is reproduced here as the
``scale=1`` default so the venue family is regenerable from repo state. The
shipped BSP itself is NEVER overwritten: every historical eval ran on those
exact bytes, so scaled variants get their own map ids (``arena_box4x``).

Scaling is HORIZONTAL ONLY (half-extent and spawn offset); the ceiling stays
at the base height so jump/rocket dynamics are unchanged — the point of a
scaled cell is longer sightlines (LG's 600u range cap, the >1000u RL lead
void), not new vertical play.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .brush import MapFile, axis_aligned_box
from .textures import MAP_TEXTURES, materialize_texture_wad

ARENA_BOX_MAP_ID = "arena_box"


@dataclass(frozen=True)
class ArenaBoxSpec:
    """Geometry contract for one sealed 1v1 box, centered on the origin."""

    half: int = 512  # interior half-extent on x and y
    height: int = 192  # interior ceiling height
    wall: int = 16  # floor/ceiling/wall thickness
    spawn_offset: int = 256  # mirrored spawns at (+-spawn_offset, 0)
    light_spacing: int = 512  # ceiling light grid pitch
    light_brightness: int = 300

    def scaled(self, scale: int) -> "ArenaBoxSpec":
        """Horizontal scale-up: extents and spawns grow, height does not."""
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        return ArenaBoxSpec(
            half=self.half * scale,
            height=self.height,
            wall=self.wall,
            spawn_offset=self.spawn_offset * scale,
            light_spacing=self.light_spacing,
            light_brightness=self.light_brightness,
        )


def build_arena_box(spec: ArenaBoxSpec = ArenaBoxSpec(), *, message: str = ARENA_BOX_MAP_ID) -> MapFile:
    arena = MapFile()
    arena.worldspawn.properties.update({"message": message, "worldtype": "2"})

    h, w, z = spec.half, spec.wall, spec.height
    # Floor, ceiling, and four sealing perimeter walls.
    arena.add_brush(axis_aligned_box(-h - w, -h - w, -w, h + w, h + w, 0, MAP_TEXTURES.floor))
    arena.add_brush(axis_aligned_box(-h - w, -h - w, z, h + w, h + w, z + w, MAP_TEXTURES.ceiling))
    arena.add_brush(axis_aligned_box(-h - w, -h - w, 0, -h, h + w, z, MAP_TEXTURES.shell))
    arena.add_brush(axis_aligned_box(h, -h - w, 0, h + w, h + w, z, MAP_TEXTURES.shell))
    arena.add_brush(axis_aligned_box(-h, -h - w, 0, h, -h, z, MAP_TEXTURES.shell))
    arena.add_brush(axis_aligned_box(-h, h, 0, h, h + w, z, MAP_TEXTURES.shell))

    # Mirrored 1v1 spawns facing each other, matching the shipped arena_box.
    arena.add_entity("info_player_start", (-spec.spawn_offset, 0, 32), angle="0")
    arena.add_entity("info_player_deathmatch", (-spec.spawn_offset, 0, 32), angle="0")
    arena.add_entity("info_player_deathmatch", (spec.spawn_offset, 0, 32), angle="180")

    # Ceiling light grid — one center light suffices at scale 1, larger cells
    # get a pitch grid so the far field is not pitch black in demos.
    coords = range(-h + spec.light_spacing // 2, h, spec.light_spacing)
    lights = [(x, y) for x in coords for y in coords] or [(0, 0)]
    for x, y in lights:
        arena.add_light((x, y, z - 32), brightness=spec.light_brightness)
    return arena


def write_arena_box(
    output_dir: Path,
    *,
    spec: ArenaBoxSpec = ArenaBoxSpec(),
    map_id: str = ARENA_BOX_MAP_ID,
) -> Path:
    """Write the deterministic source map and texture WAD; return the map path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    materialize_texture_wad(output_dir)
    map_path = output_dir / f"{map_id}.map"
    with map_path.open("w", encoding="utf-8") as handle:
        build_arena_box(spec, message=map_id).write(handle)
    return map_path
