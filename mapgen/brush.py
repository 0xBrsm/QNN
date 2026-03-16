"""Quake .map format primitives: brushes and entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import TextIO

from .textures import DEFAULT_TEXTURE_WAD, MAP_TEXTURES

# Quake engine lightmap limit: a face can have at most 18 luxels per axis.
# At texture scale S, a face of width W uses ceil(W / (16*S)) + 1 luxels.
# To stay safe: W / (16 * S) <= 16, so S >= W / 256.
# We use a slightly more conservative limit.
MAX_LUXELS = 16
LUXEL_SIZE = 16  # texels per luxel at scale 1.0


def _safe_scale(extent: int) -> float:
    """Compute minimum texture scale so a face of `extent` units fits in lightmap."""
    if extent <= MAX_LUXELS * LUXEL_SIZE:
        return 1.0
    return ceil(extent / (MAX_LUXELS * LUXEL_SIZE))


@dataclass
class Plane:
    """One face of a brush, defined by three clockwise points + texture info."""

    p1: tuple[int, int, int]
    p2: tuple[int, int, int]
    p3: tuple[int, int, int]
    texture: str = MAP_TEXTURES.floor
    x_off: int = 0
    y_off: int = 0
    rotation: float = 0.0
    x_scale: float = 1.0
    y_scale: float = 1.0

    def write(self, f: TextIO) -> None:
        p = lambda t: f"( {t[0]} {t[1]} {t[2]} )"
        f.write(
            f"{p(self.p1)} {p(self.p2)} {p(self.p3)} "
            f"{self.texture} {self.x_off} {self.y_off} "
            f"{self.rotation} {self.x_scale} {self.y_scale}\n"
        )


@dataclass
class Brush:
    """Convex solid defined by the intersection of half-spaces (planes)."""

    planes: list[Plane] = field(default_factory=list)

    def write(self, f: TextIO) -> None:
        f.write("{\n")
        for plane in self.planes:
            plane.write(f)
        f.write("}\n")


def axis_aligned_box(
    min_x: int,
    min_y: int,
    min_z: int,
    max_x: int,
    max_y: int,
    max_z: int,
    texture: str = MAP_TEXTURES.floor,
) -> Brush:
    """Create an axis-aligned box brush with safe texture scales.

    Texture scale is increased on faces that would exceed the engine's
    lightmap limit (18 luxels per axis).
    """
    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z

    # Each face is defined by the two non-normal axes.
    # -X/+X faces span Y and Z.
    sx_yz = (_safe_scale(dy), _safe_scale(dz))
    # -Y/+Y faces span X and Z.
    sy_xz = (_safe_scale(dx), _safe_scale(dz))
    # -Z/+Z faces span X and Y.
    sz_xy = (_safe_scale(dx), _safe_scale(dy))

    return Brush(
        planes=[
            # -X face
            Plane((min_x, 0, 0), (min_x, 1, 0), (min_x, 0, 1), texture,
                  x_scale=sx_yz[0], y_scale=sx_yz[1]),
            # +X face
            Plane((max_x, 0, 0), (max_x, 0, 1), (max_x, 1, 0), texture,
                  x_scale=sx_yz[0], y_scale=sx_yz[1]),
            # -Y face
            Plane((0, min_y, 0), (0, min_y, 1), (1, min_y, 0), texture,
                  x_scale=sy_xz[0], y_scale=sy_xz[1]),
            # +Y face
            Plane((0, max_y, 0), (1, max_y, 0), (0, max_y, 1), texture,
                  x_scale=sy_xz[0], y_scale=sy_xz[1]),
            # -Z face (floor)
            Plane((0, 0, min_z), (1, 0, min_z), (0, 1, min_z), texture,
                  x_scale=sz_xy[0], y_scale=sz_xy[1]),
            # +Z face (ceiling)
            Plane((0, 0, max_z), (0, 1, max_z), (1, 0, max_z), texture,
                  x_scale=sz_xy[0], y_scale=sz_xy[1]),
        ]
    )


@dataclass
class Entity:
    """A Quake entity with key-value properties and optional brushes."""

    properties: dict[str, str] = field(default_factory=dict)
    brushes: list[Brush] = field(default_factory=list)

    def write(self, f: TextIO) -> None:
        f.write("{\n")
        for key, value in self.properties.items():
            f.write(f'"{key}" "{value}"\n')
        for brush in self.brushes:
            brush.write(f)
        f.write("}\n")


@dataclass
class MapFile:
    """A complete .map file: worldspawn entity followed by point entities."""

    worldspawn: Entity = field(default_factory=lambda: Entity(
        properties={
            "classname": "worldspawn",
            "wad": DEFAULT_TEXTURE_WAD,
            "worldtype": "2",
        }
    ))
    entities: list[Entity] = field(default_factory=list)

    def add_brush(self, brush: Brush) -> None:
        """Add a brush to worldspawn."""
        self.worldspawn.brushes.append(brush)

    def add_entity(self, classname: str, origin: tuple[int, int, int], **props: str) -> None:
        """Add a point entity."""
        self.entities.append(Entity(properties={
            "classname": classname,
            "origin": f"{origin[0]} {origin[1]} {origin[2]}",
            **props,
        }))

    def add_light(self, origin: tuple[int, int, int], brightness: int = 300) -> None:
        self.add_entity("light", origin, light=str(brightness))

    def write(self, f: TextIO) -> None:
        self.worldspawn.write(f)
        for entity in self.entities:
            entity.write(f)
