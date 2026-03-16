"""BSP subdivision layout generator for procedural FFA arenas.

Strategy:
  1. Recursively subdivide a bounding rectangle into leaf rooms.
  2. Randomize floor elevation and ceiling height per room.
  3. Connect adjacent rooms with corridors through their shared wall.
  4. Build stairs in corridors that bridge height differences.
  5. Fill every BSP subdivision gap with solid (corridor-shaped holes cut out).
     This ensures the map is sealed — no void leaks between rooms.
  6. Place hazard pools (lava/water/slime) with proper floor cutouts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .brush import MapFile, axis_aligned_box
from .textures import MAP_TEXTURES

# Layout constants (Quake units).
WALL = 16
FLOOR = 16
CORRIDOR_WIDTH = 96   # wide enough for comfortable movement
MIN_ROOM_SIZE = 256    # smallest room side
MAX_ROOM_SIZE = 1024   # force splits beyond this to avoid giant rooms

# Height variation — id1 DM maps have 25-40 distinct floor levels, z_range 500-900.
# We generate per-room elevations in 16-unit steps across a wide range.
FLOOR_ELEVATION_MIN = -128
FLOOR_ELEVATION_MAX = 384
FLOOR_ELEVATION_STEP = 16  # quantize to Quake grid
CEILING_HEIGHTS = [128, 160, 192, 224, 256, 320]  # floor-to-ceiling (id1 avg ~90)
CORRIDOR_HEADROOM = 112  # minimum ceiling above corridor floor

# Navmesh / movement constants (matching Quake physics).
WALKABLE_HEIGHT = 56   # player standing height (hull 1 z-extent)
WALKABLE_CLIMB = 18    # max single-step height player can walk up
WALKABLE_RADIUS = 16   # player collision half-width

# Stair geometry.
STAIR_DEPTH = 16   # each step is 16 units deep along corridor axis
STAIR_HEIGHT = 8   # each step is 8 units tall (Quake standard)
MAX_STAIR_RISE = 96   # max height difference stairs can bridge
MIN_STEP_RUN = 16    # minimum depth per step to avoid degenerate hull expansion

# Pool geometry.
POOL_DEPTH = 48
POOL_BORDER = 64   # inset from room walls
POOL_WALL = 16     # pit wall thickness (>= 16 for clean hull expansion)


@dataclass
class Room:
    """A rectangular room in the layout."""

    x0: int
    y0: int
    x1: int
    y1: int
    z0: int = 0      # floor elevation
    z1: int = 192     # ceiling elevation

    @property
    def cx(self) -> int:
        return (self.x0 + self.x1) // 2

    @property
    def cy(self) -> int:
        return (self.y0 + self.y1) // 2

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def headroom(self) -> int:
        return self.z1 - self.z0


@dataclass
class Pool:
    """A hazard pool (lava, water, slime) recessed into a room floor."""

    x0: int
    y0: int
    x1: int
    y1: int
    texture: str  # *lava1, *04water1, *slime


@dataclass
class Corridor:
    """A corridor connecting two rooms through a shared wall."""

    room_a: int  # index into rooms list
    room_b: int
    # Corridor bounding box (interior space)
    x0: int
    y0: int
    x1: int
    y1: int
    z0: int = 0           # floor at lower end
    z1: int = 128         # ceiling
    z0_a: int = 0         # floor elevation at room_a end
    z0_b: int = 0         # floor elevation at room_b end
    axis: str = "x"       # "x" = runs east-west, "y" = runs north-south


@dataclass
class Split:
    """A BSP subdivision split — records the gap between child halves."""

    axis: str       # "x" or "y" — which axis was split
    pos: int        # coordinate where the gap starts (along split axis)
    perp_lo: int    # perpendicular extent, low end
    perp_hi: int    # perpendicular extent, high end


@dataclass
class Layout:
    rooms: list[Room] = field(default_factory=list)
    corridors: list[Corridor] = field(default_factory=list)
    splits: list[Split] = field(default_factory=list)
    pools: dict[int, Pool] = field(default_factory=dict)  # room_idx -> Pool


def subdivide(
    rng: random.Random,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    depth: int,
    max_depth: int,
) -> tuple[list[Room], list[Split]]:
    """Recursively subdivide a rectangle into rooms via BSP.

    Returns both the leaf rooms and a list of splits (gaps between children).
    Rooms larger than MAX_ROOM_SIZE are force-split even past max_depth.
    """
    w = x1 - x0
    h = y1 - y0

    can_split_x = w >= MIN_ROOM_SIZE * 2 + CORRIDOR_WIDTH
    can_split_y = h >= MIN_ROOM_SIZE * 2 + CORRIDOR_WIDTH

    # Force split if room is too large, even past max_depth.
    oversized = w > MAX_ROOM_SIZE or h > MAX_ROOM_SIZE
    at_limit = depth >= max_depth

    if (at_limit and not oversized) or (not can_split_x and not can_split_y):
        # Randomize height — quantize to grid like id1 maps.
        n_steps = (FLOOR_ELEVATION_MAX - FLOOR_ELEVATION_MIN) // FLOOR_ELEVATION_STEP
        floor_z = FLOOR_ELEVATION_MIN + rng.randint(0, n_steps) * FLOOR_ELEVATION_STEP
        ceil_h = rng.choice(CEILING_HEIGHTS)
        return [Room(x0, y0, x1, y1, z0=floor_z, z1=floor_z + ceil_h)], []

    if can_split_x and can_split_y:
        split_x = rng.random() < (0.7 if w >= h else 0.3)
    else:
        split_x = can_split_x

    if split_x:
        margin = MIN_ROOM_SIZE
        lo = x0 + margin
        hi = x1 - margin - CORRIDOR_WIDTH
        # Bias toward center to avoid extreme room sizes.
        mid = (lo + hi) // 2
        quarter = max(1, (hi - lo) // 4)
        split = rng.randint(max(lo, mid - quarter), min(hi, mid + quarter))
        left_rooms, left_splits = subdivide(rng, x0, y0, split, y1, depth + 1, max_depth)
        right_rooms, right_splits = subdivide(rng, split + CORRIDOR_WIDTH, y0, x1, y1, depth + 1, max_depth)
        my_split = Split(axis="x", pos=split, perp_lo=y0, perp_hi=y1)
        return left_rooms + right_rooms, left_splits + right_splits + [my_split]
    else:
        margin = MIN_ROOM_SIZE
        lo = y0 + margin
        hi = y1 - margin - CORRIDOR_WIDTH
        mid = (lo + hi) // 2
        quarter = max(1, (hi - lo) // 4)
        split = rng.randint(max(lo, mid - quarter), min(hi, mid + quarter))
        bot_rooms, bot_splits = subdivide(rng, x0, y0, x1, split, depth + 1, max_depth)
        top_rooms, top_splits = subdivide(rng, x0, split + CORRIDOR_WIDTH, x1, y1, depth + 1, max_depth)
        my_split = Split(axis="y", pos=split, perp_lo=x0, perp_hi=x1)
        return bot_rooms + top_rooms, bot_splits + top_splits + [my_split]


def _rooms_share_wall(a: Room, b: Room) -> str | None:
    """Check if two rooms are separated by a corridor-width gap on one axis."""
    gap = CORRIDOR_WIDTH
    if b.x0 - a.x1 == gap:
        overlap_y0 = max(a.y0, b.y0)
        overlap_y1 = min(a.y1, b.y1)
        if overlap_y1 - overlap_y0 >= CORRIDOR_WIDTH:
            return "x"
    if b.y0 - a.y1 == gap:
        overlap_x0 = max(a.x0, b.x0)
        overlap_x1 = min(a.x1, b.x1)
        if overlap_x1 - overlap_x0 >= CORRIDOR_WIDTH:
            return "y"
    return None


def connect_rooms(rng: random.Random, rooms: list[Room]) -> list[Corridor]:
    """Connect adjacent rooms with corridors, handling height differences."""
    corridors: list[Corridor] = []
    n = len(rooms)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = rooms[i], rooms[j]
            axis = _rooms_share_wall(a, b) or _rooms_share_wall(b, a)
            if axis is None:
                continue

            if axis == "x" and b.x1 < a.x0:
                a, b = b, a
                i_idx, j_idx = j, i
            elif axis == "y" and b.y1 < a.y0:
                a, b = b, a
                i_idx, j_idx = j, i
            else:
                i_idx, j_idx = i, j

            # Corridor floor = lower of the two rooms.
            # Ceiling = headroom above higher floor, clamped to not exceed
            # either room.  constrain_elevations will raise ceilings if needed.
            low_z = min(a.z0, b.z0)
            high_z = max(a.z0, b.z0)
            ceil_z = min(high_z + CORRIDOR_HEADROOM, a.z1, b.z1)

            if axis == "x":
                overlap_y0 = max(a.y0, b.y0)
                overlap_y1 = min(a.y1, b.y1)
                cy = rng.randint(overlap_y0 + CORRIDOR_WIDTH // 2,
                                 overlap_y1 - CORRIDOR_WIDTH // 2)
                corridors.append(Corridor(
                    room_a=i_idx, room_b=j_idx,
                    x0=a.x1, y0=cy - CORRIDOR_WIDTH // 2,
                    x1=b.x0, y1=cy + CORRIDOR_WIDTH // 2,
                    z0=low_z, z1=ceil_z,
                    z0_a=rooms[i_idx].z0, z0_b=rooms[j_idx].z0,
                    axis="x",
                ))
            else:
                overlap_x0 = max(a.x0, b.x0)
                overlap_x1 = min(a.x1, b.x1)
                cx = rng.randint(overlap_x0 + CORRIDOR_WIDTH // 2,
                                 overlap_x1 - CORRIDOR_WIDTH // 2)
                corridors.append(Corridor(
                    room_a=i_idx, room_b=j_idx,
                    x0=cx - CORRIDOR_WIDTH // 2, y0=a.y1,
                    x1=cx + CORRIDOR_WIDTH // 2, y1=b.y0,
                    z0=low_z, z1=ceil_z,
                    z0_a=rooms[i_idx].z0, z0_b=rooms[j_idx].z0,
                    axis="y",
                ))

    return corridors


def _generate_pools(
    rng: random.Random,
    rooms: list[Room],
) -> dict[int, Pool]:
    """Generate hazard pools for rooms. Returns dict of room_idx -> Pool.

    Chances per room: lava 15%, water 35%, slime 5%.
    At least 2 pools guaranteed per map.
    """
    pools: dict[int, Pool] = {}
    for i, room in enumerate(rooms):
        if room.width < 320 or room.height < 320:
            continue

        roll = rng.random()
        # Force hazards in first eligible rooms if none placed yet.
        force = len(pools) < 2
        if roll < 0.15:
            tex = MAP_TEXTURES.lava
        elif roll < 0.50:
            tex = MAP_TEXTURES.water
        elif roll < 0.55:
            tex = MAP_TEXTURES.slime
        elif force:
            tex = MAP_TEXTURES.water  # guarantee met: default to water
        else:
            continue

        max_pool_x = min(256, room.width - POOL_BORDER * 2)
        max_pool_y = min(256, room.height - POOL_BORDER * 2)
        if max_pool_x < 96 or max_pool_y < 96:
            continue

        pool_w = rng.randint(96, max_pool_x)
        pool_h = rng.randint(96, max_pool_y)
        px = rng.randint(room.x0 + POOL_BORDER, room.x1 - POOL_BORDER - pool_w)
        py = rng.randint(room.y0 + POOL_BORDER, room.y1 - POOL_BORDER - pool_h)

        pools[i] = Pool(px, py, px + pool_w, py + pool_h, tex)

    return pools


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _build_stairs(
    m: MapFile,
    corridor: Corridor,
    floor_tex: str = MAP_TEXTURES.floor,
) -> None:
    """Build stair-step brushes inside a corridor to bridge height differences.

    If the height difference exceeds MAX_STAIR_RISE, stairs bridge as much as
    possible and the remaining drop becomes a ledge (player falls or jumps).
    Steps are at least MIN_STEP_RUN deep and at most WALKABLE_CLIMB tall.
    """
    dz = corridor.z0_b - corridor.z0_a
    if dz == 0:
        return

    # Clamp to max stair rise — excess becomes a drop/ledge.
    bridged = min(abs(dz), MAX_STAIR_RISE)
    c = corridor
    total_run = (c.x1 - c.x0) if c.axis == "x" else (c.y1 - c.y0)

    max_steps = total_run // MIN_STEP_RUN
    # Ensure enough steps so each is within WALKABLE_CLIMB.
    min_steps = (bridged + WALKABLE_CLIMB - 1) // WALKABLE_CLIMB
    n_steps = max(min_steps, min(bridged // STAIR_HEIGHT, max_steps))
    if n_steps == 0 or n_steps > max_steps:
        return

    # Compute per-step rise to bridge the full height evenly.
    # This handles non-multiple-of-8 heights by distributing the remainder.
    rises = [bridged // n_steps] * n_steps
    remainder = bridged - rises[0] * n_steps
    for i in range(remainder):
        rises[i] += 1
    sign = 1 if dz > 0 else -1

    base_z = c.z0_a
    slab_bot = c.z0 - FLOOR
    step_run = total_run // n_steps
    cumulative_rise = 0

    for i in range(n_steps):
        # Run position along corridor axis.
        run_lo = i * step_run
        run_hi = (i + 1) * step_run if i < n_steps - 1 else total_run

        if dz > 0:
            cumulative_rise += rises[i]
            slab_top = base_z + cumulative_rise
        else:
            slab_top = base_z - cumulative_rise
            cumulative_rise += rises[i]

        if c.axis == "x":
            m.add_brush(axis_aligned_box(
                c.x0 + run_lo, c.y0, slab_bot,
                c.x0 + run_hi, c.y1, slab_top, floor_tex
            ))
        else:
            m.add_brush(axis_aligned_box(
                c.x0, c.y0 + run_lo, slab_bot,
                c.x1, c.y0 + run_hi, slab_top, floor_tex
            ))


def _build_threshold(
    m: MapFile,
    corridor: Corridor,
    rooms: list[Room],
    floor_tex: str = MAP_TEXTURES.floor,
) -> None:
    """Build solid threshold slabs where a corridor meets a higher room.

    Without these, the gap fill hole extends below the room's floor,
    exposing the void under the room and making floors appear to float.
    Skip when stairs handle the elevation change — emitting both creates
    an overlapping solid that blocks the stair run.
    """
    c = corridor
    if c.z0_a != c.z0_b:
        return  # stairs handle this corridor
    for room_idx, room_z0 in [(c.room_a, c.z0_a), (c.room_b, c.z0_b)]:
        if room_z0 <= c.z0:
            continue  # corridor floor matches or is above room floor
        # Fill from corridor floor up to room floor, spanning half the corridor.
        if c.axis == "x":
            mid_x = (c.x0 + c.x1) // 2
            if room_idx == c.room_a:
                tx0, tx1 = c.x0, mid_x
            else:
                tx0, tx1 = mid_x, c.x1
            m.add_brush(axis_aligned_box(
                tx0, c.y0, c.z0, tx1, c.y1, room_z0, floor_tex
            ))
        else:
            mid_y = (c.y0 + c.y1) // 2
            if room_idx == c.room_a:
                ty0, ty1 = c.y0, mid_y
            else:
                ty0, ty1 = mid_y, c.y1
            m.add_brush(axis_aligned_box(
                c.x0, ty0, c.z0, c.x1, ty1, room_z0, floor_tex
            ))


def _build_gap_fills(
    m: MapFile,
    layout: Layout,
    z_lo: int,
    z_hi: int,
    tex: str = MAP_TEXTURES.fill,
) -> None:
    """Fill every BSP subdivision gap with solid, cutting corridor-shaped holes.

    Each BSP split leaves a CORRIDOR_WIDTH gap.  We fill it completely, then
    punch rectangular holes where corridors pass through.  This replaces
    individual room walls and corridor side walls, guaranteeing the map is
    sealed with no void leaks.
    """
    for split in layout.splits:
        if split.axis == "x":
            gap_corrs = [c for c in layout.corridors
                         if c.axis == "x" and c.x0 == split.pos]
            _fill_gap_x(m, split, gap_corrs, z_lo, z_hi, tex)
        else:
            gap_corrs = [c for c in layout.corridors
                         if c.axis == "y" and c.y0 == split.pos]
            _fill_gap_y(m, split, gap_corrs, z_lo, z_hi, tex)


def _fill_gap_x(
    m: MapFile,
    split: Split,
    corridors: list[Corridor],
    z_lo: int,
    z_hi: int,
    tex: str,
) -> None:
    """Fill an x-axis gap (solid from split.pos to split.pos+CW in x)."""
    gx0 = split.pos
    gx1 = split.pos + CORRIDOR_WIDTH

    # Sort corridor openings by perpendicular (y) position.
    openings = sorted([(c.y0, c.y1, c.z0, c.z1) for c in corridors],
                      key=lambda g: g[0])

    cursor = split.perp_lo
    for oy0, oy1, oz0, oz1 in openings:
        # Solid before this corridor opening.
        if oy0 > cursor:
            m.add_brush(axis_aligned_box(gx0, cursor, z_lo, gx1, oy0, z_hi, tex))
        # Solid below corridor opening.
        if oz0 > z_lo:
            m.add_brush(axis_aligned_box(gx0, oy0, z_lo, gx1, oy1, oz0, tex))
        # Solid above corridor opening.
        if oz1 < z_hi:
            m.add_brush(axis_aligned_box(gx0, oy0, oz1, gx1, oy1, z_hi, tex))
        cursor = oy1
    # Solid after last corridor.
    if cursor < split.perp_hi:
        m.add_brush(axis_aligned_box(gx0, cursor, z_lo, gx1, split.perp_hi, z_hi, tex))


def _fill_gap_y(
    m: MapFile,
    split: Split,
    corridors: list[Corridor],
    z_lo: int,
    z_hi: int,
    tex: str,
) -> None:
    """Fill a y-axis gap (solid from split.pos to split.pos+CW in y)."""
    gy0 = split.pos
    gy1 = split.pos + CORRIDOR_WIDTH

    # Sort corridor openings by perpendicular (x) position.
    openings = sorted([(c.x0, c.x1, c.z0, c.z1) for c in corridors],
                      key=lambda g: g[0])

    cursor = split.perp_lo
    for ox0, ox1, oz0, oz1 in openings:
        if ox0 > cursor:
            m.add_brush(axis_aligned_box(cursor, gy0, z_lo, ox0, gy1, z_hi, tex))
        if oz0 > z_lo:
            m.add_brush(axis_aligned_box(ox0, gy0, z_lo, ox1, gy1, oz0, tex))
        if oz1 < z_hi:
            m.add_brush(axis_aligned_box(ox0, gy0, oz1, ox1, gy1, z_hi, tex))
        cursor = ox1
    if cursor < split.perp_hi:
        m.add_brush(axis_aligned_box(cursor, gy0, z_lo, split.perp_hi, gy1, z_hi, tex))


def _build_outer_shell(
    m: MapFile,
    layout: Layout,
    z_lo: int,
    z_hi: int,
) -> None:
    """Build 6 thick slabs around the arena to seal the map boundary.

    The inner faces of the shell touch the outermost room faces so there
    is no void between rooms and the shell.
    """
    # Arena boundary = outermost room faces.
    min_x = min(r.x0 for r in layout.rooms)
    max_x = max(r.x1 for r in layout.rooms)
    min_y = min(r.y0 for r in layout.rooms)
    max_y = max(r.y1 for r in layout.rooms)

    s = 32  # shell thickness
    tex = MAP_TEXTURES.shell

    # Floor slab (below everything).
    m.add_brush(axis_aligned_box(
        min_x - s, min_y - s, z_lo - s, max_x + s, max_y + s, z_lo, tex))
    # Ceiling slab (above everything).
    m.add_brush(axis_aligned_box(
        min_x - s, min_y - s, z_hi, max_x + s, max_y + s, z_hi + s, tex))
    # West wall.
    m.add_brush(axis_aligned_box(
        min_x - s, min_y - s, z_lo, min_x, max_y + s, z_hi, tex))
    # East wall.
    m.add_brush(axis_aligned_box(
        max_x, min_y - s, z_lo, max_x + s, max_y + s, z_hi, tex))
    # South wall.
    m.add_brush(axis_aligned_box(
        min_x, min_y - s, z_lo, max_x, min_y, z_hi, tex))
    # North wall.
    m.add_brush(axis_aligned_box(
        min_x, max_y, z_lo, max_x, max_y + s, z_hi, tex))


def build_room_brushes(
    m: MapFile,
    room: Room,
    pool: Pool | None = None,
    floor_tex: str = MAP_TEXTURES.floor,
    ceil_tex: str = MAP_TEXTURES.ceiling,
) -> None:
    """Build a room's floor and ceiling slabs.

    If a pool is present, the floor is split into 4 pieces around the pool
    hole, and pool geometry (pit walls, bottom, liquid brush) is emitted.
    """
    x0, y0, x1, y1 = room.x0, room.y0, room.x1, room.y1
    z0, z1 = room.z0, room.z1

    # Ceiling (always solid).
    m.add_brush(axis_aligned_box(x0, y0, z1, x1, y1, z1 + FLOOR, ceil_tex))

    if pool is None:
        # Simple solid floor.
        m.add_brush(axis_aligned_box(x0, y0, z0 - FLOOR, x1, y1, z0, floor_tex))
    else:
        # Split floor into 4 pieces around pool hole.
        p = pool
        # South strip (full width, below pool).
        if p.y0 > y0:
            m.add_brush(axis_aligned_box(x0, y0, z0 - FLOOR, x1, p.y0, z0, floor_tex))
        # North strip (full width, above pool).
        if p.y1 < y1:
            m.add_brush(axis_aligned_box(x0, p.y1, z0 - FLOOR, x1, y1, z0, floor_tex))
        # West strip (between south/north strips).
        if p.x0 > x0:
            m.add_brush(axis_aligned_box(x0, p.y0, z0 - FLOOR, p.x0, p.y1, z0, floor_tex))
        # East strip (between south/north strips).
        if p.x1 < x1:
            m.add_brush(axis_aligned_box(p.x1, p.y0, z0 - FLOOR, x1, p.y1, z0, floor_tex))

        # Pool pit: bottom slab.
        pit_bot = z0 - POOL_DEPTH
        m.add_brush(axis_aligned_box(
            p.x0, p.y0, pit_bot - FLOOR, p.x1, p.y1, pit_bot, floor_tex))
        # Pool pit: 4 walls from pit bottom to floor level.
        pw = POOL_WALL
        m.add_brush(axis_aligned_box(p.x0 - pw, p.y0 - pw, pit_bot, p.x0, p.y1 + pw, z0, floor_tex))
        m.add_brush(axis_aligned_box(p.x1, p.y0 - pw, pit_bot, p.x1 + pw, p.y1 + pw, z0, floor_tex))
        m.add_brush(axis_aligned_box(p.x0, p.y0 - pw, pit_bot, p.x1, p.y0, z0, floor_tex))
        m.add_brush(axis_aligned_box(p.x0, p.y1, pit_bot, p.x1, p.y1 + pw, z0, floor_tex))

        # Liquid brush: fills the pit, top flush with floor (+1 for engine detection).
        m.add_brush(axis_aligned_box(p.x0, p.y0, pit_bot, p.x1, p.y1, z0, p.texture))


def build_corridor_brushes(
    m: MapFile,
    corridor: Corridor,
    rooms: list[Room],
    floor_tex: str = MAP_TEXTURES.floor,
    ceil_tex: str = MAP_TEXTURES.ceiling,
) -> None:
    """Build corridor floor, ceiling, stairs, and threshold slabs.

    Side walls are handled by gap fills — the solid surrounding the
    corridor-shaped hole acts as the corridor boundary.
    """
    c = corridor
    x0, y0, x1, y1 = c.x0, c.y0, c.x1, c.y1
    z0, z1 = c.z0, c.z1

    # Base floor slab (at lowest elevation).
    m.add_brush(axis_aligned_box(x0, y0, z0 - FLOOR, x1, y1, z0, floor_tex))
    # Ceiling.
    m.add_brush(axis_aligned_box(x0, y0, z1, x1, y1, z1 + FLOOR, ceil_tex))

    # Stairs if rooms are at different heights.
    _build_stairs(m, corridor, floor_tex)

    # Threshold slabs where corridor meets a higher room.
    _build_threshold(m, corridor, rooms, floor_tex)


def _constrain_elevations(rooms: list[Room], corridors: list[Corridor]) -> None:
    """Clamp floor elevations so ALL connected rooms differ by at most max_rise.

    Without this, players drop into low rooms and can't climb back — stairs
    only bridge MAX_STAIR_RISE units.  Iterative relaxation: keep pulling
    room pairs toward each other until every corridor satisfies the constraint.
    """
    # Effective max rise: limited by stair geometry (corridor must have enough
    # run for the required number of steps).
    max_steps_per_corridor = CORRIDOR_WIDTH // MIN_STEP_RUN
    max_climbable = max_steps_per_corridor * WALKABLE_CLIMB
    max_rise = min(MAX_STAIR_RISE, max_climbable)

    max_iters = len(rooms) * 4  # converges quickly on small graphs
    for _ in range(max_iters):
        changed = False
        for c in corridors:
            a, b = rooms[c.room_a], rooms[c.room_b]
            dz = a.z0 - b.z0
            if abs(dz) <= max_rise:
                continue
            # Split the excess evenly between the two rooms.
            excess = abs(dz) - max_rise
            half = excess // 2
            rest = excess - half
            if dz > 0:
                # a is too high, b is too low — move toward each other.
                headroom_a = a.z1 - a.z0
                a.z0 -= half
                a.z1 = a.z0 + headroom_a
                headroom_b = b.z1 - b.z0
                b.z0 += rest
                b.z1 = b.z0 + headroom_b
            else:
                headroom_a = a.z1 - a.z0
                a.z0 += half
                a.z1 = a.z0 + headroom_a
                headroom_b = b.z1 - b.z0
                b.z0 -= rest
                b.z1 = b.z0 + headroom_b
            changed = True
        if not changed:
            break

    # Final enforcement: if any corridor still exceeds max_rise after
    # iterative relaxation (integer rounding residual), clamp the higher room.
    for c in corridors:
        a, b = rooms[c.room_a], rooms[c.room_b]
        dz = abs(a.z0 - b.z0)
        if dz > max_rise:
            high = a if a.z0 > b.z0 else b
            low = b if a.z0 > b.z0 else a
            headroom = high.z1 - high.z0
            high.z0 = low.z0 + max_rise
            high.z1 = high.z0 + headroom

    # Raise room ceilings where needed so corridors have CORRIDOR_HEADROOM
    # at the higher end.  Without this, low-ceiling rooms (128 headroom) with
    # large elevation differences produce corridors the player can barely fit
    # through (56-unit clearance = exactly hull height).
    for c in corridors:
        a, b = rooms[c.room_a], rooms[c.room_b]
        high_z = max(a.z0, b.z0)
        min_ceiling = high_z + CORRIDOR_HEADROOM
        if a.z1 < min_ceiling:
            a.z1 = min_ceiling
        if b.z1 < min_ceiling:
            b.z1 = min_ceiling

    # Update corridor z-values to match adjusted rooms.
    for c in corridors:
        a, b = rooms[c.room_a], rooms[c.room_b]
        c.z0_a = a.z0
        c.z0_b = b.z0
        c.z0 = min(a.z0, b.z0)
        high_z = max(a.z0, b.z0)
        c.z1 = min(high_z + CORRIDOR_HEADROOM, a.z1, b.z1)


def generate_layout(rng: random.Random, arena_size: int = 3072, max_depth: int = 3) -> Layout:
    """Generate a complete room + corridor layout."""
    half = arena_size // 2
    rooms, splits = subdivide(rng, -half, -half, half, half, depth=0, max_depth=max_depth)
    corridors = connect_rooms(rng, rooms)
    _constrain_elevations(rooms, corridors)
    pools = _generate_pools(rng, rooms)
    return Layout(rooms=rooms, corridors=corridors, splits=splits, pools=pools)


def build_layout(m: MapFile, layout: Layout) -> None:
    """Emit all geometry into a MapFile: outer shell, gap fills, rooms, corridors."""
    # Compute global z-range (all room/corridor floors and ceilings + slab thickness).
    z_lo = min(r.z0 for r in layout.rooms) - FLOOR
    z_hi = max(r.z1 for r in layout.rooms) + FLOOR
    for c in layout.corridors:
        z_lo = min(z_lo, c.z0 - FLOOR)
        z_hi = max(z_hi, c.z1 + FLOOR)
    # Account for pool pits.
    for room_idx, pool in layout.pools.items():
        pit_bot = layout.rooms[room_idx].z0 - POOL_DEPTH - FLOOR
        z_lo = min(z_lo, pit_bot)

    # 1. Outer shell — seals the arena boundary.
    _build_outer_shell(m, layout, z_lo, z_hi)

    # 2. Gap fills — solid between BSP splits with corridor holes.
    _build_gap_fills(m, layout, z_lo, z_hi)

    # 3. Room floors, ceilings, and pools.
    for i, room in enumerate(layout.rooms):
        build_room_brushes(m, room, pool=layout.pools.get(i))

    # 4. Corridor floors, ceilings, stairs, and thresholds.
    for corridor in layout.corridors:
        build_corridor_brushes(m, corridor, layout.rooms)
