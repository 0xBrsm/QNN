"""Entity placement: spawns, weapons, items, lights.

Tuned to match id1 DM map distributions:
  - Spawns: ~6-9 per map (1 per room)
  - Weapons: ~5-7 per map (~1:1 with spawns), all 6 types represented
  - Ammo: ~1-2 boxes near each weapon
  - Health: ~12-18 per map (~2 per room)
  - Armor: 3-6 per map (green/yellow common, 1 red)
  - Powerups: 0-1 quad, 0-1 pent (rare)
  - Hazards: handled by layout.py (pool geometry with floor cutouts)
"""

from __future__ import annotations

import random

from .brush import MapFile
from .layout import CORRIDOR_WIDTH, Layout, Pool, Room

# Height above floor for point entities.
# Must be > 24 to clear Quake hull 1/2 floor expansion (player mins.z = -24).
ITEM_Z_ABOVE_FLOOR = 32

# All 6 Quake weapons.
WEAPONS = [
    "weapon_supershotgun",
    "weapon_nailgun",
    "weapon_supernailgun",
    "weapon_grenadelauncher",
    "weapon_rocketlauncher",
    "weapon_lightning",
]

# Ammo paired with weapons.
AMMO_FOR_WEAPON = {
    "weapon_supershotgun": "item_shells",
    "weapon_nailgun": "item_spikes",
    "weapon_supernailgun": "item_spikes",
    "weapon_grenadelauncher": "item_rockets",
    "weapon_rocketlauncher": "item_rockets",
    "weapon_lightning": "item_cells",
}

# Armor tiers — id1 DM maps: 1-2 yellow, 0-2 green, 1 red typical.
ARMORS = ["item_armor1", "item_armor2", "item_armorInv"]

# Powerups — rare, 0-1 each per map.
POWERUPS = [
    "item_artifact_super_damage",      # quad damage
    "item_artifact_invulnerability",   # pentagram
    "item_artifact_invisibility",      # ring of shadows
]

# Margin from walls to keep entities clear.
MARGIN = 48

# Minimum spacing between placed entities (~2 player widths).
MIN_ENTITY_SPACING = 64


def _item_z(room: Room) -> int:
    """Entity Z position for a given room's floor."""
    return room.z0 + ITEM_Z_ABOVE_FLOOR


def _in_pool(x: int, y: int, pool: Pool | None) -> bool:
    """Check if an XY position overlaps a pool cutout."""
    if pool is None:
        return False
    return pool.x0 <= x <= pool.x1 and pool.y0 <= y <= pool.y1


def _in_gap_fill(x: int, y: int, layout: Layout, cw: int) -> bool:
    """Check if position is inside a gap fill solid (between BSP splits).

    Gap fills span split.pos to split.pos + corridor_width on the split axis,
    except where corridors punch through.
    """
    for s in layout.splits:
        if s.axis == "x":
            if x < s.pos or x > s.pos + cw:
                continue
            in_opening = any(
                c.axis == "x" and c.x0 == s.pos and c.y0 <= y <= c.y1
                for c in layout.corridors
            )
            if not in_opening:
                return True
        else:
            if y < s.pos or y > s.pos + cw:
                continue
            in_opening = any(
                c.axis == "y" and c.y0 == s.pos and c.x0 <= x <= c.x1
                for c in layout.corridors
            )
            if not in_opening:
                return True
    return False


def _too_close(
    x: int, y: int, placed: list[tuple[int, int]],
) -> bool:
    """Check if position is too close to any already-placed entity."""
    for px, py in placed:
        if abs(x - px) < MIN_ENTITY_SPACING and abs(y - py) < MIN_ENTITY_SPACING:
            return True
    return False


def _valid_pos(
    x: int, y: int,
    pool: Pool | None,
    layout: Layout,
    cw: int,
    placed: list[tuple[int, int]],
) -> bool:
    """Check if position is valid: not in pool, not in gap fill, not too close."""
    return (
        not _in_pool(x, y, pool)
        and not _in_gap_fill(x, y, layout, cw)
        and not _too_close(x, y, placed)
    )


def _safe_xy(
    rng: random.Random,
    room: Room,
    pool: Pool | None,
    layout: Layout,
    cw: int,
    placed: list[tuple[int, int]],
    margin: int = MARGIN,
) -> tuple[int, int]:
    """Pick a random XY inside room, avoiding pools, gap fills, and entity stacking.

    Fallback chain: random attempts -> center -> corners -> grid scan with
    spacing -> grid scan without spacing -> center (last resort).
    """
    # 20 random attempts.
    for _ in range(20):
        x = rng.randint(room.x0 + margin, room.x1 - margin)
        y = rng.randint(room.y0 + margin, room.y1 - margin)
        if _valid_pos(x, y, pool, layout, cw, placed):
            return x, y

    # Center.
    if _valid_pos(room.cx, room.cy, pool, layout, cw, placed):
        return room.cx, room.cy

    # Corners.
    corners = [
        (room.x0 + margin, room.y0 + margin),
        (room.x0 + margin, room.y1 - margin),
        (room.x1 - margin, room.y0 + margin),
        (room.x1 - margin, room.y1 - margin),
    ]
    for x, y in corners:
        if _valid_pos(x, y, pool, layout, cw, placed):
            return x, y

    # Grid scan with spacing.
    step = 32
    for sy in range(room.y0 + margin, room.y1 - margin + 1, step):
        for sx in range(room.x0 + margin, room.x1 - margin + 1, step):
            if _valid_pos(sx, sy, pool, layout, cw, placed):
                return sx, sy

    # Grid scan without spacing requirement.
    for sy in range(room.y0 + margin, room.y1 - margin + 1, step):
        for sx in range(room.x0 + margin, room.x1 - margin + 1, step):
            if not _in_pool(sx, sy, pool) and not _in_gap_fill(sx, sy, layout, cw):
                return sx, sy

    return room.cx, room.cy


def _place(
    m: MapFile,
    classname: str,
    x: int, y: int, z: int,
    placed: list[tuple[int, int]],
    **props: str,
) -> None:
    """Place an entity and record its position for deconfliction."""
    m.add_entity(classname, (x, y, z), **props)
    placed.append((x, y))


def place_spawns(
    m: MapFile,
    layout: Layout,
    rng: random.Random,
    placed: list[tuple[int, int]],
) -> None:
    """Place 1 DM spawn per room (id1: ~6-9 spawns for 6-9 rooms)."""
    cw = CORRIDOR_WIDTH
    for i, room in enumerate(layout.rooms):
        x, y = _safe_xy(rng, room, layout.pools.get(i), layout, cw, placed)
        angle = str(rng.choice([0, 90, 180, 270]))
        _place(m, "info_player_deathmatch", x, y, _item_z(room), placed, angle=angle)

    # info_player_start required for map to load.
    r = layout.rooms[0]
    x, y = _safe_xy(rng, r, layout.pools.get(0), layout, cw, placed)
    _place(m, "info_player_start", x, y, _item_z(r), placed, angle="0")


def place_weapons(
    m: MapFile,
    layout: Layout,
    rng: random.Random,
    placed: list[tuple[int, int]],
) -> None:
    """Place weapons — target ~1:1 with spawn count, all 6 types when possible.

    id1 DM: 4-7 weapons, most maps have all 6 types. RL often has 2 copies.
    """
    rooms = layout.rooms
    pools = layout.pools
    cw = CORRIDOR_WIDTH
    n_rooms = len(rooms)

    # Always place all 6 weapon types.
    weapons_to_place = list(WEAPONS)

    # If enough rooms, add a second RL (very common in id1 DM).
    if n_rooms >= 7:
        weapons_to_place.append("weapon_rocketlauncher")

    rng.shuffle(weapons_to_place)

    # Assign to rooms (round-robin if more weapons than rooms).
    room_order = list(range(n_rooms))
    rng.shuffle(room_order)

    for i, weapon in enumerate(weapons_to_place):
        room_idx = room_order[i % n_rooms]
        room = rooms[room_idx]
        pool = pools.get(room_idx)
        x, y = _safe_xy(rng, room, pool, layout, cw, placed)
        _place(m, weapon, x, y, _item_z(room), placed)

        # 1-2 ammo boxes near weapon.
        ammo = AMMO_FOR_WEAPON[weapon]
        for _ in range(rng.randint(1, 2)):
            ax = x + rng.randint(-80, 80)
            ay = y + rng.randint(-80, 80)
            ax = max(room.x0 + 32, min(room.x1 - 32, ax))
            ay = max(room.y0 + 32, min(room.y1 - 32, ay))
            if not _in_pool(ax, ay, pool) and not _in_gap_fill(ax, ay, layout, cw):
                _place(m, ammo, ax, ay, _item_z(room), placed)


def place_items(
    m: MapFile,
    layout: Layout,
    rng: random.Random,
    placed: list[tuple[int, int]],
) -> None:
    """Place health and armor pickups.

    id1 DM: ~12-18 health, 3-6 armor per map.
    """
    rooms = layout.rooms
    pools = layout.pools
    cw = CORRIDOR_WIDTH
    n_rooms = len(rooms)

    # Health: ~2-3 per room, targeting 12-18 total.
    target_health = max(10, min(20, n_rooms * 2 + rng.randint(2, 4)))
    health_placed = 0
    for i, room in enumerate(rooms):
        count = 1 if health_placed >= target_health else rng.randint(1, 3)
        pool = pools.get(i)
        for _ in range(count):
            if health_placed >= target_health:
                break
            x, y = _safe_xy(rng, room, pool, layout, cw, placed, margin=32)
            _place(m, "item_health", x, y, _item_z(room), placed)
            health_placed += 1

    # Armor: 3-6 total. 1 red, 1-2 yellow, 1-2 green.
    armor_pool: list[str] = []
    armor_pool.append("item_armorInv")  # always 1 red
    armor_pool.append("item_armor2")    # always 1 yellow
    armor_pool.append("item_armor1")    # always 1 green
    if n_rooms >= 5:
        armor_pool.append(rng.choice(["item_armor1", "item_armor2"]))
    if n_rooms >= 7:
        armor_pool.append(rng.choice(["item_armor1", "item_armor2"]))

    room_order = list(range(n_rooms))
    rng.shuffle(room_order)
    for i, armor in enumerate(armor_pool):
        room_idx = room_order[i % n_rooms]
        room = rooms[room_idx]
        x, y = _safe_xy(rng, room, pools.get(room_idx), layout, cw, placed)
        _place(m, armor, x, y, _item_z(room), placed)


def place_powerups(
    m: MapFile,
    layout: Layout,
    rng: random.Random,
    placed: list[tuple[int, int]],
) -> None:
    """Place 0-1 powerup per map. id1 DM: quad in ~4/6 maps, pent/ring rarer."""
    rooms = layout.rooms
    pools = layout.pools
    cw = CORRIDOR_WIDTH

    if rng.random() < 0.6:  # 60% chance of quad
        idx = rng.randrange(len(rooms))
        room = rooms[idx]
        x, y = _safe_xy(rng, room, pools.get(idx), layout, cw, placed)
        _place(m, "item_artifact_super_damage", x, y, _item_z(room), placed)

    if rng.random() < 0.25:  # 25% chance of pent or ring
        powerup = rng.choice([
            "item_artifact_invulnerability",
            "item_artifact_invisibility",
        ])
        idx = rng.randrange(len(rooms))
        room = rooms[idx]
        x, y = _safe_xy(rng, room, pools.get(idx), layout, cw, placed)
        _place(m, powerup, x, y, _item_z(room), placed)


def place_lights(m: MapFile, layout: Layout) -> None:
    """Place lights in rooms and corridors."""
    for room in layout.rooms:
        m.add_light((room.cx, room.cy, room.z1 - 16), brightness=300)
        if room.width > 400 and room.height > 400:
            qx = (room.x1 - room.x0) // 4
            qy = (room.y1 - room.y0) // 4
            for dx, dy in [(-qx, -qy), (qx, -qy), (-qx, qy), (qx, qy)]:
                m.add_light((room.cx + dx, room.cy + dy, room.z1 - 16), brightness=200)

    for c in layout.corridors:
        cx = (c.x0 + c.x1) // 2
        cy = (c.y0 + c.y1) // 2
        m.add_light((cx, cy, c.z1 - 8), brightness=200)


def populate(m: MapFile, layout: Layout, rng: random.Random) -> None:
    """Place all entities in the layout."""
    placed: list[tuple[int, int]] = []
    place_spawns(m, layout, rng, placed)
    place_weapons(m, layout, rng, placed)
    place_items(m, layout, rng, placed)
    place_powerups(m, layout, rng, placed)
    place_lights(m, layout)
