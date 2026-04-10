"""Quake network protocol constants and binary reader.

Shared foundation for all demo parsing — metadata extraction, classification,
and BC analysis all import from here.
"""

from __future__ import annotations

import struct
from typing import List

# ---------------------------------------------------------------------------
# Server message types (svc_*)
# ---------------------------------------------------------------------------

SVC_NOP = 1
SVC_DISCONNECT = 2
SVC_UPDATESTAT = 3
SVC_VERSION = 4
SVC_SETVIEW = 5
SVC_SOUND = 6
SVC_TIME = 7
SVC_PRINT = 8
SVC_STUFFTEXT = 9
SVC_SETANGLE = 10
SVC_SERVERINFO = 11
SVC_LIGHTSTYLE = 12
SVC_UPDATENAME = 13
SVC_UPDATEFRAGS = 14
SVC_CLIENTDATA = 15
SVC_STOPSOUND = 16
SVC_UPDATECOLORS = 17
SVC_PARTICLE = 18
SVC_DAMAGE = 19
SVC_SPAWNSTATIC = 20
SVC_SPAWNBASELINE = 22
SVC_TEMP_ENTITY = 23
SVC_SETPAUSE = 24
SVC_SIGNONNUM = 25
SVC_CENTERPRINT = 26
SVC_KILLEDMONSTER = 27
SVC_FOUNDSECRET = 28
SVC_SPAWNSTATICSOUND = 29
SVC_INTERMISSION = 30
SVC_FINALE = 31
SVC_CDTRACK = 32
SVC_SELLSCREEN = 33
SVC_CUTSCENE = 34

# ---------------------------------------------------------------------------
# Entity update bits (U_*)
# ---------------------------------------------------------------------------

U_MOREBITS = 1 << 0
U_ORIGIN1 = 1 << 1
U_ORIGIN2 = 1 << 2
U_ORIGIN3 = 1 << 3
U_ANGLE2 = 1 << 4
U_FRAME = 1 << 6
U_ANGLE1 = 1 << 8
U_ANGLE3 = 1 << 9
U_MODEL = 1 << 10
U_COLORMAP = 1 << 11
U_SKIN = 1 << 12
U_EFFECTS = 1 << 13
U_LONGENTITY = 1 << 14

# ---------------------------------------------------------------------------
# Client data bits (SU_*)
# ---------------------------------------------------------------------------

SU_VIEWHEIGHT = 1 << 0
SU_IDEALPITCH = 1 << 1
SU_PUNCH1 = 1 << 2
SU_VELOCITY1 = 1 << 5
SU_ITEMS = 1 << 9
SU_WEAPONFRAME = 1 << 12
SU_ARMOR = 1 << 13
SU_WEAPON = 1 << 14

# ---------------------------------------------------------------------------
# Temp entity groups
# ---------------------------------------------------------------------------

TE_SIMPLE_COORD = {0, 1, 2, 3, 4, 7, 8, 10, 11}
TE_BEAM = {5, 6, 9, 13}

# ---------------------------------------------------------------------------
# Stat indices
# ---------------------------------------------------------------------------

STAT_HEALTH = 0
STAT_AMMO = 3
STAT_SHELLS = 6
STAT_NAILS = 7
STAT_ROCKETS = 8
STAT_CELLS = 9

# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------


class Reader:
    """Fast binary reader for Quake demo message payloads."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos

    def byte(self) -> int:
        v = self.data[self.pos]
        self.pos += 1
        return v

    def char(self) -> int:
        return struct.unpack_from("<b", self.data, self._advance(1))[0]

    def short(self) -> int:
        return struct.unpack_from("<h", self.data, self._advance(2))[0]

    def ushort(self) -> int:
        return struct.unpack_from("<H", self.data, self._advance(2))[0]

    def long(self) -> int:
        return struct.unpack_from("<i", self.data, self._advance(4))[0]

    def float(self) -> float:
        return struct.unpack_from("<f", self.data, self._advance(4))[0]

    def coord(self) -> float:
        return self.short() * (1.0 / 8.0)

    def angle(self) -> float:
        return self.char() * (360.0 / 256.0)

    def string(self) -> str:
        start = self.pos
        while self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            if b == 0 or b == 255:
                break
        raw = self.data[start : self.pos - 1]
        return bytes(b & 0xFF for b in raw).decode("latin-1", errors="replace")

    def skip(self, n: int) -> None:
        self.pos += n

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def _advance(self, n: int) -> int:
        p = self.pos
        self.pos += n
        return p


# ---------------------------------------------------------------------------
# Shared skip helpers (used by both parser and analyze)
# ---------------------------------------------------------------------------


def skip_sound(r: Reader) -> None:
    mask = r.byte()
    if mask & 1:
        r.byte()
    if mask & 2:
        r.byte()
    r.short()
    r.byte()
    r.coord()
    r.coord()
    r.coord()


def skip_particle(r: Reader) -> None:
    r.coord()
    r.coord()
    r.coord()
    r.char()
    r.char()
    r.char()
    r.byte()
    r.byte()


def skip_damage(r: Reader) -> None:
    r.byte()
    r.byte()
    r.coord()
    r.coord()
    r.coord()


def skip_baseline(r: Reader) -> None:
    r.byte()
    r.byte()
    r.byte()
    r.byte()
    for _ in range(3):
        r.coord()
        r.angle()


def skip_temp_entity(r: Reader) -> None:
    t = r.byte()
    if t in TE_SIMPLE_COORD:
        r.coord()
        r.coord()
        r.coord()
    elif t in TE_BEAM:
        r.short()
        for _ in range(6):
            r.coord()
    elif t == 12:
        r.coord()
        r.coord()
        r.coord()
        r.byte()
        r.byte()
    else:
        raise ValueError(f"Unsupported temp entity type {t}")


def skip_serverinfo(r: Reader) -> None:
    r.long()    # protocol
    r.byte()    # maxclients
    r.byte()    # gametype
    r.string()  # level name
    while r.string():
        pass  # models
    while r.string():
        pass  # sounds


def skip_entity_update(r: Reader, bits: int) -> None:
    if bits & U_MOREBITS:
        bits |= r.byte() << 8
    if bits & U_LONGENTITY:
        r.short()
    else:
        r.byte()
    if bits & U_MODEL:
        r.byte()
    if bits & U_FRAME:
        r.byte()
    if bits & U_COLORMAP:
        r.byte()
    if bits & U_SKIN:
        r.byte()
    if bits & U_EFFECTS:
        r.byte()
    if bits & U_ORIGIN1:
        r.coord()
    if bits & U_ANGLE1:
        r.angle()
    if bits & U_ORIGIN2:
        r.coord()
    if bits & U_ANGLE2:
        r.angle()
    if bits & U_ORIGIN3:
        r.coord()
    if bits & U_ANGLE3:
        r.angle()


def skip_clientdata(r: Reader) -> None:
    """Advance reader past an SVC_CLIENTDATA message, discarding values."""
    bits = r.ushort()
    if bits & SU_VIEWHEIGHT:
        r.char()
    if bits & SU_IDEALPITCH:
        r.char()
    for axis in range(3):
        if bits & (SU_PUNCH1 << axis):
            r.char()
        if bits & (SU_VELOCITY1 << axis):
            r.char()
    r.long()  # items
    if bits & SU_WEAPONFRAME:
        r.byte()
    if bits & SU_ARMOR:
        r.byte()
    if bits & SU_WEAPON:
        r.byte()
    r.short()  # health
    r.byte()   # ammo
    for _ in range(4):
        r.byte()  # shells, nails, rockets, cells
    r.byte()  # active weapon


def read_clientdata(r: Reader) -> tuple[int, int, int, int, int, int]:
    """Parse SVC_CLIENTDATA, return (health, ammo, shells, nails, rockets, cells)."""
    bits = r.ushort()
    if bits & SU_VIEWHEIGHT:
        r.char()
    if bits & SU_IDEALPITCH:
        r.char()
    for axis in range(3):
        if bits & (SU_PUNCH1 << axis):
            r.char()
        if bits & (SU_VELOCITY1 << axis):
            r.char()
    r.long()  # items
    if bits & SU_WEAPONFRAME:
        r.byte()
    if bits & SU_ARMOR:
        r.byte()
    if bits & SU_WEAPON:
        r.byte()
    health = r.short()
    ammo = r.byte()
    shells = r.byte()
    nails = r.byte()
    rockets = r.byte()
    cells = r.byte()
    r.byte()  # activeweapon
    return health, ammo, shells, nails, rockets, cells
