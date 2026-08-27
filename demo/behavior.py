"""NQ-demo behaviour analysis: fire discipline, weapon texture, projectile lead.

Promotes two ad-hoc probes (``scripts/analysis/_vertical_dem_probe.py`` and
``scripts/analysis/_anim_dem_probe.py``) into a proper module, and adds
projectile-track / lead analysis that neither probe had.

FRAME RECONSTRUCTION
  NetQuake entity updates are deltas vs each entity's SPAWNBASELINE, not vs
  the previous update. ``demo.parser`` gets this wrong for per-frame timeline
  reconstruction (it persists previous field values instead of rebuilding
  baseline + sent-bits every update). This module follows the PROBES'
  approach instead: every update starts from ``{baseline-value or
  live-carried-value}`` and only the bits actually present in the message
  overwrite it — see ``_apply_update`` below.

PLAYER IDENTITY
  Never identify a player by modelindex persistence. A player entity can
  briefly show ``progs/h_player.mdl`` (the gib/corpse model) instead of
  ``progs/player.mdl`` around a death, and this repo's own analysis of
  tmp/a28rc1e-g.dem found the tracked entity's modelindex genuinely flip
  4 -> 6 (player.mdl -> h_player.mdl) right around the mid-demo model
  reload. The only robust invariant in NetQuake is structural: entities
  1..maxclients ARE the player client slots, for the entire demo, whether
  or not a given slot is ever used. Names come from svc_updatename
  (0-based client slot -> name string); entity number = slot + 1.

BLOCK NUMBERING
  ``blk`` is the 0-based index of the raw length-prefixed demo message
  block (first block in the file is blk 0). This is the convention that
  reproduces the numbers Brian recorded by hand against tmp/a28rc1e-g.dem
  (3812 blocks total, model-load svc_print at blk 1377, 69/50 LG beams
  owned by ent 1 before/after that block).

USAGE
    python -m demo.behavior tmp/a28rc1e-g.dem
    python -m demo.behavior tmp/a28rc1e-g.dem --split-on-model-load --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .protocol import (
    Reader,
    SVC_CENTERPRINT,
    SVC_CDTRACK,
    SVC_CLIENTDATA,
    SVC_CUTSCENE,
    SVC_DAMAGE,
    SVC_DISCONNECT,
    SVC_FINALE,
    SVC_FOUNDSECRET,
    SVC_INTERMISSION,
    SVC_KILLEDMONSTER,
    SVC_LIGHTSTYLE,
    SVC_NOP,
    SVC_PARTICLE,
    SVC_PRINT,
    SVC_SELLSCREEN,
    SVC_SERVERINFO,
    SVC_SETANGLE,
    SVC_SETPAUSE,
    SVC_SETVIEW,
    SVC_SIGNONNUM,
    SVC_SOUND,
    SVC_SPAWNBASELINE,
    SVC_SPAWNSTATIC,
    SVC_SPAWNSTATICSOUND,
    SVC_STOPSOUND,
    SVC_STUFFTEXT,
    SVC_TEMP_ENTITY,
    SVC_TIME,
    SVC_UPDATECOLORS,
    SVC_UPDATEFRAGS,
    SVC_UPDATENAME,
    SVC_UPDATESTAT,
    SVC_VERSION,
    TE_BEAM,
    U_ANGLE1,
    U_ANGLE2,
    U_ANGLE3,
    U_COLORMAP,
    U_EFFECTS,
    U_FRAME,
    U_LONGENTITY,
    U_MODEL,
    U_MOREBITS,
    U_ORIGIN1,
    U_ORIGIN2,
    U_ORIGIN3,
    U_SKIN,
    skip_clientdata,
    read_clientdata,
    skip_particle,
    skip_sound,
)

# ---------------------------------------------------------------------------
# Fire tells (see _vertical_dem_probe.py for provenance)
# ---------------------------------------------------------------------------

ATTACK_FRAME_LO, ATTACK_FRAME_HI = 103, 142  # id1 player.qc attack anim range
EF_MUZZLEFLASH = 2
FACING_YAW_ERR_DEG = 20.0
LG_RANGE_U = 600.0  # LG effective-range boundary used for distance binning

# ---------------------------------------------------------------------------
# Temp-entity codes (numbering per NQ protocol.h; TE_BEAM/TE_SIMPLE_COORD
# already live in demo.protocol -- these two are named individually here
# because projectile-impact matching cares about them specifically)
# ---------------------------------------------------------------------------

TE_EXPLOSION = 3
TE_TAREXPLOSION = 4
IMPACT_TE = {TE_EXPLOSION, TE_TAREXPLOSION}
IMPACT_MAX_BLK_GAP = 8      # ticks after last-seen the explosion may land
IMPACT_MAX_DIST_U = 300.0   # generous: rocket speed ~1000u/s outruns 1 tick

# ---------------------------------------------------------------------------
# Projectile classification (progs/*.mdl -> weapon class); matches the
# server-side table in src/engine/common/qnn_entity.c
# ---------------------------------------------------------------------------

PROJECTILE_MODELS = {
    "progs/missile.mdl": "rocket",
    "progs/grenade.mdl": "grenade",
    "progs/spike.mdl": "nail",
    "progs/s_spike.mdl": "nail",
    "progs/laser.mdl": "laser",  # not in stock id1; kept for mod compat
}
LEAD_CLASSES = ("rocket", "grenade")  # projectile_lead only makes sense here

GAP_TICKS = 5  # bridge short PVS occlusions; beyond this, assume slot reuse
AMBIGUOUS_ATTRIBUTION_DIST = 96.0
VELOCITY_LOOKBACK_TICKS = 4  # ~0.2s at 20Hz, for target-velocity estimation

# Standard id1 player.qc $frame ordering (ported from _anim_dem_probe.py)
FRAME_CATS = [
    (0, 11, "run"),
    (12, 16, "stand"),
    (17, 28, "stand"),
    (29, 40, "pain"),
    (41, 102, "death"),
    (103, 142, "attack"),
]


def frame_cat(frame: int) -> str:
    for lo, hi, name in FRAME_CATS:
        if lo <= frame <= hi:
            return name
    return "other"


_LOADED_MODEL_RE = re.compile(r"loaded\s+(models/\S+)", re.IGNORECASE)
_NAME_PREFIX_RE = re.compile(r"^([^:]+):")


def parse_model_load_text(text: str, names: Dict[int, str]) -> Optional[Tuple[str, Optional[int], Optional[int]]]:
    """Extract (model, slot, entity) from an svc_print/centerprint string
    containing "loaded models/...", resolving the "Name: ..." prefix
    against a slot(0-based) -> name map. Returns None if the text doesn't
    carry a "loaded models/" payload. entity = slot + 1 (see module
    docstring: entity number = client slot + 1)."""
    m = _LOADED_MODEL_RE.search(text)
    if not m:
        return None
    model = m.group(1).strip()
    name_m = _NAME_PREFIX_RE.match(text)
    slot = None
    entity = None
    if name_m:
        prefix = name_m.group(1).strip()
        for s, nm in names.items():
            if nm == prefix:
                slot = s
                entity = s + 1
                break
    return model, slot, entity


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelLoadEvent:
    blk: int
    t: float
    entity: Optional[int]   # slot + 1, if the name prefix resolved
    slot: Optional[int]
    model: str
    text: str


@dataclass(slots=True)
class ProjectileTrack:
    ent: int
    cls: str
    first_t: float
    first_blk: int
    first_origin: Tuple[float, float, float]
    last_t: float
    last_blk: int
    last_origin: Tuple[float, float, float]
    positions: List[Tuple[float, int, Tuple[float, float, float]]]
    owner_ent: Optional[int]
    owner_dist: Optional[float]
    ambiguous: bool
    impact_pos: Tuple[float, float, float]
    impact_t: float
    impact_blk: int
    impact_matched: bool
    impact_te: Optional[int]


@dataclass(slots=True)
class Segment:
    label: str
    blk_start: int
    blk_end: int  # exclusive
    t_start: float
    t_end: float

    def contains_blk(self, blk: int) -> bool:
        return self.blk_start <= blk < self.blk_end


@dataclass(slots=True)
class Session:
    path: str
    n_blocks: int
    frames: List[Dict[str, Any]]
    beams: List[Dict[str, Any]]
    explosions: List[Dict[str, Any]]
    projectiles: List[ProjectileTrack]
    names: Dict[int, str]
    view_ent: Optional[int]
    player_mi: Optional[int]
    maxclients: Optional[int]
    player_ents: List[int]
    models_by_index: Dict[int, str]
    model_load_events: List[ModelLoadEvent]
    segments: List[Segment]
    damage_events: List[Dict[str, Any]] = field(default_factory=list)
    health_track: List[Dict[str, Any]] = field(default_factory=list)

    def name_of(self, ent: int) -> str:
        return self.names.get(ent - 1, "?")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _apply_update(bits: int, reader: Reader, baseline: dict, prev: Optional[dict]) -> dict:
    """Rebuild one entity's live state: baseline + carried-live + sent bits.

    This is the PROBE's delta model (correct), not demo.parser's (wrong for
    per-frame reconstruction): unspecified fields persist from the entity's
    own last LIVE state (if any), falling back to its spawn baseline only
    the first time it is ever touched.
    """
    st = {
        "mi": baseline.get("mi", 0),
        "frame": baseline.get("frame", 0),
        "org": list(baseline.get("org", [0.0, 0.0, 0.0])),
        "ang": list(baseline.get("ang", [0.0, 0.0, 0.0])),
        "eff": 0,
    }
    if prev is not None:
        st["org"] = list(prev["org"])
        st["ang"] = list(prev["ang"])
        st["mi"] = prev["mi"]
        st["frame"] = prev["frame"]
        st["eff"] = prev["eff"]
    if bits & U_MODEL:
        st["mi"] = reader.byte()
    if bits & U_FRAME:
        st["frame"] = reader.byte()
    if bits & U_COLORMAP:
        reader.byte()
    if bits & U_SKIN:
        reader.byte()
    if bits & U_EFFECTS:
        st["eff"] = reader.byte()
    if bits & U_ORIGIN1:
        st["org"][0] = reader.coord()
    if bits & U_ANGLE1:
        st["ang"][0] = reader.angle()
    if bits & U_ORIGIN2:
        st["org"][1] = reader.coord()
    if bits & U_ANGLE2:
        st["ang"][1] = reader.angle()
    if bits & U_ORIGIN3:
        st["org"][2] = reader.coord()
    if bits & U_ANGLE3:
        st["ang"][2] = reader.angle()
    return st


def parse_session(path: str | Path) -> Session:
    """Parse an NQ client .dem into a Session: frames, beams, projectile
    tracks, names, and model-load-derived segments.

    Fails loud: a malformed/truncated demo raises rather than returning a
    partially-populated Session, EXCEPT for the trailing bytes of the very
    last block (real demos routinely end mid-message on disconnect), which
    is treated as a clean stop.
    """
    path = Path(path)
    payload = path.read_bytes()
    nl = payload.index(b"\n")
    body = payload[nl + 1:]
    off = 0
    blk = 0

    baselines: Dict[int, dict] = {}
    ents: Dict[int, dict] = {}
    pending: Dict[int, dict] = {}
    view_ent: Optional[int] = None
    player_mi: Optional[int] = None
    maxclients: Optional[int] = None
    models_by_index: Dict[int, str] = {}
    names: Dict[int, str] = {}
    t = 0.0
    recorder_angles = (0.0, 0.0, 0.0)

    frames: List[Dict[str, Any]] = []
    beams: List[Dict[str, Any]] = []
    explosions: List[Dict[str, Any]] = []
    damage_events: List[Dict[str, Any]] = []
    health_track: List[Dict[str, Any]] = []
    model_load_events: List[ModelLoadEvent] = []

    def flush() -> None:
        if not pending:
            return
        for e, st in pending.items():
            ents.setdefault(e, {}).update(st)
        frames.append({
            "t": t, "blk": blk, "ang": recorder_angles,
            "ents": {e: dict(v) for e, v in ents.items()},
        })
        pending.clear()

    def record_model_load(text: str) -> None:
        parsed = parse_model_load_text(text, names)
        if parsed is None:
            return
        model, slot, entity = parsed
        model_load_events.append(ModelLoadEvent(
            blk=blk, t=t, entity=entity, slot=slot, model=model, text=text,
        ))

    n_blocks = len(body)
    while off + 16 <= n_blocks:
        (msize,) = struct.unpack("<i", body[off:off + 4])
        ra = struct.unpack("<3f", body[off + 4:off + 16])
        off += 16
        if msize < 0 or off + msize > len(body):
            break
        r = Reader(body[off:off + msize])
        n = msize
        off += msize
        recorder_angles = ra
        try:
            while r.pos < n:
                cmd = r.byte()
                if cmd == 255:
                    break
                if cmd & 128:
                    bits = cmd & 127
                    if bits & U_MOREBITS:
                        bits |= r.byte() << 8
                    e = r.short() if (bits & U_LONGENTITY) else r.byte()
                    st = _apply_update(bits, r, baselines.get(e, {}), ents.get(e))
                    pending[e] = st
                    continue
                if cmd == SVC_TIME:
                    flush()
                    t = r.float()
                elif cmd == SVC_SETVIEW:
                    view_ent = r.short()
                elif cmd == SVC_SPAWNBASELINE:
                    e = r.short()
                    mi = r.byte(); fr = r.byte(); r.byte(); r.byte()
                    org = [0.0, 0.0, 0.0]; ang = [0.0, 0.0, 0.0]
                    for i in range(3):
                        org[i] = r.coord(); ang[i] = r.angle()
                    baselines[e] = {"mi": mi, "frame": fr, "org": org, "ang": ang}
                elif cmd == SVC_TEMP_ENTITY:
                    te = r.byte()
                    if te in TE_BEAM:
                        ent = r.short()
                        s = [r.coord() for _ in range(3)]
                        en = [r.coord() for _ in range(3)]
                        beams.append({"t": t, "blk": blk, "te": te, "ent": ent,
                                      "start": s, "end": en})
                    elif te in IMPACT_TE:
                        pos = [r.coord() for _ in range(3)]
                        explosions.append({"t": t, "blk": blk, "te": te, "pos": pos})
                    elif te == 12:  # TE_EXPLOSION2
                        for _ in range(3):
                            r.coord()
                        r.byte(); r.byte()
                    else:
                        for _ in range(3):
                            r.coord()
                elif cmd == SVC_SERVERINFO:
                    r.long(); maxclients = r.byte(); r.byte(); r.string()
                    mdls: List[str] = []
                    while True:
                        s = r.string()
                        if not s:
                            break
                        mdls.append(s)
                    while True:
                        s = r.string()
                        if not s:
                            break
                    for i, m in enumerate(mdls, start=1):
                        models_by_index[i] = m
                        if m == "progs/player.mdl":
                            player_mi = i
                elif cmd == SVC_UPDATENAME:
                    i = r.byte(); names[i] = r.string()
                elif cmd == SVC_CLIENTDATA:
                    _hp, _ammo, _sh, _na, _ro, _ce = read_clientdata(r)
                    health_track.append({"t": t, "health": _hp,
                                         "shells": _sh, "nails": _na,
                                         "rockets": _ro, "cells": _ce})
                elif cmd == SVC_SOUND:
                    skip_sound(r)
                elif cmd == SVC_PARTICLE:
                    skip_particle(r)
                elif cmd == SVC_DAMAGE:
                    _armor = r.byte(); _blood = r.byte()
                    _fx, _fy, _fz = r.coord(), r.coord(), r.coord()
                    damage_events.append({"t": t, "armor": _armor,
                                          "blood": _blood,
                                          "total": _armor + _blood,
                                          "from": [_fx, _fy, _fz]})
                elif cmd == SVC_SPAWNSTATIC:
                    r.byte(); r.byte(); r.byte(); r.byte()
                    for _ in range(3):
                        r.coord(); r.angle()
                elif cmd == SVC_UPDATESTAT:
                    r.byte(); r.long()
                elif cmd == SVC_UPDATEFRAGS:
                    r.byte(); r.short()
                elif cmd == SVC_UPDATECOLORS:
                    r.byte(); r.byte()
                elif cmd == SVC_SETANGLE:
                    for _ in range(3):
                        r.angle()
                elif cmd == SVC_LIGHTSTYLE:
                    r.byte(); r.string()
                elif cmd in (SVC_PRINT, SVC_CENTERPRINT):
                    s = r.string()
                    if "loaded" in s.lower():
                        record_model_load(s)
                elif cmd in (SVC_STUFFTEXT, SVC_FINALE, SVC_CUTSCENE):
                    r.string()
                elif cmd == SVC_CDTRACK:
                    r.byte(); r.byte()
                elif cmd == SVC_SPAWNSTATICSOUND:
                    for _ in range(3):
                        r.coord()
                    r.byte(); r.byte(); r.byte()
                elif cmd == SVC_SIGNONNUM:
                    r.byte()
                elif cmd == SVC_STOPSOUND:
                    r.short()
                elif cmd == SVC_SETPAUSE:
                    r.byte()
                elif cmd == SVC_VERSION:
                    r.long()
                elif cmd in (SVC_NOP, SVC_DISCONNECT, SVC_INTERMISSION,
                             SVC_KILLEDMONSTER, SVC_FOUNDSECRET,
                             SVC_SELLSCREEN):
                    pass
                else:
                    raise ValueError(f"unhandled svc {cmd} at t={t:.2f} blk={blk}")
        except (ValueError, IndexError, struct.error) as exc:
            if off >= len(body) - 1:
                break  # trailing partial block at EOF -- normal for a .dem
            raise ValueError(f"malformed demo {path} at blk={blk}: {exc}") from exc
        blk += 1
    n_blocks_total = blk
    if blk > 0:
        blk -= 1  # trailing pending updates belong to the last parsed block, not one past it
    flush()
    blk = n_blocks_total

    player_ents = sorted({
        e for f in frames for e in f["ents"]
        if maxclients is not None and 1 <= e <= maxclients
    })

    projectiles = _build_projectile_tracks(frames, player_ents, models_by_index, explosions)
    t_start = frames[0]["t"] if frames else 0.0
    t_end = frames[-1]["t"] if frames else 0.0
    segments = _compute_segments(path, model_load_events, blk, t_start, t_end)

    return Session(
        path=str(path), n_blocks=blk, frames=frames, beams=beams, explosions=explosions,
        projectiles=projectiles, names=names, view_ent=view_ent,
        player_mi=player_mi, maxclients=maxclients, player_ents=player_ents,
        models_by_index=models_by_index, model_load_events=model_load_events,
        segments=segments, damage_events=damage_events,
        health_track=health_track,
    )


# ---------------------------------------------------------------------------
# Projectile tracks
# ---------------------------------------------------------------------------


def _build_projectile_tracks(
    frames: List[Dict[str, Any]],
    player_ents: List[int],
    models_by_index: Dict[int, str],
    explosions: List[Dict[str, Any]],
) -> List[ProjectileTrack]:
    player_set = set(player_ents)
    active: Dict[int, dict] = {}
    stubs: List[dict] = []

    def close(ent: int) -> None:
        tb = active.pop(ent)
        stubs.append(tb)

    for idx, f in enumerate(frames):
        seen_this_frame = set()
        for e, st in f["ents"].items():
            if e in player_set:
                continue
            model = models_by_index.get(st["mi"])
            cls = PROJECTILE_MODELS.get(model) if model else None
            if cls is None:
                continue
            seen_this_frame.add(e)
            tb = active.get(e)
            if tb is not None and tb["cls"] == cls and idx - tb["last_idx"] <= GAP_TICKS:
                tb["positions"].append((f["t"], f["blk"], tuple(st["org"])))
                tb["last_idx"] = idx
            else:
                if tb is not None:
                    close(e)
                active[e] = {
                    "ent": e, "cls": cls, "first_idx": idx,
                    "positions": [(f["t"], f["blk"], tuple(st["org"]))],
                    "last_idx": idx,
                }
        stale = [e for e, tb in active.items()
                 if e not in seen_this_frame and idx - tb["last_idx"] > GAP_TICKS]
        for e in stale:
            close(e)
    for e in list(active.keys()):
        close(e)

    tracks: List[ProjectileTrack] = []
    for tb in stubs:
        tracks.append(_finalize_track(tb, frames, player_ents, explosions))
    tracks.sort(key=lambda tr: tr.first_t)
    return tracks


def _finalize_track(
    tb: dict, frames: List[Dict[str, Any]], player_ents: List[int],
    explosions: List[Dict[str, Any]],
) -> ProjectileTrack:
    positions = tb["positions"]
    first_t, first_blk, first_origin = positions[0]
    last_t, last_blk, last_origin = positions[-1]

    first_frame = frames[tb["first_idx"]]
    owner_ent = None
    owner_dist = None
    for pe in player_ents:
        pst = first_frame["ents"].get(pe)
        if pst is None:
            continue
        d = math.dist(pst["org"], first_origin)
        if owner_dist is None or d < owner_dist:
            owner_dist = d
            owner_ent = pe
    ambiguous = owner_ent is None or owner_dist is None or owner_dist > AMBIGUOUS_ATTRIBUTION_DIST

    impact_pos = last_origin
    impact_t = last_t
    impact_blk = last_blk
    impact_matched = False
    impact_te = None
    if tb["cls"] in LEAD_CLASSES:
        best = None
        for ex in explosions:
            if ex["te"] not in IMPACT_TE:
                continue
            gap = ex["blk"] - last_blk
            if gap < 0 or gap > IMPACT_MAX_BLK_GAP:
                continue
            d = math.dist(ex["pos"], last_origin)
            if d > IMPACT_MAX_DIST_U:
                continue
            key = (gap, d)
            if best is None or key < best[0]:
                best = (key, ex)
        if best is not None:
            ex = best[1]
            impact_pos = tuple(ex["pos"])
            impact_t = ex["t"]
            impact_blk = ex["blk"]
            impact_matched = True
            impact_te = ex["te"]

    return ProjectileTrack(
        ent=tb["ent"], cls=tb["cls"],
        first_t=first_t, first_blk=first_blk, first_origin=first_origin,
        last_t=last_t, last_blk=last_blk, last_origin=last_origin,
        positions=positions,
        owner_ent=owner_ent, owner_dist=owner_dist, ambiguous=ambiguous,
        impact_pos=impact_pos, impact_t=impact_t, impact_blk=impact_blk,
        impact_matched=impact_matched, impact_te=impact_te,
    )


# ---------------------------------------------------------------------------
# Segmenting
# ---------------------------------------------------------------------------


def _compute_segments(
    path: Path, events: List[ModelLoadEvent], n_blocks: int,
    t_start: float, t_end: float,
) -> List[Segment]:
    pre_label = path.stem or "pre-load"
    if not events:
        return [Segment(label=pre_label, blk_start=0, blk_end=n_blocks,
                         t_start=t_start, t_end=t_end)]
    segments: List[Segment] = []
    ordered = sorted(events, key=lambda e: e.blk)
    starts = [0] + [e.blk for e in ordered]
    labels = [pre_label] + [e.model for e in ordered]
    times = [t_start] + [e.t for e in ordered]
    for i, (start, label, tstart) in enumerate(zip(starts, labels, times)):
        end = starts[i + 1] if i + 1 < len(starts) else n_blocks
        tend = times[i + 1] if i + 1 < len(times) else t_end
        segments.append(Segment(label=label, blk_start=start, blk_end=end,
                                 t_start=tstart, t_end=tend))
    return segments


def _seg_frames(session: Session, segment: Optional[Segment]) -> List[Dict[str, Any]]:
    if segment is None:
        return session.frames
    return [f for f in session.frames if segment.contains_blk(f["blk"])]


def _seg_beams(session: Session, segment: Optional[Segment]) -> List[Dict[str, Any]]:
    if segment is None:
        return session.beams
    return [b for b in session.beams if segment.contains_blk(b["blk"])]


def _seg_tracks(session: Session, segment: Optional[Segment]) -> List[ProjectileTrack]:
    if segment is None:
        return session.projectiles
    return [tr for tr in session.projectiles if segment.contains_blk(tr.first_blk)]


# ---------------------------------------------------------------------------
# Generic fire tell
# ---------------------------------------------------------------------------


def _is_firing(state: dict) -> bool:
    return (ATTACK_FRAME_LO <= state["frame"] <= ATTACK_FRAME_HI) or \
           bool(state.get("eff", 0) & EF_MUZZLEFLASH)


# ---------------------------------------------------------------------------
# a. discharge_timeline
# ---------------------------------------------------------------------------


def discharge_timeline(session: Session, segment: Optional[Segment] = None) -> Dict[int, List[dict]]:
    """Per player entity: merged, time-ordered, weapon-class-labeled discharge
    events (lg beams + projectile launches + generic attack/muzzleflash
    tells not already covered by a more specific source)."""
    frames = _seg_frames(session, segment)
    beams = _seg_beams(session, segment)
    tracks = _seg_tracks(session, segment)

    specific: Dict[int, List[dict]] = defaultdict(list)
    for b in beams:
        specific[b["ent"]].append({"t": b["t"], "blk": b["blk"], "cls": "lg", "source": "beam"})
    for tr in tracks:
        if tr.owner_ent is None:
            continue
        specific[tr.owner_ent].append({
            "t": tr.first_t, "blk": tr.first_blk, "cls": tr.cls, "source": "launch",
        })

    timelines: Dict[int, List[dict]] = defaultdict(list)
    for ent in session.player_ents:
        events = list(specific.get(ent, []))
        specific_blks = sorted(e["blk"] for e in events)
        was_firing = False
        for f in frames:
            st = f["ents"].get(ent)
            firing = st is not None and _is_firing(st)
            if firing and not was_firing:
                blk = f["blk"]
                near = any(abs(blk - sb) <= 2 for sb in specific_blks)
                if not near:
                    events.append({"t": f["t"], "blk": blk, "cls": "unknown", "source": "anim"})
            was_firing = firing
        events.sort(key=lambda e: e["t"])
        timelines[ent] = events
    return dict(timelines)


# ---------------------------------------------------------------------------
# b. switch_texture
# ---------------------------------------------------------------------------


def switch_texture(session: Session, segment: Optional[Segment] = None) -> Dict[int, dict]:
    """Per entity: consecutive-discharge weapon-class transition fraction
    and run lengths, from the merged discharge_timeline."""
    timelines = discharge_timeline(session, segment)
    out: Dict[int, dict] = {}
    for ent, events in timelines.items():
        classes = [e["cls"] for e in events]
        n = len(classes)
        transitions = sum(1 for i in range(1, n) if classes[i] != classes[i - 1])
        transition_frac = transitions / (n - 1) if n > 1 else None
        run_lengths: List[int] = []
        cur = None
        cur_len = 0
        for c in classes:
            if c == cur:
                cur_len += 1
            else:
                if cur is not None:
                    run_lengths.append(cur_len)
                cur, cur_len = c, 1
        if cur is not None:
            run_lengths.append(cur_len)
        out[ent] = {
            "n_discharges": n,
            "transition_fraction": transition_frac,
            "run_lengths": run_lengths,
            "class_counts": dict(sorted(
                ((c, classes.count(c)) for c in set(classes)),
            )),
        }
    return out


# ---------------------------------------------------------------------------
# c. fire_vs_geometry
# ---------------------------------------------------------------------------


def _dz_bin(dz: float) -> str:
    if dz <= -32.0:
        return "below"
    if dz >= 32.0:
        return "above"
    return "level"


def fire_vs_geometry(session: Session, segment: Optional[Segment] = None) -> Dict[Tuple[int, int], dict]:
    """Per (shooter -> opponent) player-entity pair: fire fraction binned
    by dz (opponent_z - shooter_z, 32u edges) and by distance (<=600u vs
    >600u), gated on the shooter FACING the opponent (yaw_err <= 20deg).
    Works for any shooter perspective -- human or bot POV recordings both
    carry entity + beam + anim data the same way."""
    frames = _seg_frames(session, segment)
    players = session.player_ents
    out: Dict[Tuple[int, int], dict] = {}

    for shooter in players:
        for opp in players:
            if opp == shooter:
                continue
            rows = []
            for f in frames:
                s = f["ents"].get(shooter)
                o = f["ents"].get(opp)
                if s is None or o is None:
                    continue
                so, oo = s["org"], o["org"]
                dz = oo[2] - so[2]
                d = math.dist(so, oo)
                bearing = math.degrees(math.atan2(oo[1] - so[1], oo[0] - so[0])) % 360.0
                yaw_err = abs(((s["ang"][1] - bearing + 180.0) % 360.0) - 180.0)
                rows.append({
                    "dz": dz, "d": d, "yaw_err": yaw_err, "fire": _is_firing(s),
                })
            if not rows:
                continue
            faced = [r for r in rows if r["yaw_err"] <= FACING_YAW_ERR_DEG]
            dz_bins: Dict[str, dict] = {}
            for label in ("below", "level", "above"):
                sel = [r for r in faced if _dz_bin(r["dz"]) == label]
                dz_bins[label] = {
                    "n": len(sel),
                    "fire_fraction": (sum(r["fire"] for r in sel) / len(sel)) if sel else None,
                }
            dist_bins: Dict[str, dict] = {}
            for label, pred in (("close_le600", lambda r: r["d"] <= LG_RANGE_U),
                                 ("far_gt600", lambda r: r["d"] > LG_RANGE_U)):
                sel = [r for r in faced if pred(r)]
                dist_bins[label] = {
                    "n": len(sel),
                    "fire_fraction": (sum(r["fire"] for r in sel) / len(sel)) if sel else None,
                }
            dists = sorted(r["d"] for r in rows)
            out[(shooter, opp)] = {
                "n_mutual_frames": len(rows),
                "n_facing_frames": len(faced),
                "median_distance": dists[len(dists) // 2],
                "frac_beyond_600": sum(1 for d in dists if d > LG_RANGE_U) / len(dists),
                "dz_bins": dz_bins,
                "distance_bins": dist_bins,
            }
    return out


# ---------------------------------------------------------------------------
# d. projectile_lead
# ---------------------------------------------------------------------------


def _nearest_frame_idx(frames: List[Dict[str, Any]], t: float) -> int:
    lo, hi = 0, len(frames) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frames[mid]["t"] < t:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and abs(frames[lo - 1]["t"] - t) < abs(frames[lo]["t"] - t):
        return lo - 1
    return lo


def projectile_lead(session: Session, segment: Optional[Segment] = None) -> Dict[int, dict]:
    """Per shooter: for each rocket/grenade launch, the target's position
    at impact time vs the actual impact point -- miss distance, and the
    signed component of (impact - target_at_impact) along the target's
    velocity direction (negative = landed BEHIND the target's motion, the
    no-lead signature), binned by launch range (<600u / >=600u)."""
    tracks = [tr for tr in _seg_tracks(session, segment) if tr.cls in LEAD_CLASSES]
    frames = session.frames  # use full session for velocity continuity
    per_shooter: Dict[int, List[dict]] = defaultdict(list)

    for tr in tracks:
        if tr.owner_ent is None or tr.ambiguous:
            continue
        if not frames:
            continue
        idx = _nearest_frame_idx(frames, tr.impact_t)
        candidates = [e for e in session.player_ents
                      if e != tr.owner_ent and e in frames[idx]["ents"]]
        if not candidates:
            continue
        target = min(candidates,
                     key=lambda e: math.dist(frames[idx]["ents"][e]["org"], tr.impact_pos))
        target_now = frames[idx]["ents"][target]["org"]
        idx_before = max(0, idx - VELOCITY_LOOKBACK_TICKS)
        if target not in frames[idx_before]["ents"]:
            continue
        target_before = frames[idx_before]["ents"][target]["org"]
        dt = frames[idx]["t"] - frames[idx_before]["t"]
        if dt <= 0:
            continue
        vel = [(a - b) / dt for a, b in zip(target_now, target_before)]
        speed = math.dist((0.0, 0.0, 0.0), vel)
        miss = math.dist(tr.impact_pos, target_now)
        lead_component = None
        if speed > 1e-3:
            direction = [v / speed for v in vel]
            lead_component = sum((ip - tp) * d for ip, tp, d in
                                  zip(tr.impact_pos, target_now, direction))

        launch_idx = _nearest_frame_idx(frames, tr.first_t)
        launch_range = None
        if target in frames[launch_idx]["ents"]:
            launch_range = math.dist(tr.first_origin, frames[launch_idx]["ents"][target]["org"])

        per_shooter[tr.owner_ent].append({
            "cls": tr.cls, "target": target,
            "launch_t": tr.first_t, "impact_t": tr.impact_t,
            "impact_matched": tr.impact_matched,
            "launch_range": launch_range,
            "target_speed": speed,
            "miss_distance": miss,
            "lead_component": lead_component,
        })

    out: Dict[int, dict] = {}
    for shooter, rows in per_shooter.items():
        by_range: Dict[str, dict] = {}
        for label, pred in (("close_lt600", lambda r: r["launch_range"] is not None and r["launch_range"] < LG_RANGE_U),
                             ("far_ge600", lambda r: r["launch_range"] is not None and r["launch_range"] >= LG_RANGE_U)):
            sel = [r for r in rows if pred(r)]
            leads = sorted(r["lead_component"] for r in sel if r["lead_component"] is not None)
            misses = sorted(r["miss_distance"] for r in sel)
            by_range[label] = {
                "n": len(sel),
                "median_lead_component": leads[len(leads) // 2] if leads else None,
                "frac_behind_target": (sum(1 for x in leads if x < 0) / len(leads)) if leads else None,
                "median_miss_distance": misses[len(misses) // 2] if misses else None,
            }
        out[shooter] = {"n_launches": len(rows), "by_range": by_range, "rows": rows}
    return out


# ---------------------------------------------------------------------------
# e. anim occupancy (ported from _anim_dem_probe.py)
# ---------------------------------------------------------------------------


def anim_occupancy(session: Session, segment: Optional[Segment] = None) -> Dict[int, dict]:
    frames = _seg_frames(session, segment)
    snapshots: Dict[int, List[Tuple[float, int, Tuple[float, float, float]]]] = defaultdict(list)
    for f in frames:
        for e in session.player_ents:
            st = f["ents"].get(e)
            if st is not None:
                snapshots[e].append((f["t"], st["frame"], tuple(st["org"])))

    out: Dict[int, dict] = {}
    for ent, rows in snapshots.items():
        n = len(rows)
        if n == 0:
            continue
        cats: Dict[str, int] = defaultdict(int)
        speed_by_cat: Dict[str, List[float]] = defaultdict(list)
        seq: List[Tuple[float, int, str, float]] = []
        for i, (t, fr, org) in enumerate(rows):
            cat = frame_cat(fr)
            cats[cat] += 1
            if i > 0:
                pt, _, porg = rows[i - 1]
                dt = t - pt
                if 0 < dt < 0.5:
                    sp = math.hypot(org[0] - porg[0], org[1] - porg[1]) / dt
                    speed_by_cat[cat].append(sp)
                    seq.append((t, fr, cat, sp))
        dwells: Dict[str, List[int]] = defaultdict(list)
        flips = 0
        cur_cat, cur_len = None, 0
        for _, _, cat, _ in seq:
            if cat == cur_cat:
                cur_len += 1
            else:
                if cur_cat in ("run", "stand") and cat in ("run", "stand") and cur_cat is not None:
                    flips += 1
                if cur_cat is not None:
                    dwells[cur_cat].append(cur_len)
                cur_cat, cur_len = cat, 1
        if cur_cat is not None:
            dwells[cur_cat].append(cur_len)
        dur = rows[-1][0] - rows[0][0] if n > 1 else 0.0
        out[ent] = {
            "name": session.name_of(ent),
            "snapshots": n,
            "visible_s": round(dur, 1),
            "cat_occupancy": {k: round(v / n, 3) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
            "mean_speed_by_cat": {k: round(sum(v) / len(v), 1) for k, v in speed_by_cat.items() if v},
            "stand_run_flips": flips,
            "flips_per_s": round(flips / dur, 2) if dur else 0.0,
            "dwell_median_ticks": {k: sorted(v)[len(v) // 2] for k, v in dwells.items()},
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _run_all_analyses(session: Session, segment: Optional[Segment]) -> Dict[str, Any]:
    return {
        "discharge_timeline": discharge_timeline(session, segment),
        "switch_texture": switch_texture(session, segment),
        "fire_vs_geometry": fire_vs_geometry(session, segment),
        "projectile_lead": projectile_lead(session, segment),
        "anim_occupancy": anim_occupancy(session, segment),
    }


def _print_report(session: Session, split: bool) -> None:
    print(f"demo: {session.path}")
    print(f"blocks: {session.n_blocks}  "
          f"frames: {len(session.frames)}  beams: {len(session.beams)}  "
          f"projectiles: {len(session.projectiles)}")
    print(f"view_ent: {session.view_ent}  maxclients: {session.maxclients}  "
          f"player_mi: {session.player_mi}  names: {session.names}  "
          f"player_ents: {session.player_ents}")
    if session.model_load_events:
        print("model_load_events:")
        for ev in session.model_load_events:
            print(f"  blk={ev.blk} t={ev.t:.2f} entity={ev.entity} model={ev.model}")
    print("segments:")
    for seg in session.segments:
        print(f"  [{seg.label}] blk [{seg.blk_start},{seg.blk_end})  "
              f"t [{seg.t_start:.1f},{seg.t_end:.1f}]")

    def report_one(label: str, segment: Optional[Segment]) -> None:
        print(f"\n=== {label} ===")
        by_ent = discharge_timeline(session, segment)
        for ent, events in by_ent.items():
            counts = defaultdict(int)
            for e in events:
                counts[e["cls"]] += 1
            print(f"  discharge ent {ent} ({session.name_of(ent)}): "
                  f"n={len(events)} by_class={dict(counts)}")
        st = switch_texture(session, segment)
        for ent, info in st.items():
            print(f"  switch_texture ent {ent}: transition_frac="
                  f"{info['transition_fraction']}  runs={info['run_lengths'][:12]}"
                  f"{'...' if len(info['run_lengths']) > 12 else ''}")
        fg = fire_vs_geometry(session, segment)
        for (shooter, opp), info in fg.items():
            print(f"  fire_vs_geometry {shooter}->{opp}: n={info['n_mutual_frames']} "
                  f"med_dist={info['median_distance']:.0f} "
                  f"frac_beyond_600={info['frac_beyond_600']:.3f}")
            print("      dz_fire_frac: "
                  + " ".join(f"{k}={v['fire_fraction']}" for k, v in info["dz_bins"].items()))
        pl = projectile_lead(session, segment)
        for shooter, info in pl.items():
            print(f"  projectile_lead shooter {shooter} ({session.name_of(shooter)}): "
                  f"n_launches={info['n_launches']}")
            for rlabel, r in info["by_range"].items():
                print(f"      {rlabel}: n={r['n']} median_lead={r['median_lead_component']} "
                      f"frac_behind_target={r['frac_behind_target']} "
                      f"median_miss={r['median_miss_distance']}")
        ao = anim_occupancy(session, segment)
        for ent, info in ao.items():
            print(f"  anim ent {ent} ({info['name']}): occupancy={info['cat_occupancy']}")

    report_one("WHOLE SESSION", None)
    if split:
        for seg in session.segments:
            report_one(f"SEGMENT: {seg.label}", seg)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="NQ-demo behaviour analysis: fire discipline, weapon "
                     "switch texture, geometry-conditioned fire rate, and "
                     "projectile lead."
    )
    parser.add_argument("demo", help="path to .dem file")
    parser.add_argument("--json", metavar="PATH", help="dump full results (whole-session + all segments) as JSON")
    parser.add_argument("--split-on-model-load", action="store_true",
                         help="also print the per-segment breakdown (segments are always computed)")
    args = parser.parse_args(argv)

    session = parse_session(args.demo)
    _print_report(session, split=args.split_on_model_load)

    if args.json:
        out = {
            "session": {
                "path": session.path,
                "n_blocks": session.n_blocks,
                "n_frames": len(session.frames),
                "n_beams": len(session.beams),
                "view_ent": session.view_ent,
                "player_mi": session.player_mi,
                "maxclients": session.maxclients,
                "names": session.names,
                "player_ents": session.player_ents,
                "model_load_events": _jsonable(session.model_load_events),
                "segments": _jsonable(session.segments),
                "projectiles": _jsonable(session.projectiles),
            },
            "whole_session": _jsonable(_run_all_analyses(session, None)),
            "segments": {
                seg.label: _jsonable(_run_all_analyses(session, seg))
                for seg in session.segments
            },
        }
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
