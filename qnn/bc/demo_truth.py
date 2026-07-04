#!/usr/bin/env python3
"""Canonical byte-truth parser for QuakeWorld .qwd / MVD demos.

The demo's own ``svc_sound`` stream, parsed straight from the raw bytes, is
the ground truth for the collected attack / jump labels — it is the single
source of truth the post-collect validator (``qnn.bc.validate_labels``) and
the deep diagnostic (``scripts/analysis/sg_enumerate.py``) both compare
against. See ``src/docs/mvd-attack-audit.md``.

Promoted from the loose ``scripts/count_guncock.py`` (the linear ``parse()``
byte-reader) + the trigger-event collapse logic from the old
``scripts/audit_fire_summary.py`` (``_trigger_events`` / ``MERGE_GAP``), so
everything imports from one package module instead of
``sys.path.insert(0, 'scripts')`` copies.

The byte layout is ported 1:1 from ``src/demo/qw_classifier.c`` (the
authoritative QW parser).

Exposes:
  parse(path, target, with_times=, with_stats=)  — linear svc_sound reader
  R                                              — the byte reader
  demo_attack_sound_truth(path)                          — per-weapon sounds/triggers
  demo_jump_truth(path)                          — jump-sound count
  MERGE_GAP, trigger_events                      — think-chain collapse
"""
from __future__ import annotations

import struct
import sys

from qnn.bc.attack_vocab import CONTINUOUS, JUMP_SOUNDS, SOUND_WEAPON, WNAME

_DBG: dict[int, int] = {}
_BADBLK = [0]
DEM_CMD, DEM_READ, DEM_SET, DEM_MASK = 0, 1, 2, 7
DEM_CMD_PAYLOAD, NETCHAN_HEADER = 36, 8
SND_VOLUME, SND_ATTENUATION = 1 << 15, 1 << 14
# QW entity-update word flags (low 9 bits = entity num).
U_MOREBITS, U_REMOVE = 1 << 15, 1 << 14
U_ORIGIN1, U_ORIGIN2, U_ORIGIN3 = 1 << 9, 1 << 10, 1 << 11
U_ANGLE2, U_FRAME = 1 << 12, 1 << 13
U_ANGLE1, U_ANGLE3, U_MODEL, U_COLORMAP, U_SKIN, U_EFFECTS = (
    1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5)
# QW playerinfo flags / move-command flags.
PF_MSEC, PF_COMMAND = 1 << 0, 1 << 1
PF_VEL1, PF_MODEL, PF_SKINNUM, PF_EFFECTS, PF_WEAPONFRAME = (
    1 << 2, 1 << 5, 1 << 6, 1 << 7, 1 << 8)
CM_ANGLE1, CM_ANGLE2, CM_ANGLE3, CM_FORWARD, CM_SIDE, CM_UP, CM_BUTTONS, CM_IMPULSE = (
    1 << 0, 1 << 7, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6)

STAT_ACTIVEWEAPON = 10

# Gap (s) that separates the 0.1s player_nail/player_light think-chain stream
# from a distinct W_Attack trigger. Below it, consecutive same-class sounds are
# one trigger window; single-shot weapons fire >=0.5s apart so they are
# unaffected. See src/docs/mvd-attack-audit.md §Round 4.
MERGE_GAP = 0.15


class R:
    def __init__(self, data, pos, end):
        self.d, self.pos, self.end, self.bad = data, pos, end, False

    def _take(self, n):
        if self.pos + n > self.end:
            self.bad = True
            return b"\x00" * n
        b = self.d[self.pos:self.pos + n]
        self.pos += n
        return b

    def byte(self):  return self._take(1)[0]
    def char(self):  return struct.unpack("<b", self._take(1))[0]
    def short(self): return struct.unpack("<h", self._take(2))[0]
    def ushort(self):return struct.unpack("<H", self._take(2))[0]
    def long(self):  return struct.unpack("<i", self._take(4))[0]
    def float(self): return struct.unpack("<f", self._take(4))[0]
    def coord(self): self._take(2)
    def angle(self): self._take(1)

    def string(self):
        start = self.pos
        while self.pos < self.end:
            b = self.d[self.pos]; self.pos += 1
            if b in (0, 255):
                return self.d[start:self.pos - 1]
        self.bad = True
        return self.d[start:self.pos]


def _skip_packetentities(r, is_delta):
    if is_delta:
        r.byte()
    while not r.bad:
        word = r.ushort()
        if word == 0:
            return
        bits = word & ~511
        if bits & U_REMOVE:
            continue
        if bits & U_MOREBITS:
            bits |= r.byte()
        for m, fn in ((U_MODEL, r.byte), (U_FRAME, r.byte), (U_COLORMAP, r.byte),
                      (U_SKIN, r.byte), (U_EFFECTS, r.byte),
                      (U_ORIGIN1, r.short), (U_ANGLE1, r.char),
                      (U_ORIGIN2, r.short), (U_ANGLE2, r.char),
                      (U_ORIGIN3, r.short), (U_ANGLE3, r.char)):
            if bits & m:
                fn()


def _skip_delta_usercmd(r, protocol):
    bits = r.byte()
    if protocol <= 26:
        if bits & CM_ANGLE1: r.short()
        r.short()                       # angles[1] always
        if bits & CM_ANGLE3: r.short()
        if bits & CM_FORWARD: r.char()
        if bits & CM_SIDE:    r.char()
        if bits & CM_UP:      r.char()
        if bits & CM_BUTTONS: r.byte()
        if bits & CM_IMPULSE: r.byte()
        if bits & CM_ANGLE2:  r.byte()  # proto-26 CM_MSEC alias
    else:
        for m in (CM_ANGLE1, CM_ANGLE2, CM_ANGLE3, CM_FORWARD, CM_SIDE, CM_UP):
            if bits & m: r.short()
        if bits & CM_BUTTONS: r.byte()
        if bits & CM_IMPULSE: r.byte()
        r.byte()                        # msec always


def _skip_playerinfo(r, protocol):
    r.byte()                            # player num
    flags = r.ushort()
    r.short(); r.short(); r.short()     # origin
    r.byte()                            # frame
    if flags & PF_MSEC:    r.byte()
    if flags & PF_COMMAND: _skip_delta_usercmd(r, protocol)
    for a in range(3):
        if flags & (PF_VEL1 << a): r.short()
    for m in (PF_MODEL, PF_SKINNUM, PF_EFFECTS, PF_WEAPONFRAME):
        if flags & m: r.byte()


def _skip_temp_entity(r):
    t = r.byte()
    if t in (0, 1, 3, 4, 7, 8, 10, 11, 13):    # point-only: 3 coords
        r.coord(); r.coord(); r.coord()
    elif t in (5, 6, 9):                        # beams: short ent + 6 coords
        r.short()
        for _ in range(6): r.coord()
    elif t in (2, 12):                          # gunshot/blood: byte + 3 coords
        r.byte(); r.coord(); r.coord(); r.coord()
    else:
        r.bad = True


def parse(path, target, with_times=False, with_stats=False):
    data = open(path, "rb").read()
    pos, n = 0, len(data)
    protocol = 28
    self_slot = -1
    sound_names = {}          # precache index -> name
    target_idx = None
    by_ent = {}               # entity -> count of target sound
    all_by = {}               # (entity, sound_idx) -> count (ALL sounds)
    # (entity, sound_idx) -> [demotime, ...] for every svc_sound, so callers
    # can collapse a think-chain stream (player_nail/player_light fire a sound
    # every 0.1s) into W_Attack trigger events. Only filled when with_times.
    sound_times = {}
    # [(demotime, IT_flag), ...] — the self player's STAT_ACTIVEWEAPON broadcast
    # timeline (server-authoritative held weapon). Only filled when with_stats.
    stat_aw = []

    while pos + 5 <= n:
        demotime = struct.unpack_from("<f", data, pos)[0]
        dem_type = data[pos + 4] & DEM_MASK
        pos += 5
        if dem_type == DEM_CMD:
            pos += DEM_CMD_PAYLOAD
            continue
        if dem_type == DEM_SET:
            pos += 8
            continue
        if dem_type != DEM_READ:
            break
        if pos + 4 > n:
            break
        msg_len = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        if msg_len < 0 or pos + msg_len > n:
            break
        msg_start, msg_end = pos, pos + msg_len
        pos = msg_end
        if msg_len <= NETCHAN_HEADER:
            continue
        r = R(data, msg_start + NETCHAN_HEADER, msg_end)
        while r.pos < r.end and not r.bad:
            cmd = r.byte()
            if cmd == 1:                       # nop
                continue
            elif cmd == 2:                     # disconnect
                break
            elif cmd == 3:                     # updatestat
                si = r.byte(); sv = r.byte()
                if with_stats and si == STAT_ACTIVEWEAPON:
                    stat_aw.append((demotime, sv))
            elif cmd == 38:                    # updatestatlong
                si = r.byte(); sv = r.long()
                if with_stats and si == STAT_ACTIVEWEAPON:
                    stat_aw.append((demotime, sv))
            elif cmd == 4:                     # version
                r.long()
            elif cmd == 5:                     # setview
                r.short()
            elif cmd == 6:                     # SOUND
                channel = r.ushort()
                if channel & SND_VOLUME: r.byte()
                if channel & SND_ATTENUATION: r.byte()
                sound_num = r.byte()
                r.coord(); r.coord(); r.coord()
                ent = (channel >> 3) & 1023
                all_by[(ent, sound_num)] = all_by.get((ent, sound_num), 0) + 1
                if with_times:
                    sound_times.setdefault((ent, sound_num), []).append(demotime)
                if target_idx is not None and sound_num == target_idx:
                    by_ent[ent] = by_ent.get(ent, 0) + 1
            elif cmd == 7:                     # time
                r.float()
            elif cmd in (8, 9, 26, 31):        # print/stufftext/centerprint/finale
                if cmd == 8: r.byte()          # print level
                r.string()
            elif cmd == 10:                    # setangle
                r.angle(); r.angle(); r.angle()
            elif cmd == 11:                    # SERVERDATA
                protocol = r.long(); r.long(); r.string()
                pn = r.byte() & 0x7f
                if pn < 32:                # guard against desync garbage
                    self_slot = pn
                r.string()
                for _ in range(10): r.float()
            elif cmd in (12, 13):              # lightstyle / updatename
                r.byte(); r.string()
            elif cmd == 14:                    # updatefrags
                r.byte(); r.short()
            elif cmd == 16:                    # stopsound
                r.short()
            elif cmd == 17:                    # updatecolors
                r.byte(); r.byte()
            elif cmd == 18:                    # particle
                r.coord(); r.coord(); r.coord()
                r.char(); r.char(); r.char(); r.byte(); r.byte()
            elif cmd == 19:                    # damage
                r.byte(); r.byte(); r.coord(); r.coord(); r.coord()
            elif cmd == 20:                    # spawnstatic (baseline)
                r.byte(); r.byte(); r.byte(); r.byte()
                for _ in range(3): r.coord(); r.angle()
            elif cmd == 22:                    # spawnbaseline
                r.short()
                r.byte(); r.byte(); r.byte(); r.byte()
                for _ in range(3): r.coord(); r.angle()
            elif cmd == 23:                    # temp entity
                _skip_temp_entity(r)
            elif cmd == 24:                    # setpause
                r.byte()
            elif cmd == 25:                    # signonnum
                r.byte()
            elif cmd in (27, 28, 33, 34, 35):  # killedmonster/foundsecret/sellscreen/small+bigkick
                pass
            elif cmd == 29:                    # spawnstaticsound
                r.coord(); r.coord(); r.coord(); r.byte(); r.byte(); r.byte()
            elif cmd == 30:                    # intermission
                r.coord(); r.coord(); r.coord(); r.angle(); r.angle(); r.angle()
            elif cmd == 32:                    # cdtrack (QW: 1 byte)
                r.byte()
            elif cmd == 36:                    # updateping
                r.byte(); r.ushort()
            elif cmd == 37:                    # updateentertime
                r.byte(); r.float()
            elif cmd == 39:                    # muzzleflash
                r.short()
            elif cmd == 40:                    # updateuserinfo
                r.byte(); r.long(); r.string()
            elif cmd == 41:                    # download
                size = r.short(); r.byte()
                if size > 0:
                    r._take(size)
            elif cmd == 42:                    # playerinfo
                _skip_playerinfo(r, protocol)
            elif cmd == 43:                    # nails
                cnt = r.byte()
                for _ in range(cnt):
                    for _ in range(6): r.byte()
            elif cmd == 44:                    # chokecount
                r.byte()
            elif cmd in (45, 46):              # modellist / soundlist
                idx = r.byte()
                while not r.bad:
                    name = r.string()
                    if len(name) == 0:
                        break
                    idx += 1
                    if cmd == 46:
                        sound_names[idx] = name.decode("latin-1")
                        if name == target.encode("latin-1"):
                            target_idx = idx
                    if idx >= 511:
                        break
                r.byte()                       # continuation index
            elif cmd in (47, 48):              # packetentities / deltapacketentities
                _skip_packetentities(r, cmd == 48)
            elif cmd in (49, 50):              # maxspeed / entgravity
                r.float()
            elif cmd == 51:                    # setinfo
                r.byte(); r.string(); r.string()
            elif cmd == 52:                    # serverinfo (kv)
                r.string(); r.string()
            elif cmd == 53:                    # updatepl
                r.byte(); r.byte()
            else:
                import os
                if os.environ.get("QW_DEBUG"):
                    _DBG[cmd] = _DBG.get(cmd, 0) + 1
                break                          # unknown opcode -> stop this msg
        else:
            continue
        # inner loop broke (unknown opcode or r.bad) before msg end
        import os
        if os.environ.get("QW_DEBUG") and not r.bad:
            _BADBLK[0] += 1

    view_ent = self_slot + 1 if self_slot >= 0 else None
    if with_stats:
        return view_ent, target_idx, by_ent, sound_names, all_by, sound_times, stat_aw
    if with_times:
        return view_ent, target_idx, by_ent, sound_names, all_by, sound_times
    return view_ent, target_idx, by_ent, sound_names, all_by


def trigger_events(ts, merge_gap: float = MERGE_GAP) -> int:
    """Collapse a list of sound demotimes into W_Attack trigger events: a new
    event whenever the gap from the last counted event >= ``merge_gap``.

    Continuous weapons (NG/SNG/LG) fire a 0.1s think-chain (one projectile
    sound every ~0.1s while held); single-shot weapons fire >=0.5s apart so
    they are unaffected. See src/docs/mvd-attack-audit.md §Round 4."""
    if not ts:
        return 0
    ts = sorted(ts)
    n, last = 1, ts[0]
    for t in ts[1:]:
        if t - last >= merge_gap:
            n += 1
            last = t
    return n


# LG op-attack cadence (s) and the lhit heartbeat throttle (s) — see below.
LG_OP_CADENCE = 0.2
LG_HEARTBEAT_THROTTLE = 0.6


def lg_op_attack_count(lstart_times, lhit_times,
                       op_cadence: float = LG_OP_CADENCE,
                       burst_gap: float = LG_HEARTBEAT_THROTTLE + 0.1,
                       tail: float = 0.1) -> int:
    """Reconstruct the Thunderbolt (LG) operative-attack count from its sounds.

    LG emits NO per-shot sound: ``lstart`` plays once per discharge onset, and
    ``lhit`` is a fixed 0.6s firing HEARTBEAT — not a hit confirmation. QW
    ``W_FireLightning`` plays ``lhit`` (throttled by ``self.t_width = time+0.6``)
    BEFORE the damage trace, so it fires whenever the bolt is being discharged,
    hit or miss. Neither sound counts shots directly, so the old
    ``trigger_events(lstart+lhit)`` reference was ~half the true count.

    The collect's op-mask emits an LG op-attack every ``op_cadence`` (0.2s)
    while LG is held. Reconstruct that: group ``lstart ∪ lhit`` into bursts
    (gap <= one throttle period + margin), and for each burst emit
    ``round((span + tail) / op_cadence) + 1`` op-attacks, where ``tail``
    (~half a heartbeat) covers firing after the last observed ``lhit``.
    Physically bracketed — tail in [0, 0.2] lands within ~12% of the collected
    op-mask count, confirming the op-mask was correct and the reference was the
    artifact. See src/docs/mvd-attack-audit.md."""
    ev = sorted(set(round(t, 4) for t in lstart_times)
                | set(round(t, 4) for t in lhit_times))
    if not ev:
        return 0
    total = 0
    start = last = ev[0]
    for t in ev[1:]:
        if t - last <= burst_gap:
            last = t
        else:
            total += int(round((last - start + tail) / op_cadence)) + 1
            start = last = t
    total += int(round((last - start + tail) / op_cadence)) + 1
    return total


def demo_attack_sound_truth(path: str) -> tuple[dict[int, int], dict[int, int], int | None]:
    """Per-weapon byte-truth fire counts for one demo's view (self) entity.

    Returns ``(sounds, triggers, view_ent)``:
      sounds[cls]   — raw projectile-sound count per weapon class 1..8 (the
                      force-MVD reference: one collected event per fire SOUND).
      triggers[cls] — those sounds collapsed into W_Attack trigger pulls (the
                      QWD-path reference: continuous-weapon streams binned at
                      MERGE_GAP; single-shot SND == TRIG).
      view_ent      — the demonstrator entity (playernum+1), or None.
    """
    ve, _i, _b, names, ab, times = parse(path, "weapons/guncock.wav", with_times=True)
    n2i = {v: k for k, v in names.items()}
    sounds = {c: 0 for c in WNAME}
    cls_times: dict[int, list[float]] = {c: [] for c in WNAME}
    for sname, cls in SOUND_WEAPON.items():
        i = n2i.get(sname)
        if i is None or ve is None:
            continue
        sounds[cls] += ab.get((ve, i), 0)
        cls_times[cls].extend(times.get((ve, i), []))
    # Continuous-fire weapons need think-chain collapse; single-shot ones are
    # already 1 sound == 1 trigger (trigger_events is a no-op at >=0.5s cadence).
    triggers = {c: trigger_events(cls_times[c]) for c in WNAME}
    return sounds, triggers, ve


def demo_attack_truth(path: str, collapse_gap: float) -> tuple[dict[int, int], int | None]:
    """Per-weapon attack-event counts: the view entity's weapon-fire sounds
    collapsed at ``collapse_gap`` seconds. Returns ``(per_weapon, view_ent)``.

    This is the reference for a TRIGGER-PULL label (one event per held burst),
    not per raw projectile sound — so the demo truth is collapsed at the
    think-chain cadence. Pass ``collapse_gap=MERGE_GAP`` (~0.1s) for both the
    QWD and force-MVD paths:

      * QWD: the op-fire bit IS the held trigger pull.
      * force-MVD: the back-shift writer's attack bits bridge a continuous
        weapon's think-chain (sounds ~0.1s apart) into one contiguous run, so
        its rising edges are trigger pulls too. Full-corpus (2275 demos) the
        collected edge count is -2.78% vs the per-class MERGE_GAP reference (a
        mild over-count) but +9.24% vs raw per-projectile sounds — the raw-SND
        reference WAS the entire reported "~9% under-count" (a reference
        mismatch, not dropped labels). See src/docs/mvd-attack-audit.md.

    ``collapse_gap <= 0`` is a no-op (== raw per-sound count). Single-shot
    weapons fire >= 0.5s apart so any sane gap is a no-op for them."""
    ve, _i, _b, names, ab, times = parse(path, "weapons/guncock.wav", with_times=True)
    n2i = {v: k for k, v in names.items()}
    cls_times: dict[int, list[float]] = {c: [] for c in WNAME}
    for sname, cls in SOUND_WEAPON.items():
        i = n2i.get(sname)
        if i is None or ve is None:
            continue
        cls_times[cls].extend(times.get((ve, i), []))
    per_w = {c: (trigger_events(cls_times[c], collapse_gap) if collapse_gap > 0
                 else len(cls_times[c])) for c in WNAME}
    # LG (class 8): reconstruct op-attacks from the lstart onset + lhit
    # heartbeat (no per-shot sound), overriding the generic collapse.
    per_w[8] = lg_op_attack_count(
        times.get((ve, n2i.get("weapons/lstart.wav")), []) if ve is not None else [],
        times.get((ve, n2i.get("weapons/lhit.wav")), []) if ve is not None else [])
    return per_w, ve


def demo_jump_truth(path: str) -> tuple[int, int | None]:
    """Self jump-sound count for one demo (one player/plyrjmp8.wav per jump).

    Returns ``(n_jumps, view_ent)``. NOTE: the jump sound rides an unreliable
    PHS datagram, so this is a soft reference — used for the jump gate with a
    tolerance, not a hard 1:1 expectation."""
    ve, _i, _b, names, ab = parse(path, "weapons/guncock.wav")
    n2i = {v: k for k, v in names.items()}
    total = 0
    for sname in JUMP_SOUNDS:
        i = n2i.get(sname)
        if i is None or ve is None:
            continue
        total += ab.get((ve, i), 0)
    return total, ve


def demo_attack_jump_truth(path: str, collapse_gap: float,
                           drop_intervals: list | None = None,
                           total_frames: int = 0
                           ) -> tuple[dict[int, int], int, int | None]:
    """Single-parse combo of ``demo_attack_truth`` + ``demo_jump_truth``.

    Returns ``(attack_per_weapon, jump_count, view_ent)``. The byte parse is the
    expensive step, so computing both references from one parse is ~2x faster
    than calling the two helpers separately (each re-parses the demo). Used by
    the validator, which needs both per demo.

    When ``drop_intervals`` (a list of ``(start_frame, end_frame)`` from the
    manifest's signon/dead/intermission labels) and ``total_frames`` are given,
    sounds inside those dropped segments are EXCLUDED — so the reference counts
    only what the collect can actually label (the collect masks those segments).
    Without this, warmup/dead-time sounds (esp. bunny-hop jumps) inflate the
    reference and show as a spurious under-count. The sound time -> frame map is
    linear in the demo's max sound time (a duration proxy; approximate, since the
    collect's exact kept native-time spans aren't available here)."""
    ve, _i, _b, names, ab, times = parse(path, "weapons/guncock.wav", with_times=True)
    n2i = {v: k for k, v in names.items()}
    if drop_intervals and total_frames > 0:
        _all = [t for v in times.values() for t in v]
        _max_t = max(_all) if _all else 0.0

        def kept(t: float) -> bool:
            if _max_t <= 0:
                return True
            f = t / _max_t * total_frames
            return not any(a <= f <= b for a, b in drop_intervals)
    else:
        def kept(t: float) -> bool:
            return True

    cls_times: dict[int, list[float]] = {c: [] for c in WNAME}
    for sname, cls in SOUND_WEAPON.items():
        i = n2i.get(sname)
        if i is None or ve is None:
            continue
        cls_times[cls].extend(t for t in times.get((ve, i), []) if kept(t))
    attack = {c: (trigger_events(cls_times[c], collapse_gap) if collapse_gap > 0
                  else len(cls_times[c])) for c in WNAME}
    # LG (class 8): reconstruct op-attacks from the lstart onset + lhit
    # heartbeat (no per-shot sound) instead of the generic collapse.
    attack[8] = lg_op_attack_count(
        [t for t in times.get((ve, n2i.get("weapons/lstart.wav")), []) if kept(t)]
        if ve is not None else [],
        [t for t in times.get((ve, n2i.get("weapons/lhit.wav")), []) if kept(t)]
        if ve is not None else [])
    jump = 0
    for sname in JUMP_SOUNDS:
        i = n2i.get(sname)
        if i is None or ve is None:
            continue
        jump += sum(1 for t in times.get((ve, i), []) if kept(t))
    return attack, jump, ve


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: demo_truth.py <demo.qwd> [sound=weapons/guncock.wav]")
    path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "weapons/guncock.wav"
    view_ent, idx, by_ent, _names, _all = parse(path, target)
    print(f"demo:             {path}")
    print(f"target sound:     {target}")
    print(f"precache index:   {idx}")
    print(f"demonstrator ent: {view_ent}  (playernum+1)")
    print(f"by entity:        {dict(sorted(by_ent.items()))}")
    print(f"total (all ents): {sum(by_ent.values())}")
    if view_ent is not None:
        print(f"by DEMONSTRATOR:  {by_ent.get(view_ent, 0)}")
    sounds, triggers, _ = demo_attack_sound_truth(path)
    jumps, _ = demo_jump_truth(path)
    print("fire sounds:      ",
          {WNAME[c]: sounds[c] for c in WNAME if sounds[c]})
    print("fire triggers:    ",
          {WNAME[c]: triggers[c] for c in WNAME if triggers[c]})
    print(f"jumps:            {jumps}")


if __name__ == "__main__":
    main()
