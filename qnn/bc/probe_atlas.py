"""Generate mesh-based probe-atlas tables — one carved panorama per navmesh poly.

Drives a headless demo worker (``qw_demo_worker`` / ``nq_demo_worker``) to
build each map's Detour navmesh, then issues the ``nav_query kind=probe_atlas``
dump: every walkable poly center, raised to player-origin height, carves the
world-anchored panoramic depth atlas against the *static world only* (movers are
TOKEN_MOVER).  Payload is the same 4-bit log-ladder codec the live per-tick
atlas emits — the shared carve helper in ``qnn_spatial.c`` guarantees the codes
are byte-identical.

This replaces the corpus-derived probe placement (demo poses, Poisson-thinned)
with the load-time static table the design calls for: 0 wire bytes/frame, no
collect dependency for the panorama payload, probe positions anchored to the
map's reachable space rather than to where players happened to walk.

The navmesh only exists once a map's worldmodel is spawned, which the worker
does on demo reset — so one representative demo per map is used purely to load
geometry (``play_end=0``: zero frames collected, navmesh built in the reset).

Usage:
    PYTHONPATH=src python -m qnn.bc.probe_atlas \
        --worker assets/bin/qw_demo_worker \
        --demo-dir artifacts/corpus/qwd_probe \
        --manifest artifacts/corpus/qwd_probe_manifest.ndjson \
        --out-dir artifacts/collect/tmp/qwd_probe/navatlas
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from qnn.engine_norm import ATLAS_YAWS
from qnn.schema import SPATIAL_TOKEN_COUNT

_HELLO_TIMEOUT = 60.0
_QUERY_TIMEOUT = 300.0


def _first_demo_per_map(manifest: Path) -> dict[str, str]:
    """Map name -> a representative demo filename (first seen in the manifest)."""
    out: dict[str, str] = {}
    with open(manifest) as fh:
        for line in fh:
            e = json.loads(line)
            out.setdefault(e["map"], e["file"])
    return out


class DemoQueryWorker:
    """Headless demo worker with a background stdout drain (avoids the pipe-
    full deadlock while the worker streams the reset/collect output)."""

    def __init__(self, worker: str, game_dir: str, asset_root: Path, tick_hz: int):
        env = {**os.environ, "QUAKE_BASEDIR": str(asset_root.resolve())}
        # Bulk map-load queries can spend more than the collect watchdog's
        # 10-second default formatting one large response.  This worker is
        # query-only, so extend the guard while preserving an explicit caller
        # override.
        env.setdefault("QNN_WATCHDOG_SECONDS", str(int(_QUERY_TIMEOUT)))
        self.p = subprocess.Popen(
            [worker, "-game", game_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()
        self._send({"op": "hello", "map_id": "start", "tick_hz": tick_hz})
        self._wait(b'"ok"', _HELLO_TIMEOUT)

    def _drain(self) -> None:
        while not self._done.is_set():
            chunk = self.p.stdout.read1(65536)
            if not chunk:
                break
            with self._lock:
                self._buf.extend(chunk)

    def _send(self, d: dict) -> None:
        self.p.stdin.write((json.dumps(d) + "\n").encode())
        self.p.stdin.flush()

    def _snapshot(self) -> bytes:
        with self._lock:
            return bytes(self._buf)

    def _wait(self, marker: bytes, timeout: float) -> bytes:
        end = time.time() + timeout
        while time.time() < end:
            b = self._snapshot()
            if marker in b:
                return b
            if self.p.poll() is not None:
                raise RuntimeError(
                    "worker exited early; stderr:\n"
                    + self.p.stderr.read(4000).decode(errors="replace")
                )
            time.sleep(0.05)
        raise RuntimeError(f"timeout waiting for {marker!r}")

    def nav_query(self, demo: str, kind: str, **payload: object) -> dict:
        """Reset on ``demo`` and return one bulk ``nav_query`` result."""
        mark = len(self._snapshot())
        # play_end=0: reset + navmesh build, zero frames collected.
        self._send({"op": "collect", "demo_path": demo, "seed": 0,
                    "play_start": 0, "play_end": 0})
        self._send({"op": "nav_query", "kind": kind, **payload})
        needle = f'{{"ok":true,"query":"{kind}"'.encode()
        end = time.time() + _QUERY_TIMEOUT
        while time.time() < end:
            b = self._snapshot()
            m = b.find(needle, mark)
            if m >= 0:
                nl = b.find(b"\n", m)
                if nl > 0:
                    return json.loads(b[m:nl])["result"]
            if self.p.poll() is not None:
                raise RuntimeError(
                    f"worker exited during {kind}; stderr:\n"
                    + self.p.stderr.read(4000).decode(errors="replace")
                )
            time.sleep(0.1)
        raise RuntimeError(f"timeout waiting for {kind} response")

    def probe_atlas(
        self, demo: str, *, spacing: float = 0.0, z_spacing: float = 0.0,
    ) -> dict:
        """Dump every walkable poly's static panoramic atlas."""
        return self.nav_query(
            demo, "probe_atlas", spacing=spacing, z_spacing=z_spacing,
        )

    def close(self) -> None:
        self._done.set()
        try:
            self._send({"op": "shutdown"})
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def decode_dump(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """probe_atlas result -> (positions (P,3) f64, panoramas (P, ELEVS, YAWS) u8).

    Positions are the probe *viewpoints* — the poly center raised by
    ``z_offset`` to player-origin height, i.e. where the panorama was
    actually carved from (so relative-pose offsets against an agent pose are
    height-consistent).  Panoramas are world-anchored (carved at yaw 0) with
    4-bit codes in (elev, yaw) row-major order, one hex nibble per cell.
    Raises on any grid-shape mismatch against the schema the model expects.
    """
    if result["elevs"] != SPATIAL_TOKEN_COUNT or result["yaws"] != ATLAS_YAWS:
        raise ValueError(
            f"probe_atlas grid {result['elevs']}x{result['yaws']} != schema "
            f"{SPATIAL_TOKEN_COUNT}x{ATLAS_YAWS}"
        )
    polys = result["polys"]
    n = len(polys)
    z_offset = float(result["z_offset"])
    cell = SPATIAL_TOKEN_COUNT * ATLAS_YAWS
    positions = np.empty((n, 3), dtype=np.float64)
    panoramas = np.empty((n, SPATIAL_TOKEN_COUNT, ATLAS_YAWS), dtype=np.uint8)
    for i, p in enumerate(polys):
        positions[i] = p["center"]
        positions[i, 2] += z_offset
        hexs = p["atlas"]
        if len(hexs) != cell:
            raise ValueError(f"poly {i}: atlas {len(hexs)} chars != {cell}")
        # One hex nibble per code (0-15), (elev, yaw) row-major.
        ascii_ = np.frombuffer(hexs.encode("ascii"), dtype=np.uint8)
        codes = np.where(ascii_ <= ord("9"), ascii_ - ord("0"),
                         ascii_ - ord("a") + 10).astype(np.uint8)
        panoramas[i] = codes.reshape(SPATIAL_TOKEN_COUNT, ATLAS_YAWS)
    return positions, panoramas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True,
                        help="headless demo worker (qw_demo_worker / nq_demo_worker)")
    parser.add_argument("--demo-dir", type=Path, required=True,
                        help="directory holding the map demos (becomes -game)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=Path("assets"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tick-hz", type=int, default=20)
    parser.add_argument(
        "--spacing", type=float, default=0.0,
        help="world-grid spacing inside navmesh polys; 0 keeps one center/poly",
    )
    parser.add_argument(
        "--z-spacing", type=float, default=0.0,
        help="optional vertical probe spacing through open clearance",
    )
    parser.add_argument("--maps", nargs="*", default=None,
                        help="restrict to these maps (default: all in manifest)")
    args = parser.parse_args(argv)

    demos = _first_demo_per_map(args.manifest)
    if args.maps:
        demos = {m: demos[m] for m in args.maps}
    game_dir = os.path.relpath(args.demo_dir.absolute(), args.asset_root.absolute())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for map_name, demo in demos.items():
        # Keep bulk map exports independent.  A QW client reset can retain
        # map-owned state across demos, and a failure on one map must not
        # invalidate tables already emitted for another.
        worker = DemoQueryWorker(args.worker, game_dir, args.asset_root, args.tick_hz)
        try:
            result = worker.probe_atlas(
                demo, spacing=args.spacing, z_spacing=args.z_spacing,
            )
            positions, panoramas = decode_dump(result)
            out = args.out_dir / f"navatlas_{map_name}.json"
            with open(out, "w") as fh:
                json.dump(result, fh)
            print(f"{map_name}: {len(positions)} poly-probes "
                  f"(z_offset {result['z_offset']}) -> {out}")
        finally:
            worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
