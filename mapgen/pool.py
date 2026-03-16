"""Async pool of procedurally generated Quake maps for training.

Maintains a queue of pre-compiled BSP files.  Background threads
continuously generate new maps so that ``get()`` returns immediately
(or with minimal blocking) when an environment needs a fresh map.
"""

from __future__ import annotations

import random
import shutil
import threading
from pathlib import Path
from queue import Queue
from typing import Mapping

from .brush import MapFile
from .compile import compile_map
from .entities import populate
from .layout import build_layout, generate_layout
from .navcheck import validate_layout_graph
from .textures import materialize_texture_wad

PROCGEN_SENTINEL = "procgen"

_MAX_LAYOUT_ATTEMPTS = 10


def generate_bsp(
    seed: int,
    output_dir: Path,
    *,
    rooms: int = 3,
    arena_size: int = 3072,
) -> tuple[str, Path]:
    """Generate and compile a single procgen map.

    Returns ``(map_id, bsp_path)``.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_layout = None
    best_unreachable = float("inf")
    for attempt in range(_MAX_LAYOUT_ATTEMPTS):
        layout = generate_layout(rng, arena_size=arena_size, max_depth=rooms)
        result = validate_layout_graph(layout)
        if result.connected:
            best_layout = layout
            break
        if len(result.unreachable_rooms) < best_unreachable:
            best_unreachable = len(result.unreachable_rooms)
            best_layout = layout
        if attempt < _MAX_LAYOUT_ATTEMPTS - 1:
            rng = random.Random(seed + attempt + 1)

    layout = best_layout  # type: ignore[assignment]

    map_id = f"gen_{seed}"
    map_path = output_dir / f"{map_id}.map"

    m = MapFile()
    m.worldspawn.properties["message"] = map_id
    build_layout(m, layout)
    populate(m, layout, rng)
    materialize_texture_wad(output_dir)

    with open(map_path, "w") as f:
        m.write(f)

    bsp_path = compile_map(map_path, output_dir=output_dir)
    return map_id, bsp_path


class MapPool:
    """Pre-compiles procgen maps in background threads.

    Parameters
    ----------
    maps_dir:
        Directory to write ``.map`` and ``.bsp`` files into.  Typically
        ``$QUAKE_BASEDIR/id1/maps`` so the engine can find them.
    pool_size:
        Maximum number of ready maps buffered in the queue.
    workers:
        Number of background compilation threads.
    rooms, arena_size:
        Forwarded to the layout generator.
    """

    def __init__(
        self,
        maps_dir: Path,
        *,
        pool_size: int = 4,
        workers: int = 2,
        rooms: int = 3,
        arena_size: int = 3072,
    ) -> None:
        self._maps_dir = Path(maps_dir)
        self._maps_dir.mkdir(parents=True, exist_ok=True)
        self._rooms = rooms
        self._arena_size = arena_size
        self._queue: Queue[str] = Queue(maxsize=pool_size)
        self._rng = random.Random()
        self._rng_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._threads = [
            threading.Thread(target=self._worker_loop, daemon=True, name=f"mappool-{i}")
            for i in range(workers)
        ]
        for t in self._threads:
            t.start()

    def _next_seed(self) -> int:
        with self._rng_lock:
            return self._rng.randint(0, 2**31 - 1)

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            seed = self._next_seed()
            try:
                map_id, _ = generate_bsp(
                    seed,
                    self._maps_dir,
                    rooms=self._rooms,
                    arena_size=self._arena_size,
                )
            except Exception:
                # Compilation can fail on rare degenerate layouts; just retry.
                continue
            # Blocks if queue is full — back-pressure.
            try:
                self._queue.put(map_id, timeout=1.0)
            except Exception:
                # Queue full and timed out, or shutting down.
                # Clean up the map we just generated to avoid disk leak.
                for ext in (".bsp", ".map", ".log", ".prt"):
                    (self._maps_dir / f"{map_id}{ext}").unlink(missing_ok=True)
                continue

    def get(self, timeout: float | None = None) -> str:
        """Return the ``map_id`` of the next ready map.

        Blocks until a compiled map is available (or until *timeout*).
        """
        return self._queue.get(timeout=timeout)

    def close(self) -> None:
        self._shutdown.set()
        for t in self._threads:
            t.join(timeout=5.0)

    def __del__(self) -> None:
        self.close()
