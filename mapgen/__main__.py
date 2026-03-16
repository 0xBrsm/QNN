"""CLI entry point: python -m mapgen [options]"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from .brush import MapFile
from .compile import compile_map, find_tool
from .entities import populate
from .layout import build_layout, generate_layout
from .navcheck import validate_layout_graph
from .textures import materialize_texture_wad

MAX_ATTEMPTS = 10


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="mapgen",
        description="Procedural Quake 1 .map generator for training gyms.",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed (random if omitted)")
    p.add_argument("--rooms", type=int, default=3, help="Max BSP subdivision depth (default: 3)")
    p.add_argument("--arena-size", type=int, default=3072,
                   help="Arena bounding box side length in Quake units (default: 3072)")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output .map path (default: gen_<seed>.map)")
    p.add_argument("--compile", action="store_true",
                   help="Compile to .bsp using ericw-tools (qbsp/vis/light)")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Directory for output files (default: cwd)")
    args = p.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_path = Path(args.output) if args.output else output_dir / f"gen_{seed}.map"

    # Generate layout with connectivity validation and retry.
    best_layout = None
    best_unreachable = float("inf")

    for attempt in range(MAX_ATTEMPTS):
        layout = generate_layout(rng, arena_size=args.arena_size, max_depth=args.rooms)
        result = validate_layout_graph(layout)

        if result.connected:
            best_layout = layout
            break

        n_unreachable = len(result.unreachable_rooms)
        if n_unreachable < best_unreachable:
            best_unreachable = n_unreachable
            best_layout = layout

        if attempt < MAX_ATTEMPTS - 1:
            # Re-seed for next attempt.
            rng = random.Random(seed + attempt + 1)

    layout = best_layout  # type: ignore[assignment]
    nav = validate_layout_graph(layout)
    n_rooms = len(layout.rooms)
    n_corrs = len(layout.corridors)
    status = "connected" if nav.connected else f"{len(nav.unreachable_rooms)} unreachable"
    print(f"seed={seed}  rooms={n_rooms}  corridors={n_corrs}  {status}")

    # Build map.
    m = MapFile()
    m.worldspawn.properties["message"] = f"gen_{seed}"
    build_layout(m, layout)
    populate(m, layout, rng)

    materialize_texture_wad(map_path.parent)

    # Write .map file.
    with open(map_path, "w") as f:
        m.write(f)
    print(f"wrote {map_path}")

    # Optionally compile.
    if args.compile:
        if find_tool("qbsp") is None:
            print("error: qbsp not found on PATH. Install ericw-tools.", file=sys.stderr)
            sys.exit(1)
        bsp_path = compile_map(map_path, output_dir=output_dir)
        print(f"compiled {bsp_path}")


if __name__ == "__main__":
    main()
