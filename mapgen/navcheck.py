"""Graph-based connectivity validation for generated layouts.

BFS on the room-corridor topology, filtering corridors that aren't
physically navigable (too narrow, height difference too steep).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .layout import (
    CORRIDOR_WIDTH,
    Layout,
    MAX_STAIR_RISE,
    STAIR_HEIGHT,
    WALKABLE_CLIMB,
    WALKABLE_RADIUS,
)

MIN_STEP_RUN = 16


@dataclass
class NavcheckResult:
    connected: bool = False
    component_count: int = 0
    unreachable_rooms: list[int] = field(default_factory=list)
    error: str = ""


def validate_layout_graph(layout: Layout) -> NavcheckResult:
    """Check that all rooms are reachable via navigable corridors."""
    result = NavcheckResult()
    n = len(layout.rooms)
    if n == 0:
        result.error = "No rooms"
        return result

    # Build adjacency list, filtering unnavigable corridors.
    adj: list[list[int]] = [[] for _ in range(n)]

    for c in layout.corridors:
        if c.room_a < 0 or c.room_a >= n or c.room_b < 0 or c.room_b >= n:
            continue

        # Corridor must be wide enough for player.
        width = (c.y1 - c.y0) if c.axis == "x" else (c.x1 - c.x0)
        if width < int(WALKABLE_RADIUS * 2):
            continue

        # Height difference must be climbable.
        dz = abs(c.z0_b - c.z0_a)
        if dz > 0:
            bridged = min(dz, MAX_STAIR_RISE)
            remainder = dz - bridged
            if remainder > WALKABLE_CLIMB:
                continue
            total_run = (c.x1 - c.x0) if c.axis == "x" else (c.y1 - c.y0)
            max_steps = total_run // MIN_STEP_RUN
            min_steps = (bridged + WALKABLE_CLIMB - 1) // WALKABLE_CLIMB
            if min_steps > max_steps:
                continue

        adj[c.room_a].append(c.room_b)
        adj[c.room_b].append(c.room_a)

    # BFS from room 0.
    visited = [False] * n
    q: deque[int] = deque([0])
    visited[0] = True

    while q:
        room = q.popleft()
        for neighbor in adj[room]:
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append(neighbor)

    result.unreachable_rooms = [i for i in range(n) if not visited[i]]

    # Count connected components.
    result.component_count = 1
    for i in range(n):
        if visited[i]:
            continue
        result.component_count += 1
        cq: deque[int] = deque([i])
        visited[i] = True
        while cq:
            room = cq.popleft()
            for neighbor in adj[room]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    cq.append(neighbor)

    result.connected = len(result.unreachable_rooms) == 0
    return result
