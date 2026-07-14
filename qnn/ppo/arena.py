"""Role-neutral match/seat topology for grouped PPO arenas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ArenaSeatRole(StrEnum):
    LEARNER = "learner"
    OPPONENT_POLICY = "opponent_policy"
    ENGINE_BOT = "engine_bot"

    @property
    def externally_controlled(self) -> bool:
        return self is not ArenaSeatRole.ENGINE_BOT


@dataclass(frozen=True)
class ArenaSeat:
    server_id: int
    match_id: int
    seat_id: int
    role: ArenaSeatRole
    env_id: int | None

    @property
    def identity(self) -> tuple[int, int, int]:
        return self.server_id, self.match_id, self.seat_id


@dataclass(frozen=True)
class ArenaTopology:
    """Static seat assignment shared by bot PPO and two-policy self-play.

    ``num_lanes`` counts externally controlled trajectories.  Engine-bot
    seats occupy an upstream client slot but intentionally have no env id.
    """

    num_lanes: int
    matches_per_server: int
    seat_mode: str
    seats: tuple[ArenaSeat, ...]

    @classmethod
    def build(
        cls,
        *,
        num_lanes: int,
        matches_per_server: int = 8,
        seat_mode: str = "bot",
    ) -> "ArenaTopology":
        lanes = int(num_lanes)
        matches = int(matches_per_server)
        mode = str(seat_mode)
        if lanes < 1:
            raise ValueError("num_lanes must be >= 1")
        if not 1 <= matches <= 8:
            raise ValueError("matches_per_server must be in [1, 8]")
        if mode not in {"bot", "self_play"}:
            raise ValueError("seat_mode must be 'bot' or 'self_play'")

        external_per_match = 1 if mode == "bot" else 2
        lanes_per_server = matches * external_per_match
        if lanes % lanes_per_server:
            raise ValueError(
                f"num_lanes={lanes} must be divisible by {lanes_per_server} "
                f"for {matches} matches/server in {mode} mode"
            )

        seats: list[ArenaSeat] = []
        next_env_id = 0
        server_count = lanes // lanes_per_server
        for server_id in range(server_count):
            for match_id in range(matches):
                seats.append(
                    ArenaSeat(
                        server_id=server_id,
                        match_id=match_id,
                        seat_id=0,
                        role=ArenaSeatRole.LEARNER,
                        env_id=next_env_id,
                    )
                )
                next_env_id += 1
                if mode == "self_play":
                    seats.append(
                        ArenaSeat(
                            server_id=server_id,
                            match_id=match_id,
                            seat_id=1,
                            role=ArenaSeatRole.OPPONENT_POLICY,
                            env_id=next_env_id,
                        )
                    )
                    next_env_id += 1
                else:
                    seats.append(
                        ArenaSeat(
                            server_id=server_id,
                            match_id=match_id,
                            seat_id=1,
                            role=ArenaSeatRole.ENGINE_BOT,
                            env_id=None,
                        )
                    )

        assert next_env_id == lanes
        return cls(
            num_lanes=lanes,
            matches_per_server=matches,
            seat_mode=mode,
            seats=tuple(seats),
        )

    @property
    def server_count(self) -> int:
        return max(seat.server_id for seat in self.seats) + 1

    @property
    def external_seats(self) -> tuple[ArenaSeat, ...]:
        return tuple(seat for seat in self.seats if seat.role.externally_controlled)

    def match_seats(self, server_id: int, match_id: int) -> tuple[ArenaSeat, ArenaSeat]:
        pair = [
            seat
            for seat in self.seats
            if seat.server_id == server_id and seat.match_id == match_id
        ]
        if len(pair) != 2:
            raise KeyError((server_id, match_id))
        return pair[0], pair[1]

    def seat_for_env(self, env_id: int) -> ArenaSeat:
        if not 0 <= int(env_id) < self.num_lanes:
            raise KeyError(env_id)
        # env ids are dense in construction order; external_seats preserves it.
        return self.external_seats[int(env_id)]
