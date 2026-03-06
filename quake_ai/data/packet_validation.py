"""Packet-to-telemetry alignment checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from quake_ai.schemas import PacketEventV1, TelemetryTickV1
from quake_ai.utils.io import read_ndjson


@dataclass(slots=True)
class PacketValidationReport:
    total_packets: int
    unmatched_packets: int
    duplicate_seq: int
    out_of_window: int

    @property
    def unmatched_rate(self) -> float:
        if self.total_packets == 0:
            return 0.0
        return self.unmatched_packets / self.total_packets

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "total_packets": self.total_packets,
            "unmatched_packets": self.unmatched_packets,
            "duplicate_seq": self.duplicate_seq,
            "out_of_window": self.out_of_window,
            "unmatched_rate": self.unmatched_rate,
        }


def validate_packet_alignment(
    telemetry_path: str,
    packets_path: str,
    tick_window: int = 2,
) -> PacketValidationReport:
    telemetry = [TelemetryTickV1.from_dict(row) for row in read_ndjson(telemetry_path)]
    packets = [PacketEventV1.from_dict(row) for row in read_ndjson(packets_path)]

    ticks_by_episode: Dict[str, set[int]] = {}
    for row in telemetry:
        ticks_by_episode.setdefault(row.episode_id, set()).add(row.tick)

    seen_seq: set[tuple[str, int, str]] = set()
    unmatched = 0
    duplicates = 0
    out_of_window = 0

    for packet in packets:
        key = (packet.episode_id, packet.seq, packet.direction)
        if key in seen_seq:
            duplicates += 1
        seen_seq.add(key)

        ticks = ticks_by_episode.get(packet.episode_id, set())
        if not ticks:
            unmatched += 1
            continue

        match = False
        for delta in range(-tick_window, tick_window + 1):
            if packet.tick_estimate + delta in ticks:
                match = True
                break
        if not match:
            unmatched += 1
            out_of_window += 1

    return PacketValidationReport(
        total_packets=len(packets),
        unmatched_packets=unmatched,
        duplicate_seq=duplicates,
        out_of_window=out_of_window,
    )
