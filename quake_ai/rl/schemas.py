"""Data contracts for the Quake AI pipeline — map geometry schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_float_list(value: Iterable[Any], expected_len: int, field_name: str) -> List[float]:
    out = [float(x) for x in value]
    _require(len(out) == expected_len, f"{field_name} must have length {expected_len}")
    return out


def _as_int_list(value: Iterable[Any], field_name: str) -> List[int]:
    return [int(x) for x in value]


def _as_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), f"{field_name} must be a mapping")
    return {str(k): v for k, v in value.items()}


@dataclass(slots=True)
class RegionNode:
    region_id: int
    center: List[float]
    bounds_min: List[float]
    bounds_max: List[float]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegionNode":
        record = cls(
            region_id=int(data["region_id"]),
            center=_as_float_list(data["center"], 3, "center"),
            bounds_min=_as_float_list(data["bounds_min"], 3, "bounds_min"),
            bounds_max=_as_float_list(data["bounds_max"], 3, "bounds_max"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(self.region_id >= 0, "region_id must be >= 0")
        _require(len(self.center) == 3, "center must have length 3")
        _require(len(self.bounds_min) == 3, "bounds_min must have length 3")
        _require(len(self.bounds_max) == 3, "bounds_max must have length 3")
        for axis in range(3):
            _require(self.bounds_min[axis] <= self.bounds_max[axis], "bounds_min must be <= bounds_max")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StaticObject:
    object_id: str
    category: str
    classname: str
    region_id: int
    origin: List[float]
    angles: List[float]
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StaticObject":
        record = cls(
            object_id=str(data["object_id"]),
            category=str(data["category"]),
            classname=str(data.get("classname", "")),
            region_id=int(data["region_id"]),
            origin=_as_float_list(data["origin"], 3, "origin"),
            angles=_as_float_list(data.get("angles", [0.0, 0.0, 0.0]), 3, "angles"),
            properties=_as_mapping(data.get("properties", {}), "properties"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(bool(self.object_id), "object_id must be non-empty")
        _require(bool(self.category), "category must be non-empty")
        _require(self.region_id >= 0, "region_id must be >= 0")
        _require(len(self.origin) == 3, "origin must have length 3")
        _require(len(self.angles) == 3, "angles must have length 3")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MapState:
    map_id: str
    regions: List[RegionNode]
    static_objects: List[StaticObject]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MapState":
        record = cls(
            map_id=str(data["map_id"]),
            regions=[RegionNode.from_dict(row) for row in data.get("regions", [])],
            static_objects=[StaticObject.from_dict(row) for row in data.get("static_objects", [])],
            metadata=_as_mapping(data.get("metadata", {}), "metadata"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(bool(self.map_id), "map_id must be non-empty")
        region_ids = [region.region_id for region in self.regions]
        _require(len(region_ids) == len(set(region_ids)), "region ids must be unique")
        known_regions = set(region_ids)
        object_ids = [obj.object_id for obj in self.static_objects]
        _require(len(object_ids) == len(set(object_ids)), "static object ids must be unique")

        for region in self.regions:
            region.validate()

        for obj in self.static_objects:
            obj.validate()
            _require(obj.region_id in known_regions, "static objects must reference known regions")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
