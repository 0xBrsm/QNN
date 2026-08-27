"""ORT-driven h2h side: fight the DEPLOYED artifact.

Cross-generation h2h at the checkpoint level is blocked BY DESIGN (pre-a28
checkpoints load only from their own branches — the 2026-08-04 defensibility
gate result). What every generation shares is the deploy contract: a
self-contained exported ``.onnx`` with the decode baked in-graph, RNG carried
as loopback state, and raw native wire fields as inputs. This module drives
that artifact per lane at batch=1 with onnxruntime — the Python twin of what
``nq_client`` (src/engine/common/qnn_onnx.c) does on g4 — so an
``a26rc1b.onnx`` can fight an ``a28rc1a.onnx`` inside ``qnn.eval.h2h``'s
router, states fully isolated per lane, no torch checkpoint loading at all.

Contracts honored (single sources of truth):
  * input naming — :data:`qnn.schema.ONNX_OBS_KEY_REMAP` (the exporter's
    table, shared, not copied);
  * state carry — the ``state.loopback`` metadata declaration
    (``in=...,out=...,init=zeros|entropy|<literal>,reset=episode|persist``;
    grammar of tools/export_onnx.py:_state_loopback_decl, parsed engine-side
    by qnn_loopback_parse);
  * action rendering — the qnn.actions packet convention: ``move`` int64
    class triple → sign floats (class − 1; jump rides ``move[2] > 0.5``),
    ``look`` passes through (the graph emits the decoded look vector),
    ``attack`` is the 0..8 attack-with category.

``onnxruntime`` is not in the base venv — the repo-local ``.ort_tmp``
install (the exporter's target dir) is added to ``sys.path`` on demand.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qnn.schema import ONNX_OBS_KEY_REMAP

_ORT_LOCAL_DIR = Path(__file__).resolve().parents[3] / ".ort_tmp"


def _import_ort():
    try:
        import onnxruntime as ort  # noqa: PLC0415
        return ort
    except ImportError:
        if _ORT_LOCAL_DIR.is_dir() and str(_ORT_LOCAL_DIR) not in sys.path:
            sys.path.insert(0, str(_ORT_LOCAL_DIR))
            import onnxruntime as ort  # noqa: PLC0415
            return ort
        raise


@dataclass(frozen=True)
class LoopbackLane:
    in_name: str
    out_name: str
    init: str
    reset: str


def parse_loopback(decl: str) -> tuple[LoopbackLane, ...]:
    """Parse the ``state.loopback`` metadata declaration."""
    lanes: list[LoopbackLane] = []
    for entry in filter(None, (e.strip() for e in decl.split(";"))):
        kv: dict[str, str] = {}
        for part in entry.split(","):
            k, _, v = part.partition("=")
            kv[k.strip()] = v.strip()
        if "in" not in kv or "out" not in kv:
            raise ValueError(f"loopback entry missing in=/out=: {entry!r}")
        lanes.append(LoopbackLane(kv["in"], kv["out"],
                                  kv.get("init", "zeros"),
                                  kv.get("reset", "episode")))
    return tuple(lanes)


class OrtSide:
    """One h2h combatant backed by a deployed ONNX artifact.

    Mirrors the slice of the ``QNNPolicy`` surface the h2h router touches
    (``zero_hidden`` for ``_fresh_state``; per-side decode state lives HERE,
    keyed by lane, not in the router's ``_EpisodeState``)."""

    is_ort = True

    def __init__(self, name: str, onnx_path: Path, *, seed: int = 0):
        ort = _import_ort()
        self.name = name
        self.onnx_path = Path(onnx_path)
        # ORT's own intra/inter-op thread pool defaults to spinning across
        # (close to) all visible cores per InferenceSession — real cost for
        # these graphs (single-threaded 1996 engine code + a ~1MB GRU graph
        # at batch=1) is nowhere near that; observed ~3 cores/instance under
        # concurrent h2h runs is thread-pool spin, not compute. Gated behind
        # QNN_ORT_THREADS (unset = untouched ORT default) rather than hard-
        # coded, so this file stays behavior-identical for every existing/
        # blessed eval unless the caller opts in.
        sess_opts = ort.SessionOptions()
        threads_env = os.environ.get("QNN_ORT_THREADS")
        if threads_env is not None:
            n = int(threads_env)
            sess_opts.intra_op_num_threads = n
            sess_opts.inter_op_num_threads = n
        self.sess = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts,
            providers=["CPUExecutionProvider"])
        self.meta: dict[str, str] = dict(
            self.sess.get_modelmeta().custom_metadata_map)
        if "state.loopback" not in self.meta:
            raise ValueError(
                f"{onnx_path}: no state.loopback metadata — not a sampled "
                "deploy export (parity exports cannot be driven)")
        self.loopback = parse_loopback(self.meta["state.loopback"])
        self.tick_hz = int(float(self.meta.get("tick_hz", 20)))
        self.version = self.meta.get("version", self.onnx_path.stem)
        self._inputs = {i.name: i for i in self.sess.get_inputs()}
        self._output_names = [o.name for o in self.sess.get_outputs()]
        self._loop_in_names = {l.in_name for l in self.loopback}
        missing = self._loop_in_names - set(self._inputs)
        if missing:
            raise ValueError(f"{onnx_path}: loopback inputs {missing} not in graph")
        for core in ("move", "look", "attack"):
            if core not in self._output_names:
                raise ValueError(f"{onnx_path}: no `{core}` output — need the "
                                 "sampled deploy export")
        self._seed = int(seed)
        self._lane_state: dict[int, dict[str, np.ndarray]] = {}

    # ── QNNPolicy-surface shims for the router ────────────────────────────
    def zero_hidden(self, n: int) -> np.ndarray:
        # the router stores per-lane hidden; ours lives in _lane_state —
        # hand it an inert placeholder of the right rank.
        return np.zeros((n, 1), dtype=np.float32)

    # ── loopback state ────────────────────────────────────────────────────
    def _input_zeros(self, name: str) -> np.ndarray:
        spec = self._inputs[name]
        shape = tuple(1 if not isinstance(d, int) else d for d in spec.shape)
        return np.zeros(shape, dtype=_np_dtype(spec.type))

    def _init_lane_value(self, lane: int, lb: LoopbackLane) -> np.ndarray:
        base = self._input_zeros(lb.in_name)
        if lb.init == "zeros":
            return base
        if lb.init == "entropy":
            # deterministic per (side seed, lane, lane name): reproducible
            # runs, distinct streams per lane — never zero (xorshift lock-up).
            rs = np.random.RandomState(
                (self._seed * 1_000_003 + lane * 8191 + hash(lb.in_name) % 65521)
                & 0x7FFFFFFF)
            if np.issubdtype(base.dtype, np.integer):
                vals = rs.randint(1, np.iinfo(np.uint32).max, size=base.shape)
                return vals.astype(base.dtype)
            return rs.standard_normal(base.shape).astype(base.dtype)
        # literal CSV lanes, TILED across the buffer when it holds more
        # elements than lanes (qnn_lb_apply_init: v = csv[i % n_csv])
        vals = np.asarray([float(x) for x in lb.init.split()], dtype=np.float64)
        idx = np.arange(base.size) % vals.size
        return vals[idx].reshape(base.shape).astype(base.dtype)

    def reset_lane(self, lane: int) -> None:
        """Round boundary: episode-scoped lanes re-init, persist lanes carry
        (rng streams keep advancing — the engine's reset=persist semantics)."""
        st = self._lane_state.setdefault(lane, {})
        for lb in self.loopback:
            if lb.in_name not in st or lb.reset == "episode":
                if lb.in_name in st and lb.reset == "persist":
                    continue
                st[lb.in_name] = self._init_lane_value(lane, lb)

    # ── per-tick forward ──────────────────────────────────────────────────
    def act_lanes(self, lanes: Sequence[int],
                  obs_rows: Sequence[Mapping[str, np.ndarray]],
                  ) -> list[dict[str, Any]]:
        acts = []
        for lane, row in zip(lanes, obs_rows):
            if lane not in self._lane_state:
                self.reset_lane(lane)
            acts.append(self._act_one(lane, row))
        return acts

    def _act_one(self, lane: int, obs_row: Mapping[str, np.ndarray]) -> dict[str, Any]:
        st = self._lane_state[lane]
        feeds: dict[str, np.ndarray] = {}
        for name, spec in self._inputs.items():
            if name in self._loop_in_names:
                feeds[name] = st[name]
                continue
            key = ONNX_OBS_KEY_REMAP.get(name, name)
            if key not in obs_row:
                raise KeyError(
                    f"{self.version}: graph wants obs field {key!r} "
                    f"(input {name!r}) but the arena obs lacks it")
            feeds[name] = _fit_input(np.asarray(obs_row[key]), spec)
        outs = dict(zip(self._output_names,
                        self.sess.run(self._output_names, feeds)))
        for lb in self.loopback:
            st[lb.in_name] = np.asarray(outs[lb.out_name],
                                        dtype=st[lb.in_name].dtype).reshape(
                                            st[lb.in_name].shape)
        move = np.asarray(outs["move"]).reshape(-1)
        look = np.asarray(outs["look"]).reshape(-1)
        act: dict[str, Any] = {
            # int64 class triple → the qnn.actions sign convention
            # (class − 1 per axis, qnn_onnx.c:1326-1331; jump IS up_pos —
            # move[2] > 0.5, QNN_PackInputMask).
            "move": [float(int(c) - 1) for c in move[:3]],
            "look": [float(v) for v in look[:3]],
            "attack": int(np.asarray(outs["attack"]).reshape(-1)[0]),
        }
        if "weapon" in outs:
            # wire.9/12.x era: decided weapon impulse rides its own head
            # (h2h's attack-lane convention detector keys on this).
            act["weapon"] = int(np.asarray(outs["weapon"]).reshape(-1)[0])
        return act


def _np_dtype(ort_type: str) -> np.dtype:
    table = {
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(double)": np.float64, "tensor(int64)": np.int64,
        "tensor(int32)": np.int32, "tensor(int16)": np.int16,
        "tensor(int8)": np.int8, "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16, "tensor(uint32)": np.uint32,
        "tensor(bool)": np.bool_,
    }
    if ort_type not in table:
        raise ValueError(f"unsupported ONNX input type {ort_type}")
    return np.dtype(table[ort_type])


def _fit_input(arr: np.ndarray, spec) -> np.ndarray:
    """Cast + pad/trim one native obs field to the graph input's fixed
    (batch=1) shape. Leading dims are prepended until ranks match (the obs
    row has no batch dim); each axis is clip-copied. Slots beyond the live
    data zero-fill except ``entity_types``, whose empty-slot sentinel is −1
    (the actor-mask convention; run.py:_pad_entities_to_max)."""
    dtype = _np_dtype(spec.type)
    shape = tuple(1 if not isinstance(d, int) else d for d in spec.shape)
    out = np.zeros(shape, dtype=dtype)
    if spec.name == "entity_types":
        out.fill(-1)
    a = np.asarray(arr)
    while a.ndim < len(shape):
        a = a[None]
    if a.ndim != len(shape):
        raise ValueError(
            f"{spec.name}: obs rank {np.asarray(arr).ndim} does not fit "
            f"graph shape {shape}")
    sl = tuple(slice(0, min(ad, sd)) for ad, sd in zip(a.shape, shape))
    out[sl] = a[sl].astype(dtype, copy=False)
    return out
