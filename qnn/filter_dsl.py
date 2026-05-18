"""Shared MongoDB-style predicate evaluator over flat dicts.

Used by:
- ``qnn.bc.collect`` — manifest-entry filtering (scalar leaves, demo-level
  drop/keep decisions)
- ``qnn.bc.train`` — per-frame segment-mask derivation (array leaves,
  load-time engagement / state filters)

Both callers flatten their data into ``Mapping[str, Any]`` keyed by the
predicate's field paths and pass it here.  Operators dispatch through
numpy element-wise ops, so a Python scalar yields a Python/numpy bool
and a numpy array yields a bool-array — same code path either way.

The DSL grammar:

    predicate := {<key>: <value>, ...}     # implicit-AND of entries
    key       := "$and" | "$or" | "$not" | <field_path>
    value     := <bare-value>              # implicit $eq
               | {<op>: <arg>, ...}        # operator dict, implicit-AND
               | [predicate, ...]          # for $and / $or
               | predicate                 # for $not

Comparison ops: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin.
Logical ops:    $and, $or, $not.

Unknown ops or field paths fail loud (no silent skips).  Per
``agents/conventions.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

COMPARISON_OPS = frozenset((
    "$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte",
))
LOGICAL_OPS = frozenset(("$and", "$or", "$not"))


def _apply_op(value: Any, op: str, target: Any) -> Any:
    if op == "$eq":  return np.equal(value, target)
    if op == "$ne":  return np.not_equal(value, target)
    if op == "$in" or op == "$nin":
        if not isinstance(target, (list, tuple)):
            raise ValueError(f"{op} expects a list, got {type(target).__name__}")
        mask = np.isin(value, np.asarray(target))
        return mask if op == "$in" else np.logical_not(mask)
    if op == "$gt":  return np.greater(value, target)
    if op == "$gte": return np.greater_equal(value, target)
    if op == "$lt":  return np.less(value, target)
    if op == "$lte": return np.less_equal(value, target)
    raise ValueError(
        f"unknown comparison operator {op!r}.  Valid: {sorted(COMPARISON_OPS)}"
    )


def _eval_leaf(value: Any, predicate: Any) -> Any:
    """Apply an operator dict (or bare value = implicit $eq) to a
    resolved field value.  Multiple ops in a dict implicit-AND. """
    if not isinstance(predicate, dict):
        return _apply_op(value, "$eq", predicate)
    if not predicate:
        raise ValueError("empty operator dict {} is not a valid predicate")
    parts = [_apply_op(value, op, arg) for op, arg in predicate.items()]
    return parts[0] if len(parts) == 1 else np.logical_and.reduce(parts)


def eval_filter(data: Mapping[str, Any], predicate: Any) -> Any:
    """Evaluate a MongoDB-style predicate against a flat dict.

    Returns whatever the leaves produce — Python bool, numpy bool, or
    numpy bool-array.  Empty predicate returns True (matches all).
    Unknown field paths and operators raise (no silent skip). """
    if not isinstance(predicate, dict):
        raise ValueError(
            f"predicate must be a dict, got {type(predicate).__name__}"
        )
    if not predicate:
        return True
    parts = []
    for key, val in predicate.items():
        if key == "$and":
            if not isinstance(val, list):
                raise ValueError("$and expects a list of sub-predicates")
            parts.append(np.logical_and.reduce(
                [eval_filter(data, sub) for sub in val]
            ))
        elif key == "$or":
            if not isinstance(val, list):
                raise ValueError("$or expects a list of sub-predicates")
            parts.append(np.logical_or.reduce(
                [eval_filter(data, sub) for sub in val]
            ))
        elif key == "$not":
            parts.append(np.logical_not(eval_filter(data, val)))
        elif key.startswith("$"):
            raise ValueError(
                f"unknown logical operator {key!r}.  "
                f"Valid: {sorted(LOGICAL_OPS)}"
            )
        else:
            if key not in data:
                raise KeyError(
                    f"unknown field path {key!r} (not in flat data). "
                    f"available sample: {sorted(list(data))[:10]}"
                )
            parts.append(_eval_leaf(data[key], val))
    return parts[0] if len(parts) == 1 else np.logical_and.reduce(parts)


def validate_predicate(predicate: Any,
                       is_valid_path: Callable[[str], bool]) -> None:
    """Walk a predicate and validate caller-managed leaf paths.

    Structural validation (operator name set, $and/$or arg shapes) is
    enforced here too so a single call surfaces all spec errors at
    startup rather than at evaluation time. """
    if not isinstance(predicate, dict):
        return
    for key, val in predicate.items():
        if key in ("$and", "$or"):
            if not isinstance(val, list):
                raise ValueError(f"{key} expects a list of sub-predicates")
            for sub in val:
                validate_predicate(sub, is_valid_path)
        elif key == "$not":
            validate_predicate(val, is_valid_path)
        elif key.startswith("$"):
            raise ValueError(
                f"unknown logical operator {key!r}.  "
                f"Valid: {sorted(LOGICAL_OPS)}"
            )
        else:
            if not is_valid_path(key):
                raise ValueError(f"unknown field path {key!r}")
            if isinstance(val, dict):
                for op in val:
                    if op not in COMPARISON_OPS:
                        raise ValueError(
                            f"unknown comparison operator {op!r} in {key!r}.  "
                            f"Valid: {sorted(COMPARISON_OPS)}"
                        )
