"""Re-export of the supervised-epoch entry point and data carriers.

Kept as a stable import path; the implementation lives in
:mod:`qnn.bc.supervised_loop`.
"""

from __future__ import annotations

from qnn.bc.supervised_loop import (
    GradClipper,
    GradClipSpec,
    MidEpochState,
    PrecomputedEpisode,
    ResidentSource,
    Source,
    StreamingSource,
    make_resident_source,
    make_resident_source_from_cache,
    make_streaming_source,
    run_epoch,
)

__all__ = [
    "GradClipper",
    "GradClipSpec",
    "MidEpochState",
    "PrecomputedEpisode",
    "ResidentSource",
    "Source",
    "StreamingSource",
    "make_resident_source",
    "make_resident_source_from_cache",
    "make_streaming_source",
    "run_epoch",
]
