"""In-process hot-reload of the BC application code graph.

The ablation daemon keeps the training corpus resident in **GPU VRAM**
(:func:`qnn.bc.container.build_behavior_cloning_sources` with
``streaming=false`` materializes the cache directly to the device). A
long-lived process freezes every imported module in ``sys.modules``, so a
head / model / loss / training-step edit on disk stays invisible until the
process re-imports it. Restarting the daemon would pick up the edit but also
dump the resident corpus and pay the full VRAM reload.

:func:`reload_bc_code` purges and re-imports the ``qnn.*`` application graph
in-process so the *next* submitted job builds its model and runs its training
step from the edited source — **without touching the resident corpus**.

Boundary — what a hot-reload can and cannot pick up
---------------------------------------------------
* **Reloadable** (the common case): anything that runs *per step* — model
  forward (``qnn.model.*``), loss / metrics, the training loop, and bench head
  construction. Each job builds a *fresh* model via ``model_factory``, so no
  stale class instance survives between jobs. ``Source`` is a structural
  :class:`typing.Protocol` (no ``isinstance`` gate anywhere in the BC path),
  so reloaded training code runs against the *kept* bundle by duck-typing.

* **NOT reloadable — and silent** (guarded here): the dequantizer. The resident
  source is materialized with ``compact_dequantized=True``, which bakes
  ``qnn.model.dequant`` into the stored tensor *values*; ``ObsAccessor`` then
  passes through when obs are already dequantized, so the per-step path never
  re-runs dequant. An edit to ``dequant.py`` would therefore be applied to
  *no* live data — a silent staleness. :func:`data_layer_fingerprint` hashes
  that source so the daemon can refuse to keep a stale bundle.

* **NOT reloadable — but loud** (no guard needed): an *interface* change to the
  resident-source / ``Batch`` classes (rename a method, change a signature).
  Reloaded training code calling the new shape against the kept old instance
  raises immediately — rebuild (``reset``) and retry.

Residual caveat: other *preload-baking* logic lives inside
``qnn.bc.supervised_loop`` alongside the per-step training code (resident-source
materialization, engagement-EMA, dequant-input dropping). A semantic edit to
*those specific regions* — without an interface change — is the one case this
guard does not catch; ``reset`` after such an edit. The fingerprint is
deliberately scoped to ``dequant.py`` so that the frequent edits (train step,
metrics) that share ``supervised_loop.py`` do not force a needless rebuild.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from typing import Iterable

# Modules that must survive a reload: the running daemon, its socket client,
# and this module itself. Purging any of these would pull the rug from under
# the code performing the reload.
_KEEP_MODULES = frozenset(
    {
        "qnn.bc.ablation_daemon",
        "qnn.bc.ablation_client",
        "qnn.bc.code_reload",
    }
)

# Sources whose edits bake into the resident VRAM tensors at preload and are
# then skipped per step — i.e. silent staleness if reloaded against a kept
# bundle. See module docstring for why this is scoped to dequant only.
_DATA_LAYER_MODULES = ("qnn.model.dequant",)


def _module_source_path(module_name: str) -> str | None:
    """Resolve a module's source file without importing it."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    return spec.origin if spec is not None else None


def data_layer_fingerprint() -> str:
    """Hash the data-baking sources (see ``_DATA_LAYER_MODULES``).

    Captured when the corpus is built and re-checked on reload: a changed
    fingerprint means the resident tensors no longer reflect the source, so the
    daemon must drop the bundle rather than serve stale data.
    """
    h = hashlib.sha256()
    for name in _DATA_LAYER_MODULES:
        path = _module_source_path(name)
        h.update(name.encode("utf-8"))
        if path is None:
            h.update(b"\x00<unresolved>")
            continue
        try:
            with open(path, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"\x00<unreadable>")
    return h.hexdigest()


def _purge_targets() -> list[str]:
    return [
        name
        for name in list(sys.modules)
        if (name == "qnn" or name.startswith("qnn."))
        and name not in _KEEP_MODULES
    ]


def reload_bc_code() -> dict[str, object]:
    """Purge & re-import the ``qnn.*`` application graph in-process.

    Drops every ``qnn`` / ``qnn.*`` module from ``sys.modules`` except the
    daemon/client/reload modules, then re-imports the parent packages so the
    next ``from qnn.… import …`` in the daemon (and the per-job lazy imports in
    ``_build_job``) rebinds against fresh source. Returns a small report.

    The caller is responsible for (a) only invoking this while idle — no job
    thread may be importing or executing reloaded modules concurrently — and
    (b) re-binding any names it imported at module load (see the daemon's
    ``_import_app_symbols``); module-global bindings captured by a prior
    ``from x import name`` still point at the *old* objects until rebound.
    """
    purged = _purge_targets()
    for name in purged:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    # Re-establish the package roots so subsequent imports resolve cleanly.
    importlib.import_module("qnn")
    importlib.import_module("qnn.bc")
    return {"purged_count": len(purged), "purged": sorted(purged)}


def changed_data_layer(previous_fingerprint: str | None) -> bool:
    """True if the data-baking sources differ from ``previous_fingerprint``."""
    if previous_fingerprint is None:
        return False
    return data_layer_fingerprint() != previous_fingerprint


def keep_modules() -> Iterable[str]:
    """The reload-surviving module set (exposed for diagnostics/tests)."""
    return sorted(_KEEP_MODULES)
