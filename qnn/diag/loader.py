"""Canonical checkpoint loader for diagnostic / offline-fit scripts.

Single public entry point::

    from qnn.diag.loader import load_policy
    policy, probe = load_policy(run_dir)

This was promoted from ``qnn.bc.decode_fit._load_policy`` (which now
delegates here) so that every analysis script, bench probe, and decode-fit
sweep shares ONE load path — no drift between them.

What it does:
 * finds the best checkpoint (``best_*.pth`` → ``bc_best_model.pth`` fallback)
 * applies the pre-``attack_finished`` weapon-token 7→8 padding compat shim
 * loads via ``QNNPolicy.load`` with the probe's ``model_factory``
 * sets ``policy.model.eval()`` and ``policy.input_mask = True``
 * installs the run's polar look-grid from ``config/look_grid.json`` (if present)

Returns (policy, probe) where ``probe`` is the raw probe.json dict.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from qnn.model import look_bins as _look_bins
from qnn.model.policy import QNNPolicy


def resolve_decode_module(run_dir: Path, policy: QNNPolicy | None = None):
    """EXPLICITLY resolve (and optionally inject) a run's generation decode facade.

    ``QNNPolicy._decode()`` has NO default decode module — the module is explicit
    state, normally injected from a resolved decode config. Bare-policy entry
    points (analysis scripts that ``load_policy`` a run dir and then ``act()``)
    call this helper instead: it reads the run's OWN ``config/probe.json`` head
    set and resolves the arch from it —

      * ``move_seg`` head and/or ``attack.type == "attack_with"``
        → ``qnn.model.decode_actions``

    and RAISES when the arch cannot be determined (a24 is retired; there is no
    fallback). When ``policy`` is given the module is injected into
    ``policy._decode_mod``. Returns the resolved module.
    """
    import importlib

    run_dir = Path(run_dir)
    probe_path = run_dir / "config" / "probe.json"
    heads: dict = {}
    if probe_path.exists():
        try:
            probe = json.loads(probe_path.read_text())
            heads = (probe.get("overrides") or {}).get("heads") or {}
        except (ValueError, OSError):
            heads = {}
    is_a25 = bool(heads.get("move_seg")) or (
        (heads.get("attack") or {}).get("type") == "attack_with")
    if not is_a25:
        raise RuntimeError(
            f"{run_dir}: cannot determine the run's decode arch from "
            f"config/probe.json (no move_seg head and no attack_with attack head). "
            "a24 is retired — only a25-arch runs resolve here. Pass the run's "
            "resolved decode config instead (resolve_decode_config → "
            "resolved.decode_module → policy._decode_mod)."
        )
    mod = importlib.import_module("qnn.model.decode_actions")
    if policy is not None:
        policy._decode_mod = mod
    return mod


def load_policy(
    run_dir: Path,
    device: str | None = None,
) -> tuple[QNNPolicy, dict]:
    """Load a trained policy from a bench run directory.

    Parameters
    ----------
    run_dir:
        Path to the run directory containing ``config/probe.json`` and
        ``checkpoints/best_*.pth`` (or ``checkpoints/bc_best_model.pth``).
    device:
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).  When ``None``
        (default) the device is chosen automatically: ``"cuda"`` if a GPU is
        available, otherwise ``"cpu"``.

    Returns
    -------
    (policy, probe)
        ``policy`` is a fully-loaded :class:`~qnn.model.policy.QNNPolicy`
        in eval mode with ``input_mask=True`` and the run's polar look-grid
        installed.  ``probe`` is the raw ``probe.json`` dict.
    """
    run_dir = Path(run_dir)
    probe = json.loads((run_dir / "config" / "probe.json").read_text())
    # Probes are self-describing: the merged GraphSpec is persisted in checkpoint
    # meta ("model_graph") and QNNPolicy.load rebuilds from it. No model_factory.
    factory = None

    cks = (sorted((run_dir / "checkpoints").glob("best_*.pth"))
           or sorted((run_dir / "checkpoints").glob("bc_best_model.pth")))
    if not cks:
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir / 'checkpoints'} "
            f"(tried best_*.pth and bc_best_model.pth)"
        )

    payload = torch.load(cks[0], map_location="cpu", weights_only=False)

    # pre-attack_finished compat: weapon token was 7-wide, now 8-wide
    meta = payload.get("meta", {})
    if "model" in meta and "ffn_dim" in meta["model"]:
        meta["model"]["d_ffn"] = meta["model"].pop("ffn_dim")
    sd = payload.get("state_dict", {})
    for k, w in list(sd.items()):
        if k.endswith("weapon_builder.projs.0.weight") and w.ndim == 2 and w.shape[1] == 7:
            pad = torch.zeros(w.shape[0], 8, dtype=w.dtype)
            pad[:, :7] = w
            sd[k] = pad

    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    torch.save(payload, tmp.name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = QNNPolicy.load(tmp.name, device=device, model_factory=factory)
    policy.model.eval()
    policy.input_mask = True

    lg = run_dir / "config" / "look_grid.json"
    if lg.exists():
        g = json.loads(lg.read_text())
        _look_bins.install_polar_grid(g["mag_centers_rad"], g.get("dir_centers_rad"))

    return policy, probe
