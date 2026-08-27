"""Emit — staged decode config, gate-guarded promotion, sidecar, fit report.

The v2 emit contract (plan §P3): **no emit on gate FAIL**. The config is
STAGED first (``decode.<version>.staged.json``), validated (it must resolve
through ``qnn.model.decode_config.resolve_decode_config`` and clear the
export-gap registry), and only PROMOTED to ``decode.<version>.json`` when the
gates passed — or under an explicit ``force``, which stamps the promoted
config's provenance loudly (``forced: true``). This kills the v1
emit-despite-FAIL class where the config was written before the gate ran and
FAIL only labeled the report.

Param assembly is ported from v1 ``emit_vector_decode_config``
(qnn.eval.decode_fit_pipeline.py:2300); the emit base is the a25-native
template ``decode.base.json`` (same role as the v1 ORACLE_TEMPLATE).
"""
from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qnn.decode_fit.context import (CALIBRATION_FAMILIES, WEAPON_IMPULSE,
                                    read_json, rel_to_repo)
from qnn.decode_fit.response import WeaponPlan

_REPO = Path(__file__).resolve().parents[3]
# a25-native BASE template — the emit base (v1 ORACLE_TEMPLATE, pipeline:72).
TEMPLATE = _REPO / "src" / "qnn" / "model" / "bench" / "templates" / "decode.base.json"

# Decode knobs the eval path honors but the deploy (ONNX) export path does NOT
# thread into ExportWrapper. A non-noop value for any of these means the
# emitted config is eval-faithful but NOT deploy-faithful — staging flags it
# so we never silently ship an un-deployable config. Everything the fitter
# emits IS threaded (per-weapon gain/α vectors, universal tremor via the OU
# state loopback, turn_mag_scale, attack.*), so this registry is EMPTY — the
# mechanism stays (port of v1 NON_EXPORTED_KNOBS, decode_fit_pipeline.py:103;
# see that comment block for the history of retired/threaded knobs).
NON_EXPORTED_KNOBS: dict[str, float] = {}

_STAGED_SUFFIX = ".staged.json"

# rc / bare-tier names are EARNED, never pre-assigned (model-versioning.md):
# the rc numeral by a passed decode fit, the letter by a deploy slot, the
# bare tier by line close. Fits stage under PROVISIONAL versions; the rc
# name is applied to an already-promoted config by ``assign_rc``.
RESERVED_VERSION_RE = re.compile(r"a\d+(rc\d+[a-z]*)?")
# an assignable deploy-slot name: tier + rc numeral + single deploy letter
# (a26rc2a). Bare rcN / bare tier are aliases earned by promotion on the pi
# share, not emitted config filenames.
RC_NAME_RE = re.compile(r"a\d+rc\d+[a-z]")
PROVISIONAL_PREFIX = "prov-"


def provisional_version(model_id: str, skill: str) -> str:
    """The version a fit stages under before any rc name exists: derived from
    the run id (the only name a checkpoint has pre-pass) + the skill spec."""
    slug = str(skill).replace("=", "").replace(",", "_")
    return f"{PROVISIONAL_PREFIX}{model_id}-{slug}"


def _log(msg: str) -> None:
    print(f"[decode-fit] {msg}", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def detect_export_gaps(params: dict[str, Any]) -> list[str]:
    """Return the active decode params the ONNX export path does NOT thread.
    A non-empty list means the emitted config is eval-faithful but NOT yet
    deploy-faithful. Port of v1 decode_fit_pipeline.py:908 — reads the
    module-level ``NON_EXPORTED_KNOBS`` registry (currently empty)."""
    gaps = []
    for k, noop in NON_EXPORTED_KNOBS.items():
        v = params.get(k)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            if any(float(x) != noop for x in v):
                gaps.append(k)
        elif float(v) != noop:
            gaps.append(k)
    return gaps


def _placed_table(plans: dict[str, WeaponPlan]) -> dict[str, dict[str, Any]]:
    """The skill_vector ``placed`` table: every plan's full placement,
    INCLUDING the refusal/frontier flags and the achieved percentile."""
    return {
        w: {
            "impulse": p.impulse,
            "target_pct": p.target_pct,
            "target_hbw": p.target_hbw,
            "gain": p.gain,
            "alpha": p.alpha,
            "tremor": p.tremor,
            "pred_hbw": p.pred_hbw,
            "pred_hbw_ci": list(p.pred_hbw_ci),
            "achieved_pct": p.achieved_pct,
            "band": p.band,
            "refused": p.refused,
            "frontier_pct": p.frontier_pct,
            "frontier_hbw": p.frontier_hbw,
            "alias_of": p.alias_of,
            "notes": p.notes,
        }
        for w, p in sorted(plans.items())
    }


def stage_decode_config(ctx, plans: dict[str, WeaponPlan],
                        vectors: dict[str, Any], pins: dict[str, Any],
                        tms: float, version: str, *,
                        template: Path | None = None) -> Path:
    """Build + STAGE the decode config (``decode.<version>.staged.json`` under
    ``ctx.out_dir``) — promotion is a separate, gate-guarded step.

    Param assembly ports v1 ``emit_vector_decode_config``
    (decode_fit_pipeline.py:2300): the a25 base template, the (9,) per-impulse
    gain/α vectors, the universal scalar tremor (+ tau), the fitted
    ``look.turn_mag_scale`` substrate (never the template pin — the a24
    0.7 / a25 0.93 cross-arch class), the non-lever ``pins`` merged in
    (the live-pins attack.fire_bias_vec, zero legacy joint vector,
    move.commit_dur_tilt), and the
    ``skill_vector`` block carrying the
    plans' placed table including refusal/frontier flags.

    Validation (v1 ``_validate_decode_config``, pipeline:934): the staged file
    must pass ``resolve_decode_config`` — an unresolvable config is DELETED
    and raises (never stage garbage). ``detect_export_gaps`` (pipeline:908)
    is recorded in the staged provenance and warned loudly."""
    tpl = Path(template) if template is not None else TEMPLATE
    cfg = read_json(tpl)
    cfg["version"] = str(version)

    tremor = vectors.get("tremor", 0.0)
    if not isinstance(tremor, list):
        tremor = [float(tremor)] * 9 if float(tremor) else [0.0] * 9
    tgt_str = ", ".join(
        f"{w}=p{p.target_pct:g}" + (f"→frontier p{p.frontier_pct:g}"
                                    if p.refused else "")
        for w, p in sorted(plans.items()))
    _tr_str = ("0" if not any(tremor)
               else "[" + ",".join(f"{t:g}" for t in tremor) + "]")
    cfg["description"] = (
        f"decode-fit v2 on {ctx.model_id}: per-weapon [{tgt_str}]; per-impulse "
        f"gain/α/tremor vectors (tremor={_tr_str}); "
        f"turn_mag_scale={float(tms)} dampener (fitted substrate); "
        f"corpus_fp={str(ctx.corpus_fingerprint)[:16]} "
        f"look_grid_sha={str(ctx.look_grid_sha)[:12]}")
    cfg["look_grid"] = "config/look_grid.json"

    p = cfg.setdefault("params", {})
    # non-lever pins (invariants-before-skills: the live-pins fire-only vector,
    # move.commit_dur_tilt) merge first so the fitted levers below always win
    # a key collision.
    for k, v in (pins or {}).items():
        p[k] = v
    # the (9,) per-impulse gain/α/tremor vectors — the deliverable
    # (v1:2336-2348). Per-weapon lag was retired 7/10: never emitted.
    p["look.aim_prior_gain"] = vectors["gain"]
    p["look.aim_mag_gain"] = vectors["alpha"]
    p["look.aim_degrade_tremor_mag"] = tremor
    if vectors.get("tremor_tau") is not None:
        p["look.aim_degrade_tremor_tau"] = float(vectors["tremor_tau"])
    # the dampener the gains were fit ON — the per-model substrate, NOT the
    # template pin (the cross-arch carryover bug class; v1:2341-2346).
    p["look.turn_mag_scale"] = float(tms)

    cfg["skill_vector"] = {
        "targets": {w: pl.target_pct for w, pl in sorted(plans.items())},
        "placed": _placed_table(plans),
        "vectors": {"gain": vectors["gain"], "alpha": vectors["alpha"]},
        "tremor": {"mag": tremor,
                   "tau": p.get("look.aim_degrade_tremor_tau"),
                   "scope": "universal"},
    }

    gaps = detect_export_gaps(p)
    cfg["provenance"] = {
        **ctx.provenance(),
        "emitted_utc": _now(),
        "template": rel_to_repo(tpl),
        "export_gaps": gaps,
        "calibration_families": ["SG+SSG", "NG+SNG", "RL", "LG"],
        "staged": True,
    }

    out = Path(ctx.out_dir) / f"decode.{version}{_STAGED_SUFFIX}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")

    # v1 _validate_decode_config (pipeline:934): the staged config must resolve
    # (module imports + the fail-loud required-param manifest). Never stage an
    # unresolvable config — delete it and raise.
    from qnn.model.decode_config import resolve_decode_config
    try:
        resolve_decode_config(out)
    except Exception as e:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"staged decode config failed resolve_decode_config and was "
            f"removed ({out.name}): {e!r}") from e
    if gaps:
        _log(f"WARNING export gap — active params NOT threaded into ONNX "
             f"export: {gaps} (recorded in staged provenance)")
    _log(f"staged decode config → {rel_to_repo(out)} (version={version})")
    return out


_STYLE_KEYS = ("weapon.preference_bias_vec", "move.threat_break_hazard")


def warm_start_style(ctx, staged: Path) -> Path | None:
    """Seed the staged config's style knobs from the LATEST promoted fit of
    the same model (verify-first trim, Brian 2026-07-17): successive fits of
    one checkpoint converge to near-identical selection/switch/reactivity
    trims, so starting at the previous values makes iteration 0 a verification —
    on pass the whole trim tail is ONE eval (its npz doubles as the final
    gate eval via the existing convergence dedup). Cold model (no promoted
    fit) → None, substrate pins stand. The forced-pin fire vector is never
    warmed. The warm source is stamped into the staged provenance."""
    staged = Path(staged)
    prev = [p for p in ctx.out_dir.glob("decode.*.json")
            if not p.name.endswith(_STAGED_SUFFIX) and p != staged
            and "provenance" in read_json(p)]
    if not prev:
        return None
    src = max(prev, key=lambda p: p.stat().st_mtime)
    src_cfg = read_json(src)
    cfg = read_json(staged)
    warmed = {k: src_cfg["params"][k] for k in _STYLE_KEYS
              if k in src_cfg.get("params", {})}
    if not warmed:
        return None
    fire = warmed.get("attack.fire_bias_vec")
    if isinstance(fire, list) and len(fire) == 8:
        fire = [float(v) for v in fire]
        current = list((cfg.get("params") or {}).get(
            "attack.fire_bias_vec") or [0.0] * 8)
        source_is_family_fit = bool(
            (src_cfg.get("provenance") or {}).get("calibration_families"))
        for source, members in CALIBRATION_FAMILIES.items():
            if len(members) == 1:
                continue
            idx = WEAPON_IMPULSE[source] - 1
            # A legacy warm start may only have lifted the instrument member;
            # never let it pull the new live-pin floor downward.
            value = fire[idx] if source_is_family_fit else max(
                fire[idx], float(current[idx]))
            for member in members:
                fire[WEAPON_IMPULSE[member] - 1] = value
        warmed["attack.fire_bias_vec"] = fire
    cfg["params"].update(warmed)
    cfg.setdefault("provenance", {})["trim_warm_start"] = {
        "source": rel_to_repo(src), "keys": sorted(warmed)}
    staged.write_text(json.dumps(cfg, indent=2) + "\n")
    _log(f"trim warm-start ← {rel_to_repo(src)} ({sorted(warmed)})")
    return src


def promote_decode_config(staged: Path, *, gate_passed: bool,
                          force: bool = False,
                          forced_note: str | None = None,
                          style_flags: list[str] | None = None) -> Path:
    """Promote ``decode.<version>.staged.json`` → ``decode.<version>.json``
    (same dir) ONLY when the gates passed — or under ``force``, which stamps
    ``{"forced": true, "forced_note": …}`` into the promoted provenance block
    (loud, per plan: --force exists for debugging). Otherwise raise
    RuntimeError: **no emit on gate FAIL**. ``style_flags`` (channels whose
    fitted-vs-native style spend exceeded margin) are stamped into the
    promoted provenance — the artifact carries its own training-target
    register (flags never block promotion; Brian 2026-07-16)."""
    staged = Path(staged)
    if not staged.name.endswith(_STAGED_SUFFIX):
        raise ValueError(f"not a staged decode config: {staged} "
                         f"(expected *{_STAGED_SUFFIX})")
    if not (gate_passed or force):
        raise RuntimeError(
            f"refusing to promote {staged.name}: gate_passed=False and no "
            "force — no emit on gate FAIL (plan §P3). Fix the fit / re-run "
            "the gates, or promote with force=True (debugging only; stamped "
            "loudly in provenance).")
    promoted = staged.with_name(
        staged.name[: -len(_STAGED_SUFFIX)] + ".json")
    cfg = read_json(staged)
    prov = cfg.setdefault("provenance", {})
    prov["staged"] = False
    prov["promoted_utc"] = _now()
    prov["gate_passed"] = bool(gate_passed)
    if style_flags:
        prov["style_flags"] = sorted(style_flags)
        prov["style_flags_note"] = (
            "fitted-vs-native band-v5 style spend over margin on these "
            "channels — flagged training targets, not gate failures "
            "(skill is never capped for style)")
    if force:
        prov["forced"] = True
        prov["forced_note"] = forced_note or (
            f"promoted with force=True (gate_passed={bool(gate_passed)}) — "
            "debugging only; NOT gate-blessed")
    promoted.write_text(json.dumps(cfg, indent=2) + "\n")
    _log(f"promoted decode config → {rel_to_repo(promoted)}"
         + (" [FORCED]" if force else ""))
    return promoted


def assign_rc(source: Path, rc: str, *, replace: bool = False,
              force: bool = False) -> Path:
    """Mark a PROMOTED provisional config with its earned rc name: copy
    ``decode.<provisional>.json`` → ``decode.<rc>.json`` (same dir) with the
    ``version`` field re-stamped and the assignment recorded in provenance
    (``rc_source`` / ``provisional_version`` / ``rc_assigned_utc``). The
    source file is stamped ``rc_assigned: <rc>`` so its promotion is visible
    in place.

    Guards (model-versioning.md — rc names are earned, never pre-assigned):

    * source must be PROMOTED (never a ``.staged.json``) with
      ``gate_passed: true`` and not force-promoted — no rc name on a fit
      that did not pass, unless ``force`` (stamped loudly);
    * ``rc`` must be a full deploy-slot name (``a26rc2a``) — bare rcN and
      bare tier are share-side promotion aliases, not emitted configs;
    * an existing target is only overwritten under ``replace`` — the
      re-emit-same-letter path for a superseded NEVER-shipped artifact.
      Replacing a deployed config's file is caller error; check the pi
      share before passing ``replace``."""
    source = Path(source)
    if not RC_NAME_RE.fullmatch(rc):
        raise ValueError(
            f"not an assignable rc name: {rc!r} (expected tier+rc+deploy "
            f"letter, e.g. a26rc2a — bare rcN / bare tier are promotion "
            f"aliases, never emitted filenames)")
    if source.name.endswith(_STAGED_SUFFIX):
        raise ValueError(
            f"{source.name} is STAGED — rc names are assigned to PROMOTED "
            f"configs only (gate pass promotes; then assign)")
    cfg = read_json(source)
    prov = cfg.get("provenance") or {}
    if prov.get("staged", True) or "promoted_utc" not in prov:
        raise RuntimeError(
            f"refusing rc assignment: {source.name} has no promotion "
            f"provenance — only a gate-promoted config earns an rc name")
    if not prov.get("gate_passed") and not force:
        raise RuntimeError(
            f"refusing rc assignment: {source.name} has gate_passed="
            f"{prov.get('gate_passed')!r} — the rc numeral is earned by a "
            f"PASSED fit (force=True for debugging only, stamped loudly)")
    if prov.get("forced") and not force:
        raise RuntimeError(
            f"refusing rc assignment: {source.name} was force-promoted "
            f"(not gate-blessed); re-run the fit or force the assignment")
    if prov.get("rc_assigned") and not force:
        raise RuntimeError(
            f"refusing rc assignment: {source.name} already assigned "
            f"{prov['rc_assigned']!r} — one fit, one rc name (a refit "
            f"promotes its own provisional config first)")
    target = source.with_name(f"decode.{rc}.json")
    if target == source:
        raise ValueError(f"source already carries the rc name: {source}")
    replaced = target.exists()
    if replaced and not replace:
        raise RuntimeError(
            f"refusing rc assignment: {rel_to_repo(target)} exists. If it "
            f"was NEVER deployed this is the re-emit-same-letter path — "
            f"verify against the pi share, then pass replace=True. If it "
            f"WAS deployed, the next letter is the correct name.")
    out = dict(cfg)
    out["version"] = rc
    out["provenance"] = {
        **prov,
        "rc_assigned_utc": _now(),
        "rc_source": rel_to_repo(source),
        "provisional_version": cfg.get("version"),
        **({"replaced_unshipped": True} if replaced else {}),
        **({"rc_forced": True} if force else {}),
    }
    # ``rc_assigned`` marks a SOURCE as having been promoted (stamped below);
    # it is never correct on the rc artifact itself, which IS the rc. Drop any
    # value inherited through the provenance dict — a hand-tuned probe chain
    # copies its ancestor's block wholesale, so a config descended from an
    # already-assigned config carries a stale letter. Caught 2026-08-21: the
    # a28rc1g candidate arrived claiming rc_assigned="a28rc1f" (inherited from
    # decode.prov-…-SGp100_NGp90_RLp90_LGp90.json via the lockout probe chain),
    # which would have shipped an artifact naming itself the PREVIOUS deploy.
    out["provenance"].pop("rc_assigned", None)
    target.write_text(json.dumps(out, indent=2) + "\n")
    cfg.setdefault("provenance", {})["rc_assigned"] = rc
    source.write_text(json.dumps(cfg, indent=2) + "\n")
    _log(f"assigned rc name {rc} → {rel_to_repo(target)}"
         + (" [REPLACED unshipped artifact]" if replaced else "")
         + (" [FORCED]" if force else ""))
    return target


def write_sidecar(ctx, doc: dict[str, Any], out_path: Path) -> Path:
    """Write the skill-curve sidecar (schema ``skill_curve_v3``). The content
    is assembled by the CLI — this only wraps schema + provenance and writes.
    Port of v1 write_skill_curve_sidecar (decode_fit_pipeline.py:926)."""
    out_path = Path(out_path)
    out = dict(doc)
    out["schema"] = "skill_curve_v3"
    out.setdefault("provenance", ctx.provenance())
    out.setdefault("generated_utc", _now())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    _log(f"wrote skill-curve sidecar → {rel_to_repo(out_path)}")
    return out_path


def fit_report(ctx, *, fits: dict[str, Any],
               plans: dict[str, WeaponPlan] | dict[str, dict],
               budget: dict[str, Any], gates: dict[str, Any],
               emit: dict[str, Any] | None = None,
               notes: Any = None) -> dict[str, Any]:
    """Assemble the final fit-report skeleton — pure dict assembly, the CLI
    passes the parts (fits summary, resolved plans, budget ledger, gate
    reports, emit info). The result label folds the gate statuses: only a
    complete set of PASS statuses is PASS; FAIL, missing, or skipped evidence
    is FAIL. There is intentionally no ``PASS-PENDING-VALIDATION`` state: that
    a27 escape hatch emitted rc1a without ever measuring combat behavior."""
    statuses = [g.get("status") for g in (gates or {}).values()
                if isinstance(g, dict)]
    if statuses and all(s == "PASS" for s in statuses):
        label = "PASS"
    else:
        label = "FAIL"
    plans_out = {
        w: (dataclasses.asdict(p) if dataclasses.is_dataclass(p) else p)
        for w, p in (plans or {}).items()
    }
    return {
        "model_id": ctx.model_id,
        "run_dir": str(getattr(ctx, "run_dir", "")),
        "generated_utc": _now(),
        "fits": fits,
        "plans": plans_out,
        "budget": budget,
        "gates": gates,
        "emit": emit,
        "notes": notes,
        "provenance": ctx.provenance(),
        "result": label,
    }
