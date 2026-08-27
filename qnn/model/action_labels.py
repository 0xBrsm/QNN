"""Action-label contract registry — what a head's target MEANS, versioned.

The observation side already solved this problem: a model declares the
entity stream it wants *at a pinned policy version* (`entities:
{policy: "v3"}`, qnn/obs_api.py), policy versions pin semantics rather
than byte layout, and a served policy is never retired.  This module is
the same idea for the *action* side.

── Why this exists ────────────────────────────────────────────────────
The 9-class attack-with selector is ONE architecture (AttackWithHead)
that has been trained against three different label semantics across
generations.  Until this registry, which semantics you got was implied
by the graph *slot name* you happened to write:

    heads: {"attack": {"type": "attack_with"}}   → target from act_attack
    heads: {"weapon": {"type": "attack_with"}}   → target from act_weapon

Same module, same weights, same hyperparameters — different label
column, selected by a name, validated by nothing.  A27 retired
`act_weapon` (invariant 7: no equipped weapon or carried weapon intent),
so an a27 probe that named the `weapon` slot silently trained on an
all-zero column: every positive frame masked to ignore_index, the head
learned class 0 only, and the failure surfaced as a plausible-looking
`acc_attack: 0.0` rather than an error.  See
runs/bc/bench/purecombat_w132s2_seed43 (`n_weapon_valid: 0.0`).

The fix is to make the binding explicit and reviewable: a selector head
DECLARES its label contract by name, the contract names the action
columns it reads, and an unknown or unsatisfiable contract is a hard
error at spec-compile time.  The slot name goes back to meaning only
"which logits tensor" — never "which label".

── Contract naming ────────────────────────────────────────────────────
`<family>.v<N>`.  The family is the label's subject (`attack`, `weapon`);
the version pins semantics.  ANY semantic change is a NEW version — the
old one keeps serving, because checkpoints on disk were trained against
it and cross-line eval must keep loading them.  Never edit a contract in
place; add a row.

Derive functions are NOT held here.  This is a leaf module (it imports
nothing from qnn at runtime) so the graph/spec layer can validate a
declaration without dragging in torch or the head modules.  Each head
module registers its own derive beside the code that owns it, via
:func:`register_label_derive` — the same decentralized-registry idiom as
qnn.model.node_registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Derive signature: (actions, device, valid_flat, **kw) -> (N,) long target
# with _IGNORE (-100) on frames the contract declines to score.
LabelDerive = Callable[..., Any]


@dataclass(frozen=True)
class LabelContract:
    """One pinned action-label semantics — a registry row.

    ``columns`` is the set of action streams the derive reads.  It is the
    contract's real precondition: a corpus that does not carry them (or
    carries them dead) cannot satisfy this contract, and the epoch-1
    label guard checks exactly this.
    """

    name: str
    family: str
    #: Prose description, quoted verbatim in fail-loud messages.
    label: str
    #: Number of categorical classes; 0 = not categorical (binary/regression).
    classes: int
    #: Action columns the derive reads, in no particular order.
    columns: tuple[str, ...]
    #: True for the categorical weapon/attack selector family — the heads
    #: that read a readout/token edge rather than the motor edge set.
    selector: bool
    #: Generations that trained against this contract, for error messages.
    generations: tuple[str, ...]


# ── The registry ────────────────────────────────────────────────────
#
# Rows are append-only.  A NEW semantics is a NEW row, never an edit:
# checkpoints on disk name the contract they were trained against, and
# cross-line eval (qnn.eval.h2h) loads a25/a26 checkpoints against a27
# code.  Retiring a row breaks those loads silently.

_CONTRACTS: dict[str, LabelContract] = {}


def _row(contract: LabelContract) -> LabelContract:
    if contract.name in _CONTRACTS:
        raise RuntimeError(f"duplicate label contract {contract.name!r}")
    _CONTRACTS[contract.name] = contract
    return contract


#: A27 (current).  The sole weapon decision is the 9-class attack-with
#: head and the label IS the action column: 0 = no attack, 1..8 = the
#: impulse class QC W_Attack actually advanced on a discharge frame.
#: No carried intent, no equipped-state repair
#: (agents/plans/a27-pure-combat-substrate.md invariant 7).
ATTACK_V1 = _row(LabelContract(
    name="attack.v1", family="attack",
    label="A27 discharge-only attack-with (act_attack is the class)",
    classes=9, columns=("attack",), selector=True,
    generations=("a27",),
))

#: A25/A26.  The class comes from `act_weapon` — de-scripted deliberate
#: SELECT-intent, carried forward across frames — gated to frames where
#: `act_attack > 0`.  Frames whose intent weapon cannot fire are
#: self-healed to the held weapon when the arsenal obs is available.
#: Wire-shim only for new work; a25/a26 checkpoints require it.
WEAPON_V2 = _row(LabelContract(
    name="weapon.v2", family="weapon",
    label="A25/A26 carried select-intent attack-with (act_weapon gated by act_attack)",
    classes=9, columns=("weapon", "attack"), selector=True,
    generations=("a25", "a26"),
))

#: Pre-A25.  The 8-class HELD-weapon head: `act_weapon` is the raw engine
#: weapon byte every frame (1..8 → class 0..7); no-weapon frames are
#: dropped.  Not a discharge label at all — it predicts what is equipped.
WEAPON_V1 = _row(LabelContract(
    name="weapon.v1", family="weapon",
    label="pre-A25 held-weapon 8-class (act_weapon is the equipped byte)",
    classes=8, columns=("weapon",), selector=True,
    generations=("a22", "a23", "a24"),
))

#: The binary attack head — `act_attack` read as a float BCE target with
#: the distance-shoulder weighting.  Registered so a graph can name it
#: and so the epoch-1 guard knows its column; its derive still lives
#: inline in QNNPolicy (it carries the shoulder/precompute machinery and
#: is not a categorical selector).
ATTACK_V0 = _row(LabelContract(
    name="attack.v0", family="attack",
    label="binary attack (act_attack as a BCE target with distance shoulder)",
    classes=0, columns=("attack",), selector=False,
    generations=("a22", "a23", "a24", "a25", "a26"),
))


# ── Derive registration ─────────────────────────────────────────────

_DERIVES: dict[str, LabelDerive] = {}


def register_label_derive(name: str):
    """Register the target-derivation for contract ``name``.

    Applied beside the function that owns the semantics, so the contract
    row and its implementation cannot drift apart silently.
    """

    def bind(fn: LabelDerive) -> LabelDerive:
        if name not in _CONTRACTS:
            raise RuntimeError(
                f"cannot register a derive for unknown label contract {name!r}; "
                f"add its row to qnn.model.action_labels first "
                f"(known: {sorted(_CONTRACTS)})"
            )
        if name in _DERIVES and _DERIVES[name] is not fn:
            raise RuntimeError(f"duplicate derive for label contract {name!r}")
        _DERIVES[name] = fn
        return fn

    return bind


# ── Lookup ──────────────────────────────────────────────────────────


def contract(name: str) -> LabelContract:
    """Return the contract row, or raise naming the known rows.

    Fail loud: an unknown contract is a corrupt or hand-edited graph
    spec, never something to default.
    """
    try:
        return _CONTRACTS[name]
    except KeyError:
        raise KeyError(
            f"unknown action-label contract {name!r}; known contracts: "
            f"{sorted(_CONTRACTS)}"
        ) from None


def derive_for(name: str) -> LabelDerive:
    """Return the registered derive for ``name``.

    A contract with no derive means its owning head module was never
    imported — a build/import-order bug, not a config error.
    """
    contract(name)  # validate the name first, for the better message
    try:
        return _DERIVES[name]
    except KeyError:
        raise RuntimeError(
            f"label contract {name!r} has no registered derive; its head "
            f"module was not imported (registered: {sorted(_DERIVES)})"
        ) from None


def contracts_for_family(family: str) -> tuple[str, ...]:
    return tuple(sorted(n for n, c in _CONTRACTS.items() if c.family == family))


def selector_contracts() -> tuple[str, ...]:
    return tuple(sorted(n for n, c in _CONTRACTS.items() if c.selector))


def all_contracts() -> tuple[str, ...]:
    return tuple(sorted(_CONTRACTS))
