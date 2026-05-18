"""Diagnostic tools for trained QNN policies.

Answers "where is capacity bound, where is it slack, are 8-epoch comparisons
reliable" — not mechanistic interpretability.

Submodules import lazily; ``import qnn.diag`` itself has no torch dependency.
For tools that need a model: ``from qnn.diag import ablation, gradients, ...``
For history-only / data-analysis tools: ``from qnn.diag import history, convergence``.

Public submodules:
    qnn.diag.history       — per-head loss curves from bc_history.json (no torch)
    qnn.diag.convergence   — slope / asymptote / "is comparison reliable?" (no torch)
    qnn.diag.rank          — SVD effective rank of Linear weights
    qnn.diag.ablation      — submodule zero-out, layer ablation table
    qnn.diag.gradients     — per-param grad norms, dead-param detection
    qnn.diag.participation — PR + dead-unit + fire-rate stats on head bottlenecks
    qnn.diag.attention     — per-attention-head entropy, similarity, ablation
    qnn.diag.pruning       — per-neuron pruning sensitivity (slow, opt-in)
    qnn.diag.linear_probe  — frozen-trunk → linear classifier (slow, opt-in)
    qnn.diag.report        — aggregator → markdown + JSON
    qnn.diag.cli           — entry point: ``python -m qnn.diag --checkpoint …``
"""
