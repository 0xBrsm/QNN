"""Aggregate the diagnostic suite into a single markdown + JSON report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from qnn.diag import (
    ablation,
    attention,
    convergence,
    data,
    gradients,
    history,
    linear_probe,
    participation,
    pruning,
    rank,
)


def run_report(
    *,
    checkpoint: Path,
    data_dir: Path,
    run_dir: Path | None = None,
    max_val_shards: int = 1,
    max_val_episodes: int | None = 8,
    skip: set[str] | None = None,
    include: set[str] | None = None,
) -> dict[str, Any]:
    """Run the full diagnostic suite and return a structured dict.

    ``skip`` is a set of section names to omit; supported:
    ``{"history", "rank", "ablation", "gradients",
       "participation", "pruning", "attention", "linear_probe"}``.

    Slow tools (``pruning``, ``linear_probe``) are skipped by default unless
    explicitly enabled via the ``include`` arg of ``run_report``.
    """
    from qnn.model.policy import QNNPolicy

    skip = skip or set()
    include = include or set()
    # Slow tools — skip unless explicitly included.
    SLOW = {"pruning", "linear_probe"}
    skip |= (SLOW - include)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
    }

    # ── Loss history (free, no model needed) ──────────────────────────────
    if run_dir is not None and "history" not in skip:
        hist = history.load_history(run_dir)
        if hist:
            report["history"] = {
                "n_epochs": len(hist),
                "best_epoch": history.best_epoch(hist),
                "still_improving": history.still_improving(hist),
                "train_val_gap": history.train_val_gap_progression(hist),
                "per_head_curves": history.per_head_curves(hist),
            }
            report["convergence"] = convergence.reliability_report(hist)

    # Load model
    print(f"[diag] Loading checkpoint {checkpoint}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = QNNPolicy.load(str(checkpoint), device=device)
    policy.model.eval()

    # ── Effective rank (no data needed) ──────────────────────────────────
    if "rank" not in skip:
        print("[diag] Computing effective rank of all Linear layers...", flush=True)
        report["rank"] = rank.all_linear_ranks(policy.model)

    # Load val data (used by ablation + gradients)
    if {"ablation", "gradients"} - skip:
        print(f"[diag] Loading val data ({max_val_shards} shards, {max_val_episodes} episodes max)...", flush=True)
        episodes = data.load_val_episodes(
            Path(data_dir),
            split="val",
            max_shards=max_val_shards,
            max_episodes=max_val_episodes,
        )
        report["val_episodes_used"] = len(episodes)

        # ── Layer ablation ────────────────────────────────────────────────
        if "ablation" not in skip and episodes:
            print(f"[diag] Running layer ablation on {len(episodes)} episodes...", flush=True)
            report["ablation"] = ablation.layer_ablation_table(policy, episodes)

        # ── Gradient diagnostics (one episode) ───────────────────────────
        if "gradients" not in skip and episodes:
            print("[diag] Computing gradient norms (one supervised batch)...", flush=True)
            policy.model.train()  # need grad
            grad_rows = gradients.per_parameter_grad_norms(policy, episodes[0])
            report["gradients"] = {
                "by_param": grad_rows[:50],  # first 50 to keep report compact
                "by_module": gradients.aggregate_by_module(grad_rows, depth=2),
                "summary": gradients.gradient_health_summary(grad_rows),
            }
            policy.model.eval()

        # ── Participation ratio / dead units on head bottlenecks ────────
        if "participation" not in skip and episodes:
            print("[diag] Computing participation ratio + dead-unit stats on head bottlenecks...", flush=True)
            report["participation"] = participation.head_bottleneck_report(policy, episodes)

        # ── Per-attention-head diagnostics ──────────────────────────────
        if "attention" not in skip and episodes:
            print("[diag] Capturing per-attention-head weights...", flush=True)
            report["attention_patterns"] = attention.attention_pattern_summary(policy, episodes)
            print("[diag] Per-attention-head ablation...", flush=True)
            report["attention_ablation"] = attention.per_attention_head_ablation(policy, episodes)

        # ── Per-neuron pruning sensitivity (slow, opt-in) ───────────────
        if "pruning" not in skip and episodes:
            print("[diag] Per-head pruning sensitivity (slow)...", flush=True)
            report["pruning_summary"] = pruning.head_bottleneck_pruning_summary(policy, episodes)

        # ── Linear probe (slow, opt-in) ─────────────────────────────────
        if "linear_probe" not in skip and episodes:
            print("[diag] Linear probe on head-input features (slow)...", flush=True)
            train_eps = data.load_val_episodes(
                Path(data_dir),
                split="train",
                max_shards=max_val_shards,
                max_episodes=max_val_episodes,
            )
            report["linear_probe"] = linear_probe.linear_probe_report(
                policy, train_eps, episodes,
            )

    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the structured report as readable markdown."""
    lines: list[str] = []
    lines.append(f"# Diagnostic report\n")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    lines.append(f"- Data: `{report['data_dir']}`")
    lines.append(f"- Val episodes used for ablation/gradients: {report.get('val_episodes_used', 0)}")
    lines.append("")

    if "history" in report:
        h = report["history"]
        lines.append("## Training history\n")
        lines.append(f"- Total epochs: {h['n_epochs']}")
        if h.get("best_epoch"):
            be = h["best_epoch"]
            lines.append(f"- Best epoch: {be.get('epoch')} (val_loss={be.get('val_loss'):.4f})")
        lines.append(f"- Still improving at end (last 2 epochs):")
        for k, v in h.get("still_improving", {}).items():
            lines.append(f"  - {k}: {'yes' if v else 'no'}")
        gap = h.get("train_val_gap", [])
        if gap:
            lines.append(f"- Train/val gap by epoch: {[(e, round(g, 4)) for e, g in gap]}")
        lines.append("")

    if "convergence" in report:
        c = report["convergence"]
        lines.append("## Convergence reliability\n")
        slope = c.get("slope_last3")
        lines.append(f"- Late-epoch slope (val_loss / epoch): {slope:+.4f}" if slope is not None else "- Slope: unavailable")
        lines.append(f"- Convergence class: **{c['convergence']}**  "
                     f"(thresholds: converged < 0.002, near < 0.005, descending ≥ 0.005)")
        asym = c.get("asymptote_fit")
        if asym:
            lines.append(f"- Projected asymptote (loss_inf): {asym['loss_inf']:.4f}  "
                         f"(fit RMS: {asym['residual_rms']:.4f})")
            if c.get("last_value") is not None:
                gap_to_asym = c["last_value"] - asym["loss_inf"]
                lines.append(f"- Gap from current to projected asymptote: {gap_to_asym:+.4f}")
        if c["convergence"] == "descending":
            lines.append("- ⚠ This run is still descending; comparisons against converged runs are biased.")
        lines.append("")

    if "rank" in report:
        lines.append("## Effective rank of Linear layers\n")
        lines.append("Sorted by frac (fraction of full rank used) ascending — most-overparameterized first.\n")
        lines.append("| name | shape | full | effective | frac | params |")
        lines.append("|------|-------|------|-----------|------|--------|")
        for r in report["rank"]:
            lines.append(
                f"| `{r['name']}` | {r['shape']} | {r['full_rank']} | {r['effective_rank']} | {r['frac']:.2f} | {r['n_params']:,} |"
            )
        lines.append("")

    if "ablation" in report:
        lines.append("## Layer ablation (zero-out)\n")
        lines.append("Larger delta = module is more essential. Sorted by delta desc.\n")
        lines.append("| module | baseline | ablated | delta |")
        lines.append("|--------|----------|---------|-------|")
        for r in report["ablation"]:
            err = r.get("error")
            if err:
                lines.append(f"| `{r['name']}` | {r['baseline_loss']:.4f} | error | — |  ")
                lines.append(f"  - error: {err}")
            else:
                lines.append(f"| `{r['name']}` | {r['baseline_loss']:.4f} | {r['ablated_loss']:.4f} | {r['delta']:+.4f} |")
        lines.append("")

    if "gradients" in report:
        g = report["gradients"]
        lines.append("## Gradient norms (one supervised batch)\n")
        s = g.get("summary", {})
        if s:
            lines.append("**Summary:**")
            lines.append(f"- Layers with grad: {s['n_layers']}")
            lines.append(f"- Max / median / min grad norm: {s['max_grad_norm']:.4f} / {s['median_grad_norm']:.4f} / {s['min_grad_norm']:.4e}")
            lines.append(f"- Fraction of layers with grad < 1% of max: {s['frac_below_1pct_of_max']:.1%}")
            lines.append(f"- Fraction of layers with zero grad: {s['frac_zero']:.1%}")
            lines.append("")
        lines.append("**By module (aggregated, top 20 by grad norm):**\n")
        lines.append("| module | grad_norm | param_norm | n_params |")
        lines.append("|--------|-----------|------------|----------|")
        for r in g.get("by_module", [])[:20]:
            lines.append(f"| `{r['name']}` | {r['grad_norm']:.4f} | {r['param_norm']:.4f} | {r['n_params']:,} |")
        lines.append("")

    if "participation" in report:
        lines.append("## Head bottleneck activation stats\n")
        lines.append("Participation ratio is *one* signal — not a primary cut decision. See ablation/pruning for that.\n")
        lines.append("| head | B | frames | PR | eff_rank_1pct | dead | rare | always_on | weak_cols |")
        lines.append("|------|---|--------|----|---------------|------|------|-----------|-----------|")
        for name, r in report["participation"].items():
            lines.append(
                f"| `{name}` | {r['d_hidden']} | {r['frames_analyzed']} | "
                f"{r['participation_ratio']:.1f} | {r['effective_rank_1pct']} | "
                f"{r['dead_units']} | {r['rare_units']} | {r['always_on_units']} | "
                f"{r['weak_cols_lt_10pct_max']} |"
            )
        lines.append("")

    if "attention_patterns" in report:
        lines.append("## Attention-head patterns\n")
        lines.append("Low entropy = sharply focused; high entropy near max = diffuse. "
                     "Off-diag similarity ~1.0 = redundant heads.\n")
        lines.append("| layer | n_heads | seq_len | mean_entropy | max_possible | offdiag_sim |")
        lines.append("|-------|---------|---------|--------------|--------------|-------------|")
        for name, r in report["attention_patterns"].items():
            sim = r.get("cross_head_offdiag_sim_mean")
            sim_str = f"{sim:.3f}" if sim is not None and sim == sim else "—"  # nan check
            lines.append(
                f"| `{name}` | {r['n_heads']} | {r['seq_len']} | "
                f"{r['mean_entropy']:.3f} | {r['max_possible_entropy']:.3f} | {sim_str} |"
            )
        lines.append("")

    if "attention_ablation" in report:
        lines.append("## Per-attention-head ablation\n")
        lines.append("Zeroed each head's V projection + out_proj input columns; larger delta = more essential.\n")
        lines.append("| layer | head_idx | baseline | ablated | delta |")
        lines.append("|-------|----------|----------|---------|-------|")
        for r in report["attention_ablation"]:
            lines.append(
                f"| `{r['layer']}` | {r['head_idx']} | {r['baseline_loss']:.4f} | "
                f"{r['ablated_loss']:.4f} | {r['delta']:+.4f} |"
            )
        lines.append("")

    if "pruning_summary" in report:
        lines.append("## Per-head bottleneck pruning sensitivity\n")
        lines.append("How many neurons account for 90% of impact, and how many are effectively redundant?\n")
        lines.append("| head | neurons | max_delta | median_delta | k_at_90% | n_redundant<.001 |")
        lines.append("|------|---------|-----------|--------------|----------|-----------------|")
        for name, r in report["pruning_summary"].items():
            lines.append(
                f"| `{name}` | {r['n_neurons']} | {r['max_delta']:.4f} | "
                f"{r['median_delta']:.4f} | {r['k_at_90pct']} | {r['n_redundant_lt_0.001']} |"
            )
        lines.append("")

    if "linear_probe" in report:
        lp = report["linear_probe"]
        lines.append("## Linear probe (frozen-encoder → linear classifier)\n")
        lines.append("Compare to trained-head F1: gap = how much the head's nonlinearity is doing.\n")
        for k, v in lp.items():
            if isinstance(v, float):
                lines.append(f"- `{k}`: probe_f1 = **{v:.4f}**")
            else:
                lines.append(f"- `{k}`: {v}")
        lines.append("")

    return "\n".join(lines)
