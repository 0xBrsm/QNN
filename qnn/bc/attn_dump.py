"""Dump attention weights from a trained BC model on val data.

Reports where the self token (readout) attends: entities, spatial,
action history, or self.  Helps diagnose whether the model is using
entity information for look predictions.

Usage (in training container):
    python -m qnn.bc.attn_dump runs/bc/<run>/checkpoints/bc_best_model.pth \
        --bc-data-dir assets/collect/prod --device cpu --frames 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qnn.model.policy import QNNPolicy
from qnn.model.transformer import TransformerBlock


@contextmanager
def capture_attention(model: QNNPolicy):
    """Temporarily hook all TransformerBlocks to capture attention weights."""
    weights: list[list[torch.Tensor]] = []  # [layer][batch call] = (batch, n_heads, seq, seq)
    handles = []

    for i, block in enumerate(model.model.trunk.blocks):
        layer_weights: list[torch.Tensor] = []
        weights.append(layer_weights)

        def make_hook(lw):
            def hook_fn(module, args, kwargs):
                # Override need_weights to True
                kwargs = dict(kwargs) if kwargs else {}
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return args, kwargs
            return hook_fn

        def make_post_hook(lw):
            def post_hook(module, args, kwargs, output):
                # output is (attn_out, attn_weights)
                if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                    lw.append(output[1].detach().cpu())
                return output
            return post_hook

        h1 = block.attn.register_forward_pre_hook(make_hook(layer_weights), with_kwargs=True)
        h2 = block.attn.register_forward_hook(make_post_hook(layer_weights), with_kwargs=True)
        handles.extend([h1, h2])

    try:
        yield weights
    finally:
        for h in handles:
            h.remove()


def load_val_episodes(bc_data_dir: str, max_frames: int):
    """Load val episodes up to max_frames total."""
    val_dir = Path(bc_data_dir) / "precomputed_val"
    manifest = json.loads((val_dir / "manifest.json").read_text())

    obs_list = []
    action_list = []
    total = 0

    if isinstance(manifest, dict) and manifest.get("format") == "sharded_v1":
        for shard in manifest["shards"]:
            obs_arrays = {k: np.load(val_dir / v, mmap_mode="r") for k, v in shard["obs"].items()}
            act_arrays = {k: np.load(val_dir / v, mmap_mode="r") for k, v in shard["actions"].items()}
            start = 0
            for ep_len in shard["episode_lengths"]:
                if total >= max_frames:
                    break
                take = min(ep_len, max_frames - total)
                obs_list.append({k: v[start:start+take] for k, v in obs_arrays.items()})
                action_list.append({k: v[start:start+take] for k, v in act_arrays.items()})
                total += take
                start += ep_len
    else:
        for entry in manifest:
            if total >= max_frames:
                break
            obs = {k: np.load(val_dir / v, mmap_mode="r") for k, v in entry["obs"].items()}
            acts = {k: np.load(val_dir / v, mmap_mode="r") for k, v in entry["actions"].items()}
            take = min(obs[next(iter(obs))].shape[0], max_frames - total)
            obs_list.append({k: v[:take] for k, v in obs.items()})
            action_list.append({k: v[:take] for k, v in acts.items()})
            total += take

    return obs_list, action_list, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to bc_best_model.pth")
    parser.add_argument("--bc-data-dir", default="assets/collect/prod")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to analyze")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model = QNNPolicy.load(args.checkpoint, device=args.device)
    model.model.eval()

    n_layers = len(model.model.trunk.blocks)
    has_ah = model.model.trunk.tokenizer.action_history_tokens > 0
    ah_count = model.model.trunk.tokenizer.action_history_tokens

    # Token layout: [self(1), spatial(9), action_history(ah_count), entities(16)]
    self_idx = 0
    spatial_start = 1
    spatial_end = 10  # 1..9
    if has_ah:
        ah_start = 10
        ah_end = 10 + ah_count
        entity_start = ah_end
    else:
        ah_start = ah_end = 10
        entity_start = 10
    entity_end = entity_start + 16

    print(f"Token layout: self=0, spatial=1..9, ah={ah_start}..{ah_end-1}, entity={entity_start}..{entity_end-1}")
    print(f"Layers: {n_layers}, heads: {model.n_heads}")
    print()

    obs_episodes, act_episodes, total_frames = load_val_episodes(args.bc_data_dir, args.frames)
    print(f"Loaded {total_frames} frames")

    # Run forward passes and collect attention
    all_attn = [[] for _ in range(n_layers)]  # [layer] -> list of (n_heads, seq, seq)

    with torch.inference_mode():
        for obs_ep in obs_episodes:
            n = obs_ep[next(iter(obs_ep))].shape[0]
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                batch_obs = {k: torch.from_numpy(np.asarray(v[start:end])).to(model.device) for k, v in obs_ep.items()}

                with capture_attention(model) as weights:
                    model.model.trunk(batch_obs)

                for layer_idx in range(n_layers):
                    if weights[layer_idx]:
                        # (batch, n_heads, seq, seq)
                        w = weights[layer_idx][0]
                        all_attn[layer_idx].append(w)

    # Analyze: for the self token (row 0), where does it attend?
    print("\n=== Self token attention (readout) ===")
    print("Where the self token looks when producing the readout vector.\n")

    for layer_idx in range(n_layers):
        if not all_attn[layer_idx]:
            print(f"Layer {layer_idx}: no attention captured")
            continue

        # Concatenate across batches: (total_frames, n_heads, seq, seq)
        cat = torch.cat(all_attn[layer_idx], dim=0)
        # Self token attention: row 0 -> (total_frames, n_heads, seq)
        self_attn = cat[:, :, 0, :]  # (frames, heads, seq)

        for head in range(self_attn.shape[1]):
            head_attn = self_attn[:, head, :]  # (frames, seq)
            avg = head_attn.mean(dim=0)  # (seq,)

            self_weight = avg[self_idx].item()
            spatial_weight = avg[spatial_start:spatial_end].sum().item()
            ah_weight = avg[ah_start:ah_end].sum().item() if has_ah else 0.0
            entity_weight = avg[entity_start:entity_end].sum().item()

            print(f"Layer {layer_idx} Head {head}:")
            print(f"  self:     {self_weight:.4f}")
            print(f"  spatial:  {spatial_weight:.4f}  (9 tokens, avg {spatial_weight/9:.4f})")
            if has_ah:
                print(f"  action_h: {ah_weight:.4f}  ({ah_count} tokens, avg {ah_weight/ah_count:.4f})")
            print(f"  entity:   {entity_weight:.4f}  (16 tokens, avg {entity_weight/16:.4f})")

            # Top 5 individual token attention
            top5 = avg.topk(5)
            token_names = []
            for idx in top5.indices.tolist():
                if idx == 0:
                    name = "self"
                elif spatial_start <= idx < spatial_end:
                    name = f"spatial_{idx - spatial_start}"
                elif has_ah and ah_start <= idx < ah_end:
                    name = f"ah_{idx - ah_start}"
                elif entity_start <= idx < entity_end:
                    name = f"entity_{idx - entity_start}"
                else:
                    name = f"tok_{idx}"
                token_names.append(f"{name}={avg[idx]:.4f}")
            print(f"  top 5:    {', '.join(token_names)}")
            print()

    # Also check: on frames with large turns, does attention shift?
    print("=== Attention by turn magnitude ===")
    all_look = np.concatenate([ep["look"] for ep in act_episodes])
    turn_mag = np.sqrt(all_look[:, 1]**2 + all_look[:, 2]**2)
    turn_deg = np.degrees(np.arcsin(np.clip(turn_mag, 0, 1)))

    for layer_idx in range(n_layers):
        if not all_attn[layer_idx]:
            continue
        cat = torch.cat(all_attn[layer_idx], dim=0)
        self_attn = cat[:, :, 0, :]  # (frames, heads, seq)
        # Use first head only for simplicity
        head_attn = self_attn[:, 0, :].numpy()[:len(turn_deg)]

        bins = [("0-5°", 0, 5), ("5-15°", 5, 15), ("15°+", 15, 180)]
        print(f"Layer {layer_idx} Head 0 — entity attention by turn magnitude:")
        for tag, lo, hi in bins:
            mask = (turn_deg >= lo) & (turn_deg < hi)
            if mask.sum() == 0:
                continue
            ent_attn = head_attn[mask, entity_start:entity_end].sum(axis=1).mean()
            ah_attn = head_attn[mask, ah_start:ah_end].sum(axis=1).mean() if has_ah else 0
            print(f"  {tag:6s}  entity={ent_attn:.4f}  ah={ah_attn:.4f}  (n={mask.sum()})")
        print()


if __name__ == "__main__":
    main()
