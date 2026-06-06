"""Standalone causal probe for the BC target head.

Isolates the target-prediction problem from the full BC trunk: can a model
predict the opt3-labeler's per-frame slot choice from *causal* observations
alone, closing the ~2.5% labeler-vs-engine-slot-0 gap?

Inputs (per frame):
  - per-slot scalars (16, 19) — rel/dist/vel/path/eta/facing/recency etc.
  - per-slot type id (16,) — embedded
  - per-slot enemy flag (16,) — derived from team scalar
  - self scalars (16,)
  - self movement id (3-way one-hot)
  - self weapon id (embedded)
  - action context: look (3) + fire (1)

Architecture:
  - per-slot per-frame MLP (no time mixing) -> per-slot embedding (32)
  - flatten per-slot embeddings + concat global features -> frame vector
  - causal TCN (left-pad only) over time -> 16-way slot logits per frame

Loss: CE with ignore_index=-100.
Eval is bucketed:
  - bucket A: target == engine slot 0 (trivial, ~97.5% of labeled frames)
  - bucket B: target != engine slot 0 (the interesting ~2.5%)

Usage:
    PYTHONPATH=src python -m qnn.labeler.probes.target_head_probe \
        --data-dir artifacts/collect/qwd \
        --output   runs/probe/target_v0 \
        --epochs 3 --chunk-len 512 --max-shards 3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qnn.vocab import MAX_TOKEN_OBJECTS, TOKEN_ACTOR


_ACTOR_TEAM_OFFSET = 16
_TEAM_TEAMMATE_VALUE = 1.0

N_SLOTS = MAX_TOKEN_OBJECTS         # 16
N_SLOT_SCALARS = 19
N_SELF_SCALARS = 16
N_TYPE_VOCAB = 16                   # token types fit easily here
N_WEAPON_VOCAB = 16                 # quake weapon ids fit easily here


# ── per-frame feature build ──────────────────────────────────────────────────

def _enemy_flag(entity_types: np.ndarray, entity_scalars: np.ndarray) -> np.ndarray:
    """(T,16) float32. 1 iff token is an enemy actor (actor & not teammate)."""
    is_actor = entity_types == TOKEN_ACTOR
    team = entity_scalars[:, :, _ACTOR_TEAM_OFFSET].astype(np.float32)
    is_teammate = team == _TEAM_TEAMMATE_VALUE
    return (is_actor & ~is_teammate).astype(np.float32)


# ── model ─────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """1D conv with left-pad only — output at t depends only on inputs <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TargetHeadProbe(nn.Module):
    def __init__(
        self,
        slot_embed_dim: int = 32,
        type_embed_dim: int = 4,
        weapon_embed_dim: int = 8,
        channels: int = 128,
        n_layers: int = 7,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.type_embed = nn.Embedding(N_TYPE_VOCAB, type_embed_dim)
        self.weapon_embed = nn.Embedding(N_WEAPON_VOCAB, weapon_embed_dim)

        slot_input_dim = N_SLOT_SCALARS + type_embed_dim + 1  # +1 enemy flag
        self.slot_mlp = nn.Sequential(
            nn.Linear(slot_input_dim, slot_embed_dim),
            nn.GELU(),
            nn.Linear(slot_embed_dim, slot_embed_dim),
        )

        # Per-frame feature: per-slot embeddings flattened + self + action
        frame_input_dim = (
            N_SLOTS * slot_embed_dim
            + N_SELF_SCALARS
            + 3                    # self_movement_id one-hot
            + weapon_embed_dim
            + 3                    # look
            + 1                    # fire
        )

        self.in_proj = nn.Sequential(
            nn.Linear(frame_input_dim, channels),
            nn.GELU(),
        )

        layers: list[nn.Module] = []
        for i in range(n_layers):
            dilation = 2 ** i
            layers += [
                CausalConv1d(channels, channels, kernel_size, dilation=dilation),
                nn.GELU(),
                nn.Dropout(p_drop),
            ]
        self.tcn = nn.Sequential(*layers)

        self.head = nn.Linear(channels, N_SLOTS)

        self._kernel_size = kernel_size
        self._n_layers = n_layers

    @property
    def receptive_field(self) -> int:
        """Causal RF (frames into the past, inclusive of t)."""
        return 1 + (self._kernel_size - 1) * (2 ** self._n_layers - 1)

    def forward(
        self,
        slot_scalars: torch.Tensor,    # (B, T, 16, 19)
        slot_types: torch.Tensor,      # (B, T, 16)        long
        slot_enemy: torch.Tensor,      # (B, T, 16)
        self_scalars: torch.Tensor,    # (B, T, 16)
        movement_oh: torch.Tensor,     # (B, T, 3)
        weapon_id: torch.Tensor,       # (B, T)            long
        look: torch.Tensor,            # (B, T, 3)
        fire: torch.Tensor,            # (B, T)
    ) -> torch.Tensor:
        B, T = slot_scalars.shape[:2]

        slot_t_emb = self.type_embed(slot_types)                              # (B,T,16,Et)
        slot_in = torch.cat(
            [slot_scalars, slot_t_emb, slot_enemy.unsqueeze(-1)], dim=-1
        )                                                                      # (B,T,16,S+Et+1)
        slot_emb = self.slot_mlp(slot_in)                                     # (B,T,16,Es)
        slot_flat = slot_emb.reshape(B, T, -1)                                # (B,T,16*Es)

        weap_emb = self.weapon_embed(weapon_id)                               # (B,T,Ew)
        fire_in = fire.unsqueeze(-1).float()                                  # (B,T,1)

        frame = torch.cat(
            [slot_flat, self_scalars, movement_oh, weap_emb, look, fire_in],
            dim=-1,
        )                                                                      # (B,T,F)

        h = self.in_proj(frame)                                                # (B,T,C)
        h = h.transpose(1, 2)                                                  # (B,C,T)
        h = self.tcn(h)
        h = h.transpose(1, 2)                                                  # (B,T,C)
        return self.head(h)                                                    # (B,T,16)


# ── dataset ───────────────────────────────────────────────────────────────────

@dataclass
class _Shard:
    """mmap views into one shard."""
    entity_scalars: np.ndarray   # (T,16,19) fp16
    entity_types:   np.ndarray   # (T,16)    int8
    entity_ids:     np.ndarray   # (T,16,3)  uint8
    self_scalars:   np.ndarray   # (T,16)    fp16
    movement_id:    np.ndarray   # (T,1)     int32
    weapon_id:      np.ndarray   # (T,1)     int32
    look:           np.ndarray   # (T,3)     fp16
    fire:           np.ndarray   # (T,)      uint8
    target:         np.ndarray   # (T,)      int64
    episode_lengths: list[int]
    # Per-slot keep mask derived from the configured token_mask predicate,
    # or None if no masking is configured.  Stored at load time so the
    # per-chunk slice in __getitem__ is a cheap boolean index.
    token_keep:     "np.ndarray | None" = None


def _load_split(
    split_dir: Path,
    max_shards: int | None = None,
    token_mask: "dict | None" = None,
) -> list[_Shard]:
    manifest = json.loads((split_dir / "manifest.json").read_text())
    shards = manifest["shards"][:max_shards] if max_shards else manifest["shards"]
    out: list[_Shard] = []
    for sh_idx in range(len(shards)):
        tag = f"shard{sh_idx:06d}"
        etyp = np.load(split_dir / f"{tag}_obs_entity_types.npy", mmap_mode="r")
        eids = np.load(split_dir / f"{tag}_obs_entity_ids.npy", mmap_mode="r")
        escal = np.load(split_dir / f"{tag}_obs_entity_scalars_raw.npy", mmap_mode="r")
        s_scal = np.load(split_dir / f"{tag}_obs_self_scalars.npy", mmap_mode="r")
        mid = np.load(split_dir / f"{tag}_obs_self_movement_id.npy", mmap_mode="r")
        wid = np.load(split_dir / f"{tag}_obs_self_weapon_id.npy", mmap_mode="r")
        look = np.load(split_dir / f"{tag}_act_look.npy", mmap_mode="r")
        fire = np.load(split_dir / f"{tag}_act_fire.npy", mmap_mode="r")
        target = np.load(split_dir / f"{tag}_act_target.npy", mmap_mode="r")

        token_keep = None
        if token_mask:
            from qnn.bc.token_filter import _flatten_per_token_arrays
            from qnn import filter_dsl
            flat = _flatten_per_token_arrays({
                "entity_types": etyp, "entity_ids": eids,
            })
            token_keep = np.asarray(
                filter_dsl.eval_filter(flat, token_mask), dtype=bool
            )

        out.append(_Shard(
            entity_scalars=escal,
            entity_types=etyp,
            entity_ids=eids,
            self_scalars=s_scal,
            movement_id=mid,
            weapon_id=wid,
            look=look,
            fire=fire,
            target=target,
            episode_lengths=shards[sh_idx]["episode_lengths"],
            token_keep=token_keep,
        ))
    return out


class _ChunkedDataset(Dataset):
    """Episodes chunked into fixed-length windows. Tail padded with -100 targets."""

    def __init__(self, shards: list[_Shard], chunk_len: int,
                 drop_all_ignore: bool = True) -> None:
        self.shards = shards
        self.chunk_len = chunk_len
        self.index: list[tuple[int, int, int, int]] = []  # (sh, ep_start, ck_start, valid_len)
        for sh_idx, sh in enumerate(shards):
            start = 0
            for length in sh.episode_lengths:
                T = int(length)
                if T <= 0:
                    start += T
                    continue
                ep_target = np.asarray(sh.target[start:start + T])
                n_full = T // chunk_len
                for c in range(n_full):
                    if drop_all_ignore and (ep_target[c * chunk_len:(c + 1) * chunk_len] == -100).all():
                        continue
                    self.index.append((sh_idx, start, c * chunk_len, chunk_len))
                tail = T - n_full * chunk_len
                if tail > 0:
                    if not drop_all_ignore or not (ep_target[n_full * chunk_len:] == -100).all():
                        self.index.append((sh_idx, start, n_full * chunk_len, tail))
                start += T

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sh_idx, ep_start, ck_start, valid_len = self.index[idx]
        sh = self.shards[sh_idx]
        s = ep_start + ck_start
        e = s + valid_len
        L = self.chunk_len

        slot_scalars = torch.zeros(L, N_SLOTS, N_SLOT_SCALARS, dtype=torch.float32)
        slot_types   = torch.zeros(L, N_SLOTS, dtype=torch.long)
        slot_enemy   = torch.zeros(L, N_SLOTS, dtype=torch.float32)
        self_scalars = torch.zeros(L, N_SELF_SCALARS, dtype=torch.float32)
        movement_oh  = torch.zeros(L, 3, dtype=torch.float32)
        weapon_id    = torch.zeros(L, dtype=torch.long)
        look         = torch.zeros(L, 3, dtype=torch.float32)
        fire         = torch.zeros(L, dtype=torch.float32)
        target       = torch.full((L,), -100, dtype=torch.long)

        es = np.asarray(sh.entity_scalars[s:e], dtype=np.float32)
        et_raw = np.asarray(sh.entity_types[s:e], dtype=np.int64)
        ss = np.asarray(sh.self_scalars[s:e], dtype=np.float32)
        mi = np.asarray(sh.movement_id[s:e], dtype=np.int64).reshape(-1)
        wi = np.asarray(sh.weapon_id[s:e], dtype=np.int64).reshape(-1).clip(0, N_WEAPON_VOCAB - 1)
        lk = np.asarray(sh.look[s:e], dtype=np.float32)
        fr = np.asarray(sh.fire[s:e], dtype=np.float32)
        tg = np.asarray(sh.target[s:e], dtype=np.int64)

        # Apply token_mask if configured: zero out scalars and types on
        # rejected slots, and clear targets that pointed into them.
        if sh.token_keep is not None:
            keep = sh.token_keep[s:e]                       # (chunk_T, 16) bool
            drop = ~keep
            if drop.any():
                es = es.copy(); es[drop] = 0.0
                et_raw = et_raw.copy(); et_raw[drop] = -1
                rows = np.arange(et_raw.shape[0])
                slot_idx = np.clip(tg, 0, N_SLOTS - 1)
                masked_target = (tg != -100) & (et_raw[rows, slot_idx] == -1)
                if masked_target.any():
                    tg = tg.copy()
                    tg[masked_target] = -100

        # Enemy flag derived from (possibly masked) per-slot scalars +
        # types, so it stays consistent with the rest of the per-slot
        # feature vector.
        en = _enemy_flag(et_raw.astype(np.int8), es).astype(np.float32)
        et = et_raw.clip(0, N_TYPE_VOCAB - 1)

        slot_scalars[:valid_len] = torch.from_numpy(es)
        slot_types[:valid_len]   = torch.from_numpy(et)
        slot_enemy[:valid_len]   = torch.from_numpy(en)
        self_scalars[:valid_len] = torch.from_numpy(ss)
        movement_oh[:valid_len, 0] = torch.from_numpy((mi == 0).astype(np.float32))
        movement_oh[:valid_len, 1] = torch.from_numpy((mi == 1).astype(np.float32))
        movement_oh[:valid_len, 2] = torch.from_numpy((mi >= 2).astype(np.float32))
        weapon_id[:valid_len]    = torch.from_numpy(wi)
        look[:valid_len]         = torch.from_numpy(lk)
        fire[:valid_len]         = torch.from_numpy(fr)
        target[:valid_len]       = torch.from_numpy(tg)

        return {
            "slot_scalars": slot_scalars,
            "slot_types":   slot_types,
            "slot_enemy":   slot_enemy,
            "self_scalars": self_scalars,
            "movement_oh":  movement_oh,
            "weapon_id":    weapon_id,
            "look":         look,
            "fire":         fire,
            "target":       target,
        }


# ── eval ──────────────────────────────────────────────────────────────────────

@dataclass
class BucketStats:
    n: int = 0
    correct: int = 0
    # predicted == 0 frequency (the strict-engine baseline matches slot 0)
    pred_zero: int = 0
    # how often label == 0 (sanity)
    label_zero: int = 0
    # full confusion of {label_slot -> pred_slot} counts for the labels
    # that actually occur in the bucket; key is (label, pred).
    confusion: dict = None

    def __post_init__(self):
        if self.confusion is None:
            self.confusion = {}

    def acc(self) -> float:
        return 100.0 * self.correct / max(self.n, 1)


def _accumulate(stats_all: BucketStats, stats_a: BucketStats, stats_b: BucketStats,
                target: torch.Tensor, pred: torch.Tensor) -> None:
    """Bucket: A = target==0 (trivial), B = target!=0 (disagreement case)."""
    mask = target != -100
    if not mask.any():
        return
    t = target[mask]
    p = pred[mask]
    correct = (t == p).int()
    n = int(mask.sum().item())
    c = int(correct.sum().item())
    stats_all.n += n
    stats_all.correct += c
    stats_all.pred_zero += int((p == 0).sum().item())
    stats_all.label_zero += int((t == 0).sum().item())

    is_a = t == 0
    is_b = ~is_a
    stats_a.n += int(is_a.sum().item())
    stats_a.correct += int(correct[is_a].sum().item())
    stats_a.pred_zero += int((p[is_a] == 0).sum().item())
    stats_b.n += int(is_b.sum().item())
    stats_b.correct += int(correct[is_b].sum().item())
    stats_b.pred_zero += int((p[is_b] == 0).sum().item())

    # Confusion accumulators (B only — A is trivial; pred==0 covers the bulk).
    t_b = t[is_b].cpu().numpy()
    p_b = p[is_b].cpu().numpy()
    for lt, lp in zip(t_b.tolist(), p_b.tolist()):
        stats_b.confusion[(lt, lp)] = stats_b.confusion.get((lt, lp), 0) + 1


# ── train / eval loop ─────────────────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@dataclass
class TrainConfig:
    data_dir: Path
    output: Path
    chunk_len: int = 512
    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    channels: int = 128
    n_layers: int = 7
    kernel_size: int = 3
    dropout: float = 0.1
    num_workers: int = 0
    max_shards_train: int | None = None
    max_shards_val: int | None = None
    seed: int = 17
    # Class-imbalance handling. 97.5% of labeled frames have target=0, so
    # vanilla CE collapses to "always predict 0". Down-weight slot 0 to push
    # the model to actually predict the 2.5% disagreement cases.
    slot0_weight: float = 0.05
    autocast_dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"
    # Optional per-slot mask predicate (qnn.bc.token_filter).  Applied at
    # shard load time; rejected slots are zeroed in the feature build.
    token_mask: "dict | None" = None


def _run_eval(model: TargetHeadProbe, loader: DataLoader,
              device: torch.device) -> tuple[BucketStats, BucketStats, BucketStats, dict]:
    """Returns (all, bucket_A, bucket_B, threshold_curve).

    threshold_curve maps a confidence threshold tau to the practical policy:
        if model_argmax != 0 and softmax(model_argmax) >= tau:
            predict model_argmax
        else:
            predict 0
    For each tau, records (A_acc, B_acc, override_rate).
    """
    model.eval()
    all_, a, b = BucketStats(), BucketStats(), BucketStats()
    taus = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    # For each tau accumulate: (n_total, correct, overrides, override_correct,
    #                          A_correct_under_policy, B_correct_under_policy, A_n, B_n)
    curve: dict = {tau: {"A_n": 0, "A_correct": 0,
                          "B_n": 0, "B_correct": 0,
                          "overrides": 0, "override_correct": 0,
                          "total": 0} for tau in taus}
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=(device.type == "cuda")
            ):
                logits = model(
                    batch["slot_scalars"], batch["slot_types"], batch["slot_enemy"],
                    batch["self_scalars"], batch["movement_oh"],
                    batch["weapon_id"], batch["look"], batch["fire"],
                )
            logits = logits.float()
            pred = logits.argmax(dim=-1)
            _accumulate(all_, a, b, batch["target"], pred)

            # Threshold-policy eval.
            probs = F.softmax(logits, dim=-1)
            argmax_p = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            target = batch["target"]
            mask = target != -100
            tg = target[mask]
            pr = pred[mask]
            pp = argmax_p[mask]
            for tau in taus:
                # Override slot 0 only when argmax!=0 AND confidence>=tau.
                override = (pr != 0) & (pp >= tau)
                policy_pred = torch.where(override, pr, torch.zeros_like(pr))
                is_a = tg == 0
                is_b = ~is_a
                a_correct = ((policy_pred == tg) & is_a).sum().item()
                b_correct = ((policy_pred == tg) & is_b).sum().item()
                rec = curve[tau]
                rec["A_n"] += int(is_a.sum().item())
                rec["B_n"] += int(is_b.sum().item())
                rec["A_correct"] += int(a_correct)
                rec["B_correct"] += int(b_correct)
                rec["overrides"] += int(override.sum().item())
                rec["override_correct"] += int(((policy_pred == tg) & override).sum().item())
                rec["total"] += int(mask.sum().item())
    return all_, a, b, curve


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, cuda available: {torch.cuda.is_available()}")

    train_dir = cfg.data_dir / "precomputed_train"
    val_dir   = cfg.data_dir / "precomputed_val"

    if cfg.token_mask:
        print(f"token_mask: {cfg.token_mask}")
    t0 = time.time()
    train_shards = _load_split(train_dir, cfg.max_shards_train, token_mask=cfg.token_mask)
    val_shards   = _load_split(val_dir,   cfg.max_shards_val,   token_mask=cfg.token_mask)
    print(f"loaded train shards={len(train_shards)} val shards={len(val_shards)} "
          f"in {time.time()-t0:.1f}s")

    train_ds = _ChunkedDataset(train_shards, cfg.chunk_len)
    val_ds   = _ChunkedDataset(val_shards,   cfg.chunk_len)
    print(f"chunks: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=False, pin_memory=(device.type == "cuda"),
    )

    model = TargetHeadProbe(
        channels=cfg.channels, n_layers=cfg.n_layers,
        kernel_size=cfg.kernel_size, p_drop=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,}  causal RF={model.receptive_field} frames")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.output.mkdir(parents=True, exist_ok=True)

    class_weights = torch.ones(N_SLOTS, dtype=torch.float32, device=device)
    class_weights[0] = cfg.slot0_weight
    print(f"class weights: slot0={cfg.slot0_weight}, slot1..15=1.0")

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[cfg.autocast_dtype]
    use_autocast = (cfg.autocast_dtype != "fp32" and device.type == "cuda")
    print(f"autocast: {cfg.autocast_dtype} (enabled={use_autocast})")

    log: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        t_ep = time.time()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            target = batch["target"]
            if (target != -100).sum() == 0:
                continue
            with torch.amp.autocast(device_type=device.type,
                                     dtype=autocast_dtype, enabled=use_autocast):
                logits = model(
                    batch["slot_scalars"], batch["slot_types"], batch["slot_enemy"],
                    batch["self_scalars"], batch["movement_oh"],
                    batch["weapon_id"], batch["look"], batch["fire"],
                )                                            # (B,T,16)
                loss = F.cross_entropy(
                    logits.reshape(-1, N_SLOTS),
                    target.reshape(-1),
                    ignore_index=-100,
                    weight=class_weights,
                )
            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            running += float(loss.item())
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  ep{epoch} batch {n_batches}/{len(train_loader)} "
                      f"loss={running/n_batches:.4f}")

        train_loss = running / max(n_batches, 1)
        all_, a, b, curve = _run_eval(model, val_loader, device)
        elapsed = time.time() - t_ep
        baseline_b = 100.0 * b.pred_zero / max(b.n, 1)  # what % of B did model predict 0?
        # Per-label-slot breakdown of bucket B: how often does the model
        # nail each non-zero slot vs collapse it back to slot 0?
        b_per_label = {}
        for (lt, lp), c in b.confusion.items():
            d = b_per_label.setdefault(lt, {"n": 0, "right": 0, "to_zero": 0})
            d["n"] += c
            if lp == lt:
                d["right"] += c
            if lp == 0:
                d["to_zero"] += c
        per_label_str = "  ".join(
            f"L{lt}: {d['right']}/{d['n']} ({100*d['right']/d['n']:.1f}%, "
            f"->0 {100*d['to_zero']/d['n']:.1f}%)"
            for lt, d in sorted(b_per_label.items())
        )

        print(
            f"epoch {epoch}  loss={train_loss:.4f}  "
            f"val_all={all_.acc():.2f}%  bucket_A(target=0)={a.acc():.2f}%  "
            f"bucket_B(target!=0)={b.acc():.2f}%  "
            f"({b.n} frames; model pred 0 on B: {baseline_b:.1f}%)  "
            f"{elapsed:.1f}s"
        )
        print(f"  bucket B per-label: {per_label_str}")
        print(f"  threshold policy (override slot 0 when argmax!=0 and p>=tau):")
        print(f"    {'tau':>5}  {'A_acc%':>7}  {'B_acc%':>7}  {'overall%':>8}  "
              f"{'overrides':>10}  {'override_acc%':>12}")
        for tau, rec in curve.items():
            A_acc = 100 * rec["A_correct"] / max(rec["A_n"], 1)
            B_acc = 100 * rec["B_correct"] / max(rec["B_n"], 1)
            overall = 100 * (rec["A_correct"] + rec["B_correct"]) / max(rec["total"], 1)
            ov = rec["overrides"]
            ov_acc = 100 * rec["override_correct"] / max(ov, 1)
            print(f"    {tau:>5.2f}  {A_acc:>6.2f}  {B_acc:>6.2f}  {overall:>7.2f}  "
                  f"{ov:>10d}  {ov_acc:>12.2f}")
        log.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_all_acc": all_.acc(), "val_all_n": all_.n,
            "bucket_A_acc": a.acc(), "bucket_A_n": a.n,
            "bucket_B_acc": b.acc(), "bucket_B_n": b.n,
            "bucket_B_pred_zero_pct": baseline_b,
            "bucket_B_per_label": b_per_label,
            "threshold_curve": {str(k): v for k, v in curve.items()},
        })
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": {
                "chunk_len": cfg.chunk_len, "channels": cfg.channels,
                "n_layers": cfg.n_layers, "kernel_size": cfg.kernel_size,
                "dropout": cfg.dropout,
            },
        }, cfg.output / "latest.pt")
        (cfg.output / "log.json").write_text(json.dumps(log, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output",   type=Path, required=True)
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--epochs",    type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--channels",   type=int, default=128)
    p.add_argument("--n-layers",   type=int, default=7)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-shards", type=int, default=None,
                   help="limit train+val to first N shards (smoke test)")
    p.add_argument("--max-shards-train", type=int, default=None)
    p.add_argument("--max-shards-val",   type=int, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--slot0-weight", type=float, default=0.05,
                   help="CE weight on the dominant slot 0 class (down-weight to expose 2.5% minority)")
    p.add_argument("--autocast-dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--token-mask", type=str, default=None,
                   help="Path to a JSON file with a token_mask predicate "
                        "(qnn.bc.token_filter spec). None = no mask.")
    args = p.parse_args()
    token_mask = json.loads(Path(args.token_mask).read_text()) if args.token_mask else None

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output=args.output,
        chunk_len=args.chunk_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        channels=args.channels,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        num_workers=args.num_workers,
        max_shards_train=args.max_shards_train if args.max_shards_train is not None else args.max_shards,
        max_shards_val=args.max_shards_val if args.max_shards_val is not None else args.max_shards,
        seed=args.seed,
        slot0_weight=args.slot0_weight,
        autocast_dtype=args.autocast_dtype,
        token_mask=token_mask,
    )
    train(cfg)


if __name__ == "__main__":
    main()
