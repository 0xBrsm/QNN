"""Mixture-of-vMF look head — continuous, multimodal directional distribution.

The human-likeness alternative to binning the look turn-delta: instead of a
categorical over discretized bins, predict a mixture of K von Mises–Fisher
components over the unit sphere (the turn-delta direction). Each component is a
(weight, mean direction μ, concentration κ). This keeps binning's advantages —
multimodality (one component per flick/hold/track mode; "hold" = a high-κ
component at forward) and multi-scale spread (heterogeneous κ replaces foveated
bin spacing) — WITHOUT quantization, and is PPO-ready (closed-form log-prob).

  mlp_in = features            (for look_cls: MLP(GRU(CLS)); flat trunk readout)
  h      = mlp(features) → per component: 1 mix logit + 3 mean params + 1 κ param

The head emits the mixture params (consumed by the look loss as the vMF NLL) plus
a deterministic look_predict (top-weight component's μ) for diagnostics. ACTING
samples via qnn.model.vmf.mixture_sample → a 3D unit vector for the engine.

vMF is isotropic — it cannot bend the yaw≫pitch ellipse within one component; the
Kent distribution would. See qnn.model.vmf.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.look_head import LookHeadInput, LookHeadOutput
from qnn.model.vmf import KAPPA_MAX, KAPPA_MIN


class PureVMFLookHead(nn.Module):
    """Look head: MLP(features) → mixture of K vMF over S² (no prior, no pointer)."""

    def __init__(self, in_dim: int, d_hidden: int, activation: str, n_components: int = 3) -> None:
        super().__init__()
        self.k = int(n_components)
        # per component: mix logit (1) + mean direction (3) + concentration (1)
        self.mlp = make_head_mlp(in_dim, self.k * 5, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        h = self.mlp(inp.features).reshape(*inp.features.shape[:-1], self.k, 5)
        mix_logits = h[..., 0]                                              # (B*, K)
        mu = h[..., 1:4]
        mu = mu / mu.norm(dim=-1, keepdim=True).clamp(min=1e-6)             # (B*, K, 3) unit
        kappa = F.softplus(h[..., 4]).clamp(KAPPA_MIN, KAPPA_MAX)           # (B*, K) > 0

        # Deterministic readout (diagnostics only — ACTING samples the mixture).
        top = mix_logits.argmax(dim=-1, keepdim=True)                      # (B*, 1)
        look_predict = torch.gather(mu, -2, top.unsqueeze(-1).expand(*top.shape, 3)).squeeze(-2)
        zero = torch.zeros_like(look_predict)
        return LookHeadOutput(
            look_predict=look_predict, look_prior=zero, look_delta=zero,
            look_vmf_mix=mix_logits, look_vmf_mu=mu, look_vmf_kappa=kappa,
        )

    def look_loss(self, logits, look_label, valid, compute_metrics):
        """vMF mixture NLL on the valid look frames (the look-loss hook contract).

        `logits` carries this head's forwarded outputs; `look_label` is the
        normalized unit turn-delta on the valid frames. Emits the additive NLL
        sufficient stats (lookdist_nll_*) so the loop derives look_dll = log(4π) −
        mean_NLL (Δloglik over the uniform-on-sphere baseline). Lives in bench — canonical
        never needs a vMF branch. See qnn.model.vmf.
        """
        from qnn.model.vmf import mixture_log_prob, mixture_nll
        mix = logits["_look_vmf_mix"].reshape(-1, self.k)[valid]
        mu = logits["_look_vmf_mu"].reshape(-1, self.k, 3)[valid]
        kappa = logits["_look_vmf_kappa"].reshape(-1, self.k)[valid]
        loss = mixture_nll(look_label, mix, mu, kappa)
        metrics = {}
        if compute_metrics:
            with torch.no_grad():
                nll_sum = -mixture_log_prob(look_label, mix, mu, kappa).sum()
                metrics["lookdist_nll_sum"] = nll_sum.detach()
                metrics["lookdist_nll_n"] = look_label.new_tensor(float(look_label.shape[0]))
        return loss, metrics
