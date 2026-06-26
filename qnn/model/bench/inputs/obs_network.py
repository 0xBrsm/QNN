"""Bench Network wrapper that enters the promoted ObsAccessor scope."""

from __future__ import annotations

from qnn.model.network import Network
from qnn.model.tokens.obs_accessor import obs_accessor_scope_from_obs


class BenchObsNetwork(Network):
    """Network that enters the ObsAccessor scope around ``forward``."""

    def forward(  # type: ignore[override]
        self,
        obs,
        hidden=None,
        reset_mask=None,
    ):
        with obs_accessor_scope_from_obs(obs):
            return super().forward(obs=obs, hidden=hidden, reset_mask=reset_mask)
