"""v5.2 scale-specific proxy registry.

Only checkpoints with a known action catalog are registered.  Missing or
unvalidated scales deliberately return an inactive proxy so callers can use
the deterministic search without special-case branching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.optimizer.proxy_model import MarginalGainProxy


DEFAULT_SCALE_CHECKPOINTS = {
    20: "proxy_mlp_4070_pilot.pt",
    50: "proxy_mlp_4070_50_path_pilot.pt",
}


class MultiScaleProxyRegistry:
    """Load at most one calibrated proxy per graph scale."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        checkpoints: dict[int, str] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.checkpoints = dict(checkpoints or DEFAULT_SCALE_CHECKPOINTS)
        self._cache: dict[tuple[int, tuple[str, ...]], MarginalGainProxy] = {}

    def for_graph(
        self,
        graph: nx.DiGraph,
        actions: Sequence[DefenseAction],
    ) -> MarginalGainProxy:
        """Return a scale-specific proxy, cached by scale and action catalog."""
        scale = graph.number_of_nodes()
        action_ids = tuple(action.id for action in actions)
        key = (scale, action_ids)
        if key in self._cache:
            return self._cache[key]
        checkpoint_name = self.checkpoints.get(scale)
        checkpoint = self.model_dir / checkpoint_name if checkpoint_name else self.model_dir / f"proxy_missing_{scale}.pt"
        proxy = MarginalGainProxy(checkpoint, actions)
        self._cache[key] = proxy
        return proxy

    def status(self, graph: nx.DiGraph, actions: Sequence[DefenseAction]) -> dict[str, object]:
        """Expose an auditable gate status without running inference."""
        proxy = self.for_graph(graph, actions)
        catalog_match = tuple(action.id for action in actions) == proxy.checkpoint_action_ids
        return {
            "scale": graph.number_of_nodes(),
            "checkpoint": proxy.checkpoint.name,
            "model_type": proxy.model_type,
            "loaded": proxy.active,
            "compatible": proxy.compatible(graph) and catalog_match,
            "reason": proxy.reason if proxy.active else proxy.reason,
            "checkpoint_scale": proxy.checkpoint_scale,
            "catalog_match": catalog_match,
        }

