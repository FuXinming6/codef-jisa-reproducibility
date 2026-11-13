"""Optional inference adapter for the trained marginal-gain proxy.

The proxy only ranks feasible candidate actions.  The real two-stage
evaluator remains responsible for selecting and validating the final defense
portfolio.  This module is deliberately optional so the offline demo still
works when PyTorch or the checkpoint is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import portfolio_cost

try:  # PyTorch is an optional demo acceleration dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on minimal installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class _ProxyMLP(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int) -> None:
        if nn is None:  # pragma: no cover - guarded by MarginalGainProxy
            return
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class MarginalGainProxy:
    """Load and apply the v2.7 20-node marginal-gain checkpoint."""

    def __init__(self, checkpoint: str | Path, actions: Sequence[DefenseAction]) -> None:
        self.checkpoint = Path(checkpoint)
        self.actions = tuple(actions)
        self.model = None
        self.feature_names: list[str] = []
        self.feature_means: list[float] = []
        self.feature_scales: list[float] = []
        self.model_type = "unknown"
        self.checkpoint_action_ids: tuple[str, ...] = ()
        self.checkpoint_scale: int | None = None
        self.reason = "not_loaded"
        if torch is None:
            self.reason = "torch_unavailable"
            return
        if not self.checkpoint.exists():
            self.reason = "checkpoint_missing"
            return
        try:
            try:
                payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            except TypeError:  # PyTorch versions before weights_only existed.
                payload = torch.load(self.checkpoint, map_location="cpu")
            names = payload.get("feature_names", [])
            means = payload.get("feature_means", [])
            scales = payload.get("feature_scales", [])
            if not names or len(names) != len(means) or len(names) != len(scales):
                raise ValueError("checkpoint feature metadata is incomplete")
            model = _ProxyMLP(len(names))
            model.load_state_dict(payload["state_dict"])
            model.eval()
            self.model = model
            self.model_type = str(payload.get("model_type", "marginal_gain_mlp"))
            self.feature_names = [str(name) for name in names]
            self.feature_means = [float(value) for value in means]
            self.feature_scales = [max(float(value), 1e-8) for value in scales]
            candidate_ids = tuple(
                name.removeprefix("candidate_id_")
                for name in self.feature_names
                if name.startswith("candidate_id_")
            )
            self.checkpoint_action_ids = candidate_ids
            raw_scale = payload.get("scale")
            if raw_scale is not None:
                self.checkpoint_scale = int(raw_scale)
            else:
                self.checkpoint_scale = {12: 20, 25: 50}.get(len(candidate_ids))
            self.reason = "ready"
        except Exception as error:  # pragma: no cover - surfaced in diagnostics
            self.model = None
            self.reason = f"load_error:{error}"

    @property
    def active(self) -> bool:
        return self.model is not None

    def compatible(self, graph: nx.DiGraph) -> bool:
        """Require the graph scale encoded by the checkpoint when available."""
        if not self.active:
            return False
        return self.checkpoint_scale is None or graph.number_of_nodes() == self.checkpoint_scale

    @staticmethod
    def _action_signal(graph: nx.DiGraph, action: DefenseAction) -> tuple[float, float, float]:
        if isinstance(action.target, tuple):
            source, target = action.target
            edge = graph.get_edge_data(source, target, {})
            local = float(edge.get("risk", 0.0))
            asset = float(graph.nodes[target].get("asset_value", 0.0)) if target in graph else 0.0
            degree = float(graph.degree(target)) if target in graph else 0.0
            return local, asset, degree
        node = str(action.target)
        if node not in graph:
            return 0.0, 0.0, 0.0
        local = sum(float(data.get("risk", 0.0)) for _, _, data in graph.in_edges(node, data=True))
        local += sum(float(data.get("risk", 0.0)) for _, _, data in graph.out_edges(node, data=True))
        return local, float(graph.nodes[node].get("asset_value", 0.0)), float(graph.degree(node))

    def _features(self, graph: nx.DiGraph, current_ids: frozenset[str], candidate: DefenseAction) -> list[float]:
        chosen = [action for action in self.actions if action.id in current_ids]
        local, asset, degree = self._action_signal(graph, candidate)
        type_value = getattr(candidate.action_type, "value", str(candidate.action_type))
        values: dict[str, float] = {
            "node_count": float(graph.number_of_nodes()),
            "edge_count": float(graph.number_of_edges()),
            "total_asset_value": sum(float(data.get("asset_value", 0.0)) for _, data in graph.nodes(data=True)),
            "mean_edge_risk": sum(float(data.get("risk", 0.0)) for _, _, data in graph.edges(data=True)) / max(graph.number_of_edges(), 1),
            "state_action_count": float(len(chosen)),
            "state_cost": float(portfolio_cost(chosen)),
            "state_effect_sum": sum(float(action.effect) for action in chosen),
            "state_delay_mean": sum(float(action.response_delay) for action in chosen) / max(len(chosen), 1),
            "candidate_cost": float(candidate.cost),
            "candidate_effect": float(candidate.effect),
            "candidate_delay": float(candidate.response_delay),
            "candidate_local_risk": local,
            "candidate_asset_value": asset,
            "candidate_degree": degree,
            "candidate_is_edge": float(isinstance(candidate.target, tuple)),
            "candidate_is_node_hardening": float(type_value == "node_hardening"),
            "candidate_is_edge_restriction": float(type_value == "edge_restriction"),
            "candidate_is_node_isolation": float(type_value == "node_isolation"),
            "scale": float(graph.number_of_nodes()),
        }
        values.update({f"state_selected_{action.id}": float(action.id in current_ids) for action in self.actions})
        values.update({f"candidate_id_{action.id}": float(action.id == candidate.id) for action in self.actions})
        return [values.get(name, 0.0) for name in self.feature_names]

    def rank_actions(self, graph: nx.DiGraph, current_ids: frozenset[str] = frozenset()) -> tuple[list[str], dict[str, float]]:
        """Return action IDs in predicted marginal-gain order and raw scores."""
        if not self.compatible(graph):
            return [action.id for action in self.actions], {}
        matrix = [self._features(graph, current_ids, action) for action in self.actions]
        normalized = [
            [(value - mean) / scale for value, mean, scale in zip(row, self.feature_means, self.feature_scales)]
            for row in matrix
        ]
        with torch.no_grad():
            scores = self.model(torch.tensor(normalized, dtype=torch.float32)).tolist()
        ranking = sorted(zip(self.actions, scores), key=lambda item: (-float(item[1]), item[0].id))
        return [action.id for action, _ in ranking], {action.id: round(float(score), 6) for action, score in ranking}
