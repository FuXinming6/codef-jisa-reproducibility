"""Calibration-only contextual selection of an initial/response budget split.

The selector learns a mapping from *pre-attack graph features* to the split
that performed best on separate calibration graphs.  At inference time it
only consumes the unseen graph's features; it never evaluates alternative
splits on that graph.  This is deliberately a small, dependency-free kNN
baseline, not an attack-graph evaluator or a claim of a novel learning method.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Hashable, Mapping, Sequence


Split = Hashable


def extract_budget_context_features(graph: object) -> dict[str, float]:
    """Extract deterministic, pre-attack context features from a graph.

    Features summarize static topology, configured edge-risk values, critical
    asset metadata, and structural ingress-to-critical reachability. The
    function does not inspect attack paths, realized attack observations,
    random seeds, or defense outcomes.
    """
    try:
        node_rows = list(graph.nodes(data=True))  # type: ignore[attr-defined]
        edge_rows = list(graph.edges(data=True))  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("graph must provide NetworkX-compatible nodes(data=True) and edges(data=True)") from exc

    node_ids = [node for node, _ in node_rows]
    node_count = len(node_rows)
    edge_count = len(edge_rows)
    node_data = {node: data for node, data in node_rows}
    in_degrees = {node: 0 for node in node_ids}
    out_degrees = {node: 0 for node in node_ids}
    edge_risks: list[float] = []
    adjacency: dict[object, list[object]] = {node: [] for node in node_ids}
    for source, target, data in edge_rows:
        out_degrees[source] = out_degrees.get(source, 0) + 1
        in_degrees[target] = in_degrees.get(target, 0) + 1
        adjacency.setdefault(source, []).append(target)
        risk = float(data.get("risk", 0.0))
        if not isfinite(risk):
            raise ValueError("graph edge risk values must be finite")
        edge_risks.append(risk)

    ingress = [node for node, data in node_rows if data.get("type") == "internet"]
    if not ingress:
        ingress = [node for node in node_ids if in_degrees.get(node, 0) == 0]
    critical = [node for node, data in node_rows if bool(data.get("critical", False))]

    reachable_pairs = 0
    for source in ingress:
        seen = {source}
        stack = [source]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        reachable_pairs += sum(target in seen for target in critical)
    pair_count = len(ingress) * len(critical)

    mean_edge_risk = sum(edge_risks) / len(edge_risks) if edge_risks else 0.0
    edge_risk_std = (
        sqrt(sum((risk - mean_edge_risk) ** 2 for risk in edge_risks) / len(edge_risks))
        if edge_risks else 0.0
    )
    mean_critical_value = (
        sum(float(node_data[node].get("asset_value", 1.0)) for node in critical) / len(critical)
        if critical else 0.0
    )
    if not isfinite(mean_critical_value):
        raise ValueError("critical asset values must be finite")

    possible_directed_edges = node_count * max(0, node_count - 1)
    return {
        "node_count": float(node_count),
        "edge_count": float(edge_count),
        "edge_density": edge_count / possible_directed_edges if possible_directed_edges else 0.0,
        "mean_in_degree": sum(in_degrees.values()) / node_count if node_count else 0.0,
        "mean_out_degree": sum(out_degrees.values()) / node_count if node_count else 0.0,
        "ingress_count": float(len(ingress)),
        "critical_count": float(len(critical)),
        "critical_asset_share": len(critical) / node_count if node_count else 0.0,
        "ingress_critical_pair_reachability": reachable_pairs / pair_count if pair_count else 0.0,
        "mean_edge_risk": mean_edge_risk,
        "max_edge_risk": max(edge_risks, default=0.0),
        "edge_risk_std": edge_risk_std,
        "mean_critical_asset_value": mean_critical_value,
    }


@dataclass(frozen=True)
class BudgetSplitExample:
    """One calibration graph and its measured risk for each candidate split.

    ``features`` must be available before an attack is observed. ``risks``
    should contain outcomes measured by the same evaluator for every candidate
    split on this calibration graph. Do not construct examples from test-graph
    split evaluations.
    """

    features: Mapping[str, float]
    risks: Mapping[Split, float]


class ContextualBudgetSplitSelector:
    """Choose a budget split by standardized, inverse-distance kNN regression.

    Each split's risk is predicted independently from the ``k`` nearest
    calibration graphs, then the split with the lowest predicted risk is
    selected. Feature means and population standard deviations are learned
    from calibration features only. Exact feature matches are averaged among
    themselves, avoiding arbitrary influence from nearby points.

    The public inference API accepts features only, making it impossible for
    this class to run candidate-split evaluations on the test graph.
    """

    def __init__(self, n_neighbors: int = 5, *, distance_epsilon: float = 1e-12) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if not isfinite(distance_epsilon) or distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be a finite positive number")
        self.n_neighbors = int(n_neighbors)
        self.distance_epsilon = float(distance_epsilon)
        self._feature_names: tuple[str, ...] = ()
        self._means: tuple[float, ...] = ()
        self._scales: tuple[float, ...] = ()
        self._features: tuple[tuple[float, ...], ...] = ()
        self._risks: tuple[Mapping[Split, float], ...] = ()
        self._splits: tuple[Split, ...] = ()
        self._fitted = False

    @staticmethod
    def _numeric_features(features: Mapping[str, float], names: Sequence[str] | None = None) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if not isinstance(features, Mapping) or not features:
            raise ValueError("features must be a non-empty mapping of pre-attack numeric features")
        feature_names = tuple(sorted(features)) if names is None else tuple(names)
        if set(features) != set(feature_names):
            missing = sorted(set(feature_names) - set(features))
            extra = sorted(set(features) - set(feature_names))
            raise ValueError(f"feature keys differ from calibration schema (missing={missing}, extra={extra})")
        values: list[float] = []
        for name in feature_names:
            try:
                value = float(features[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"feature {name!r} must be numeric") from exc
            if not isfinite(value):
                raise ValueError(f"feature {name!r} must be finite")
            values.append(value)
        return feature_names, tuple(values)

    def fit(self, examples: Sequence[BudgetSplitExample]) -> "ContextualBudgetSplitSelector":
        """Fit from calibration-graph feature/outcome records only."""
        if not examples:
            raise ValueError("at least one calibration example is required")

        first_names, _ = self._numeric_features(examples[0].features)
        split_order = tuple(examples[0].risks.keys())
        if not split_order:
            raise ValueError("each calibration example must include at least one candidate split")
        if len(set(split_order)) != len(split_order):
            raise ValueError("candidate split identifiers must be unique")

        vectors: list[tuple[float, ...]] = []
        risk_rows: list[dict[Split, float]] = []
        for index, example in enumerate(examples):
            _, vector = self._numeric_features(example.features, first_names)
            if set(example.risks) != set(split_order):
                raise ValueError(f"calibration example {index} has a different candidate split set")
            normalized_risks: dict[Split, float] = {}
            for split in split_order:
                try:
                    risk = float(example.risks[split])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"risk for split {split!r} must be numeric") from exc
                if not isfinite(risk):
                    raise ValueError(f"risk for split {split!r} must be finite")
                normalized_risks[split] = risk
            vectors.append(vector)
            risk_rows.append(normalized_risks)

        means = tuple(sum(row[j] for row in vectors) / len(vectors) for j in range(len(first_names)))
        scales_list: list[float] = []
        for j, mean in enumerate(means):
            variance = sum((row[j] - mean) ** 2 for row in vectors) / len(vectors)
            scale = sqrt(variance)
            scales_list.append(scale if scale > self.distance_epsilon else 1.0)

        self._feature_names = first_names
        self._means = means
        self._scales = tuple(scales_list)
        self._features = tuple(vectors)
        self._risks = tuple(risk_rows)
        self._splits = split_order
        self._fitted = True
        return self

    def predict_risks(self, features: Mapping[str, float]) -> dict[Split, float]:
        """Estimate split risks from calibration neighbors for an unseen graph."""
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
        _, query = self._numeric_features(features, self._feature_names)
        standardized_query = tuple((value - mean) / scale for value, mean, scale in zip(query, self._means, self._scales))
        distances: list[tuple[float, int]] = []
        for index, row in enumerate(self._features):
            standardized_row = tuple((value - mean) / scale for value, mean, scale in zip(row, self._means, self._scales))
            distance = sqrt(sum((left - right) ** 2 for left, right in zip(standardized_query, standardized_row)))
            distances.append((distance, index))
        distances.sort(key=lambda item: (item[0], item[1]))

        exact = [index for distance, index in distances if distance <= self.distance_epsilon]
        if exact:
            neighbors = [(1.0, index) for index in exact]
        else:
            neighbors = [
                (1.0 / max(distance, self.distance_epsilon), index)
                for distance, index in distances[: min(self.n_neighbors, len(distances))]
            ]
        total_weight = sum(weight for weight, _ in neighbors)
        return {
            split: sum(weight * self._risks[index][split] for weight, index in neighbors) / total_weight
            for split in self._splits
        }

    def predict(self, features: Mapping[str, float]) -> Split:
        """Return the lowest predicted-risk split; ties follow calibration order."""
        estimates = self.predict_risks(features)
        return min(self._splits, key=lambda split: (estimates[split], self._splits.index(split)))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The fixed pre-attack feature schema learned from calibration data."""
        if not self._fitted:
            raise RuntimeError("fit must be called before reading the feature schema")
        return self._feature_names

    @property
    def calibration_size(self) -> int:
        """Number of calibration graphs retained by the fitted selector."""
        if not self._fitted:
            raise RuntimeError("fit must be called before reading calibration size")
        return len(self._features)
