"""Produce descriptive family summaries from the current-code runtime repeat."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "graph_medians.csv").open(newline="", encoding="utf-8") as stream:
        graph_rows = list(csv.DictReader(stream))
    if len(graph_rows) != 18:
        raise RuntimeError(f"Expected 18 graph medians, got {len(graph_rows)}")

    output = []
    family_order = ("er_dag", "hierarchical", "multipath")
    metrics = ("wall_seconds", "process_cpu_seconds")
    for family_index, family in enumerate(family_order):
        selected = [row for row in graph_rows if row["family"] == family]
        if len(selected) != 6:
            raise RuntimeError(f"Expected six graphs in {family}, got {len(selected)}")
        for metric_index, metric in enumerate(metrics):
            ratios = [float(row["pairstart_" + metric]) / float(row["codef_" + metric])
                      for row in selected]
            deltas = [float(row["codef_" + metric]) - float(row["pairstart_" + metric])
                      for row in selected]
            rng = random.Random(2026100502 + family_index * 10 + metric_index)
            boot = []
            for _ in range(10000):
                picks = [rng.randrange(6) for _ in range(6)]
                boot.append(math.exp(statistics.fmean(math.log(ratios[i]) for i in picks)))
            output.append({
                "family": family,
                "metric": metric,
                "graphs": 6,
                "geometric_pairstart_over_codef": math.exp(statistics.fmean(map(math.log, ratios))),
                "pointwise_graph_bootstrap_ci_95": [quantile(boot, .025), quantile(boot, .975)],
                "codef_faster_tied_slower": [
                    sum(delta < -1e-9 for delta in deltas),
                    sum(abs(delta) <= 1e-9 for delta in deltas),
                    sum(delta > 1e-9 for delta in deltas),
                ],
                "mean_codef_minus_pairstart_seconds": statistics.fmean(deltas),
            })
    (run_dir / "family_summary.json").write_text(
        json.dumps({"descriptive_only": True, "bootstrap_draws": 10000,
                    "base_seed": 2026100502, "results": output}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
