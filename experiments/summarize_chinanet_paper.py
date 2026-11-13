"""Summarize paired paper runs with bootstrap intervals and exact sign tests."""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def _bootstrap_ci(values: list[float], seed: int, samples: int = 4000) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def _sign_test_p(wins: int, losses: int) -> float:
    count = wins + losses
    if count == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(count, index) for index in range(tail + 1)) / (2 ** count)
    return min(1.0, 2.0 * probability)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_case: dict[tuple[str, int], dict[str, dict[int, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("method"):
            key = (row["family"], int(row["scale"]))
            by_case[key][row["method"]][int(row["seed"])] = row
    summaries: list[dict[str, object]] = []
    for family, scale in sorted(by_case):
        methods = by_case[(family, scale)]
        codef = methods.get("CoDef", {})
        for baseline in ("Greedy", "RiskTopK", "MultiStart-Greedy", "PairStart-Greedy"):
            comparator = methods.get(baseline, {})
            seeds = sorted(set(codef) & set(comparator))
            if not seeds:
                continue
            deltas = [float(codef[seed]["objective"]) - float(comparator[seed]["objective"]) for seed in seeds]
            time_ratios = [float(codef[seed]["runtime_seconds"]) / max(float(comparator[seed]["runtime_seconds"]), 1e-12) for seed in seeds]
            wins = sum(delta < -1e-12 for delta in deltas)
            ties = sum(abs(delta) <= 1e-12 for delta in deltas)
            losses = sum(delta > 1e-12 for delta in deltas)
            ci_low, ci_high = _bootstrap_ci(deltas, seed=scale * 10_000 + sum(ord(char) for char in family + baseline))
            summaries.append({
                "family": family,
                "scale": scale,
                "baseline": baseline,
                "paired_seeds": len(seeds),
                "codef_mean_objective": round(statistics.mean(float(codef[seed]["objective"]) for seed in seeds), 8),
                "baseline_mean_objective": round(statistics.mean(float(comparator[seed]["objective"]) for seed in seeds), 8),
                "mean_paired_delta_codef_minus_baseline": round(statistics.mean(deltas), 8),
                "delta_bootstrap_95ci_low": round(ci_low, 8),
                "delta_bootstrap_95ci_high": round(ci_high, 8),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "exact_sign_test_p": round(_sign_test_p(wins, losses), 8),
                "mean_codef_runtime_seconds": round(statistics.mean(float(codef[seed]["runtime_seconds"]) for seed in seeds), 6),
                "mean_baseline_runtime_seconds": round(statistics.mean(float(comparator[seed]["runtime_seconds"]) for seed in seeds), 6),
                "mean_runtime_ratio": round(statistics.mean(time_ratios), 3),
                "synthetic_data_only": True,
            })
    # Control family-wise error over all topology-by-baseline sign tests.
    ordered = sorted(range(len(summaries)), key=lambda index: summaries[index]["exact_sign_test_p"])
    running_max = 0.0
    test_count = len(ordered)
    for rank, index in enumerate(ordered, start=1):
        raw_p = float(summaries[index]["exact_sign_test_p"])
        running_max = max(running_max, min(1.0, (test_count - rank + 1) * raw_p))
        summaries[index]["holm_adjusted_p"] = round(running_max, 8)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", default="summary")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize(rows)
    if not summary:
        raise SystemExit("No paired CoDef/baseline results found")
    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / f"chinanet_{args.tag}_summary.csv"
    md_out = out_dir / f"chinanet_{args.tag}_report.md"
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# ChinaNet 2026 CoDef 实验汇总",
        "",
        "风险目标越低越好。每行比较同一拓扑、规模和随机种子上的配对结果；bootstrap 区间基于逐种子目标差值。符号检验忽略平局。当前拓扑为合成仿真数据，不代表真实企业网络测量。",
        "",
        "|拓扑|规模|基线|配对种子|CoDef 胜/平/负|平均配对差 (CoDef−基线)|95% bootstrap CI|精确符号检验 p（Holm 校正）|CoDef/基线耗时比|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"|{row['family']}|{row['scale']}|{row['baseline']}|{row['paired_seeds']}|{row['wins']}/{row['ties']}/{row['losses']}|"
            f"{row['mean_paired_delta_codef_minus_baseline']}|[{row['delta_bootstrap_95ci_low']}, {row['delta_bootstrap_95ci_high']}]|"
            f"{row['exact_sign_test_p']} ({row['holm_adjusted_p']})|{row['mean_runtime_ratio']}x|"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- `CoDef` 将普通 Greedy 方案纳入保底候选，因此对 Greedy 的不劣性是实现设计保证；质量优势应重点看 MultiStart-Greedy 与 PairStart-Greedy。",
        "- 小样本下 p 值和置信区间仅作初步描述；投稿前应完成预先设定的独立种子规模，并报告所有拓扑族和规模。",
        "- 所有拓扑均为可复现的合成模型，结论仅适用于本文拓扑生成规则、动作目录和预算设定。",
        "- 本比较是配对实验；若输入 CSV 包含部分方法或部分种子，汇总只使用两方法共有的种子。",
    ])
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {csv_out}")
    print(f"Saved {md_out}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
