"""Audit the completed JISA synthetic parameter-sensitivity experiment."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "chinanet_paper" / "cost_effect_value_sensitivity_20261112"
RAW = OUT / "runs.csv"
PROTOCOL = OUT / "protocol.json"
SUMMARY = OUT / "summary.json"
PAPER_SUMMARY = OUT / "paper_summary.json"
PRIMARY = ROOT / "results" / "chinanet_paper" / "chinanet_main_50_20seed_runs.csv"
TOLERANCE = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    paper_summary = json.loads(PAPER_SUMMARY.read_text(encoding="utf-8"))
    with RAW.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with PRIMARY.open(newline="", encoding="utf-8") as stream:
        primary_rows = list(csv.DictReader(stream))

    families = protocol["families"]
    seeds = [int(seed) for seed in protocol["seeds"]]
    conditions = protocol["conditions"]
    methods = protocol["methods"]
    expected_keys = set(itertools.product(families, seeds, conditions, methods))
    observed_keys = [
        (row["family"], int(row["seed"]), row["condition"], row["method"])
        for row in rows
    ]
    counts = Counter(observed_keys)
    duplicate_keys = [key for key, count in counts.items() if count != 1]
    missing_keys = sorted(expected_keys - set(observed_keys))
    unexpected_keys = sorted(set(observed_keys) - expected_keys)

    paired_by_input: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    paired_parameter_mismatches = 0
    invalid_numeric_rows = 0
    out_of_range_factors = 0
    baseline_lookup = {
        (row["family"], int(row["seed"]), row["method"]): float(row["objective"])
        for row in primary_rows
        if row["method"] in {"CoDef", "PairStart-Greedy"}
    }
    base_comparisons = 0
    base_mismatches: list[dict[str, object]] = []

    def check_factor(factor: float, should_vary: bool) -> bool:
        if should_vary:
            return 0.85 - 1e-12 <= factor <= 1.15 + 1e-12
        return abs(factor - 1.0) <= 1e-12

    for row in rows:
        try:
            objective = float(row["objective"])
            runtime = float(row["runtime_seconds"])
            factors = json.loads(row["parameter_values_json"])
            if not math.isfinite(objective) or not math.isfinite(runtime) or runtime <= 0:
                invalid_numeric_rows += 1
            condition = row["condition"]
            cost_varies = condition in {"cost_only", "joint"}
            effect_varies = condition in {"effect_only", "joint"}
            asset_varies = condition in {"asset_only", "joint"}
            for value in factors["cost_multipliers"].values():
                if not check_factor(float(value), cost_varies):
                    out_of_range_factors += 1
            for value in factors["effect_multipliers"].values():
                if not check_factor(float(value), effect_varies):
                    out_of_range_factors += 1
            for value in factors["asset_value_multipliers"].values():
                if not check_factor(float(value), asset_varies):
                    out_of_range_factors += 1
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            invalid_numeric_rows += 1
            continue

        pair_key = (row["family"], int(row["seed"]), condition)
        paired_by_input[pair_key].append(row)
        if condition == "base":
            key = (row["family"], int(row["seed"]), row["method"])
            reference = baseline_lookup.get(key)
            if reference is None:
                base_mismatches.append({"key": key, "reason": "missing primary reference"})
            else:
                base_comparisons += 1
                delta = abs(objective - reference)
                if delta > TOLERANCE:
                    base_mismatches.append({"key": key, "absolute_delta": delta})

    for pair_rows in paired_by_input.values():
        if len(pair_rows) != 2:
            paired_parameter_mismatches += 1
            continue
        if pair_rows[0]["input_fingerprint"] != pair_rows[1]["input_fingerprint"]:
            paired_parameter_mismatches += 1
        if pair_rows[0]["parameter_values_json"] != pair_rows[1]["parameter_values_json"]:
            paired_parameter_mismatches += 1

    source_mismatches = []
    for relative_path, expected_hash in protocol["source_sha256"].items():
        source_path = Path(relative_path)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if not source_path.exists() or sha256(source_path) != expected_hash:
            source_mismatches.append(relative_path)
    primary_recorded_hash = next(
        (
            expected_hash
            for relative_path, expected_hash in protocol["source_sha256"].items()
            if Path(relative_path).as_posix().replace("\\", "/")
            == "results/chinanet_paper/chinanet_main_50_20seed_runs.csv"
        ),
        None,
    )
    primary_hash_matches = primary_recorded_hash is not None and sha256(PRIMARY) == primary_recorded_hash

    summary_rows = summary.get("raw_rows")
    summary_expected = summary.get("expected_raw_rows")
    checks = {
        "physical_rows_complete": len(rows) == len(expected_keys),
        "unique_expected_keys": not duplicate_keys and not missing_keys and not unexpected_keys,
        "method_pairs_share_identical_inputs": paired_parameter_mismatches == 0,
        "finite_objectives_and_positive_runtimes": invalid_numeric_rows == 0,
        "perturbation_factors_obey_protocol": out_of_range_factors == 0,
        "base_reproduces_primary_objectives": base_comparisons == len(families) * len(seeds) * len(methods)
        and not base_mismatches,
        "protocol_source_hashes_match": not source_mismatches,
        "primary_input_hash_matches_protocol": primary_hash_matches,
        "summary_matches_complete_raw_data": summary_rows == len(rows) == summary_expected,
        "publication_summary_matches_complete_raw_data": paper_summary.get("raw_rows") == len(rows) == paper_summary.get("expected_raw_rows"),
        "protocol_declares_frozen_design": protocol.get("design_frozen_before_execution") is True,
    }
    verification = {
        "experiment": protocol["experiment"],
        "passed": all(checks.values()),
        "checks": checks,
        "raw_rows": len(rows),
        "expected_rows": len(expected_keys),
        "base_primary_comparisons": base_comparisons,
        "maximum_base_absolute_difference": max(
            [
                abs(
                    float(row["objective"])
                    - baseline_lookup[(row["family"], int(row["seed"]), row["method"])]
                )
                for row in rows
                if row["condition"] == "base"
                and (row["family"], int(row["seed"]), row["method"]) in baseline_lookup
            ]
            or [0.0]
        ),
        "duplicate_keys": duplicate_keys,
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "base_mismatches": base_mismatches,
        "paired_input_mismatch_count": paired_parameter_mismatches,
        "invalid_numeric_row_count": invalid_numeric_rows,
        "out_of_range_factor_count": out_of_range_factors,
        "source_hash_mismatches": source_mismatches,
        "primary_input_hash_matches": primary_hash_matches,
        "tolerance": TOLERANCE,
    }
    out_path = OUT / "verification.json"
    out_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2, ensure_ascii=False))
    if not verification["passed"]:
        raise SystemExit("parameter-sensitivity integrity audit failed")


if __name__ == "__main__":
    main()
