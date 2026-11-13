# CoDef anonymous reproducibility artifact

This archive contains the paper-branch source code, experiment protocols, public topology input, graph-level outputs, and analysis scripts needed to inspect the reported controlled-simulation results. It contains no author identities. The Chinanet input is licensed CC BY 4.0 and supplies connectivity/provider labels only; all security annotations and actions are synthetic.

## License note

The Chinanet topology is attributed under CC BY 4.0 in the included source protocol and data record. This review archive does not assign a reuse license to the original CoDef code; the authors should specify the code license in the public release.

## Main included experiments

- Primary 50-node paired comparisons and same-generator hierarchical holdout.
- 100-node follow-up, matched mechanism ablation, public Chinanet single-structure check, and one-hop fixed-policy prefix-offset analysis.
- Static-CoDef beam-ranking ablation on the 60 primary graphs.
- Deterministic replay of the seed-5682 mechanism trace used in Figure 3, including profile-level footholds and response sets.
- Fixed-policy adaptive response-pool sensitivity at base sizes 2/4/8 on 12 selected graphs; portfolios are not reoptimized.
- Newly recorded 200-node/32-action and 1,000-node/100-action stress runs, including right-censored records and worker logs.
- Current-code single-host runtime-repeat records.

The large-catalog stress rerun is a newly specified protocol and does not reproduce older timing rows for which raw records could not be located. In the new rerun, both methods were right-censored at 1,000 nodes under 135 seconds; those results are reported as censored, not as infeasibility.

## Reproduction

Run from the archive root with Python 3.12 and the packages listed in `requirements.txt`. Exact per-batch environments are recorded in the included protocol/admission JSON files. The primary, ablation, public-topology, and large-catalog stress batches used Windows 11, CPython 3.12.7, NetworkX 3.3, PyYAML 6.0.2, and psutil 5.9.0 on an Intel Core i7-14700HX; stress workers were pinned to one logical CPU with numerical-library threads set to one. The separate 18-graph current-code runtime repeat used Windows 11, native CPython 3.12.14, NetworkX 3.6.1, PyYAML 6.0.3, and psutil 7.2.2 on the same CPU model. Do not pool absolute runtimes across these batches; each comparison uses its own recorded protocol.

See `experiments/chinanet_paper_protocol.py` for the deterministic topology/catalog generator. Each result folder contains the specific protocol and hashes where applicable. `ARTIFACT_SHA256.json` contains SHA-256 hashes for archived files other than itself.

Figure 3's seed-5682 trace can be reproduced with `experiments/run_jisa_seed5682_trace.py`; the figure can then be rendered from the archived JSON using `experiments/render_jisa_seed5682_trace_figure.py` (Pillow is listed in `requirements.txt`).

## Stress-run source snapshot

The large-catalog stress run records the SHA-256 of the exact solver source used. That pre-Static-CoDef version is included at `results/chinanet_paper/large_catalog_stress_20261112/source_snapshot/src/optimizer/attacker_uncertainty.py`; its hash is `0e067d8f22a6ecd29bf4d14ce30a5753ae581d9d07be15407138e4e6aa4c63c3`. The current source also contains the later optional static-risk beam-ranking ablation; its default response-aware mode is unchanged.

## Synthetic parameter-sensitivity analysis

The archive includes a predeclared reoptimization probe on the same 60 fixed 50-node graph/edge-risk instances used by the primary panel. It varies synthetic action costs, applicable action effects, and non-ingress asset values by independent deterministic factors in [0.85, 1.15], separately and jointly. These are model-sensitivity draws, not empirical calibration or additional topology samples. The protocol, per-run CSV, raw runner summary/report, publication-facing absolute-change summary/report, runner, summarizer, and integrity-audit script are archived together. After reproducing the batch, run `python experiments/audit_jisa_parameter_sensitivity.py` to check row coverage, paired inputs, source hashes, and exact reproduction of the primary base-condition objectives.
