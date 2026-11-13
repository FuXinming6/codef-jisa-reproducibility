# JISA synthetic parameter sensitivity

The analysis re-optimizes the same 60 fixed 50-node topology/edge-risk instances under the registered parameter perturbations. These are sensitivity draws, not 60 additional topologies, empirical attribute estimates, or confirmatory tests.

Pairwise delta is CoDef objective minus PairStart-Greedy under the same perturbed evaluator; lower favors CoDef. Ties use an absolute tolerance of 1e-12. Relative objective changes compare each method with its own base-condition result on the same graph.

|Condition|N|CoDef lower / tie / higher|Mean paired ΔJ|Median paired ΔJ|Median relative ΔJ CoDef|Median relative ΔJ PairStart|
|---|---:|---:|---:|---:|---:|---:|
|base|60|2/57/1|-0.003100|0.000000|0.00%|0.00%|
|cost_only|60|1/58/1|0.000246|0.000000|0.00%|0.00%|
|effect_only|60|3/56/1|-0.002682|0.000000|0.00%|0.00%|
|asset_only|60|3/55/2|-0.000451|0.000000|4.36%|3.51%|
|joint|60|1/57/2|-0.000790|0.000000|3.27%|3.07%|

No inferential p-values are reported: the perturbed parameter sets are deterministic model-sensitivity probes on repeated base graphs, not independent samples from an estimated population.
