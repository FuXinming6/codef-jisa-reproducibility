# Public-topology structural validation

## Scope

This stress test uses the 42-node, 66-link `Chinanet` structure from Internet Topology Zoo. The public data provide connectivity and provider labels only. Edge risks, asset values, security roles, critical assets, attack direction, and defense actions are deterministic synthetic annotations. The result is therefore **not** a real China Telecom attack graph or a field validation.

Dataset: https://doi.org/10.25909/30153949.v1; license: CC BY 4.0. S. Knight, H. X. Nguyen, N. Falkner, R. Bowden, and M. Roughan, The Internet Topology Zoo, IEEE JSAC 29(9), 2011, doi:10.1109/JSAC.2011.111002.

The graph is mapped once to a 66-edge DAG with four provider-labelled international ingress nodes and two fixed central critical nodes (Xi'an and Tianjin). Thirty predeclared seeds perturb edge risk only. Every method receives the same graph instance, 12 actions, 4:4 budget, path cap 4, response pool 2, constraints, and three attacker profiles. Seeds measure sensitivity on one fixed structure; they are not 30 independent real networks.

## Paired results

Negative delta means CoDef has lower (better) objective. Numerical ties use 1e-12; practical equivalence uses |delta| <= 0.001. Bootstrap intervals are paired across the 30 risk perturbations. Sign tests omit numerical ties.

|Comparator|N|CoDef win/tie/loss|Practically equivalent|Mean delta|Median delta|95% paired bootstrap CI|Exact sign p (Holm)|Runtime ratio|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Greedy|30|4/26/0|27|-0.00027766|0.00000000|[-0.00063248, -0.00001589]|0.125 (0.5)|23.29x|
|RiskTopK|30|27/3/0|3|-0.03410205|-0.03187191|[-0.04251892, -0.02575554]|1.49012e-08 (7.45058e-08)|36.24x|
|MultiStart-Greedy|30|1/29/0|29|-0.00003769|0.00000000|[-0.00011307, 0.00000000]|1 (1)|3.16x|
|PairStart-Greedy|30|0/30/0|30|0.00000000|0.00000000|[0.00000000, 0.00000000]|1 (1)|0.97x|
|Exact|30|0/30/0|30|0.00000000|0.00000000|[0.00000000, 0.00000000]|1 (1)|0.53x|

## Interpretation boundary

Against the strongest heuristic PairStart-Greedy, CoDef records 0/30/0 wins/ties/losses. Against Exact, it records 0/30/0. If these comparisons are dominated by ties, the supported claim is portability to a public external structure and exact/strong-baseline parity under this protocol—not superior solution quality.

This experiment complements rather than replaces the three-family synthetic benchmark. It reduces the risk that the implementation only works on the project's own graph generator, while leaving real vulnerability calibration and operational validation as future work.
