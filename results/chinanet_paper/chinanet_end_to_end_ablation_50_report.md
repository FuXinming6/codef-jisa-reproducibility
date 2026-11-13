# ChinaNet 50-node end-to-end mechanism ablation

Each condition re-runs the same bounded CoDef initial-portfolio search on the exact same generated instance. The independent unit is one family/seed graph, not one attacker profile.

Expected 20 seeds/family; completed paired instances: er_dag=20, hierarchical=20, multipath=20. Complete default sweep: True.
Scale=50; total budget=8; phase-one path cap=4; adaptive response-pool base size=2; beam width=2; profiles=risk_max, shortest_path, asset_targeted; objective=mean + 0.5*(worst−mean) + 0.2*sample-SD.
Git HEAD=4c40ffdfcc9e5dd78af0a802a23724fbff8ab717; source SHA-256=39cb9b5cf5df7216336cf34cf6ac729614dbfc58d385656e00386e700ba04580.

## Conditions

- `full`: B1/B2=4:4, local response-action eligibility, attacker continuation replanning.
- `no_reserve`: B1/B2=8:0, local response-action eligibility, replanning. Total budget is unchanged; the full budget is available before attack, but the optimizer is not forced to spend every unit.
- `no_replanning`: B1/B2=4:4 and local response-action eligibility, with no post-response route expansion. For each observed prefix, one response is chosen robustly against the worst case in the capped set of committed suffixes sharing that prefix; the response is not selected for a single hidden suffix. Blocked attempted suffixes receive zero terminal risk.
- `global_action_scope`: B1/B2=4:4 and replanning, but response actions may target any catalog item. This is a response-authority relaxation, not a deployment-realistic baseline or an information-only change. Because the eligible catalog is larger, adaptive-diverse re-ranking/truncation can also change the effective response candidate pool.

All arms preserve the same graph generator and edge-risk perturbation seeds as the main 50-node benchmark (`graph_seed=seed+50*1009+family_index*7919`; risk seed `seed+200003`), action catalog, constraints, evaluator, path cap, response-pool ranking/truncation, and bounded beam search. `no_reserve` compares budget timing/allocation, not the abstract value of additional total resources.

## Paired results

Delta is `ablation − full`; positive means the ablation has a higher/worse objective. Intervals are nominal, pointwise 95% paired percentile-bootstrap CIs for the mean delta over graph seeds; they are not multiplicity-adjusted. Exact sign tests of paired directions omit ties and are Holm-corrected over the nine planned family-by-ablation contrasts. The CIs and sign tests target different summaries/hypotheses; use the Holm-adjusted tests for planned directional decisions and the CIs as effect-size uncertainty.

|Topology|Ablation|N|Full mean|Ablation mean|Mean Δ|Median Δ|95% paired bootstrap CI|Ablation better/tie/worse|Sign p (Holm)|Ablation mean runtime (s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|er_dag|no_reserve|20|0.32254450|0.35578465|0.03324015|0.02250508|[0.01747653, 0.05121176]|1/5/14|0.000976562 (0.0078125)|4.0747|
|er_dag|no_replanning|20|0.32254450|0.31919056|-0.00335394|0.00000000|[-0.01990468, 0.01233372]|7/8/5|0.774414 (1)|2.2444|
|er_dag|global_action_scope|20|0.32254450|0.32142110|-0.00112340|0.00000000|[-0.00271067, 0.00000925]|5/14/1|0.21875 (1)|5.4967|
|hierarchical|no_reserve|20|0.11353852|0.12767998|0.01414146|0.00591561|[-0.00815622, 0.03769828]|8/0/12|0.503445 (1)|2.0595|
|hierarchical|no_replanning|20|0.11353852|0.10499314|-0.00854537|0.00000000|[-0.01948563, -0.00014220]|5/12/3|0.726562 (1)|1.2595|
|hierarchical|global_action_scope|20|0.11353852|0.07592073|-0.03761779|-0.00948309|[-0.06111997, -0.01684069]|14/5/1|0.000976562 (0.0078125)|2.1948|
|multipath|no_reserve|20|0.19820173|0.22883380|0.03063206|0.01124494|[0.01028094, 0.05685022]|3/1/16|0.00442505 (0.0265503)|21.4746|
|multipath|no_replanning|20|0.19820173|0.15226531|-0.04593642|-0.02449000|[-0.08077604, -0.01611859]|11/4/5|0.210114 (1)|11.8690|
|multipath|global_action_scope|20|0.19820173|0.15724634|-0.04095539|-0.04252824|[-0.05587865, -0.02637315]|15/5/0|6.10352e-05 (0.000549316)|13.9580|

## Per-profile mean terminal risk

|Topology|Variant|N|risk_max|shortest_path|asset_targeted|
|---|---|---:|---:|---:|---:|
|er_dag|full|20|0.32699403|0.22631893|0.26239404|
|er_dag|no_reserve|20|0.35987242|0.35152017|0.29305535|
|er_dag|no_replanning|20|0.31352101|0.32630864|0.23573355|
|er_dag|global_action_scope|20|0.32334877|0.22719258|0.25770150|
|hierarchical|full|20|0.09438725|0.05910666|0.10445526|
|hierarchical|no_reserve|20|0.13072786|0.10224629|0.11418240|
|hierarchical|no_replanning|20|0.08035198|0.07777120|0.09093659|
|hierarchical|global_action_scope|20|0.05510224|0.04294352|0.07077541|
|multipath|full|20|0.20551237|0.07379871|0.17293938|
|multipath|no_reserve|20|0.23011053|0.23011053|0.20517264|
|multipath|no_replanning|20|0.12891827|0.12891827|0.12601419|
|multipath|global_action_scope|20|0.15981192|0.06340240|0.14774389|

## Control reproduction and instance checks

Each arm stores graph, action-catalog, constraint, and combined SHA-256 fingerprints. The runner rejects fingerprint differences between arms for one family/seed.
The `full` arm is checked against the saved main benchmark by paired objective and initial action IDs when the reference row exists. This checks that the new control reproduces the registered main protocol, while the experiment itself is rerun under one working-tree source fingerprint.

## Interpretation boundary

The experiment measures conditional mechanism contributions under the current synthetic topology generators, three heuristic attacker profiles, candidate caps, response-pool heuristic, and beam width. It does not prove global optimality, Stackelberg equilibrium, real-world risk reduction, or generalization to operational networks. A non-significant contrast is not proof that a mechanism has no value; interactions and bounded-search approximation can mask effects.

Raw paired runs: `chinanet_end_to_end_ablation_50_runs.csv`.
