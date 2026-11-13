# JISA synthetic parameter sensitivity

The same 60 fixed 50-node graph/edge-risk instances are reoptimized under the registered conditions. These perturbations are synthetic sensitivity probes, not empirical estimates, independent topology samples, or confirmatory tests.

Pairwise ΔJ is CoDef minus PairStart-Greedy, so negative values favor CoDef. Lower/tie/higher counts use an absolute tolerance of 1e-12. Within-method changes are absolute objective differences from the same graph's base condition; this avoids undefined relative changes where base J is zero.

|Condition|Family|N|CoDef lower / tie / higher|Mean paired ΔJ|Median absolute ΔJ vs base (CoDef / PairStart)|
|---|---|---:|---:|---:|---:|
|Base|Overall|60|2/57/1|-0.003100|0.000000 / 0.000000|
|Base|ER-like|20|0/20/0|+0.000000|0.000000 / 0.000000|
|Base|Hierarchical|20|1/19/0|-0.000560|0.000000 / 0.000000|
|Base|Multipath|20|1/18/1|-0.008740|0.000000 / 0.000000|
|Costs ±15%|Overall|60|1/58/1|+0.000246|0.000000 / 0.000000|
|Costs ±15%|ER-like|20|0/20/0|+0.000000|0.000000 / 0.000000|
|Costs ±15%|Hierarchical|20|1/19/0|-0.000262|0.000000 / 0.000000|
|Costs ±15%|Multipath|20|0/19/1|+0.001001|0.000000 / 0.000000|
|Effects ±15%|Overall|60|3/56/1|-0.002682|0.000453 / 0.001070|
|Effects ±15%|ER-like|20|0/20/0|+0.000000|0.000076 / 0.000076|
|Effects ±15%|Hierarchical|20|1/19/0|-0.000150|0.003940 / 0.002958|
|Effects ±15%|Multipath|20|2/17/1|-0.007895|0.000000 / 0.000000|
|Assets ±15%|Overall|60|3/55/2|-0.000451|0.022498 / 0.022498|
|Assets ±15%|ER-like|20|1/19/0|-0.000828|0.030907 / 0.030907|
|Assets ±15%|Hierarchical|20|1/18/1|-0.000089|0.009562 / 0.008530|
|Assets ±15%|Multipath|20|1/18/1|-0.000437|0.023406 / 0.023406|
|Joint ±15%|Overall|60|1/57/2|-0.000790|0.020540 / 0.020540|
|Joint ±15%|ER-like|20|0/20/0|+0.000000|0.013737 / 0.013737|
|Joint ±15%|Hierarchical|20|0/19/1|+0.003708|0.020680 / 0.020680|
|Joint ±15%|Multipath|20|1/18/1|-0.006077|0.023661 / 0.023661|

No inferential p-values are reported because the parameter settings are controlled probes on the same graph instances.
