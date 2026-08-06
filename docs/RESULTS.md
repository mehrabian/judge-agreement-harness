# Results

Filled from `results/*.json` only — every number here regenerates from a command named next to it.

## Headline: judge↔human agreement (`make reproduce`, `make reproduce-live`)

| Judge | Agreement (ties) | Agreement (non-tie) | κ | 95% CI (non-tie) | n |
|---|---|---|---|---|---|
| GPT-4 released | TBD | TBD | TBD | TBD | TBD |
| This judge | TBD | TBD | TBD | TBD | TBD |
| Human↔human | TBD | TBD | — | TBD | TBD |

Gap vs reported (~85% non-tie): TBD + explanation.

## Ablations (`python -m eval.ablations --probe all`)

Each probe states its falsifier; a probe whose falsifier fired is reported as such.

### Position bias
Hypothesis: verdicts depend on answer order. Falsifier: flip rate at noise level and
first-position win rate ≈ 50% among decisive pairs.
Result: TBD

### Swap-consistency mitigation
Hypothesis: two-order aggregation raises human alignment. Falsifier: no gain beyond CI overlap.
Result: TBD (incl. tie-rate cost of the mitigation)

### Verbosity
Longer-answer win rate — human: TBD · this judge: TBD · GPT-4 released: TBD.
The judge−human gap is the bias estimate; raw rate alone is confounded (longer may be better).

### Deterministic checks vs judge (per category)
| Category | Feature model acc. | Judge agreement | Gap |
|---|---|---|---|
| TBD | | | |

Reading: categories where the gap ≈ 0 are where assertions/regex checks suffice and a judge
is unjustified cost; the large-gap categories are the judge's actual value.

### Pointwise vs pairwise (probe, n≈50)
Disagreement rate and the pattern in disagreeing items: TBD

## Error analysis
Pattern in judge↔human disagreements (sampled and read, not just counted): TBD

## Cost and latency (`python -m eval.report --runs`)
Total spend: TBD · calls: TBD · p50/p95 latency: TBD · cost per 1k judged pairs: TBD
