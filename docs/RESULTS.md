# Results

Filled from `results/*.json` only — every number here regenerates from a command named next to it.

## Headline: judge↔human agreement (`make reproduce`, `make reproduce-live`)

| Judge | Agreement (ties) | Agreement (non-tie) | κ | 95% CI (non-tie) | n |
|---|---|---|---|---|---|
| GPT-4 released | 66.8% | 85.8% | 0.53 | [82.8, 88.7] | 882 / 549 |
| This judge (`claude-sonnet-4-6`) | 67.5% | 88.1% | 0.76 | [84.0, 91.9] | 406 / 269 |
| Human↔human | 64.2% | 82.6% | — | — | 606 / 386 pairs |

Gap vs reported (~85% non-tie): GPT-4 recomputed **+0.8 pts** — within noise; tie filter + turn-1 + per-vote unit match the paper.

Baselines (`make baselines`): random expected 33.3%; longer-answer non-tie 69.1% (κ=0.38).

## Ablations (`python -m eval.ablations --probe all`)

### Position bias
Hypothesis: verdicts depend on answer order. Falsifier: flip rate at noise level and
first-position win rate ≈ 50% among decisive pairs.
**Result:** flip rate **4.9%** (n_decisive=268); first-position win **49.1%**. Position bias is small on this model; falsifier approximately holds.

### Swap-consistency mitigation
Hypothesis: two-order aggregation raises human alignment. Falsifier: no gain beyond CI overlap.
**Result:** order-ab 87.4% [83.3, 91.2]; order-ba 86.0% [81.8, 89.9]; aggregated 88.1% [84.0, 91.9].
Gain vs better single-order is **+0.7 pts** and inside CI overlap — falsifier fires for a *material* improvement claim. Tie rate cost: 9% → 15%.

### Verbosity
Longer-answer win rate — human: **69.1%** · this judge: **75.6%** · GPT-4 released: **74.3%**.
Judge−human gap = **+6.5 pts** (mild extra verbosity preference vs humans).

### Deterministic checks vs judge (per category)
Command regenerates via `results/ablations.json` / `fit_and_score`.

| Category | Feature model acc. | n |
|---|---|---|
| writing | 51.3% | 158 |
| roleplay | 53.4% | 208 |
| extraction | 47.8% | 113 |
| reasoning | 50.0% | 150 |
| math | 63.9% | 147 |
| coding | 60.8% | 148 |
| stem | 44.5% | 182 |
| humanities | 43.8% | 178 |
| **overall** | **51.7%** | 1284 |

Length coefficient = +0.62. Math/coding are where surface features help most; writing/humanities stay near chance — judge territory. Overall 51.7% vs judge↔human 88.1%: checks do **not** replace the judge on this set.

### Pointwise vs pairwise (probe)
Command: `bash run_protocol_probe.sh` (`eval "$(vault env)"`).
Sample: 50 pairs drawn from the live-judged set (seed=0).
**Result:** disagreement rate **18%** (9/50); pointwise↔pairwise agree **82%**.
Pointwise ties 11 vs pairwise ties 8 — protocols diverge on a non-trivial minority of items.

## Error analysis
Parse-failure rate: **0.0%** (604 calls). Unparseable → tie rule unused this run.

## Cost and latency (`python -m eval.report --runs`)
Total spend (est.): **$3.50** · calls: **604** · p50/p95 latency: **6297 / 9986 ms** ·
**$44.90 per 1k judged pairs** (provider sheet: $3/$15 per MTok for Sonnet).
