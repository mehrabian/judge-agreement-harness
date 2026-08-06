# Evaluation

## Task
Given two model responses to the same MT-Bench question, predict which one the expert human
annotators preferred (or tie). A judge is scored by its agreement with those labels.

## Data
`lmsys/mt_bench_human_judgments` (CC-BY-4.0), revision pinned in `data/README.md`.
Splits: `human` (expert pairwise votes) and `gpt4_pair` (GPT-4 verdicts on the same pairs).
Filter: turn 1 only. Row counts after each filter are printed by `make data` and recorded here:

| Filter | Rows kept |
|---|---|
| raw `human` split | TBD |
| turn == 1 | TBD |
| after tie normalization | TBD |

## Unit of comparison and tie rule
TBD before any judge runs — one (question_id, model_a, model_b) item vs one human vote; tie-label
normalization. Both choices change n and the with-ties agreement; documented here with the
row counts they keep, before any judge is run.

## Metrics
- Percent agreement, computed both with ties included and on the non-tie subset (the two
  setups reported upstream; the gap between them is large and reported, never averaged).
- Cohen's κ alongside every agreement number — chance-corrected for each judge's verdict
  distribution; the primary alignment claim.
- Uncertainty: percentile bootstrap over items, n=10,000, seed=0, 95% intervals.

## What the metric fails to capture
Agreement with human *preference*, not correctness: a judge that shares human verbosity bias
scores higher, not lower. The per-category feature-model comparison exists to expose where
preference is predictable from surface features alone.

## CI checks (`make gate`)
1. Unit tests, including hand-computable agreement/κ fixtures.
2. GPT-4↔human agreement recomputed from released data == committed reference exactly.
3. Judge↔human agreement from cached verdicts ≥ threshold (TBD: rule, who set it, CI basis).
4. Verdict-distribution total-variation distance vs committed reference ≤ tolerance (TBD).
