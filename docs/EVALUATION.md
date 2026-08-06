# Evaluation

## Task
Given two model responses to the same MT-Bench question, predict which one the expert human
annotators preferred (or tie). A judge is scored by its agreement with those labels.

## Data
`lmsys/mt_bench_human_judgments` (CC-BY-4.0), revision pinned in `src/data.py` (`DATASET_REVISION`)
and `data/README.md`.
Splits: `human` (expert pairwise votes) and `gpt4_pair` (GPT-4 verdicts on the same pairs).
Filter: turn 1 only. Row counts after each filter are printed by `make data` and recorded here:

| Filter | Rows kept |
|---|---|
| raw `human` split | 3355 |
| turn == 1 | 1689 |
| after tie normalization | 1689 (labels → {model_a, model_b, tie}) |
| raw `gpt4_pair` split | 2400 |
| gpt4_pair turn == 1 | 1200 |

## Unit of comparison and tie rule
- **Unit:** one comparison = one human vote × one other-judge verdict on the same
  `(question_id, model_a, model_b)`. Per-vote (Zheng et al.), not majority.
- **Tie rule:** any `winner` whose lowercased form starts with `tie` (including
  `tie (bothbad)`) is normalized to `tie` before agreement is computed.
- **Without ties:** drop a comparison if *either* side said `tie`.

## Metrics
- Percent agreement, computed both with ties included and on the non-tie subset (the two
  setups reported upstream; the gap between them is large and reported, never averaged).
- Cohen's κ alongside every agreement number — chance-corrected for each judge's verdict
  distribution; the primary alignment claim.
- Uncertainty: percentile bootstrap over aligned rows, n=10,000, seed=0, 95% intervals.

## What the metric fails to capture
Agreement with human *preference*, not correctness: a judge that shares human verbosity bias
scores higher, not lower. The per-category feature-model comparison exists to expose where
preference is predictable from surface features alone.

## CI checks (`make gate`)
1. Unit tests, including hand-computable agreement/κ fixtures.
2. GPT-4↔human agreement recomputed from released data == committed reference exactly.
3. Judge↔human agreement from cached verdicts ≥ threshold:
   **floor = committed non-tie agree − half the bootstrap CI width**
   (operator decision; current floor **0.8416** from live 0.8810 with CI [0.8401, 0.9190] —
   see `results/reference.json`).
4. Verdict-distribution total-variation distance vs committed reference ≤ **0.15**.
