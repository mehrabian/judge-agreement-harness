# Decisions

Append-only. One entry per real choice: alternatives, reason, the number that settled it.

## 2026-08-06 — Anchor protocol: MT-Bench pairwise with released human labels
Alternatives: Chatbot Arena conversations (larger, crowdsourced, noisier labels); building a
custom preference set (no external reference number).
Reason: expert labels, released GPT-4 verdicts enabling a zero-API reference reproduction, and
a published agreement figure (~85% non-tie) to validate the harness against before trusting it
on a new judge.

## 2026-08-06 — Judge prompt fetched from upstream, not copied
Alternatives: vendor the prompt text into the repo.
Reason: comparability with reported numbers depends on the exact template; fetching
`pair-v2` from the FastChat repo at a pinned commit removes transcription drift and makes the
provenance auditable.

## 2026-08-06 — Deterministic CI gate on cached verdicts
Alternatives: live judge calls in CI.
Reason: live calls make the gate flaky (provider variance, rate limits) and put secrets and
per-push cost into CI; regressions worth catching are in harness code and prompts, which
cached verdicts expose deterministically.

## 2026-08-06 — Tie rule: normalize all `tie*` labels to `tie`
Alternatives: keep `tie (bothbad)` distinct; drop ties from all metrics.
Reason: upstream non-tie agreement drops any comparison where either side said any tie
variant; collapsing variants keeps n comparable to the paper without inventing a fourth class.
Row counts recorded in docs/EVALUATION.md after `make data`.

## 2026-08-06 — Unit of comparison: per-vote (not majority human)
Alternatives: majority vote per (question_id, model_a, model_b).
Reason: Zheng et al. compare per human vote; majority invents a third judge and shrinks n.
Cost: items with more votes weigh more.

## 2026-08-06 — Parse failures treated as ties
Alternatives: drop the row; count as a win for a side.
Reason: matches upstream FastChat behavior; dropping shrinks n silently; counting as a win
biases position. Rate reported in docs/RESULTS.md after the live run.

## 2026-08-06 — Judge model: claude-sonnet-4-6
Alternatives: gpt-4o-mini (cheaper), claude-sonnet-4-20250514 (404 on this account).
Reason: operator chose Claude; `claude-sonnet-4-6` is the model id used elsewhere on this
machine and accepted by the Anthropic API. Exact string logged in every run record.

## 2026-08-06 — Subsample: 300 pairs stratified by question category
Alternatives: uniform over human rows; full set.
Reason: uniform over-represents chattiest model pairs; full set blows the API budget.
Category = (question_id - qmin) // 10 with qmin = min turn-1 id (81), 8 MT-Bench categories.
Seed=0. Drawn ids recoverable from the run log.

## 2026-08-06 — Feature model: logistic regression, drop ties, binary target
Alternatives: three-class; gradient boosting.
Reason: target matches non-tie judge scoring; logistic stays inspectable (length coefficient
is itself a finding). 5-fold stratified CV.

## 2026-08-06 — Gate floor = committed non-tie agree − half CI width
Alternatives: fixed 0.80; committed value − full CI width.
Reason: half-width is outside the noise band of the committed estimate without waiting for
a many-sigma drop. Who decided: project operator. TV distance max = 0.15.

## 2026-08-06 — Gate always recomputes live agreement from verdict parquet
Alternatives: trust `agreement_cached-live_human.json` if present.
Reason: a stale JSON lets a verdict regression keep the gate green; the drill only
fires when agreement is recomputed from `results/verdicts/live_aggregated.parquet`.
Regression drill: `--regress 0.25` drops non-tie agree from 88.1% → 75.6% (below floor 0.842).
