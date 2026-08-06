# Scope

## Implemented
- Pairwise judge protocol from Zheng et al. (2306.05685): paper's `pair-v2` prompt fetched
  verbatim from the upstream repo at build time, temperature 0, verdicts in {A, B, tie},
  each pair evaluated in both answer orders, swap-inconsistent results scored as ties.
- Validation against the released human labels: percent agreement with and without ties,
  Cohen's κ, percentile bootstrap CIs over items (seeded).
- Offline reproduction of the GPT-4↔human agreement figure from the dataset's released
  `gpt4_pair` verdicts — zero-API, used as the CI reference point.
- Bias probes: position (verdict flip rate under order swap), verbosity (longer-answer win
  rate vs the human rate), protocol (pointwise vs pairwise disagreement on a probe set).
- Deterministic-features baseline: logistic model on length/format/refusal features,
  compared to the judge per question category.
- CI gate on cached verdicts; per-call cost and latency logging.

## Out of scope
- Elo / Bradley–Terry leaderboard construction — a ranking question, orthogonal to judge validity.
- Judge fine-tuning (Prometheus-style open evaluators) — different cost class; the harness
  would evaluate such a judge unchanged.
- Turn-2 (multi-turn) evaluation — turn-1 pairs only.
- Ensembles of judges; non-English data.

## Reduced from the original
- Live judge runs on a stratified 300-pair subsample (~600 calls) rather than the full set;
  API budget. Effect: wider CIs, reported everywhere a number appears.
- One judge model under test rather than the paper's several; the released GPT-4 verdicts
  provide the second reference judge at zero cost.
