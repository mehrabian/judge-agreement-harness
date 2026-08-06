# judge-agreement-harness

Validation harness for LLM-as-judge pipelines: measures how well a pairwise LLM judge
agrees with 3.3K expert human preference labels (MT-Bench), quantifies its position and
verbosity biases, and locates the boundary where deterministic checks predict human
preference as well as a judge does.

## Results

| Judge | Agreement w/ humans (non-tie) | Cohen's κ | Notes |
|---|---|---|---|
| Random | TBD | ~0 | floor |
| Longer-answer heuristic | TBD | TBD | the honest baseline |
| Deterministic feature model | TBD | TBD | length/format/refusal features, 5-fold CV |
| This judge (`JUDGE_MODEL`, both orders + consistency) | TBD | TBD | 300-pair stratified subsample |
| GPT-4 (released verdicts, Zheng et al.) | TBD (~85% reported) | TBD | recomputed from released data |
| Human ↔ human | TBD (~81% reported) | — | ceiling |

TBD: one-sentence ablation finding (swap-consistency effect, with CI).

## Quickstart

```bash
pip install -e ".[dev]"
make data          # pull lmsys/mt_bench_human_judgments (pinned revision)
make reproduce     # GPT-4 vs human agreement + kappa from released data, no API key needed
make judge         # run your own judge (set OPENAI_API_KEY or ANTHROPIC_API_KEY, JUDGE_MODEL)
make gate          # the CI check: deterministic eval on cached verdicts
```

## How it works

`eval/` is the measurement rig (agreement with/without ties, Cohen's κ, seeded bootstrap
CIs over items); `src/` is the judge under test (paper-faithful pairwise prompt, temperature 0,
each pair judged in both answer orders, swap-inconsistent verdicts scored as ties). CI reruns
the offline eval on every push and fails on agreement regression or verdict-distribution shift.

## Limitations

- Live-judge numbers are on a 300-pair stratified subsample; bootstrap CIs are reported and wide relative to few-point effects.
- Turn-1 conversations only; single judge model; English only.
- Agreement is measured against human *preference*, which is not correctness — categories with objective answers are where the deterministic feature model closes most of the gap.

## Reference

Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023
(arXiv:2306.05685); data: `lmsys/mt_bench_human_judgments` (CC-BY-4.0). Agreement-metric
framing follows Thakur et al., *Judging the Judges* (arXiv:2406.12624).
Independent implementation; not affiliated with the authors.
