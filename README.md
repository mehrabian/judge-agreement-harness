# judge-agreement-harness

Validation harness for LLM-as-judge pipelines: measures how well a pairwise LLM judge
agrees with 3.3K expert human preference labels (MT-Bench), quantifies its position and
verbosity biases, and locates the boundary where deterministic checks predict human
preference as well as a judge does.

## Results

| Judge | Agreement w/ humans (non-tie) | Cohen's κ | Notes |
|---|---|---|---|
| Random | 33.3% | ~0 | floor (uniform over {a,b,tie}) |
| Longer-answer heuristic | 69.1% | 0.38 | the honest baseline |
| Deterministic feature model | 51.7% acc | — | length/format/refusal, 5-fold CV |
| This judge (`claude-sonnet-4-6`, both orders + consistency) | **88.1%** [84.0, 91.9] | 0.76 | 300-pair stratified subsample |
| GPT-4 (released verdicts, Zheng et al.) | 85.8% [82.8, 88.7] | 0.53 | recomputed from released data |
| Human ↔ human | 82.6% | — | ceiling |

Ablation (swap-consistency): aggregated 88.1% vs order-ab 87.4% / order-ba 86.0% — gain inside CI overlap; tie rate rises 9%→15%. Position flip rate 4.9%, first-position win 49.1%.

## Quickstart

```bash
pip install -e ".[dev]"   # or: export PYTHONPATH=.
make data                 # pull lmsys/mt_bench_human_judgments (pinned revision)
make reproduce            # GPT-4 vs human, $0
make baselines
# live judge (vault-injected key; never paste secrets):
eval "$(vault env)" && make judge
make reproduce-live
make gate
```

## How it works

`eval/` is the measurement rig (agreement with/without ties, Cohen's κ, seeded bootstrap
CIs); `src/` is the judge under test (paper-faithful pairwise prompt, temperature 0,
each pair judged in both answer orders, swap-inconsistent verdicts scored as ties). CI
reruns the offline eval on cached verdicts every push.

## Limitations

- Live-judge numbers are on a 300-pair stratified subsample; bootstrap CIs are wide relative to few-point effects.
- Turn-1 conversations only; single judge model (`claude-sonnet-4-6`); English only.
- Agreement is measured against human *preference*, not correctness — feature model closes more of the gap on math/coding (~61–64%) than writing/humanities (~44–51%).

## Reference

Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023
(arXiv:2306.05685); data: `lmsys/mt_bench_human_judgments` (CC-BY-4.0). Agreement-metric
framing follows Thakur et al., *Judging the Judges* (arXiv:2406.12624).
Independent implementation; not affiliated with the authors.
