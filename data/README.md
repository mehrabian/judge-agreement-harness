# Data

## Source
`lmsys/mt_bench_human_judgments` — Hugging Face dataset released with Zheng et al.,
*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (NeurIPS 2023).
https://huggingface.co/datasets/lmsys/mt_bench_human_judgments

## License
CC-BY-4.0 — redistribution and derivative analysis permitted with attribution. Attribution in
the repo README. Raw conversations are re-downloaded by `make data`, not committed.

## Contents
- `human` split (~3.3K rows): expert pairwise preferences over responses from six models
  (GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B) on the 80 MT-Bench questions.
- `gpt4_pair` split (~2.4K rows): GPT-4's pairwise verdicts on the same pairs.
- Fields: `question_id`, `model_a`, `model_b`, `winner`, `judge`, `conversation_a`,
  `conversation_b`, `turn`.

## Download
```bash
make data
# = python -m src.data --download   (pins DATASET_REVISION, writes data/processed/*.parquet)
```
Pinned revision: `f7d2896d2cc5d80f8b55c2bbc722613555233c25` (`DATASET_REVISION` in `src/data.py`).
FastChat judge prompts pinned at commit `7ad1d6386288ba1a7862c11feb673425713eea5b`.


## Judge prompt provenance
`pair-v2` pairwise template (and `single-v1` for the pointwise probe) fetched at a pinned
commit from `lm-sys/FastChat` `fastchat/llm_judge/data/judge_prompts.jsonl`, cached under
`data/prompts/`. See `src/prompts.py`.

## Filtering
Turn 1 only. Tie-label normalization documented in `docs/EVALUATION.md`. Row counts after
each filter step are printed by `make data` and recorded in `docs/EVALUATION.md`.
