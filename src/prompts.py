"""Fetch the paper's judge prompt templates verbatim from upstream, at a pinned commit.

The templates are NOT vendored into this repo on purpose: comparability with the reported
numbers depends on the exact wording, and fetching from source removes transcription drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

# Pinned FastChat commit so pair-v2 / single-v1 cannot drift under us.
FASTCHAT_COMMIT = "7ad1d6386288ba1a7862c11feb673425713eea5b"
PROMPTS_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/"
    f"{FASTCHAT_COMMIT}/fastchat/llm_judge/data/judge_prompts.jsonl"
)
CACHE = Path("data/prompts/judge_prompts.jsonl")

PAIRWISE_KEY = "pair-v2"    # pairwise: verdicts [[A]] / [[B]] / [[C]]
POINTWISE_KEY = "single-v1"  # pointwise probe: rating 1-10


def get_prompt(key: str = PAIRWISE_KEY) -> dict:
    """Return the upstream prompt template dict (fields incl. system_prompt, prompt_template)."""
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(PROMPTS_URL, timeout=30)
        r.raise_for_status()
        CACHE.write_bytes(r.content)
    for line in CACHE.read_text().splitlines():
        obj = json.loads(line)
        if obj.get("name") == key:
            return obj
    raise KeyError(f"prompt {key!r} not found in {CACHE}")


if __name__ == "__main__":
    p = get_prompt()
    print(p["system_prompt"][:400], "...")
