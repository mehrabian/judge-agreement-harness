"""Live pairwise judge runner. Boilerplate (client, retries, logging) is complete;
the genuine decisions are marked TODO and unanswered."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data import load
from src.parse import parse_verdict
from src.prompts import get_prompt

load_dotenv()
RUNS = Path("results/runs")

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.0
MAX_TOKENS = 1024


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
def _call(messages: list[dict]) -> dict:
    """One chat completion. Reads OPENAI_API_KEY or ANTHROPIC_API_KEY; OpenAI-style first."""
    if os.environ.get("OPENAI_API_KEY"):
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": JUDGE_MODEL, "messages": messages,
                  "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        return {"text": d["choices"][0]["message"]["content"], "usage": d.get("usage", {})}
    if os.environ.get("ANTHROPIC_API_KEY"):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = [m for m in messages if m["role"] != "system"]
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01"},
            json={"model": JUDGE_MODEL, "system": system, "messages": user,
                  "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        return {"text": d["content"][0]["text"], "usage": d.get("usage", {})}
    raise RuntimeError("No API key found: set OPENAI_API_KEY or ANTHROPIC_API_KEY")


def build_messages(question: str, answer_a: str, answer_b: str) -> list[dict]:
    """Fill the upstream pair-v2 template. Do not reword — see docs/DECISIONS.md."""
    tpl = get_prompt()
    user = tpl["prompt_template"].format(question=question, answer_a=answer_a, answer_b=answer_b)
    return [{"role": "system", "content": tpl["system_prompt"]},
            {"role": "user", "content": user}]


def select_pairs(n: int, seed: int = 0) -> pd.DataFrame:
    """Choose which human-labelled pairs the live judge sees.

    TODO(judge runs): subsample design.
      Uniform sampling over rows over-represents the model pairs with the most votes and
      the chattiest models. A stratified draw over (question category, model pair) keeps
      every category and matchup represented at n=300. Categories come from question_id
      ranges (80 questions x 8 categories, 10 each — see the MT-Bench question file).
      Whatever you do: seed it, log the design, and keep the drawn ids in the run log so
      the exact subsample is recoverable.
    """
    raise NotImplementedError


def run(n_pairs: int, both_orders: bool = True) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{JUDGE_MODEL}.jsonl"
    pairs = select_pairs(n_pairs)
    done = 0
    with out.open("a") as f:
        for _, row in pairs.iterrows():
            orders = [("a", "b"), ("b", "a")] if both_orders else [("a", "b")]
            for first, second in orders:
                q = row[f"conversation_{first}"][0]["content"]
                ans_1 = row[f"conversation_{first}"][1]["content"]
                ans_2 = row[f"conversation_{second}"][1]["content"]
                t0 = time.perf_counter()
                resp = _call(build_messages(q, ans_1, ans_2))
                latency_ms = (time.perf_counter() - t0) * 1000
                rec = {
                    "question_id": int(row["question_id"]),
                    "model_a": row["model_a"], "model_b": row["model_b"],
                    "order": f"{first}{second}",
                    "verdict_raw": resp["text"][-200:],
                    "verdict": parse_verdict(resp["text"]),
                    "judge_model": JUDGE_MODEL, "temperature": TEMPERATURE,
                    "usage": resp["usage"], "latency_ms": round(latency_ms, 1),
                    "prompt_hash": hashlib.sha256(
                        json.dumps(build_messages(q, ans_1, ans_2)).encode()).hexdigest()[:12],
                    "commit": _git_commit(),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(rec) + "\n")
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(pairs)} pairs")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--both-orders", action="store_true")
    a = ap.parse_args()
    run(a.pairs, a.both_orders)
