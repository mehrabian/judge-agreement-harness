"""Cost/latency summary from run logs. Complete — no open decisions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fill from your provider's current price sheet; used for estimates only — the invoice is
# the source of truth for the number that goes in docs/RESULTS.md.
PRICE_PER_MTOK = {"input": None, "output": None}  # TODO(judge runs): set for JUDGE_MODEL


def summarize_runs() -> None:
    frames = [pd.read_json(p, lines=True) for p in sorted(Path("results/runs").glob("*.jsonl"))]
    if not frames:
        raise SystemExit("no run logs")
    df = pd.concat(frames)
    lat = df["latency_ms"]
    usage = pd.json_normalize(df["usage"])
    print(f"calls: {len(df)}")
    print(f"parse failures: {(df['verdict'].isna()).mean():.1%}")
    print(f"latency p50/p95: {lat.quantile(.5):.0f} / {lat.quantile(.95):.0f} ms")
    for col in usage.columns:
        print(f"tokens {col}: {usage[col].sum():,}")
    if all(PRICE_PER_MTOK.values()):
        cost = (usage.filter(like="prompt").sum().sum() * PRICE_PER_MTOK["input"]
                + usage.filter(like="completion").sum().sum() * PRICE_PER_MTOK["output"]) / 1e6
        print(f"estimated cost: ${cost:.2f}  "
              f"(${cost / df['question_id'].nunique() * 1000:.2f} per 1k judged pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="store_true")
    if ap.parse_args().runs:
        summarize_runs()
