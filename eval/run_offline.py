"""Offline agreement runs: released GPT-4 verdicts, or cached live-judge verdicts. $0.

Complete once eval.agreement is implemented — this file is orchestration only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.agreement import agreement, align, summarize
from eval.bootstrap import bootstrap_ci
from src.data import load

RESULTS = Path("results")
VERDICTS = Path("results/verdicts")


def load_cached_live() -> pd.DataFrame:
    """Latest run log -> aggregated verdict table; also write committed parquet for CI.

    Aggregation rule (swap consistency) lives in eval/ablations.py:aggregate_orders so the
    on/off comparison uses the same code path.
    """
    from eval.ablations import aggregate_orders

    VERDICTS.mkdir(parents=True, exist_ok=True)
    committed = VERDICTS / "live_aggregated.parquet"
    runs = sorted(Path("results/runs").glob("*.jsonl"))
    # Prefer non-empty run logs (failed smokes leave 0-byte files).
    runs = [p for p in runs if p.stat().st_size > 0]
    if runs:
        df = pd.read_json(runs[-1], lines=True)
        agg = aggregate_orders(df)
        agg.to_parquet(committed, index=False)
        return agg
    if committed.exists():
        return pd.read_parquet(committed)
    raise SystemExit("no run logs — `make judge` first")


def main(judge: str) -> None:
    human = load("human")
    other = load("gpt4_pair") if judge == "gpt4_pair" else load_cached_live()
    aligned = align(human, other)
    res = summarize(aligned)
    lo, hi = bootstrap_ci(aligned, lambda d: agreement(d, drop_ties=True)["agree"])
    res["no_ties"]["ci95"] = [lo, hi]
    res["n_items"] = len(aligned)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"agreement_{judge}_human.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["gpt4_pair", "cached-live"], required=True)
    main(ap.parse_args().judge)
