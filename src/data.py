"""Download and normalize lmsys/mt_bench_human_judgments. Complete — no decisions here
except DATASET_REVISION, which must be pinned on first successful download."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET = "lmsys/mt_bench_human_judgments"
# Pin after first download: `huggingface-cli scan-cache` or the commit hash shown on the HF
# dataset page. An unpinned dataset means reruns may not land on the same rows.
DATASET_REVISION: str | None = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"

PROCESSED = Path("data/processed")


def download() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for split in ("human", "gpt4_pair"):
        ds = load_dataset(DATASET, split=split, revision=DATASET_REVISION)
        df = ds.to_pandas()
        n_raw = len(df)
        df_t1 = df[df["turn"] == 1].copy()
        # Normalize tie variants so downstream code sees only {model_a, model_b, tie}.
        winners = df_t1["winner"].astype(str).str.strip()
        df_t1["winner"] = winners.where(~winners.str.lower().str.startswith("tie"), "tie")
        df_t1.to_parquet(PROCESSED / f"{split}_turn1.parquet", index=False)
        print(f"{split}: raw={n_raw} turn1={len(df_t1)} "
              f"winners={df_t1['winner'].value_counts().to_dict()}")
    print("Record the row counts above in docs/EVALUATION.md.")


def load(split: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{split}_turn1.parquet")


def peek(n: int) -> None:
    df = load("human")
    cols = ["question_id", "model_a", "model_b", "winner", "judge", "turn"]
    print(df[cols].head(n).to_string())
    print("\nDistinct winner labels:", sorted(df["winner"].unique()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--peek", type=int, default=0)
    args = ap.parse_args()
    if args.download:
        download()
    if args.peek:
        peek(args.peek)
