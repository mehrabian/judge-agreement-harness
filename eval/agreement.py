"""Agreement between two judges. The core of the project — own every line."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

KEY = ["question_id", "model_a", "model_b"]


def _normalize_winner(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    tie_mask = out.str.lower().str.startswith("tie")
    return out.where(~tie_mask, "tie")


def _verdict_col(df: pd.DataFrame) -> str:
    if "winner" in df.columns:
        return "winner"
    if "verdict" in df.columns:
        return "verdict"
    raise KeyError(f"expected 'winner' or 'verdict'; got {list(df.columns)}")


def align(j1: pd.DataFrame, j2: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on KEY. Per-vote: each human vote is one comparison (paper)."""
    left = j1.copy()
    right = j2.copy()
    c1, c2 = _verdict_col(left), _verdict_col(right)
    left["verdict_1"] = _normalize_winner(left[c1])
    right["verdict_2"] = _normalize_winner(right[c2])
    right_dedup = right.drop_duplicates(subset=KEY, keep="first")
    merged = left.merge(right_dedup[KEY + ["verdict_2"]], on=KEY, how="inner")
    if merged.empty:
        raise ValueError("align produced 0 rows — check KEY overlap")
    return merged.reset_index(drop=True)


def agreement(aligned: pd.DataFrame, drop_ties: bool) -> dict:
    """Percent agreement + kappa. without-ties drops rows where EITHER said tie."""
    if "verdict_1" not in aligned.columns or "verdict_2" not in aligned.columns:
        raise KeyError("aligned frame must have verdict_1 and verdict_2")
    df = aligned.copy()
    if drop_ties:
        df = df[(df["verdict_1"] != "tie") & (df["verdict_2"] != "tie")]
    n = len(df)
    if n == 0:
        raise ValueError("no comparisons left after tie filter")
    agree = float((df["verdict_1"] == df["verdict_2"]).mean())
    labels = sorted(set(df["verdict_1"]) | set(df["verdict_2"]))
    kappa = float(cohen_kappa_score(df["verdict_1"], df["verdict_2"], labels=labels))
    return {"agree": agree, "kappa": kappa, "n": n}


def summarize(aligned: pd.DataFrame) -> dict:
    out = {}
    for drop_ties in (False, True):
        key = "no_ties" if drop_ties else "with_ties"
        out[key] = agreement(aligned, drop_ties=drop_ties)
    return out
