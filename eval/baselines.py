"""Reference points every judge must beat (or, for the human ceiling, approach)."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from eval.agreement import KEY, _normalize_winner, agreement, align
from src.data import load

RESULTS = Path("results")


def _answer_len(conversation) -> int:
    if conversation is None or len(conversation) < 2:
        raise ValueError("expected conversation with turn-1 answer at index 1")
    content = conversation[1].get("content")
    if content is None:
        raise ValueError("conversation turn-1 missing content")
    return len(content)


def longer_answer_judge(df: pd.DataFrame) -> pd.Series:
    """Verdict = whichever response is longer (characters of the turn-1 answer)."""
    verdicts = []
    for _, row in df.iterrows():
        la = _answer_len(row["conversation_a"])
        lb = _answer_len(row["conversation_b"])
        if la > lb:
            verdicts.append("model_a")
        elif lb > la:
            verdicts.append("model_b")
        else:
            verdicts.append("tie")
    return pd.Series(verdicts, index=df.index, name="verdict")


def random_judge_expected(df: pd.DataFrame) -> float:
    """Expected agreement of a uniform random {a, b, tie} judge with human votes.

    E[agree] = sum_c p_human(c) * (1/3). Computed from observed freqs after
    tie normalization (equals 1/3 when all three labels appear).
    """
    winners = _normalize_winner(df["winner"])
    p = winners.value_counts(normalize=True)
    classes = ["model_a", "model_b", "tie"]
    return float(sum(p.get(c, 0.0) * (1.0 / 3.0) for c in classes))


def human_human(df: pd.DataFrame) -> dict:
    """Agreement between pairs of human votes on the same item — the ceiling."""
    winners = _normalize_winner(df["winner"])
    work = df[KEY].copy()
    work["winner"] = winners
    pairs_with: list[bool] = []
    pairs_no: list[bool] = []
    for _, g in work.groupby(KEY):
        votes = g["winner"].tolist()
        if len(votes) < 2:
            continue
        for v1, v2 in itertools.combinations(votes, 2):
            pairs_with.append(v1 == v2)
            if v1 != "tie" and v2 != "tie":
                pairs_no.append(v1 == v2)
    if not pairs_with:
        raise ValueError("no items with >=2 human votes for human↔human")
    out = {
        "with_ties": float(sum(pairs_with) / len(pairs_with)),
        "n_pairs_with_ties": len(pairs_with),
    }
    if not pairs_no:
        raise ValueError("no non-tie human vote pairs")
    out["no_ties"] = float(sum(pairs_no) / len(pairs_no))
    out["n_pairs_no_ties"] = len(pairs_no)
    return out


if __name__ == "__main__":
    df = load("human")
    longer = df[["question_id", "model_a", "model_b"]].copy()
    longer["verdict"] = longer_answer_judge(df).to_numpy()
    aligned = align(df, longer)
    longer_res = {
        "with_ties": agreement(aligned, drop_ties=False),
        "no_ties": agreement(aligned, drop_ties=True),
    }
    out = {
        "random_expected": random_judge_expected(df),
        "longer_answer": longer_res,
        "human_human": human_human(df),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
