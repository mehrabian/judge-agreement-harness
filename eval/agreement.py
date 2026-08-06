"""Agreement between two judges. The core of the project — own every line."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

KEY = ["question_id", "model_a", "model_b"]  # turn already filtered to 1


def align(j1: pd.DataFrame, j2: pd.DataFrame) -> pd.DataFrame:
    """Inner-join two verdict tables on KEY so both judges scored the same items.

    TODO(harness): the human split has MULTIPLE votes per item.
      Options:
        (a) per-vote comparison — each human vote is one comparison (upstream does this);
            inflates n, items with more votes weigh more.
        (b) majority vote per item — one comparison per item; you invented a third judge
            ("majority human") and n shrinks.
      Pick one, defend it, log it. The paper's number is only comparable under (a).
    """
    raise NotImplementedError


def agreement(aligned: pd.DataFrame, drop_ties: bool) -> dict:
    """Percent agreement and Cohen's kappa for columns verdict_1 / verdict_2.

    TODO(harness): 'without ties' upstream means dropping comparisons where EITHER side said
      tie — not just where they disagree about ties. Get this exactly right; it is the
      single most common source of a 5-point discrepancy vs the reported figure.
    """
    raise NotImplementedError


def summarize(aligned: pd.DataFrame) -> dict:
    out = {}
    for drop_ties in (False, True):
        key = "no_ties" if drop_ties else "with_ties"
        out[key] = agreement(aligned, drop_ties=drop_ties)
    return out
