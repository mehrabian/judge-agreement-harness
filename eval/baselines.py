"""Reference points every judge must beat (or, for the human ceiling, approach)."""
from __future__ import annotations

import json
from pathlib import Path

from src.data import load

RESULTS = Path("results")


def longer_answer_judge(df) -> "pd.Series":  # noqa: F821
    """Verdict = whichever response is longer (characters of the turn-1 answer).

    TODO(before first run): implement, then compute agreement with human votes via eval.agreement.
      This is the honest baseline: it quantifies how far length alone gets you, and its
      number is the denominator for every verbosity-bias conversation later.
    """
    raise NotImplementedError


def random_judge_expected(df) -> float:
    """Expected agreement of a uniform random {a, b, tie} judge with the human votes.

    TODO(before first run): compute from the human verdict distribution, don't assume 1/3 or 1/2.
    """
    raise NotImplementedError


def human_human(df) -> float:
    """Agreement between pairs of human votes on the same item — the ceiling.

    TODO(before first run): items with >=2 votes; all unordered vote pairs within an item; same
      with/without-ties setups as everywhere else.
    """
    raise NotImplementedError


if __name__ == "__main__":
    df = load("human")
    out = {
        "random_expected": random_judge_expected(df),
        "longer_answer": None,  # fill from agreement() once implemented
        "human_human": human_human(df),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
