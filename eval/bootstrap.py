"""Seeded percentile bootstrap over items. Complete — no open decisions."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

N_BOOT = 10_000
SEED = 0


def bootstrap_ci(
    df: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% percentile CI of `stat` under resampling rows (items) with replacement.

    Resampling is over ITEMS, not votes: the randomness we care about is which questions
    and pairs ended up in the benchmark, not annotator noise within an item.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([stat(df.iloc[i]) for i in idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
