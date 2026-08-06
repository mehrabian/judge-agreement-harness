"""Bias probes and the checks-vs-judge boundary. Each probe: what varies, what's fixed,
what falsifies. Fill the design lines in docs/RESULTS.md before implementing the probe."""
from __future__ import annotations

import argparse

import pandas as pd


def aggregate_orders(runs: pd.DataFrame) -> pd.DataFrame:
    """Two-order verdicts -> one verdict per item.

    TODO(judge runs): the swap-consistency rule from the anchor's appendix —
      consistent verdict -> keep it; inconsistent -> tie. Map order 'ba' verdicts back to
      the canonical a/b frame BEFORE comparing (a 'model_a' verdict in the swapped order
      is a vote for the answer shown first, i.e. canonical model_b).
    """
    raise NotImplementedError


def probe_position(runs: pd.DataFrame) -> dict:
    """Flip rate under order swap; first-position win rate among decisive verdicts.
    Falsifier: flip rate at noise level AND first-position rate ~= 50%."""
    raise NotImplementedError


def probe_swap(runs: pd.DataFrame, human: pd.DataFrame) -> dict:
    """Agreement with humans: order-1 only vs order-2 only vs aggregated. Plus tie-rate cost.
    Falsifier: no gain beyond CI overlap."""
    raise NotImplementedError


def probe_verbosity(runs: pd.DataFrame, human: pd.DataFrame, gpt4: pd.DataFrame) -> dict:
    """Longer-answer win rate under each judge; the judge - human gap is the bias estimate."""
    raise NotImplementedError


def probe_features(human: pd.DataFrame) -> dict:
    """Deterministic checks vs judge, per category — see eval/features.py."""
    raise NotImplementedError


def probe_protocol(n: int = 50) -> dict:
    """Pointwise (single-v1, 1-10) vs pairwise verdicts on n items; disagreement pattern."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["position", "swap", "verbosity", "features",
                                        "protocol", "all"], required=True)
    args = ap.parse_args()
    raise SystemExit("wire probes as they are implemented; write results/ablations.json")
