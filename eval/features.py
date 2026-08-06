"""Deterministic features + linear model predicting the human winner.

The point of this file: locate where human preference is predictable from computable
surface checks alone — those are the cases where an LLM judge is unjustified cost.
Feature extraction is boilerplate (complete); the modelling choices are yours.
"""
from __future__ import annotations

import re

import pandas as pd

REFUSAL_RE = re.compile(r"\b(I can't|I cannot|I'm sorry|as an AI)\b", re.I)
CODE_RE = re.compile(r"```")
LIST_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s", re.M)
HEADER_RE = re.compile(r"^#{1,6}\s", re.M)


def answer_features(text: str) -> dict:
    return {
        "len_chars": len(text),
        "n_code_blocks": len(CODE_RE.findall(text)) // 2,
        "n_list_items": len(LIST_RE.findall(text)),
        "n_headers": len(HEADER_RE.findall(text)),
        "refusal": bool(REFUSAL_RE.search(text)),
    }


def pair_features(row: pd.Series) -> dict:
    a = answer_features(row["conversation_a"][1]["content"])
    b = answer_features(row["conversation_b"][1]["content"])
    f = {f"{k}_ratio": (a[k] + 1) / (b[k] + 1) for k in ("len_chars", "n_code_blocks",
                                                          "n_list_items", "n_headers")}
    f["refusal_a"], f["refusal_b"] = a["refusal"], b["refusal"]
    return f


def fit_and_score(df: pd.DataFrame, seed: int = 0) -> dict:
    """Predict human winner from pair_features, 5-fold CV, accuracy per question category.

    TODO(analysis): decide the target encoding (drop ties? three-class?) — it must MATCH how the
      judge is scored or the comparison is apples to oranges. Logistic regression on
      purpose: calibrated-ish, inspectable coefficients (the coefficient table is itself a
      finding — what does the length coefficient say?). Record in docs/DECISIONS.md.
    """
    raise NotImplementedError
