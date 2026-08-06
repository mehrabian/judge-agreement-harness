"""Deterministic features + linear model predicting the human winner.

The point of this file: locate where human preference is predictable from computable
surface checks alone — those are the cases where an LLM judge is unjustified cost.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from eval.agreement import KEY, _normalize_winner

REFUSAL_RE = re.compile(r"\b(I can't|I cannot|I'm sorry|as an AI)\b", re.I)
CODE_RE = re.compile(r"```")
LIST_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s", re.M)
HEADER_RE = re.compile(r"^#{1,6}\s", re.M)

FEATURE_COLS = [
    "len_chars_ratio",
    "n_code_blocks_ratio",
    "n_list_items_ratio",
    "n_headers_ratio",
    "refusal_a",
    "refusal_b",
]

# MT-Bench categories: question_id 1-80, 10 per category.
CATEGORY_NAMES = [
    "writing",
    "roleplay",
    "extraction",
    "reasoning",
    "math",
    "coding",
    "stem",
    "humanities",
]


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
    f = {
        f"{k}_ratio": (a[k] + 1) / (b[k] + 1)
        for k in ("len_chars", "n_code_blocks", "n_list_items", "n_headers")
    }
    f["refusal_a"], f["refusal_b"] = a["refusal"], b["refusal"]
    return f


def _category(qid: int, qmin: int = 81) -> str:
    """Map dataset question_id (turn-1: 81-160) onto the 8 MT-Bench categories."""
    idx = (int(qid) - qmin) // 10
    if idx < 0 or idx >= len(CATEGORY_NAMES):
        raise ValueError(f"question_id {qid} outside expected category range from qmin={qmin}")
    return CATEGORY_NAMES[idx]


def fit_and_score(df: pd.DataFrame, seed: int = 0) -> dict:
    """Predict human winner from pair_features, 5-fold CV, accuracy per category.

    Target encoding: drop ties, binary model_a vs model_b — matches the non-tie
    agreement setup used for the judge. Logistic regression: calibrated-ish,
    inspectable coefficients.
    """
    qmin = int(df["question_id"].min())
    rows = []
    for _, row in df.iterrows():
        w = _normalize_winner(pd.Series([row["winner"]])).iloc[0]
        if w == "tie":
            continue
        feats = pair_features(row)
        feats["target"] = 1 if w == "model_a" else 0
        feats["category"] = _category(row["question_id"], qmin=qmin)
        for k in KEY:
            feats[k] = row[k]
        rows.append(feats)
    if not rows:
        raise ValueError("no non-tie human votes for feature model")
    data = pd.DataFrame(rows)
    X = data[FEATURE_COLS].astype(float).values
    y = data["target"].astype(int).values
    if len(np.unique(y)) < 2:
        raise ValueError("need both classes for logistic regression")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.full(len(y), -1)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_tr, y[train_idx])
        preds[test_idx] = clf.predict(X_te)
    if (preds < 0).any():
        raise RuntimeError("CV left some rows unpredicted")

    overall = float((preds == y).mean())
    per_cat = {}
    for cat in CATEGORY_NAMES:
        mask = data["category"].values == cat
        if mask.sum() == 0:
            continue
        per_cat[cat] = {
            "acc": float((preds[mask] == y[mask]).mean()),
            "n": int(mask.sum()),
        }

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Xs, y)
    coefs = {c: float(v) for c, v in zip(FEATURE_COLS, clf.coef_[0])}

    return {
        "overall_acc": overall,
        "n": int(len(y)),
        "per_category": per_cat,
        "coefficients": coefs,
        "target": "binary_model_a_vs_model_b_drop_ties",
        "model": "LogisticRegression_5fold_CV",
        "qmin": qmin,
    }
