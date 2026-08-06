"""Bias probes and the checks-vs-judge boundary. Each probe: what varies, what's fixed,
what falsifies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.agreement import KEY, _normalize_winner, agreement, align
from eval.bootstrap import bootstrap_ci
from eval.features import fit_and_score
from src.data import load

RESULTS = Path("results")
RUNS = Path("results/runs")


def _canonical_verdict(order: str, verdict: str | None) -> str:
    """Map a raw order-local verdict into the canonical model_a/model_b frame.

    In order 'ba', the answer shown first is canonical model_b, so a 'model_a'
    token means canonical model_b.
    """
    if verdict is None:
        return "tie"
    v = str(verdict)
    if v.lower().startswith("tie") or v == "tie":
        return "tie"
    if order == "ab":
        return v
    if order == "ba":
        if v == "model_a":
            return "model_b"
        if v == "model_b":
            return "model_a"
        return "tie"
    raise ValueError(f"unknown order {order!r}")


def _load_latest_runs() -> pd.DataFrame:
    runs = sorted(RUNS.glob("*.jsonl"))
    if not runs:
        raise FileNotFoundError("no run logs in results/runs/ — run `make judge` first")
    return pd.read_json(runs[-1], lines=True)


def aggregate_orders(runs: pd.DataFrame) -> pd.DataFrame:
    """Two-order verdicts -> one verdict per item (swap-consistency).

    Consistent (same canonical winner) -> keep it; inconsistent -> tie.
    """
    required = {"question_id", "model_a", "model_b", "order", "verdict"}
    missing = required - set(runs.columns)
    if missing:
        raise KeyError(f"runs missing columns: {sorted(missing)}")
    rows = []
    for key, g in runs.groupby(KEY, sort=False):
        by_order = {}
        for _, r in g.iterrows():
            by_order[r["order"]] = _canonical_verdict(r["order"], r["verdict"])
        if "ab" not in by_order or "ba" not in by_order:
            # Single-order run: keep what we have (mapped if present).
            v = by_order.get("ab") or by_order.get("ba")
            if v is None:
                continue
            winner = v
        else:
            va, vb = by_order["ab"], by_order["ba"]
            winner = va if va == vb else "tie"
        rows.append({
            "question_id": key[0],
            "model_a": key[1],
            "model_b": key[2],
            "verdict": winner,
            "winner": winner,
        })
    if not rows:
        raise ValueError("aggregate_orders produced 0 items")
    return pd.DataFrame(rows)


def _single_order(runs: pd.DataFrame, order: str) -> pd.DataFrame:
    sub = runs[runs["order"] == order].copy()
    if sub.empty:
        raise ValueError(f"no rows with order={order!r}")
    sub = sub.drop_duplicates(subset=KEY, keep="first")
    sub["verdict"] = [
        _canonical_verdict(o, v) for o, v in zip(sub["order"], sub["verdict"])
    ]
    sub["winner"] = sub["verdict"]
    return sub[KEY + ["verdict", "winner"]].reset_index(drop=True)


def probe_position(runs: pd.DataFrame) -> dict:
    """Flip rate under order swap; first-position win rate among decisive verdicts."""
    pairs = []
    for key, g in runs.groupby(KEY, sort=False):
        by_order = {r["order"]: r["verdict"] for _, r in g.iterrows()}
        if "ab" not in by_order or "ba" not in by_order:
            continue
        # Position frame: model_a token = first shown answer won.
        first_ab = by_order["ab"]  # None / model_a / model_b / tie
        first_ba = by_order["ba"]
        # Flip in canonical space.
        ca = _canonical_verdict("ab", first_ab)
        cb = _canonical_verdict("ba", first_ba)
        flipped = ca != cb and ca != "tie" and cb != "tie"
        # First-position win: decisive when the judge picked A or B (not tie/None).
        first_wins = []
        for raw in (first_ab, first_ba):
            if raw in ("model_a", "model_b"):
                first_wins.append(raw == "model_a")
        pairs.append({
            "flipped_decisive": flipped,
            "both_decisive": ca != "tie" and cb != "tie",
            "inconsistent": ca != cb,
            "first_wins": first_wins,
        })
    if not pairs:
        raise ValueError("no two-order pairs for position probe")
    both = [p for p in pairs if p["both_decisive"]]
    flip_rate = float(np.mean([p["flipped_decisive"] for p in both])) if both else float("nan")
    first_all = [fw for p in pairs for fw in p["first_wins"]]
    first_pos_rate = float(np.mean(first_all)) if first_all else float("nan")
    inconsistent_rate = float(np.mean([p["inconsistent"] for p in pairs]))
    return {
        "n_pairs": len(pairs),
        "n_both_decisive": len(both),
        "flip_rate_decisive": flip_rate,
        "first_position_win_rate": first_pos_rate,
        "inconsistent_rate": inconsistent_rate,
        "falsifier": "flip_rate ~ noise and first_position_win_rate ~ 0.5",
    }


def probe_swap(runs: pd.DataFrame, human: pd.DataFrame) -> dict:
    """Agreement with humans: order-1 only vs order-2 only vs aggregated."""
    out = {}
    for name, table in [
        ("order_ab", _single_order(runs, "ab")),
        ("order_ba", _single_order(runs, "ba")),
        ("aggregated", aggregate_orders(runs)),
    ]:
        aligned = align(human, table)
        res = {
            "with_ties": agreement(aligned, drop_ties=False),
            "no_ties": agreement(aligned, drop_ties=True),
        }
        lo, hi = bootstrap_ci(aligned, lambda d: agreement(d, drop_ties=True)["agree"])
        res["no_ties"]["ci95"] = [lo, hi]
        out[name] = res
        out[f"{name}_tie_rate"] = float((table["verdict"] == "tie").mean())
    out["falsifier"] = "aggregated no_ties agree within CI of single-order"
    return out


def _longer_win_rate(df: pd.DataFrame, verdict_col: str) -> dict:
    for col in ("conversation_a", "conversation_b", verdict_col):
        if col not in df.columns:
            raise KeyError(
                f"_longer_win_rate missing {col!r}; columns={list(df.columns)}"
            )
    wins = 0
    decisive = 0
    for _, row in df.iterrows():
        v = _normalize_winner(pd.Series([row[verdict_col]])).iloc[0]
        if v == "tie":
            continue
        la = len(row["conversation_a"][1]["content"])
        lb = len(row["conversation_b"][1]["content"])
        if la == lb:
            continue
        longer = "model_a" if la > lb else "model_b"
        decisive += 1
        if v == longer:
            wins += 1
    if decisive == 0:
        raise ValueError("no decisive unequal-length pairs")
    return {"longer_win_rate": wins / decisive, "n": decisive}


def probe_verbosity(runs: pd.DataFrame, human: pd.DataFrame, gpt4: pd.DataFrame) -> dict:
    """Longer-answer win rate under each judge; judge−human gap = bias estimate."""
    agg = aggregate_orders(runs)
    for frame in (agg, human, gpt4):
        frame["question_id"] = frame["question_id"].astype(int)
    conv = human.drop_duplicates(subset=KEY, keep="first")[
        KEY + ["conversation_a", "conversation_b"]
    ].copy()
    human_u = human[KEY + ["winner"]].copy()
    human_u["verdict"] = _normalize_winner(human_u["winner"])
    gpt4_u = gpt4[KEY + ["winner"]].copy()
    gpt4_u["verdict"] = _normalize_winner(gpt4_u["winner"])
    agg_m = agg[KEY + ["verdict"]].merge(conv, on=KEY, how="inner")
    human_m = human_u.merge(conv, on=KEY, how="inner")
    gpt4_m = gpt4_u.merge(conv, on=KEY, how="inner")
    if agg_m.empty:
        raise ValueError("verbosity merge: no overlap between aggregated verdicts and human convs")
    h = _longer_win_rate(human_m, "verdict")
    j = _longer_win_rate(agg_m, "verdict")
    g = _longer_win_rate(gpt4_m, "verdict")
    return {
        "human": h,
        "this_judge": j,
        "gpt4_released": g,
        "judge_minus_human": j["longer_win_rate"] - h["longer_win_rate"],
        "note": "gap vs humans is the bias estimate; raw rate confounded by quality",
    }


def probe_features(human: pd.DataFrame) -> dict:
    return fit_and_score(human)


def probe_protocol(n: int = 50) -> dict:
    """Pointwise (single-v1, 1-10) vs pairwise on n items from the live-judged set.

    Samples from cached aggregated pairwise verdicts (so comparison n matches),
    then scores those same pairs pointwise. Uses live API if keys present.
    """
    import os
    import re

    from src.judge import _call
    from src.prompts import POINTWISE_KEY, get_prompt

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        return {"skipped": True, "reason": "no API key for pointwise probe"}

    if not RUNS.exists() or not list(RUNS.glob("*.jsonl")):
        raise FileNotFoundError("need results/runs/*.jsonl — run make judge first")

    pairwise = aggregate_orders(_load_latest_runs())
    human = load("human")
    conv = human.drop_duplicates(subset=KEY, keep="first")[
        KEY + ["conversation_a", "conversation_b"]
    ].copy()
    conv["question_id"] = conv["question_id"].astype(int)
    pairwise = pairwise.copy()
    pairwise["question_id"] = pairwise["question_id"].astype(int)
    pool = pairwise.merge(conv, on=KEY, how="inner")
    if len(pool) < n:
        raise ValueError(
            f"need {n} live-judged pairs with conversations, found {len(pool)}"
        )
    items = pool.sample(n=n, random_state=0).reset_index(drop=True)

    score_re = re.compile(r"\[\[(\d+)\]\]")
    tpl = get_prompt(POINTWISE_KEY)
    rows = []
    for i, row in items.iterrows():
        q = row["conversation_a"][0]["content"]
        scores = {}
        for side, conv_key in (("a", "conversation_a"), ("b", "conversation_b")):
            ans = row[conv_key][1]["content"]
            user = tpl["prompt_template"].format(question=q, answer=ans)
            messages = [
                {"role": "system", "content": tpl["system_prompt"]},
                {"role": "user", "content": user},
            ]
            resp = _call(messages)
            m = score_re.findall(resp["text"])
            if not m:
                scores[side] = None
            else:
                scores[side] = int(m[-1])
        if scores["a"] is None or scores["b"] is None:
            pointwise = "tie"
        elif scores["a"] > scores["b"]:
            pointwise = "model_a"
        elif scores["b"] > scores["a"]:
            pointwise = "model_b"
        else:
            pointwise = "tie"
        rows.append({
            "question_id": int(row["question_id"]),
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "score_a": scores["a"],
            "score_b": scores["b"],
            "pointwise": pointwise,
            "pairwise": row["verdict"],
        })
        if (i + 1) % 10 == 0:
            print(f"protocol {i + 1}/{n}")
    point_df = pd.DataFrame(rows)
    agree_mask = point_df["pointwise"] == point_df["pairwise"]
    return {
        "n": len(point_df),
        "disagreement_rate": float((~agree_mask).mean()),
        "agree": {
            "agree": float(agree_mask.mean()),
            "n": len(point_df),
        },
        "disagree_n": int((~agree_mask).sum()),
        "pointwise_verdicts": point_df["pointwise"].value_counts().to_dict(),
        "pairwise_verdicts": point_df["pairwise"].value_counts().to_dict(),
        "falsifier": "protocols diverge on writing/roleplay more than extraction",
        "sample": "live_judged_pairs_seed0",
    }


def _run_all() -> dict:
    human = load("human")
    gpt4 = load("gpt4_pair")
    runs = _load_latest_runs()
    out = {
        "position": probe_position(runs),
        "swap": probe_swap(runs, human),
        "verbosity": probe_verbosity(runs, human, gpt4),
        "features": probe_features(human),
        "protocol": probe_protocol(50),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ablations.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--probe",
        choices=["position", "swap", "verbosity", "features", "protocol", "all"],
        required=True,
    )
    args = ap.parse_args()
    human = load("human")
    if args.probe == "all":
        print(json.dumps(_run_all(), indent=2, default=str))
    elif args.probe == "position":
        print(json.dumps(probe_position(_load_latest_runs()), indent=2))
    elif args.probe == "swap":
        print(json.dumps(probe_swap(_load_latest_runs(), human), indent=2))
    elif args.probe == "verbosity":
        print(
            json.dumps(
                probe_verbosity(_load_latest_runs(), human, load("gpt4_pair")),
                indent=2,
            )
        )
    elif args.probe == "features":
        print(json.dumps(probe_features(human), indent=2, default=str))
    elif args.probe == "protocol":
        print(json.dumps(probe_protocol(50), indent=2, default=str))
