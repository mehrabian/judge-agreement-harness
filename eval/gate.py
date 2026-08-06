"""Deterministic CI gate. Runs on cached verdicts only — no API calls, no secrets, no flake."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eval.agreement import agreement, align
from eval.ablations import aggregate_orders
from src.data import load

REFERENCE = Path("results/reference.json")
VERDICTS = Path("results/verdicts/live_aggregated.parquet")
AGREEMENT_LIVE = Path("results/agreement_cached-live_human.json")
AGREEMENT_GPT4 = Path("results/agreement_gpt4_pair_human.json")

# Threshold rule: committed non-tie agree minus half the CI width.
# Set after first reproduce-live; documented in docs/EVALUATION.md.
AGREEMENT_FLOOR: float | None = None
TV_DISTANCE_MAX: float | None = 0.15


def _load_reference() -> dict:
    if not REFERENCE.exists():
        raise FileNotFoundError(
            f"missing {REFERENCE} — run make reproduce, then write reference.json"
        )
    return json.loads(REFERENCE.read_text())


def check_offline_reproduction() -> bool:
    """Recompute GPT-4↔human agreement; must equal committed reference EXACTLY."""
    human = load("human")
    gpt4 = load("gpt4_pair")
    aligned = align(human, gpt4)
    res = {
        "with_ties": agreement(aligned, drop_ties=False),
        "no_ties": agreement(aligned, drop_ties=True),
        "n_items": len(aligned),
    }
    ref = _load_reference()
    gpt4_ref = ref["gpt4_pair_human"]
    ok = (
        abs(res["no_ties"]["agree"] - gpt4_ref["no_ties"]["agree"]) < 1e-12
        and abs(res["with_ties"]["agree"] - gpt4_ref["with_ties"]["agree"]) < 1e-12
        and abs(res["no_ties"]["kappa"] - gpt4_ref["no_ties"]["kappa"]) < 1e-12
        and res["n_items"] == gpt4_ref["n_items"]
    )
    print(json.dumps({"check": "offline_reproduction", "ok": ok, "got": res}))
    return ok


def check_live_agreement() -> bool:
    """Recompute agreement from cached verdict parquet; must stay >= AGREEMENT_FLOOR.

    Always recomputes from VERDICTS (not the stale agreement JSON) so a regression
    drill or silent verdict edit is visible to the gate.
    """
    global AGREEMENT_FLOOR
    ref = _load_reference()
    if AGREEMENT_FLOOR is None:
        live_ref = ref.get("live_human")
        if live_ref is None:
            print(json.dumps({"check": "live_agreement", "ok": False,
                              "reason": "no live_human in reference and no AGREEMENT_FLOOR"}))
            return False
        ci = live_ref["no_ties"].get("ci95")
        if not ci or len(ci) != 2:
            raise ValueError("reference live_human.no_ties.ci95 required to set floor")
        half_width = (ci[1] - ci[0]) / 2
        AGREEMENT_FLOOR = live_ref["no_ties"]["agree"] - half_width
    if not VERDICTS.exists():
        print(json.dumps({"check": "live_agreement", "ok": False,
                          "reason": f"missing {VERDICTS}"}))
        return False
    human = load("human")
    live = pd.read_parquet(VERDICTS)
    aligned = align(human, live)
    got = agreement(aligned, drop_ties=True)["agree"]
    ok = got >= AGREEMENT_FLOOR
    print(json.dumps({
        "check": "live_agreement",
        "ok": ok,
        "got": got,
        "floor": AGREEMENT_FLOOR,
    }))
    return ok


def _verdict_dist(df: pd.DataFrame, col: str = "verdict") -> pd.Series:
    s = df[col] if col in df.columns else df["winner"]
    counts = s.value_counts(normalize=True)
    for lab in ("model_a", "model_b", "tie"):
        if lab not in counts:
            counts[lab] = 0.0
    return counts[["model_a", "model_b", "tie"]]


def check_verdict_distribution() -> bool:
    """TV distance between cached verdict distribution and reference <= max."""
    ref = _load_reference()
    ref_dist = pd.Series(ref["live_verdict_dist"])
    if not VERDICTS.exists():
        print(json.dumps({"check": "verdict_distribution", "ok": False,
                          "reason": f"missing {VERDICTS}"}))
        return False
    live = pd.read_parquet(VERDICTS)
    got = _verdict_dist(live)
    tv = 0.5 * float(np.abs(got - ref_dist.reindex_like(got).fillna(0)).sum())
    max_tv = TV_DISTANCE_MAX if TV_DISTANCE_MAX is not None else 0.15
    ok = tv <= max_tv
    print(json.dumps({
        "check": "verdict_distribution",
        "ok": ok,
        "tv": tv,
        "max": max_tv,
        "got": got.to_dict(),
    }))
    return ok


def regress(fraction: float) -> None:
    """Deliberately perturb `fraction` of cached verdicts (CI red-drill)."""
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")
    if not VERDICTS.exists():
        raise FileNotFoundError(VERDICTS)
    df = pd.read_parquet(VERDICTS)
    col = "verdict" if "verdict" in df.columns else "winner"
    rng = np.random.default_rng(0)
    n = len(df)
    k = max(1, int(round(n * fraction)))
    idx = rng.choice(n, size=k, replace=False)
    flipped = []
    for i in idx:
        old = df.at[df.index[i], col]
        options = [x for x in ("model_a", "model_b", "tie") if x != old]
        new = options[int(rng.integers(0, len(options)))]
        df.at[df.index[i], col] = new
        if "winner" in df.columns:
            df.at[df.index[i], "winner"] = new
        flipped.append({"i": int(i), "old": old, "new": new})
    out = VERDICTS.with_name("live_aggregated_REGRESSED.parquet")
    df.to_parquet(out, index=False)
    # Also overwrite the gate input so the next gate run fails.
    df.to_parquet(VERDICTS, index=False)
    print(json.dumps({"regressed": k, "path": str(VERDICTS), "flips": flipped[:20]}))


def write_reference_from_results() -> None:
    """Helper: build reference.json from current agreement json + verdicts."""
    if not AGREEMENT_GPT4.exists():
        raise FileNotFoundError(AGREEMENT_GPT4)
    gpt4 = json.loads(AGREEMENT_GPT4.read_text())
    ref: dict = {"gpt4_pair_human": gpt4}
    if AGREEMENT_LIVE.exists() and VERDICTS.exists():
        live = json.loads(AGREEMENT_LIVE.read_text())
        ref["live_human"] = live
        v = pd.read_parquet(VERDICTS)
        ref["live_verdict_dist"] = _verdict_dist(v).to_dict()
        ci = live["no_ties"].get("ci95")
        if ci:
            half = (ci[1] - ci[0]) / 2
            ref["agreement_floor_rule"] = "committed_no_ties_agree - half_ci_width"
            ref["agreement_floor"] = live["no_ties"]["agree"] - half
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(json.dumps(ref, indent=2))
    print(f"wrote {REFERENCE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regress", type=float, default=None)
    ap.add_argument("--write-reference", action="store_true")
    ap.add_argument(
        "--offline-only",
        action="store_true",
        help="Only check GPT-4↔human reproduction (skip live checks when no verdicts yet)",
    )
    args = ap.parse_args()
    if args.write_reference:
        write_reference_from_results()
        sys.exit(0)
    if args.regress is not None:
        regress(args.regress)
        sys.exit(0)
    checks = [check_offline_reproduction()]
    ref = _load_reference()
    has_live = VERDICTS.exists() and "live_human" in ref
    if args.offline_only or not has_live:
        if not has_live:
            print(json.dumps({
                "check": "live_skipped",
                "reason": "no results/verdicts/live_aggregated.parquet + live_human in reference — run make judge first",
            }))
        ok = all(checks)
    else:
        ok = all(checks + [check_live_agreement(), check_verdict_distribution()])
    print(json.dumps({"gate": "pass" if ok else "FAIL"}))
    sys.exit(0 if ok else 1)
