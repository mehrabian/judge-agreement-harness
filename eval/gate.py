"""Deterministic CI gate. Runs on cached verdicts only — no API calls, no secrets, no flake."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REFERENCE = Path("results/reference.json")  # committed numbers the gate defends

# TODO(gate config): thresholds. The tradeoff: a threshold inside the bootstrap CI fires false
# alarms at a rate you can compute from the CI width; one far outside it misses real
# regressions. Candidate rule: committed value minus half the CI width. Whatever you pick,
# document rule + who decided + basis in docs/EVALUATION.md.
AGREEMENT_FLOOR: float | None = None
TV_DISTANCE_MAX: float | None = None


def check_offline_reproduction() -> bool:
    """Recompute GPT-4<->human agreement; must equal the committed reference EXACTLY
    (same data revision + same code => same number; any drift is a code regression)."""
    raise NotImplementedError


def check_live_agreement() -> bool:
    """Cached live-judge agreement >= AGREEMENT_FLOOR."""
    raise NotImplementedError


def check_verdict_distribution() -> bool:
    """Total variation distance between cached verdict distribution and reference <= max."""
    raise NotImplementedError


def regress(fraction: float) -> None:
    """Deliberately perturb `fraction` of cached verdicts (the CI red-drill). Prints what
    it changed so the drill is auditable; never run on main."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regress", type=float, default=None)
    args = ap.parse_args()
    if args.regress:
        regress(args.regress)
        sys.exit(0)
    ok = all([check_offline_reproduction(), check_live_agreement(),
              check_verdict_distribution()])
    print(json.dumps({"gate": "pass" if ok else "FAIL"}))
    sys.exit(0 if ok else 1)
