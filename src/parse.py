"""Verdict parsing. Strict on purpose: a lenient parser hides judge failures as verdicts."""
from __future__ import annotations

import re

# Upstream format: the judge must end with [[A]], [[B]] or [[C]] (tie).
VERDICT_RE = re.compile(r"\[\[(A|B|C)\]\]")

LABEL = {"A": "model_a", "B": "model_b", "C": "tie"}

# Parse-failure rule (docs/DECISIONS.md): unparseable output is treated as tie
# (matches Zheng et al.). Failures are counted and the rate is reported.
PARSE_FAILURE_AS = "tie"


def parse_verdict(text: str) -> str | None:
    """Return 'model_a' | 'model_b' | 'tie', or None when no verdict token is found."""
    matches = VERDICT_RE.findall(text)
    if not matches:
        return None
    # Last match wins: judges sometimes quote the format before deciding.
    return LABEL[matches[-1]]


def verdict_or_tie(text: str) -> str:
    """Parsed verdict, or PARSE_FAILURE_AS when the token is missing."""
    v = parse_verdict(text)
    if v is None:
        return PARSE_FAILURE_AS
    return v
