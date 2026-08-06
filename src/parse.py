"""Verdict parsing. Strict on purpose: a lenient parser hides judge failures as verdicts."""
from __future__ import annotations

import re

# Upstream format: the judge must end with [[A]], [[B]] or [[C]] (tie).
VERDICT_RE = re.compile(r"\[\[(A|B|C)\]\]")

LABEL = {"A": "model_a", "B": "model_b", "C": "tie"}


def parse_verdict(text: str) -> str | None:
    """Return 'model_a' | 'model_b' | 'tie', or None when no verdict token is found.

    TODO(judge runs): decide what a None becomes downstream.
      Upstream treats unparseable output as a tie; counting it as a win for either side
      biases the judge toward whichever position the failure correlates with, and dropping
      the row silently shrinks n. Whatever you choose: count the failures, report the rate
      in docs/RESULTS.md, and record the rule in docs/DECISIONS.md.
    """
    matches = VERDICT_RE.findall(text)
    if not matches:
        return None
    # Last match wins: judges sometimes quote the format before deciding.
    return LABEL[matches[-1]]
