# Decisions

Append-only. One entry per real choice: alternatives, reason, the number that settled it.

## 2026-08-06 — Anchor protocol: MT-Bench pairwise with released human labels
Alternatives: Chatbot Arena conversations (larger, crowdsourced, noisier labels); building a
custom preference set (no external reference number).
Reason: expert labels, released GPT-4 verdicts enabling a zero-API reference reproduction, and
a published agreement figure (~85% non-tie) to validate the harness against before trusting it
on a new judge.

## 2026-08-06 — Judge prompt fetched from upstream, not copied
Alternatives: vendor the prompt text into the repo.
Reason: comparability with reported numbers depends on the exact template; fetching
`pair-v2` from the FastChat repo at a pinned commit removes transcription drift and makes the
provenance auditable.

## 2026-08-06 — Deterministic CI gate on cached verdicts
Alternatives: live judge calls in CI.
Reason: live calls make the gate flaky (provider variance, rate limits) and put secrets and
per-push cost into CI; regressions worth catching are in harness code and prompts, which
cached verdicts expose deterministically.

<!-- Entries below are added during the build, at decision time: tie rule and unit of
     comparison; conflicting-human-votes rule; judge model choice; subsample design;
     parse-failure rule (+ measured rate); gate threshold (+ CI basis); feature set. -->
