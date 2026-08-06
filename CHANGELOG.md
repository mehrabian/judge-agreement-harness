## CHANGELOG
- What: Vault-wired live judge (`claude-sonnet-4-6`), 300x2 run, ablations, CI gate green, README/RESULTS filled. Model id corrected after dated sonnet-4 returned HTTP 404. `run_judge.sh` uses `eval "$(vault env)"` (no plaintext secrets file). Protocol probe rerun at n=50 (disagreement 18%).
- Why: Complete live judge evaluation with vault-injected Anthropic key; no plaintext `.env`.
- Files: run_judge.sh, run_protocol_probe.sh, .envrc, .gitignore, src/judge.py, eval/*, docs/*, README.md, results/*
