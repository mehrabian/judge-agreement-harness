#!/usr/bin/env bash
set -eo pipefail

cd /mnt/c/mnt/00-judge-agreement-harness
export PATH="$HOME/.local/bin:$PATH"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate train-spectral-model-production
export PYTHONPATH=.
export JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-6}"

eval "$(vault env)"
echo "key_len=${#ANTHROPIC_API_KEY} model=$JUDGE_MODEL"

if [[ ${#ANTHROPIC_API_KEY} -lt 80 ]]; then
  echo "ERROR: ANTHROPIC_API_KEY too short after vault env"
  exit 1
fi

echo "=== pointwise protocol probe (n=50, live-judged sample) ==="
python - <<'PY'
import json
from pathlib import Path
from eval.ablations import probe_protocol

proto = probe_protocol(50)
Path("results/protocol_probe.json").write_text(json.dumps(proto, indent=2, default=str))
abl_path = Path("results/ablations.json")
abl = json.loads(abl_path.read_text()) if abl_path.exists() else {}
abl["protocol"] = proto
abl_path.write_text(json.dumps(abl, indent=2, default=str))
print(json.dumps(proto, indent=2, default=str))
PY

echo "PROTOCOL_DONE"
