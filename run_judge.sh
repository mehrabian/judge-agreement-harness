#!/usr/bin/env bash
set -eo pipefail

cd /mnt/c/mnt/00-judge-agreement-harness
export PATH="$HOME/.local/bin:$PATH"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate train-spectral-model-production
export PYTHONPATH=.
export JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-6}"

if [[ -f .env ]]; then
  set -a && source .env && set +a
elif command -v vault >/dev/null 2>&1; then
  eval "$(vault env)"
elif [[ -f /mnt/c/mnt/mt-gyre-mini-upstream/.env ]]; then
  set -a && source /mnt/c/mnt/mt-gyre-mini-upstream/.env && set +a
fi

echo "JUDGE_MODEL=$JUDGE_MODEL"
echo "ANTHROPIC key_len=${#ANTHROPIC_API_KEY}"

if [[ ${#ANTHROPIC_API_KEY} -lt 80 ]]; then
  echo "ERROR: set ANTHROPIC_API_KEY via vault env, .env, or gyre .env (expected len>=80)."
  exit 1
fi

PAIRS="${PAIRS:-300}"
echo "Running judge on $PAIRS pairs x both orders..."
python -m src.judge --pairs "$PAIRS" --both-orders

echo "=== reproduce-live ==="
python -m eval.run_offline --judge cached-live

echo "=== write reference ==="
python -m eval.gate --write-reference

echo "=== ablations ==="
python -m eval.ablations --probe all

echo "=== report ==="
python -m eval.report --runs

echo "=== gate ==="
python -m eval.gate

echo "JUDGE_DONE"
