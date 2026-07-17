#!/usr/bin/env bash
# Re-run the Agentic Effort data collection (paper App. H) from scratch.
#
# This QUERIES THE MODELS. It is only needed to regenerate the raw data; to just
# reproduce the figures from the committed data, run reproduce_plots.sh (no API).
#
# The code is a verbatim copy of the experiment as run for the paper: model
# calls go through localrouter -> a LiteLLM proxy (see llm_providers.py), with
# no reasoning/thinking parameter set, so each model uses its provider default
# (Claude: thinking off; GPT-5.5 and Gemini: their default reasoning; Gemini is
# routed via OpenRouter).
#
# Requirements:
#   pip install localrouter openai pandas pyyaml scipy matplotlib adjustText
#   export LITELLM_API_KEY=...            # the LiteLLM proxy key
#   export LITELLM_BASE_URL=...           # optional; defaults to the paper's proxy
#
# Idempotent: both samplers skip (outcome, sample) pairs already present in the
# data/ submodule, so with the committed data in place this is a no-op and makes
# no API calls. Raw data is written into the value_leakage_data submodule at
#   ../data/final_data/agentic_effort/<slug>/{agentic,liking}/data.csv
set -euo pipefail
cd "$(dirname "$0")"

DATA="../data/final_data/agentic_effort"   # the value_leakage_data submodule

MODELS=(
  "anthropic/claude-opus-4-8|claude-opus-4-8"
  "openai/gpt-5.5|gpt-5.5"
  "openrouter/google/gemini-3.1-pro-preview|gemini-3.1-pro"
)

# --- Agentic persistence: n=50 rollouts/outcome, then cap at 300 turns ---
for entry in "${MODELS[@]}"; do
  routing_id="${entry%%|*}"
  slug="${entry##*|}"
  python scripts/extend_agentic_n50.py \
    --model "$routing_id" \
    --data "$DATA/$slug/agentic/data.csv" \
    --target 50 --parallel 12
  python scripts/apply_turn_cap.py \
    --data "$DATA/$slug/agentic/data.csv" --cap 300
done

# --- Stated liking: 0-100 rating, n=50/outcome (flat agentic outcome set) ---
python scripts/run_liking_newmodels.py \
  --models anthropic/claude-opus-4-8 openai/gpt-5.5 openrouter/google/gemini-3.1-pro-preview \
  --parallel 20 --max-tokens 2048
