#!/usr/bin/env bash
# Reproduce the Agentic Effort experiments (paper App. H): agentic persistence
# and stated liking, for the three paper models.
#
# This directory is a verbatim copy of the experiment code as it was run for the
# paper. It is self-contained and does NOT use the repo's shared/ infrastructure.
# Model calls go through localrouter -> a LiteLLM proxy (see llm_providers.py),
# exactly as in the original runs: no reasoning/thinking parameter is set, so
# each model uses its provider default (Claude: thinking off; GPT-5.5 and
# Gemini: their default reasoning). Gemini is routed via OpenRouter.
#
# Requirements:
#   pip install localrouter openai pandas pyyaml scipy matplotlib adjustText
#   export LITELLM_API_KEY=...            # the LiteLLM proxy key
#   export LITELLM_BASE_URL=...           # optional; defaults to the paper's proxy
#
# Outputs:
#   results/<slug>/{agentic,liking}/data.csv   (gitignored; the paper data lives
#                                                in the separate data repository)
#   figures + LaTeX under results/paper/
set -euo pipefail
cd "$(dirname "$0")"

# routing id (as sent to the proxy)          display slug (results/<slug>/...)
MODELS=(
  "anthropic/claude-opus-4-8|claude-opus-4-8"
  "openai/gpt-5.5|gpt-5.5"
  "openrouter/google/gemini-3.1-pro-preview|gemini-3.1-pro"
)

# --- Agentic persistence: collect n=50 rollouts/outcome, then cap at 300 turns ---
for entry in "${MODELS[@]}"; do
  routing_id="${entry%%|*}"
  slug="${entry##*|}"
  python scripts/extend_agentic_n50.py \
    --model "$routing_id" \
    --data "results/$slug/agentic/data.csv" \
    --target 50 --parallel 12
  python scripts/apply_turn_cap.py \
    --data "results/$slug/agentic/data.csv" --cap 300
done

# --- Stated liking: 0-100 rating, n=50/outcome (flat agentic outcome set) ---
python scripts/run_liking_newmodels.py \
  --models anthropic/claude-opus-4-8 openai/gpt-5.5 openrouter/google/gemini-3.1-pro-preview \
  --parallel 20 --max-tokens 2048

# --- Figures ---
python scripts/build_measures_newmodels.py            # results/<slug>/correlations/measures.csv
python scripts/plot_paper_figures.py                  # Fig 1 (persistence) + Fig 2 (liking vs persistence)
for entry in "${MODELS[@]}"; do
  python scripts/plot_boxplots.py --model "${entry##*|}"   # appendix distributions
done
python scripts/plot_cap_sensitivity.py                # appendix: gemini cap sensitivity
python scripts/make_rollout_figure_tex.py             # example rollout as LaTeX
