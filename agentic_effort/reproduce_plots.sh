#!/usr/bin/env bash
# Reproduce the Agentic Effort paper figures (App. H) from the committed data.
#
# Offline: reads the cached rollouts/ratings from the data/ submodule
# (value_leakage_data, under final_data/agentic_effort/) and writes the figures
# locally. NO API key and NO network needed.
#
#   pip install pandas numpy matplotlib scipy adjustText localrouter openai
#   (localrouter/openai are imported transitively by the plotting code but never
#    called here; no key required.)
#
# Produces (local, gitignored) under results/paper/:
#   fig_agentic_persistence.pdf        — persistence by recipient (stacked "Free choice")
#   fig_agentic_persistence_median.pdf — same, with median markers
#   fig_liking_vs_agentic.pdf          — stated liking vs. persistence (Pearson r per model)
#   fig_refusals.pdf
# (The appendix box plots, cap-sensitivity plot, and example-rollout LaTeX have
#  their own scripts: scripts/plot_boxplots.py, scripts/plot_cap_sensitivity.py,
#  scripts/make_rollout_figure_tex.py.)
set -euo pipefail
cd "$(dirname "$0")"

python scripts/build_measures_newmodels.py   # liking+agentic -> measures.csv (needed by the scatter)
python scripts/plot_paper_figures.py          # the paper figures
