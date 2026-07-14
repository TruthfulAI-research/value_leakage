"""Threshold-sweep experiment variants.

A `thresholds` spec is a dict `{"pct_range": (lo_pct, hi_pct), "n": N}`.
Thresholds are computed as np.linspace over baseline-distribution values
between the lo/hi percentiles. For a single-median threshold use
pct_range=(50, 50) and n=1.
"""

_PROMPT_KEYS = [
    "v1_giraffes",
    "v1_zills",
    # "v1_whale",
    "v1_bridge",
    "v1_crochet",
    "v1_tbc",
    "v1_maiden",
    "v1_turns",
    "v1_orangecars",
    "v1_windowdays",
]

_MEDIAN_SPEC = {"pct_range": (50, 50), "n": 1}

_SIZES = {
    "median":       {"n_baseline": 1000, "n_per_threshold": 1000},
    "median_small": {"n_baseline": 100,  "n_per_threshold": 100},
}

THRESHOLD_EXPERIMENTS = {}
for _pk in _PROMPT_KEYS:
    _name = _pk.removeprefix("v1_")
    for _size_suffix, _size_cfg in _SIZES.items():
        THRESHOLD_EXPERIMENTS[f"{_name}_{_size_suffix}"] = {
            "prompts": [_pk],
            **_size_cfg,
            "thresholds": _MEDIAN_SPEC,
        }

# Multi-prompt: all prompts in a single experiment.
THRESHOLD_EXPERIMENTS["main_experiment"] = {
    "prompts": _PROMPT_KEYS,
    "n_baseline": 100,
    "n_per_threshold": 100,
    "thresholds": _MEDIAN_SPEC,
}
THRESHOLD_EXPERIMENTS["main_experiment_small"] = {
    "prompts": _PROMPT_KEYS,
    "n_baseline": 10,
    "n_per_threshold": 10,
    "thresholds": _MEDIAN_SPEC,
}

# Variant conditions of main_experiment_small: same routing mechanism, but
# different "Note:" hint framings. Equal-charity variants present two equally
# good donation targets; the concrete variant names a real-world good/bad
# charity pair (AMF vs Kids Wish Network).
_VARIANTS = [
    "equal_dwb_imc",
    "equal_amf_mc",
    "equal_dr_ac",
    "concrete_amf_kw",
    "concrete_amf_burn",
    "good_only",
    "accurate",
]
for _v in _VARIANTS:
    THRESHOLD_EXPERIMENTS[f"main_experiment_small_{_v}"] = {
        "prompts": [f"{_pk}_{_v}" for _pk in _PROMPT_KEYS],
        "n_baseline": 10,
        "n_per_threshold": 10,
        "thresholds": _MEDIAN_SPEC,
    }

# Full-size companion to main_experiment_small_accurate: same `_accurate`
# prompts, but at main_experiment's 100/100 sample counts.
THRESHOLD_EXPERIMENTS["main_experiment_accurate"] = {
    "prompts": [f"{_pk}_accurate" for _pk in _PROMPT_KEYS],
    "n_baseline": 100,
    "n_per_threshold": 100,
    "thresholds": _MEDIAN_SPEC,
}
