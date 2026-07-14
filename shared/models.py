"""Per-model configuration: backend selection and sampling parameters."""

from pathlib import Path

_SYSTEM_PROMPTS_DIR = Path(__file__).parent / "system_prompts"

# claude.ai substitutes {{currentDateTime}} server-side. Pinned (not derived from
# datetime.now()) so the cache hash stays stable across runs — bump manually.
_CURRENT_DATETIME = "Thursday, April 30, 2026"


def _load_system_prompt(filename):
    return (_SYSTEM_PROMPTS_DIR / filename).read_text().replace(
        "{{currentDateTime}}", _CURRENT_DATETIME,
    )

_CLAUDE_DEFAULTS = dict(
    backend="claude",
    max_tokens=16000,
    temperature=1,
    max_concurrent=100,
    budget_tokens=10000,
    thinking_display="summarized",
)

_OPENAI_DEFAULTS = dict(
    backend="openai",
    max_tokens=16000,
    temperature=1,
    max_concurrent=200,
)

_TINKER_DEFAULTS = dict(
    backend="tinker",
    max_tokens=16000,
    temperature=1,
    max_concurrent=50,
)

_GEMINI_DEFAULTS = dict(
    backend="gemini",
    max_tokens=16000,
    temperature=1,
    max_concurrent=50,
    stream=True,
)

MODELS = {
    # --- Claude ---
    "claude-sonnet-4": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-sonnet-4-20250514",
        "display_name": "claude-sonnet-4",
        "thinking_type": "enabled",
    },
    "claude-opus-4.6-high": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-6",
        "display_name": "claude-opus-4-6-high",
        "thinking_type": "enabled",
        "effort": "high",
    },
    "claude-opus-4.6-max": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-6",
        "display_name": "claude-opus-4-6-max",
        "thinking_type": "enabled",
        "effort": "max",
    },
    "claude-opus-4.8-low": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-8",
        "display_name": "claude-opus-4-8-low",
        "thinking_type": "adaptive",  # enabled unsupported on 4.8; effort drives thinking
        "effort": "low",
    },
    "claude-opus-4.8-medium": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-8",
        "display_name": "claude-opus-4-8-medium",
        "thinking_type": "adaptive",  # enabled unsupported on 4.8; effort drives thinking
        "effort": "medium",
    },
    "claude-opus-4.8-high": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-8",
        "display_name": "claude-opus-4-8-high",
        "thinking_type": "adaptive",  # enabled unsupported on 4.8; effort drives thinking
        "effort": "high",
    },
    "claude-opus-4.8-max": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-8",
        "display_name": "claude-opus-4-8-max",
        "thinking_type": "adaptive",  # enabled unsupported on 4.8; effort drives thinking
        "effort": "max",
        # Same headroom as claude-opus-4.7-max: at max effort the default 16k
        # is not enough for a non-trivial fraction of responses.
        "max_tokens": 64000,
    },
    "claude-opus-4.8-xhigh": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-8",
        "display_name": "claude-opus-4-8-xhigh",
        "thinking_type": "adaptive",  # enabled unsupported on 4.8; effort drives thinking
        "effort": "xhigh",
        "max_tokens": 64000,  # headroom like the max variant (xhigh is high-effort)
    },
    "claude-fable-5-high": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-fable-5",
        "display_name": "claude-fable-5-high",
        "thinking_type": "adaptive",  # same surface as 4.7/4.8: adaptive only
        "effort": "high",
    },
    "claude-opus-4.7-low": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-7",
        "display_name": "claude-opus-4-7-low",
        "thinking_type": "adaptive",  # enabled doesn't work with 4.7
        "effort": "low",
    },
    "claude-opus-4.7-high": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-7",
        "display_name": "claude-opus-4-7-high",
        "thinking_type": "adaptive",  # enabled doesn't work with 4.7
        "effort": "high",
    },
    "claude-opus-4.7-xhigh": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-7",
        "display_name": "claude-opus-4-7-xhigh",
        "thinking_type": "adaptive", # shame we can't turn this off.
        "effort": "xhigh",
    },
    "claude-opus-4.7-max": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-7",
        "display_name": "claude-opus-4-7-max",
        "thinking_type": "adaptive",
        "effort": "max",
        # The default 16k is not enough in ~12% of cases.
        "max_tokens": 64000,
    },
    "claude-opus-4.5-high": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-5",
        "display_name": "claude-opus-4-5-high",
        "thinking_type": "enabled",
        "effort": "high",
    },
    "claude-sonnet-4.5": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-sonnet-4-5",
        "display_name": "claude-sonnet-4-5",
        "thinking_type": "enabled",
    },
    "claude-sonnet-4.6": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-sonnet-4-6",
        "display_name": "claude-sonnet-4-6",
        "thinking_type": "enabled",
        "effort": "high",
    },
    # THIS IS THE NAME WE'LL USE
    # Instead of the one w/o the -high suffix.
    "claude-sonnet-4.6-high": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-sonnet-4-6",
        "display_name": "claude-sonnet-4-6-high",
        "thinking_type": "enabled",
        "effort": "high",
    },
    "claude-opus-4.1": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-1",
        "display_name": "claude-opus-4-1",
        "thinking_type": "enabled",
    },
    "claude-opus-4.1-claudeai": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-1",
        "display_name": "claude-opus-4-1 (claude.ai sysprompt)",
        "thinking_type": "enabled",
        "system_prompt": _load_system_prompt("claude_opus_4_1.txt"),
    },
    "claude-opus-4.6-high-claudeai": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-6",
        "display_name": "claude-opus-4-6-high (claude.ai sysprompt)",
        "thinking_type": "enabled",
        "effort": "high",
        "system_prompt": _load_system_prompt("claude_opus_4_6.txt"),
    },
    "claude-opus-4.7-high-claudeai": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-7",
        "display_name": "claude-opus-4-7-high (claude.ai sysprompt)",
        "thinking_type": "adaptive",
        "effort": "high",
        "system_prompt": _load_system_prompt("claude_opus_4_7.txt"),
    },
    "claude-opus-4.7-max-claudeai": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-7",
        "display_name": "claude-opus-4-7-max (claude.ai sysprompt)",
        "thinking_type": "adaptive",
        "effort": "max",
        "system_prompt": _load_system_prompt("claude_opus_4_7.txt"),
    },
    "claude-opus-4.5-high-claudeai": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-5",
        "display_name": "claude-opus-4-5-high (claude.ai sysprompt)",
        "thinking_type": "enabled",
        "effort": "high",
        "system_prompt": _load_system_prompt("claude_opus_4_5.txt"),
    },

    # --- Gemini ---
    "gemini-3.1-pro-high": {
        **_GEMINI_DEFAULTS,
        "model": "gemini-3.1-pro-preview",
        "display_name": "gemini-3.1-pro-high",
        "max_tokens": 20000,
        "thinking_level": "high",
    },
    "gemini-3.1-pro-medium": {
        **_GEMINI_DEFAULTS,
        "model": "gemini-3.1-pro-preview",
        "display_name": "gemini-3.1-pro-medium",
        "thinking_level": "medium",
    },
    "gemini-2.5-pro": {
        **_GEMINI_DEFAULTS,
        "model": "gemini-2.5-pro",
        "display_name": "gemini-2.5-pro",
        "thinking_budget": -1,
    },
    "gemini-3-flash-high": {
        **_GEMINI_DEFAULTS,
        "model": "gemini-3-flash-preview",
        "display_name": "gemini-3-flash-high",
        "thinking_level": "high",
    },
    "gemini-3.5-flash-high": {
        **_GEMINI_DEFAULTS,
        "model": "gemini-3.5-flash",
        "display_name": "gemini-3.5-flash-high",
        "max_tokens": 20000,
        "thinking_level": "high",
    },

    # --- OpenAI ---
    "gpt-5.1-medium": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.1",
        "display_name": "gpt-5.1-medium",
        "reasoning_effort": "medium",
        "reasoning_summary": "auto",
    },
    "gpt-5.2-medium": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.2",
        "display_name": "gpt-5.2-medium",
        "reasoning_effort": "medium",
        "reasoning_summary": "auto",
    },
    "gpt-5.4-low": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.4",
        "display_name": "gpt-5.4-low",
        "reasoning_effort": "low",
        "reasoning_summary": "auto",
    },
    "gpt-5.4-medium": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.4",
        "display_name": "gpt-5.4-medium",
        "reasoning_effort": "medium",
        "reasoning_summary": "auto",
    },
    "gpt-5.4-mini": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.4-mini",
        "display_name": "gpt-5.4-mini",
        "reasoning_effort": "medium",
        "reasoning_summary": "auto",
    },
    "gpt-5.4-nano": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.4-nano",
        "display_name": "gpt-5.4-nano",
        "reasoning_effort": "medium",
        "reasoning_summary": "auto",
    },
    "gpt-5.4-xhigh": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.4",
        "display_name": "gpt-5.4-xhigh",
        "reasoning_effort": "xhigh",
        "reasoning_summary": "auto",
    },
    "gpt-5.5-instant": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.5",
        "display_name": "gpt-5.5-instant",
        "reasoning_effort": "none",
        "reasoning_summary": "auto",
    },
    "gpt-5.5-low": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.5",
        "display_name": "gpt-5.5-low",
        "reasoning_effort": "low",
        "reasoning_summary": "auto",
    },
    "gpt-5.5-medium": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.5",
        "display_name": "gpt-5.5-medium",
        "reasoning_effort": "medium",
        "reasoning_summary": "auto",
    },
    "gpt-5.5-high": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.5",
        "display_name": "gpt-5.5-high",
        "reasoning_effort": "high",
        "reasoning_summary": "auto",
        # At high effort gpt-5.5 can burn 10k+ hidden reasoning tokens; the
        # default 16k max_output_tokens leaves no headroom for the visible
        # answer and we'd get the same empty-answer pattern as Claude max.
        # gpt-5.5's published cap is 128k; 32k is plenty.
        "max_tokens": 64000,
    },
    "gpt-5.5-xhigh": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.5",
        "display_name": "gpt-5.5-xhigh",
        "reasoning_effort": "xhigh",
        "reasoning_summary": "auto",
        # xhigh burns even more hidden reasoning than high, so it needs at
        # least the same headroom (see gpt-5.5-high note); 64k leaves plenty
        # of room for the tiny visible answer. NB: this value is part of the
        # rollout cache key (runner._prompt_hash / _model_hashable). The
        # ai_bubble caches (final_data/cache/gpt-5.5-xhigh/) were originally
        # migrated at the inherited 16k default and have been re-keyed to this
        # 64k hash so they stay consistent with gpt-5.5-high — the rollouts
        # themselves were verified complete (0 empty / 0 truncated answers),
        # so the re-key is bookkeeping only.
        "max_tokens": 64000,
    },
    "gpt-5.6-sol-medium": {
        **_OPENAI_DEFAULTS,
        # Sol is the flagship tier of the gpt-5.6 family (Terra/Luna below it);
        # effort ladder none/low/medium/high/xhigh/max, same Responses API.
        "model": "gpt-5.6-sol",
        "display_name": "gpt-5.6-sol-medium",
        "reasoning_effort": "medium",
        "reasoning_summary": "auto",
    },

    # --- Tinker ---
    "kimi-k2.5": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://652451f1-64d7-5951-98cb-b17b5cc98b2e:train:0/sampler_weights/base-untrained-Kimi-K2.5",
        "display_name": "Kimi-K2.5",
        "renderer": "kimi_k25",
    },
    "kimi-k2.6": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://a4cc9a6e-a51d-5429-9eec-f2e718f53470:train:0/sampler_weights/base-untrained-Kimi-K2.6",
        "display_name": "Kimi-K2.6",
        # K2.6's chat template matches K2.5's for single-turn (only differences
        # are multi-turn thinking preservation), so the K2.5 renderer is safe.
        "renderer": "kimi_k25",
        # Default 16k truncates the CoT mid-stream on ~33% of hard prompts
        # Tinker doesn't allow more than 32k context.
        "max_tokens": 32000,
    },
    # same for me. I will probably not actually use these. 
    "kimi-k2.6-harry": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://9b20633b-83f9-59c6-81ed-126501eb8ea2:train:0/sampler_weights/base-untrained-Kimi-K2.6",
        "display_name": "Kimi-K2.6-harry",
        "renderer": "kimi_k25",
        "max_tokens": 32000,
    },
    "deepseek-v3.1": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://db241947-3326-5a1d-9eba-6be0306ea06c:train:0/sampler_weights/base-untrained-DeepSeek-V3.1",
        "display_name": "DeepSeek-V3.1",
        "renderer": "deepseek_v3_thinking",
    },
    "qwen3.5-397": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://f24bea7f-e84b-5130-b80d-1e92550abb70:train:0/sampler_weights/base-untrained-Qwen3.5-397B-A17B",
        "display_name": "Qwen3.5-397B-A17B",
        "renderer": "qwen3_5",
    },
    "qwen3.5-397-harry": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://0ae88ee2-4a2e-5bec-9353-a08529db8e1f:train:0/sampler_weights/base-untrained-Qwen3.5-397B-A17B",
        "display_name": "Qwen3.5-397B-A17B-harry",
        "renderer": "qwen3_5",
    },
    "qwen3.6-35": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://885f73d1-541a-50e5-8c61-0638c2219d99:train:0/sampler_weights/base-untrained-Qwen3.6-35B-A3B",
        "display_name": "Qwen3.6-35B-A3B",
        # No qwen3_6 renderer ships with tinker_cookbook. For single-turn use
        # the 3.5 and 3.6 chat templates differ only in a multi-turn
        # preserve_thinking flag and a tool-call arg-serialization tweak,
        # neither of which affects our usage, so qwen3_5 is safe.
        "renderer": "qwen3_5",
    },
    "qwen3.5-35": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://d3ed2c8c-55f5-5ab7-895e-0e0c9fc73cbb:train:0/sampler_weights/base-untrained-Qwen3.5-35B-A3B",
        "display_name": "Qwen3.5-35B-A3B",
        "renderer": "qwen3_5",
    },
    "nemotron3-120b": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://a5ff6920-18a4-5a47-97b7-0c09c0c3b7c7:train:0/sampler_weights/base-untrained-NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "display_name": "Nemotron-3-Super-120B",
        "renderer": "nemotron3",
    },
    "gpt-oss-120b": {
        **_TINKER_DEFAULTS,
        "model_path": "tinker://e0225af7-3cc4-53f0-9f4d-54b67660bfbe:train:0/sampler_weights/base-untrained-gpt-oss-120b",
        "display_name": "gpt-oss-120b",
        "renderer": "gpt_oss",
    },
}
