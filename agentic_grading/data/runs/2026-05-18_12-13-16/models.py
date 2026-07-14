"""Per-model configuration: backend selection and sampling parameters.

Minimal subset of ../giraffes/shared/models.py — only the models this
eval needs. The OpenAI models are non-reasoning, so for a fair comparison
Claude Sonnet 4.6 also runs with thinking disabled here. Gemini 2.5 Pro
keeps thinking enabled (thinking_budget=-1 = unlimited) because that's
the production behaviour for that model — disable it via
`thinking_budget=0` if you want strict apples-to-apples non-reasoning
comparison.
"""

_CLAUDE_DEFAULTS = dict(
    backend="claude",
    max_tokens=32000,
    temperature=1,
    max_concurrent=100,
    thinking_display="summarized",
)

_OPENAI_DEFAULTS = dict(
    backend="openai",
    max_tokens=32000,
    temperature=1,
    max_concurrent=200,
)

_GEMINI_DEFAULTS = dict(
    backend="gemini",
    max_tokens=16000,
    temperature=1,
    max_concurrent=50,
    stream=True,
)

MODELS = {
    # OPENAI
    "gpt-4o-mini": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-4o-mini",
        "display_name": "gpt-4o-mini",
    },
    "gpt-4o": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-4o",
        "display_name": "gpt-4o",
    },
    "gpt-4.1": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-4.1",
        "display_name": "gpt-4.1",
    },
    "gpt-5.1": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.1",
        "display_name": "gpt-5.1",
        "reasoning_effort": "none",
    },
    "gpt-5.5": {
        **_OPENAI_DEFAULTS,
        "model": "gpt-5.5",
        "display_name": "gpt-5-high",
        "reasoning_effort": "none",
    },
    
    # CLAUDE
    "claude-haiku-4.5": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-haiku-4-5",
        "display_name": "claude-haiku-4.5",
        "thinking_type": "disabled",
    },
    "claude-sonnet-4.6": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-sonnet-4-6",
        "display_name": "claude-sonnet-4.6",
        "thinking_type": "disabled",
    },
    "claude-opus-4.6": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-6",
        "display_name": "claude-opus-4.6",
        "thinking_type": "disabled",
    },
    "claude-opus-4.7": {
        **_CLAUDE_DEFAULTS,
        "model": "claude-opus-4-7",
        "display_name": "claude-opus-4.7",
        "thinking_type": "disabled",
    },

    # GEMINI
    "gemini-2.5-pro": {
        **_GEMINI_DEFAULTS,
        "model": "gemini-2.5-pro",
        "display_name": "gemini-2.5-pro",
        # -1 = unlimited thinking budget (Gemini's default for 2.5 Pro).
        "thinking_budget": -1,
    },
}
