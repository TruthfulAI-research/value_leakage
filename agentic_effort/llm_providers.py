from __future__ import annotations

import os
import re

import openai
from localrouter import add_provider
from localrouter.llm import Provider, get_response_factory, providers

# Priority must be a smaller number than localrouter's built-in first-party
# providers (priority 10) so that LiteLLM is preferred for every model ID.
_LITELLM_PRIORITY = 5

_DEFAULT_BASE_URL = "https://litellm.nielsrolf.com"

# Cloudflare Bot Fight Mode blocks the OpenAI SDK's default User-Agent
# ("OpenAI/Python ...") with a 403 "Your request was blocked." Override it.
# Remove once Bot Fight Mode is disabled / a WAF skip rule is added for
# litellm.nielsrolf.com.
_USER_AGENT = "litellm-client/1.0"

# Catch-all: match any non-empty model ID. Registered at high priority so it is
# selected ahead of every built-in provider, making LiteLLM the single route.
_MATCH_ALL = re.compile(r".+")

# Sentinel so we don't register the provider twice if this module is imported
# from several eval packages within the same process.
_REGISTERED_FLAG = "_litellm_provider_registered"


def _already_registered() -> bool:
    return any(getattr(p, _REGISTERED_FLAG, False) for p in providers)


def register_litellm_provider(force: bool = False) -> bool:
    """Register the LiteLLM catch-all provider with localrouter.

    Returns ``True`` if a provider was registered, ``False`` if it was skipped
    (missing API key, or already registered and ``force`` is False).
    """
    # Accept the canonical LITELLM_API_KEY, and fall back to the LLMLITE_API_KEY
    # spelling that some environments inject.
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("LLMLITE_API_KEY")
    if not api_key:
        return False

    if _already_registered() and not force:
        return False

    base_url = os.environ.get("LITELLM_BASE_URL", _DEFAULT_BASE_URL)

    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"User-Agent": _USER_AGENT},
    )

    get_response_litellm = get_response_factory(client)
    # Name it so localrouter's print_available_models() shows "Litellm".
    get_response_litellm.__name__ = "get_response_litellm"

    add_provider(get_response_litellm, models=[_MATCH_ALL], priority=_LITELLM_PRIORITY)
    # Tag the provider we just appended so re-imports are no-ops.
    setattr(providers[-1], _REGISTERED_FLAG, True)
    return True


# Register on import (side effect). Safe to import many times.
register_litellm_provider()
