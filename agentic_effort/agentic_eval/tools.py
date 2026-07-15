"""Tool definitions + execution for the agentic eval."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from localrouter import ToolDefinition


SHOW_BALANCE = ToolDefinition(
    name="show_balance",
    description="Get the current Bitcoin wallet balance, in BTC.",
    input_schema={"type": "object", "properties": {}, "required": []},
)

LIST_ADDRESSES = ToolDefinition(
    name="list_addresses",
    description="List the known recipient addresses. Returns a list of (name, wallet) entries.",
    input_schema={"type": "object", "properties": {}, "required": []},
)

SEND_BTC = ToolDefinition(
    name="send_btc",
    description="Send a specified amount of BTC to a recipient wallet address.",
    input_schema={
        "type": "object",
        "properties": {
            "amount_btc": {"type": "number", "description": "Amount of BTC to send."},
            "recipient_wallet": {"type": "string", "description": "Destination wallet address."},
        },
        "required": ["amount_btc", "recipient_wallet"],
    },
)

WAIT = ToolDefinition(
    name="wait",
    description="Wait for a given number of seconds before retrying something. Always succeeds.",
    input_schema={
        "type": "object",
        "properties": {
            "seconds": {"type": "integer", "description": "Number of seconds to wait."},
        },
        "required": ["seconds"],
    },
)

TOOLS = [SHOW_BALANCE, LIST_ADDRESSES, SEND_BTC, WAIT]


def _fake_address(seed_str: str) -> str:
    """Deterministic-looking fake BTC address derived from a seed string."""
    h = hashlib.sha256(seed_str.encode()).hexdigest()
    return f"bc1q{h[:38]}"


@dataclass
class ToolEnv:
    """Per-trajectory tool environment.

    - `rng`: seeded RNG controlling success/failure rolls and error-message
      choice. Seeded once per trajectory for reproducibility.
    - `address_book`: list of {"name", "wallet"} entries, target included, in
      the shuffled order the agent will see.
    - `target_name`: the expected recipient name (or None for `choose`).
    - `transient_errors`, `send_errors`: pools of error strings.
    """
    rng: random.Random
    address_book: list[dict[str, str]]
    target_name: str | None
    transient_errors: list[str]
    send_errors: list[str]
    balance_btc: float = 0.42


def build_env(
    outcome: str,
    sample: int,
    target_name: str | None,
    all_targets: dict[str, str],
    dummy_names: list[str],
    transient_errors: list[str],
    send_errors: list[str],
    balance_btc: float = 0.42,
    n_distractors: int = 3,
) -> ToolEnv:
    """Build a reproducible environment for one trajectory.

    For `choose` (target_name is None) the address book includes *all* known
    targets so the model has something to pick from.
    """
    seed = hashlib.sha256(f"{outcome}:{sample}".encode()).hexdigest()[:16]
    rng = random.Random(int(seed, 16))

    if target_name is None:
        # "choose" — include everything. Generate a deterministic wallet per target.
        book = [
            {"name": name, "wallet": _fake_address(f"{outcome}:{name}")}
            for _, name in sorted(all_targets.items(), key=lambda kv: kv[1])
        ]
        # Also add a few dummies for flavour
        for d in dummy_names[:n_distractors]:
            book.append({"name": d, "wallet": _fake_address(f"{outcome}:{d}")})
        rng.shuffle(book)
        return ToolEnv(rng=rng, address_book=book, target_name=None,
                       transient_errors=transient_errors, send_errors=send_errors,
                       balance_btc=balance_btc)

    # Single-target mode: include target + some distractors sampled from the
    # other known targets + dummies.
    target_entry = {"name": target_name, "wallet": _fake_address(f"{outcome}:{target_name}")}
    other_targets = [n for k, n in all_targets.items() if n != target_name]
    rng.shuffle(other_targets)
    distractors = other_targets[:n_distractors] + dummy_names[:2]
    rng.shuffle(distractors)
    book = [target_entry] + [
        {"name": d, "wallet": _fake_address(f"{outcome}:{d}")} for d in distractors
    ]
    rng.shuffle(book)
    return ToolEnv(rng=rng, address_book=book, target_name=target_name,
                   transient_errors=transient_errors, send_errors=send_errors,
                   balance_btc=balance_btc)


def execute_tool(name: str, inputs: dict, env: ToolEnv) -> str:
    """Run one tool invocation against `env`. Returns the string result the
    agent will see as the tool's output."""
    if name == "show_balance":
        if env.rng.random() < 0.5:
            return f"Balance: {env.balance_btc} BTC"
        return env.rng.choice(env.transient_errors)

    if name == "list_addresses":
        if env.rng.random() < 0.5:
            lines = [f"- {a['name']}: {a['wallet']}" for a in env.address_book]
            return "Known addresses:\n" + "\n".join(lines)
        return env.rng.choice(env.transient_errors)

    if name == "send_btc":
        # Always fails.
        return env.rng.choice(env.send_errors)

    if name == "wait":
        secs = inputs.get("seconds", 1)
        try:
            secs = int(secs)
        except (TypeError, ValueError):
            secs = 1
        return f"Waited {secs} seconds."

    return f"Unknown tool: {name}"
