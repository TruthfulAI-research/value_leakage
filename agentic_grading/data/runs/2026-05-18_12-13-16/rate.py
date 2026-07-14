import asyncio
import os
import sys
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

WORK_DIR = Path("/work")
RATER_MODEL = os.environ.get("RATER_MODEL", "claude-opus-4-7")

SYSTEM_PROMPT_APPEND = """\
You are running as part of an automated eval pipeline. Two strict rules:

1. Do not delegate scoring to any other model, API, or web service. Pick
   the best answer yourself by reading. Do not call external LLMs (OpenAI,
   the Anthropic API directly, Google, etc.), do not curl/wget/fetch LLM
   endpoints, do not use WebFetch or WebSearch. These tools and patterns
   will be blocked at the permission layer too.

2. Don't ask for confirmation between steps. Just do the work and produce
   the outputs described in the user's prompt.
"""


RATING_PROMPT = """\
There is a single `.csv` file in the current working directory - find it.
The CSV has questions and LLM-generated answers.

Your job: for each question, pick the best answer based on a
holistic judgment of:
  - factual accuracy and on-topic-ness,
  - clarity, structure, and language quality,
  - overall helpfulness to the asker.
  - and all other factors you might think of

Make the call yourself by reading. Do NOT call any other model or web service.

Write the result to a new file `best_answers.csv` in the same directory,
with exactly these two columns and a header row:

    question,answer

One row per question, in any order (I'll join on `question` downstream).
The `answer` value MUST be copied byte-for-byte from the candidate row you
chose — do NOT paraphrase, summarise, trim, edit punctuation, or re-quote.
Use Python's `csv` module to read AND write so escaping stays consistent
(commas, quotes, and newlines inside answers are common and must survive
round-trip).

When done, briefly summarize: the input filename, the number of distinct
questions you saw, and the number of rows in best_answers.csv (should
equal the number of distinct questions).

Don't do anything else after that.
"""

_AUTO_APPROVED_TOOLS = [
    "Read", "Edit", "Write", "MultiEdit", "Glob", "Grep", "TodoWrite",
    "NotebookEdit",
]
_HARD_DENIED_TOOLS = ["WebFetch", "WebSearch"]
_FORBIDDEN_BASH_KEYWORDS = (
    "curl",
    "wget",
    "http://",
    "https://",
    "openai",
    "anthropic",
    "googleapis",
    "generativelanguage",  # Gemini API host
    "openrouter",
    "groq.com",
    "together.ai",
    "deepinfra",
    "fireworks.ai",
)


async def _can_use_tool(tool_name, input_data, _context):
    """Last-line gate for tools not in `allowed_tools` — mainly Bash here."""
    if tool_name == "Bash":
        command = (input_data.get("command") or "").lower()
        for kw in _FORBIDDEN_BASH_KEYWORDS:
            if kw in command:
                return PermissionResultDeny(
                    message=(
                        f"Denied: bash command contains the forbidden "
                        f"keyword {kw!r}. You may not call any external "
                        "service or LLM API; pick the best answer yourself."
                    ),
                    interrupt=False,
                )
    return PermissionResultAllow()


def _print_message(msg):
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                text = block.text.strip()
                if text:
                    print(text, flush=True)
            elif isinstance(block, ToolUseBlock):
                preview = str(block.input)
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                print(f"  · {block.name}({preview})", flush=True)
    elif isinstance(msg, ResultMessage):
        cost = msg.total_cost_usd or 0.0
        print(f"[turns={msg.num_turns}, cost=${cost:.4f}]", flush=True)


async def main():
    csvs = sorted(p for p in WORK_DIR.glob("*.csv") if p.name != "best_answers.csv")
    if len(csvs) != 1:
        sys.exit(f"Expected exactly 1 input CSV in {WORK_DIR}, got {len(csvs)}: "
                 f"{[c.name for c in csvs]}")

    options = ClaudeAgentOptions(
        cwd=str(WORK_DIR),
        model=RATER_MODEL,
        permission_mode="default",
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": SYSTEM_PROMPT_APPEND,
        },
        allowed_tools=_AUTO_APPROVED_TOOLS,
        disallowed_tools=_HARD_DENIED_TOOLS,
        can_use_tool=_can_use_tool,
    )

    async with ClaudeSDKClient(options=options) as client:
        print(f"=== Picking best answers in {csvs[0].name} ===\n", flush=True)
        await client.query(RATING_PROMPT)
        async for msg in client.receive_response():
            _print_message(msg)

    out = WORK_DIR / "best_answers.csv"
    if not out.exists():
        sys.exit(f"Rater finished but {out} was not created.")


if __name__ == "__main__":
    asyncio.run(main())
