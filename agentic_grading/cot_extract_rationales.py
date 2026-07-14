"""Extract rationales + chain-of-thought + picks for every trial in a run.

Output: /tmp/trials_<rater>.jsonl
One record per trial:
{
  "rater":          "claude" | "codex" | "qwen",
  "run_dir":        "<path>",
  "seed":           "rep_222333",
  "rater_text":     "<all assistant-visible text concatenated>",
  "thinking_text":  "<all CoT/thinking content>",
  "picks": [
     {"q_idx": 0, "question": "...", "picked_label": "gpt-4o", "picked_real": "claude-haiku-4.5", "picked_sample_idx": "1"},
     ...
  ],
  "n_picks":              10,
  "n_anthropic_picks":    int,
  "n_openai_picks":       int,
  "frac_anthropic":       float,
  "rater_chars":          int,
  "thinking_chars":       int,
}
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

ANTHROPIC_PREFIXES = ("claude",)
OPENAI_PREFIXES = ("gpt", "openai")
GOOGLE_PREFIXES = ("gemini", "palm", "bard")
ALIBABA_PREFIXES = ("qwen",)

# Houses we track explicitly. Anything that doesn't match -> "other".
TRACKED_HOUSES = ("anthropic", "openai", "google", "alibaba")

def label_house(label: str) -> str:
    s = label.lower()
    if s.startswith(ANTHROPIC_PREFIXES):
        return "anthropic"
    if any(s.startswith(p) for p in OPENAI_PREFIXES):
        return "openai"
    if s.startswith(GOOGLE_PREFIXES):
        return "google"
    if s.startswith(ALIBABA_PREFIXES):
        return "alibaba"
    return "other"


def extract_claude(log_path: Path) -> tuple[str, str, dict]:
    """Return (rater_text, thinking_text, behavior_dict)."""
    rater_parts = []
    think_parts = []
    n_assistant = 0
    n_tool_use = 0
    tool_names = Counter()
    with log_path.open() as f:
        for raw in f:
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            if ev.get("msg_type") != "AssistantMessage":
                continue
            n_assistant += 1
            for blk in ev.get("blocks") or []:
                btype = (blk.get("type") or "").lower()
                if btype in ("textblock", "text"):
                    txt = blk.get("text") or ""
                    if txt:
                        rater_parts.append(txt)
                elif btype in ("thinkingblock", "thinking"):
                    txt = blk.get("thinking") or blk.get("text") or ""
                    if txt:
                        think_parts.append(txt)
                elif btype in ("tool_use", "tooluse"):
                    n_tool_use += 1
                    name = blk.get("name") or "?"
                    tool_names[name] += 1
    behavior = {
        "n_assistant_turns": n_assistant,
        "n_tool_use": n_tool_use,
        "tool_names": dict(tool_names.most_common()),
    }
    return "\n\n".join(rater_parts), "\n\n".join(think_parts), behavior


def extract_codex(log_path: Path) -> tuple[str, str, dict]:
    rater_parts = []
    think_parts = []
    n_agent_messages = 0
    n_command_exec = 0
    n_web_search = 0
    n_turns = 0
    cmd_first_words = Counter()
    with log_path.open() as f:
        for raw in f:
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            ttop = ev.get("type")
            if ttop == "turn.completed":
                n_turns += 1
            if ttop != "item.completed":
                continue
            item = ev.get("item") or {}
            itype = item.get("type")
            if itype == "agent_message":
                txt = item.get("text") or ""
                if txt:
                    rater_parts.append(txt)
                    n_agent_messages += 1
            elif itype == "reasoning":
                txt = item.get("text") or item.get("summary") or ""
                if not txt:
                    # Some codex versions carry the summary as a list of
                    # {type,text} blocks under "content" instead of "text".
                    content = item.get("content")
                    if isinstance(content, list):
                        txt = "\n".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict)
                        ).strip()
                    elif isinstance(content, str):
                        txt = content
                if txt:
                    think_parts.append(txt)
            elif itype == "command_execution":
                n_command_exec += 1
                cmd = (item.get("command") or "").strip().lower()
                if cmd:
                    fw = cmd.split(maxsplit=1)[0]
                    cmd_first_words[fw] += 1
            elif itype == "web_search":
                n_web_search += 1
    behavior = {
        "n_agent_messages": n_agent_messages,
        "n_command_exec": n_command_exec,
        "n_web_search": n_web_search,
        "n_turns": n_turns,
        "cmd_first_words": dict(cmd_first_words.most_common(10)),
    }
    return "\n\n".join(rater_parts), "\n\n".join(think_parts), behavior


def extract_qwen(log_path: Path) -> tuple[str, str, dict]:
    """Qwen rate_qwen.py writes one Header + one Final line per trial
    (the Final line carries the full final messages list — qwen_agent
    yields cumulative state, so the last yield subsumes every prior).
    For older runs that streamed per-chunk we also accept "Chunk" /
    "agent_messages" as a fallback and take the highest chunk_idx.
    """
    rater_parts = []
    think_parts = []
    n_assistant = 0
    n_tool_calls = 0
    tool_names = Counter()
    last_snapshot = None
    last_chunk_idx = -1
    with log_path.open() as f:
        for raw in f:
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            mt = ev.get("msg_type")
            if mt == "Final":
                last_snapshot = ev.get("messages") or []
                break  # Final always wins
            if mt in ("Chunk", "agent_messages"):
                idx = ev.get("chunk_idx") or 0
                if idx >= last_chunk_idx:
                    last_chunk_idx = idx
                    last_snapshot = ev.get("messages") or []
    if last_snapshot:
        for msg in last_snapshot:
            if msg.get("role") != "assistant":
                continue
            n_assistant += 1
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                rater_parts.append(content)
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                think_parts.append(reasoning)
            fc = msg.get("function_call")
            if isinstance(fc, dict):
                n_tool_calls += 1
                tool_names[fc.get("name") or "?"] += 1
            elif isinstance(fc, list):
                for tc in fc:
                    n_tool_calls += 1
                    if isinstance(tc, dict):
                        tool_names[tc.get("name") or "?"] += 1
    behavior = {
        "n_assistant_turns": n_assistant,
        "n_tool_calls": n_tool_calls,
        "tool_names": dict(tool_names.most_common()),
    }
    return "\n\n".join(rater_parts), "\n\n".join(think_parts), behavior


def join_picks(replicates_dir: Path, rated_dir: Path, seed_name: str) -> list[dict]:
    """Match best_answers.csv rows back to source rep_<seed>.csv rows
    on (question, answer)."""
    src_csv = replicates_dir / f"{seed_name}.csv"
    if not src_csv.exists():
        # Some runs keep only the per-trial input copy
        src_csv = rated_dir / seed_name / f"{seed_name}.csv"
    pick_csv = rated_dir / seed_name / "best_answers.csv"
    if not src_csv.exists() or not pick_csv.exists():
        return []

    with src_csv.open() as f:
        src_rows = list(csv.DictReader(f))
    with pick_csv.open() as f:
        pick_rows = list(csv.DictReader(f))

    # build lookup keyed on (question, answer) -> {real_model, model, sample_idx}
    by_qa = {}
    for r in src_rows:
        by_qa[(r["question"], r["answer"])] = r

    picks = []
    for q_idx, p in enumerate(pick_rows):
        key = (p.get("question", ""), p.get("answer", ""))
        src = by_qa.get(key)
        if src is None:
            picks.append({
                "q_idx": q_idx,
                "question": p.get("question", ""),
                "picked_label": None,
                "picked_real": None,
                "picked_sample_idx": None,
                "match_failed": True,
            })
            continue
        picks.append({
            "q_idx": q_idx,
            "question": p.get("question", ""),
            "picked_label": src.get("model"),
            "picked_real": src.get("real_model"),
            "picked_sample_idx": src.get("sample_idx"),
            "match_failed": False,
        })
    return picks


def process_run(rater: str, run_dir: Path, out_path: Path, extractor):
    n_written = 0
    with out_path.open("w") as out:
        for seed_dir in sorted((run_dir / "rated").glob("rep_*")):
            log_path = seed_dir / "rater_log.jsonl"
            if not log_path.exists():
                continue
            seed = seed_dir.name
            rater_text, thinking_text, behavior = extractor(log_path)
            picks = join_picks(run_dir / "replicates", run_dir / "rated", seed)
            houses = [label_house(p.get("picked_label") or "") for p in picks]
            picks_by_house = {h: 0 for h in TRACKED_HOUSES}
            picks_by_house["other"] = 0
            for h in houses:
                picks_by_house[h] = picks_by_house.get(h, 0) + 1
            n_anthro = picks_by_house.get("anthropic", 0)
            n_openai = picks_by_house.get("openai", 0)
            n_picks = sum(1 for h in houses if h != "other")
            rec = {
                "rater": rater,
                "run_dir": str(run_dir),
                "seed": seed,
                "rater_text": rater_text,
                "thinking_text": thinking_text,
                "behavior": behavior,
                "picks": picks,
                "n_picks": n_picks,
                "n_anthropic_picks": n_anthro,
                "n_openai_picks": n_openai,
                "picks_by_house": picks_by_house,
                "frac_anthropic": (n_anthro / n_picks) if n_picks else None,
                "rater_chars": len(rater_text),
                "thinking_chars": len(thinking_text),
            }
            out.write(json.dumps(rec) + "\n")
            n_written += 1
    print(f"wrote {n_written} trials -> {out_path}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "claude":
        run = Path(sys.argv[2])
        out = Path(sys.argv[3])
        process_run("claude", run, out, extract_claude)
    elif cmd == "codex":
        run = Path(sys.argv[2])
        out = Path(sys.argv[3])
        process_run("codex", run, out, extract_codex)
    elif cmd == "qwen":
        run = Path(sys.argv[2])
        out = Path(sys.argv[3])
        process_run("qwen", run, out, extract_qwen)
    else:
        print(f"unknown cmd {cmd}", file=sys.stderr)
        sys.exit(2)
