"""Render one real agentic rollout as a standalone LaTeX figure, in the spirit
of the interactive rollout-explorer widget: agent text per turn, tool calls
with their outputs, failures in red, the voluntary stop in yellow.

The output compiles on its own (pdflatex, Overleaf) and the part between the
%% BEGIN FIGURE / %% END FIGURE markers can be \\input into a paper.

    python scripts/make_rollout_figure_tex.py \
        --model-slug claude-opus-4-8 --outcome humane_league --sample 9 \
        --out results/paper/fig_rollout_example.tex
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "final_data" / "agentic_effort"
_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def esc(text: str) -> str:
    out = "".join(TEX_SPECIALS.get(ch, ch) for ch in text)
    # Markdown remnants and unicode the paper font won't have.
    out = out.replace("✅", r"\textcolor{RoOkGreen}{$\surd$}~")
    out = out.replace("❌", r"\textcolor{RoFailRed}{$\times$}~")
    out = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", out)
    out = re.sub(r"`(.+?)`", lambda m: r"\texttt{" + short_wallet(m.group(1)) + "}", out)
    # Shorten bare wallet addresses so they never overflow a column.
    out = re.sub(r"bc1q[0-9a-f]{20,}", lambda m: short_wallet(m.group(0)), out)
    out = out.replace("—", "---").replace("–", "--")
    return out


def esc_block(text: str) -> str:
    """Escape + preserve line structure: blank lines stay paragraph breaks,
    single newlines become forced line breaks."""
    paragraphs = []
    for para in text.split("\n\n"):
        lines = [esc(l) for l in para.splitlines() if l.strip()]
        paragraphs.append(" \\\\\n".join(lines))
    return "\n\n".join(p for p in paragraphs if p)


def short_wallet(w: str) -> str:
    return w if len(w) <= 16 else f"{w[:10]}\\dots {w[-4:]}"


def call_sig(name: str, inputs: dict) -> str:
    parts = []
    for k, v in (inputs or {}).items():
        v = short_wallet(str(v)) if k == "recipient_wallet" else esc(str(v))
        parts.append(f"{esc(k)}={v}")
    return rf"\texttt{{{esc(name)}({', '.join(parts)})}}"


def is_failure(output: str) -> bool:
    lowered = output.lower()
    return any(w in lowered for w in ("rejected", "not accepted", "error", "fail", "unavailable", "timeout", "backpressure"))


def build_task_prompt(outcome: str) -> str:
    """Reconstruct the exact initial user message (agentic_eval.eval logic)."""
    from agentic_eval.eval import load_config
    cfg = load_config()
    if outcome == "choose":
        return cfg.choose_task
    return cfg.task_template.format(target=cfg.targets[outcome])


def render(transcript: list[dict], header: str, task_prompt: str | None = None) -> str:
    lines: list[str] = []
    lines.append(r"\begin{robox}{" + header + "}")
    if task_prompt:
        lines.append(r"\par\vspace{2pt}\noindent\textsf{\tiny\bfseries\textcolor{RoUserBlue}{USER}}\par\vspace{1pt}\noindent")
        lines.append(r"\begin{rouser}")
        lines.append(r"\footnotesize " + esc_block(task_prompt))
        lines.append(r"\end{rouser}")
    for t in transcript:
        turn = t.get("turn", 0)
        lines.append(rf"\roturn{{{turn + 1}}}")
        if t.get("text", "").strip():
            lines.append(esc_block(t["text"].strip()))
            lines.append("")
        outputs = {i: o for i, o in enumerate(t.get("tool_outputs", []))}
        for i, call in enumerate(t.get("tools", [])):
            output = outputs.get(i, {}).get("output", "")
            failed = call["name"] == "send_btc" and is_failure(output)
            style = "rotoolfail" if failed else "rotoolok"
            lines.append(rf"\begin{{{style}}}")
            lines.append(rf"\rocall{{{call_sig(call['name'], call.get('input') or {})}}}")
            if output:
                lines.append(rf"\roout{{{esc_block(output.strip())}}}")
            lines.append(rf"\end{{{style}}}")
        if not t.get("tools"):
            lines.append(r"\rostops")
    lines.append(r"\end{robox}")
    return "\n".join(lines)


MACROS = r"""% --- rollout-figure macros (self-contained; only needs \usepackage[most]{tcolorbox}
% and xcolor, both already in the paper preamble; all names are Ro/ro-prefixed
% to avoid collisions) ------------------------------------------------------
\definecolor{RoOkGreen}{HTML}{2F9E44}
\definecolor{RoFailRed}{HTML}{C92A2A}
\definecolor{RoStopYellow}{HTML}{E67700}
\definecolor{RoToolGray}{HTML}{555555}
\definecolor{RoBoxBorder}{HTML}{999999}
\definecolor{RoUserBlue}{HTML}{1971C2}
\newtcolorbox{robox}[1]{
  enhanced, colback=white, colframe=RoBoxBorder, boxrule=0.6pt,
  arc=2pt, left=6pt, right=6pt, top=2pt, bottom=2pt,
  title={\footnotesize\bfseries #1}, coltitle=black, colbacktitle=black!6,
  fonttitle=\sffamily, fontupper=\footnotesize}
\newtcolorbox{rotoolfail}{colback=RoFailRed!6, colframe=RoFailRed!70, boxrule=0pt,
  leftrule=2pt, arc=1pt, left=5pt, right=4pt, top=2pt, bottom=2pt,
  before skip=1.5pt, after skip=1.5pt}
\newtcolorbox{rotoolok}{colback=black!4, colframe=RoToolGray!70, boxrule=0pt,
  leftrule=2pt, arc=1pt, left=5pt, right=4pt, top=2pt, bottom=2pt,
  before skip=1.5pt, after skip=1.5pt}
\newtcolorbox{rouser}{colback=RoUserBlue!7, colframe=RoUserBlue!70, boxrule=0pt,
  leftrule=2pt, arc=1pt, left=5pt, right=4pt, top=2pt, bottom=2pt,
  before skip=1.5pt, after skip=1.5pt}
\newcommand{\roturn}[1]{\par\vspace{2pt}\noindent\textsf{\tiny\bfseries\textcolor{RoToolGray}{TURN #1}}\par\vspace{1pt}\noindent}
\newcommand{\rocall}[1]{\noindent\scriptsize #1\par}
\newcommand{\roout}[1]{\noindent\scriptsize\itshape\textcolor{RoToolGray}{$\hookrightarrow$ #1}\par}
\newcommand{\rostops}{\begin{tcolorbox}[colback=RoStopYellow!8, colframe=RoStopYellow!80,
  boxrule=0pt, leftrule=2pt, arc=1pt, left=5pt, right=4pt, top=2pt, bottom=2pt,
  before skip=3pt, after skip=3pt]
  \scriptsize\sffamily\textcolor{RoStopYellow}{\textbf{agent stops}
  (turn ends with no tool call)}\end{tcolorbox}}
% ---------------------------------------------------------------------------
"""

FIGURE_HEADER = r"""% Auto-generated by scripts/make_rollout_figure_tex.py.
% A complete figure: \input this file (or paste it) anywhere in the document.
% Only needs \usepackage[most]{tcolorbox} and xcolor, both already loaded in
% the paper preamble. All helper macros are defined INSIDE the figure group
% (Ro/ro-prefixed), so they stay local and cannot clash with the paper.
"""

PREAMBLE = r"""% Auto-generated by scripts/make_rollout_figure_tex.py — standalone preview
% of fig_rollout_example_figure.tex.
\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{xcolor}
\usepackage[most]{tcolorbox}
\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-slug", default="claude-opus-4-8")
    parser.add_argument("--outcome", default="humane_league")
    parser.add_argument("--sample", type=int, default=9)
    parser.add_argument("--out", default=str(_RESULTS_ROOT / "paper" / "fig_rollout_example.tex"))
    args = parser.parse_args()

    df = pd.read_csv(str(_DATA_ROOT / args.model_slug / "agentic" / "data.csv"))
    row = df[(df["outcome"] == args.outcome) & (df["sample"] == args.sample)].iloc[0]
    transcript = json.loads(row["transcript_json"])

    header = (
        f"Agentic persistence rollout --- {args.model_slug}, "
        f"target: {esc(args.outcome)} (sample {args.sample}, "
        f"{int(row['n_send_attempts'])} send attempts, stop: {esc(row['stop_reason'])})"
    )
    body = render(transcript, header, task_prompt=build_task_prompt(args.outcome))
    caption = (
        r"\caption{A representative agentic-persistence rollout ("
        + esc(args.model_slug) + ", target: " + esc(args.outcome)
        + r"). The agent retries the always-failing \texttt{send\_btc} tool "
        + r"with increasing waits before giving up voluntarily.}"
        + "\n" + r"\label{fig:rollout-example}"
    )
    figure = (FIGURE_HEADER + "\\begin{figure}[t]\n" + MACROS + "\n"
              + body + "\n" + caption + "\n\\end{figure}\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure_path = out.with_name(out.stem + "_figure.tex")
    figure_path.write_text(figure)
    out.write_text(PREAMBLE + figure + POSTAMBLE)
    print(f"wrote {figure_path} (pure figure) and {out} (standalone preview), {len(transcript)} turns")


if __name__ == "__main__":
    main()
