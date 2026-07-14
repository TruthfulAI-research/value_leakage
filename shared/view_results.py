"""Open the llmcomp Streamlit viewer on a results DataFrame.

Split out from `shared.runner` so the rest of the runner can be llmcomp-free.
The only llmcomp dependency that survives in `shared/` is the viewer
(`llmcomp.Question.view`); import this module only when you actually want to
launch the viewer.
"""

import socket

from llmcomp import Question


def _free_port(preferred=8501):
    """Return ``preferred`` if free, else any free port the OS gives us."""
    for p in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port available")


def view_results(view_df, port=None):
    """Open the llmcomp viewer on ``view_df``.

    Picks a free port (8501 if available, else whatever the OS hands us)
    so a stale viewer running on the default port doesn't make the
    streamlit subprocess fail with CalledProcessError.
    """
    view_df = view_df.copy()
    view_df["api_kwargs"] = view_df["prompt"].apply(
        lambda p: {"messages": [{"role": "user", "content": p}]}
    )
    view_df["answer"] = view_df.apply(
        lambda r: f"<thinking>\n{r['reasoning']}\n</thinking>\n\n{r['answer']}"
        if r["reasoning"] else r["answer"],
        axis=1,
    )
    view_df.drop(columns=["reasoning", "prompt"], inplace=True)
    Question.view(view_df, port=port or _free_port())
