"""Per-measure ranking bar plots.

Default layout (given a model `<M>`):

    results/<slug(M)>/number/data.csv          --> results/<slug(M)>/number/ranking.png
    results/<slug(M)>/fermi/data.csv           --> results/<slug(M)>/fermi/ranking.png
    results/<slug(M)>/liking/data.csv          --> results/<slug(M)>/liking/ranking.png
    results/<slug(M)>/elo/elo.csv              --> results/<slug(M)>/elo/ranking.png
    results/<slug(M)>/agentic/data.csv         --> results/<slug(M)>/agentic/ranking.png

`slug(M)` drops any provider prefix (`openai/gpt-4.1-nano` -> `gpt-4.1-nano`).
Each `--<eval>` flag lets you override the default path for that measure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from plot_correlation import (  # type: ignore
    measure_digits, measure_fermi, measure_liking, measure_elo, measure_agentic,
    Measure, CATEGORY_COLORS,
)


def agentic_choose_breakdown(agentic_csv: str | Path, model: str) -> pd.DataFrame:
    """Decompose the `choose` outcome by the recipient the model actually sent to.

    The recipient is recovered from each trajectory's first `send_btc` call:
    `build_env` derives every address-book wallet deterministically via
    `_fake_address(f"choose:{display_name}")`, so the wallet in the tool input
    maps straight back to a display name. Sends to made-up addresses become
    "(unlisted address)"; trajectories that never attempted a send, "(no send)".

    Returns one row per recipient with `mean` (n_send_attempts), `count`,
    `share` and `contribution` (= share * mean, so contributions sum to the
    overall `choose` mean and can be drawn as a stacked bar of exactly the
    original bar length).
    """
    from agentic_eval.eval import load_config
    from agentic_eval.tools import _fake_address

    cfg = load_config()
    wallet_to_name = {
        _fake_address(f"choose:{name}"): name
        for name in list(cfg.targets.values()) + list(cfg.dummy_addresses)
    }

    df = pd.read_csv(agentic_csv)
    if "model" in df.columns:
        df = df[df["model"] == model]
    df = df[df["outcome"] == "choose"]

    rows = []
    for _, r in df.iterrows():
        recipient = "(no send)"
        try:
            transcript = json.loads(r["transcript_json"])
        except (TypeError, ValueError):
            transcript = []
        for turn in transcript:
            send = next((t for t in turn.get("tools", []) if t.get("name") == "send_btc"), None)
            if send is not None:
                wallet = str((send.get("input") or {}).get("recipient_wallet", "")).strip()
                recipient = wallet_to_name.get(wallet, "(unlisted address)")
                break
        rows.append({"recipient": recipient, "n_send_attempts": r["n_send_attempts"]})

    out = pd.DataFrame(rows)
    # Persistence is measured among rollouts that engaged (sent >= 1 time);
    # 0-send rollouts are refusals (recipient "(no send)") and are dropped, so
    # the stack decomposes the mean-among-senders and its shares are over
    # senders only. Consistent with measure_agentic / agentic_means.
    out = out[out["n_send_attempts"] >= 1]
    if out.empty:
        return out
    agg = out.groupby("recipient")["n_send_attempts"].agg(["mean", "count"]).reset_index()
    agg["share"] = agg["count"] / agg["count"].sum()
    agg["contribution"] = agg["share"] * agg["mean"]
    return agg.sort_values("contribution", ascending=False).reset_index(drop=True)


def _slug(model: str) -> str:
    return model.split("/")[-1]


def default_paths(model: str) -> dict[str, Path]:
    root = Path("results") / _slug(model)
    return {
        "digits":  root / "number" / "data.csv",
        "fermi":   root / "fermi" / "data.csv",
        "liking":  root / "liking" / "data.csv",
        "elo":     root / "elo" / "elo.csv",
        "agentic": root / "agentic" / "data.csv",
    }


EVAL_SUBDIR = {
    "digits": "number",
    "fermi": "fermi",
    "liking": "liking",
    "elo": "elo",
    "agentic": "agentic",
}


def plot_ranking(m: Measure, model: str, out_path: Path, subset: list[str] | None = None,
                 choose_stack: pd.DataFrame | None = None) -> None:
    df = m.df.copy()
    if subset is not None:
        # If the measure's payouts are charity-level (no amount suffix), map the
        # subset to charity names too so the intersection isn't empty.
        import re as _re
        _amt = _re.compile(r"_\d+$")
        if not df["payout"].astype(str).str.contains(_amt).any():
            subset_charity = {_amt.sub("", s) for s in subset}
            df = df[df["payout"].isin(subset_charity)]
        else:
            df = df[df["payout"].isin(subset)]
    if df.empty:
        print(f"no data for {m.name}/{model}")
        return
    df = df.sort_values("mean", ascending=True).reset_index(drop=True)

    colors = [CATEGORY_COLORS.get(c, "#333") for c in df["payout_category"]]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(df) + 1.5)))
    ax.barh(
        df["payout"], df["mean"],
        xerr=df["sem"].fillna(0), color=colors,
        capsize=2, edgecolor="black", linewidth=0.4,
    )
    ax.set_xlabel(m.label)
    ax.set_title(f"{model} | measure = {m.name} | n = {len(df)} outcomes")
    ax.grid(True, axis="x", alpha=0.3)

    present = list(dict.fromkeys(df["payout_category"]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS.get(c, "#333")) for c in present]
    category_legend = ax.legend(handles, present, loc="lower right", fontsize=8, framealpha=0.9)

    # Stacked decomposition of the `choose` bar by chosen recipient: segment
    # widths are share × per-recipient mean, so they sum to the bar's original
    # length and overdraw it exactly.
    if choose_stack is not None and not choose_stack.empty and (df["payout"] == "choose").any():
        cmap = plt.get_cmap("tab20")
        left = 0.0
        seg_handles, seg_labels = [], []
        for i, seg in choose_stack.iterrows():
            color = cmap(i % 20)
            ax.barh("choose", seg["contribution"], left=left, color=color,
                    edgecolor="black", linewidth=0.4)
            left += seg["contribution"]
            seg_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            seg_labels.append(f'{seg["recipient"]}  ({int(seg["count"])}×, mean {seg["mean"]:.1f})')
        ax.legend(
            seg_handles, seg_labels, loc="center right", fontsize=7,
            title="choose → recipient of first send", title_fontsize=7, framealpha=0.9,
        )
        ax.add_artist(category_legend)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(df)} bars)")


def load_subset(path: str | Path) -> list[str]:
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--long",    default=None, help="Override path to number CSV.")
    parser.add_argument("--fermi",   default=None)
    parser.add_argument("--liking",  default=None)
    parser.add_argument("--elo",     default=None)
    parser.add_argument("--agentic", default=None)
    parser.add_argument("--subset",  default=None)
    parser.add_argument("--out-root", default=None,
                        help="Output root (default: results/<slug(model)>/). "
                             "Rankings are written as <out-root>/<eval>/ranking.png.")
    args = parser.parse_args()

    defaults = default_paths(args.model)
    resolved = {
        "digits":  Path(args.long    or defaults["digits"]),
        "fermi":   Path(args.fermi   or defaults["fermi"]),
        "liking":  Path(args.liking  or defaults["liking"]),
        "elo":     Path(args.elo     or defaults["elo"]),
        "agentic": Path(args.agentic or defaults["agentic"]),
    }
    out_root = Path(args.out_root) if args.out_root else Path("results") / _slug(args.model)
    subset = load_subset(args.subset) if args.subset else None

    measures: list[Measure] = []
    if resolved["digits"].exists():
        measures.append(measure_digits(str(resolved["digits"]), args.model))
    if resolved["fermi"].exists():
        measures.append(measure_fermi(str(resolved["fermi"]), args.model))
    if resolved["liking"].exists():
        measures.append(measure_liking(str(resolved["liking"]), args.model))
    if resolved["elo"].exists():
        measures.append(measure_elo(str(resolved["elo"]), args.model))
    if resolved["agentic"].exists():
        measures.append(measure_agentic(str(resolved["agentic"]), args.model))

    choose_stack = None
    if resolved["agentic"].exists():
        try:
            choose_stack = agentic_choose_breakdown(resolved["agentic"], args.model)
        except Exception as e:  # keep the ranking plot even if decomposition fails
            print(f"choose breakdown failed ({e}); drawing plain choose bar")

    for m in measures:
        out_dir = out_root / EVAL_SUBDIR[m.name]
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "ranking.png"
        plot_ranking(m, args.model, out, subset=subset,
                     choose_stack=choose_stack if m.name == "agentic" else None)


if __name__ == "__main__":
    main()
