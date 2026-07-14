"""Shared figure styling for the final_scripts plots.

Importing this module applies one consistent set of matplotlib rcParams to every
final-figure script, so the whole paper shares a single sizing scheme. Change a
number here to restyle every figure at once -- this is the single source of truth.

    TICK_FS        x / y tick labels                       (xtick/ytick.labelsize)
    AXIS_LABEL_FS  x / y axis labels                       (axes.labelsize)
    TITLE_FS       subplot titles, i.e. ax.set_title       (axes.titlesize)
    SUPTITLE_FS    figure suptitles, i.e. fig.suptitle     (figure.titlesize)
    LEGEND_FS      legend text                             (legend.fontsize)
    VALUE_FS       numeric value labels drawn on bars      (ax.text -- NOT an
                   rcParam, so pass fontsize=VALUE_FS explicitly at those sites)
    HEADER_FS      in-plot family/group header labels      (ax.text -- NOT an
                   rcParam, so pass fontsize=HEADER_FS explicitly at those sites)
    ANNOT_FS       in-plot annotations / stat boxes         (ax.text -- NOT an
                   rcParam, so pass fontsize=ANNOT_FS explicitly at those sites)
    COUNT_FS       small per-bar n= / count annotations     (ax.text -- NOT an
                   rcParam, so pass fontsize=COUNT_FS explicitly at those sites)

So that the rcParams above actually take effect, call sites should NOT pass an
explicit ``fontsize=`` for ticks / axis labels / titles / suptitles / legends --
let them inherit. Bar value labels and family/group headers (ax.text) pass
``fontsize=VALUE_FS`` / ``fontsize=HEADER_FS`` / ``fontsize=ANNOT_FS`` explicitly.
"""
import matplotlib as mpl

TICK_FS = 11
AXIS_LABEL_FS = 12
TITLE_FS = 14
SUPTITLE_FS = 16
LEGEND_FS = 11
VALUE_FS = 10
HEADER_FS = 13
ANNOT_FS = 11
COUNT_FS = 8

mpl.rcParams.update({
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "axes.labelsize": AXIS_LABEL_FS,
    "axes.titlesize": TITLE_FS,
    "figure.titlesize": SUPTITLE_FS,
    "legend.fontsize": LEGEND_FS,
})
