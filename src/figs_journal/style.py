"""Shared style for the paper-B journal figures (vector PDF, Type-42 fonts, >= 7 pt, Okabe-Ito colour-blind-safe palette)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = f"{ROOT}/results/T6/figs_journal"
COL1, COL2 = 3.5, 7.0  # inches: single / double column

# Okabe-Ito
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    p = f"{OUT}/{name}"
    fig.savefig(p)
    print("wrote", p)
