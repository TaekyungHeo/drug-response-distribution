"""Shared matplotlib style for paper figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers 'nature' style

PALETTE = [
    "#0072B2",  # Okabe-Ito blue
    "#D55E00",  # Okabe-Ito vermillion
    "#009E73",  # Okabe-Ito bluish green
    "#E69F00",  # Okabe-Ito orange
    "#56B4E9",  # Okabe-Ito sky blue
    "#CC79A7",  # Okabe-Ito reddish purple
    "#8A8A8A",  # gray
    "#000000",  # black
]

BLUE = "#0072B2"
RED = "#D55E00"
GREEN = "#009E73"
TAN = "#E69F00"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GRAY = "#8A8A8A"
LGRAY = "#BDBDBD"
BLACK = "#000000"
TEXT = "#222222"
PINK = "#CC79A7"

MM_PER_INCH = 25.4


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH


SINGLE = mm_to_in(89)
ONE_AND_HALF = mm_to_in(128)
FULL = mm_to_in(183)
MAX_HEIGHT = mm_to_in(170)

LINE = 0.55
THIN = 0.4
MARKER = 3.2


def apply() -> None:
    plt.style.use("nature")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "regular",
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.prop_cycle": mpl.cycler("color", PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": LINE,
        "axes.edgecolor": BLACK,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.major.width": LINE,
        "ytick.major.width": LINE,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "lines.linewidth": LINE,
        "patch.linewidth": THIN,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": False,
        "legend.borderpad": 0.25,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.35,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def panel_label(ax: mpl.axes.Axes, letter: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="right",
        color=BLACK,
        clip_on=False,
    )


def title(ax: mpl.axes.Axes, text: str) -> None:
    ax.set_title(text, loc="center", pad=3, fontweight="bold")


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", which="major", pad=2)


def savefig(fig: mpl.figure.Figure, path, **kwargs) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.035, **kwargs)
