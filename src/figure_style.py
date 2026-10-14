from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "navy": "#0F4D92",
    "blue": "#3775BA",
    "orange": "#E28E2C",
    "green": "#4C956C",
    "gold": "#D4A72C",
    "purple": "#9A4D8E",
    "grey": "#767676",
    "light_grey": "#CFCECE",
    "black": "#272727",
    "red": "#B64342",
    "teal": "#42949E",
    "pale_blue": "#DCE8F3",
    "pale_red": "#F6CFCB",
    "pale_gold": "#F2E6D9",
}


def set_nature_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(
        x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top"
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, base, dpi=600):
    for ext in ("png", "pdf", "svg", "tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(f"{base}.{ext}", **kwargs)
    plt.close(fig)
