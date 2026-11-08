"""Make the report's two figures from the five-seed results.

Figure 1 (fig_tradeoff): carbon emissions against average discomfort. Each
seed is a small dot; each configuration's mean is a large marker with +-1
standard-deviation bars. The four efficient configurations are highlighted.
RBC lies far outside the plotted range and is noted at the edge.

Figure 2 (fig_buildings): discomfort in each building, mean +-1 standard
deviation over the five seeds, for the coordination comparison and the four
fair-reward variants.

    .venv/Scripts/python.exe scripts/make_report_figures.py --output-dir ../figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_RUN = PROJECT_ROOT / "outputs" / "seed_matrix_summary" / "per_run.csv"

# Colours and ink from the data-viz reference palette (light mode, white page).
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
BUILDING_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 8,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "pdf.fonttype": 42,
    "savefig.facecolor": SURFACE,
})


def style_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def save(fig, output_dir: Path, name: str) -> None:
    for suffix in (".pdf", ".png"):
        fig.savefig(output_dir / f"{name}{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tradeoff_figure(runs: pd.DataFrame, output_dir: Path) -> None:
    # (config, label, efficient, label offset in points, horizontal alignment, leader line)
    configs = [
        ("marlisa", "Standard MARLISA", True, (12, -5), "left", False),
        ("independent_sac_training_encoders", "Independent SAC", False, (12, 0), "left", False),
        ("fair_worst_case", "Fair: worst-building only", True, (51, 0), "left", True),
        ("fair_carbon_aware_marlisa", "Fair carbon-aware MARLISA", False, (12, 3), "left", False),
        ("fair_disparity", "Fair: gap only", False, (-10, 0), "right", False),
        ("fair_no_fairness", "Fair: no fairness term", True, (-10, 0), "right", False),
        ("carbon_aware_marlisa", "Carbon-aware\nMARLISA", True, (0, -16), "center", False),
    ]
    # A broken x-axis: carbon-only MARLISA sits far left of the cluster at ~175 kg.
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(6.3, 3.3), sharey=True, gridspec_kw={"width_ratios": [1, 4.2], "wspace": 0.05},
    )
    for ax in (left, right):
        style_axes(ax, grid_axis="both")
    left.set_xlim(79, 89)
    left.set_xticks([80, 85])
    right.set_xlim(150, 197)
    right.set_xticks([150, 160, 170, 180, 190])
    left.set_ylim(0, 100)
    left.set_yticks([0, 25, 50, 75, 100])
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(left=False)
    for ax, x_pos in ((left, 1), (right, 0)):
        ax.plot([x_pos], [0], transform=ax.transAxes, marker=[(-1, -1.5), (1, 1.5)], markersize=7,
                linestyle="none", color=AXIS, mec=AXIS, mew=0.9, clip_on=False, zorder=5)

    label_box = dict(boxstyle="square,pad=0.08", fc=SURFACE, ec="none", alpha=0.85)
    for config, label, efficient, offset, align, leader in configs:
        sub = runs[runs["config"] == config]
        x = sub["carbon_kgco2"]
        y = sub["average_discomfort"] * 100
        ax = left if x.mean() < 120 else right
        colour = ACCENT if efficient else MUTED
        ax.scatter(x, y, s=12, color=colour, alpha=0.3, linewidths=0, zorder=2)
        ax.errorbar(
            x.mean(), y.mean(), yerr=y.std(), fmt="o", ms=7, mfc=colour, mec=SURFACE,
            mew=1.5, ecolor=colour, elinewidth=1.1, capsize=0, zorder=3,
        )
        ax.annotate(
            label, (x.mean(), y.mean()), xytext=offset, textcoords="offset points",
            ha=align, va="center", fontsize=8, color=INK_2, zorder=4, bbox=label_box,
            arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=0, shrinkB=5) if leader else None,
        )

    rbc = runs[runs["config"] == "rbc"].iloc[0]
    right.annotate(
        f"RBC: {rbc['carbon_kgco2']:.1f} kg, {rbc['average_discomfort'] * 100:.1f}%  →",
        (196.5, rbc["average_discomfort"] * 100), ha="right", va="center", fontsize=8, color=INK_2,
    )

    fig.supxlabel("Carbon emissions over the test period (kg CO$_2$)", fontsize=8, color=INK_2, y=0.0)
    left.set_ylabel("Average discomfort (% of occupied hours)")
    handles = [
        Line2D([], [], marker="o", ls="", ms=7, mfc=ACCENT, mec=SURFACE, mew=1.5, label="Efficient configurations"),
        Line2D([], [], marker="o", ls="", ms=7, mfc=MUTED, mec=SURFACE, mew=1.5, label="Other configurations"),
    ]
    right.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False, labelcolor=INK_2)
    save(fig, output_dir, "fig_tradeoff")


def rounded_bar(ax, x_centre: float, height: float, width: float, colour: str, radius_pt: float = 1.5) -> None:
    """Draw a bar with rounded top corners and a square base."""
    fig = ax.figure
    fig.canvas.draw()
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    radius_px = radius_pt * fig.dpi / 72
    rx = radius_px * (x1 - x0) / box.width
    ry = radius_px * (y1 - y0) / box.height
    left = x_centre - width / 2
    if height <= 2 * ry:
        ax.add_patch(Rectangle((left, 0), width, height, color=colour, lw=0, zorder=2))
        return
    ax.add_patch(FancyBboxPatch(
        (left, 0), width, height, boxstyle=f"round,pad=0,rounding_size={rx}",
        mutation_aspect=ry / rx, color=colour, lw=0, zorder=2,
    ))
    ax.add_patch(Rectangle((left, 0), width, height - ry, color=colour, lw=0, zorder=2))


def buildings_figure(runs: pd.DataFrame, output_dir: Path) -> None:
    groups = [
        ("independent_sac_training_encoders", "Independent\nSAC"),
        ("marlisa", "Standard\nMARLISA"),
        ("fair_no_fairness", "Fair: no\nfairness term"),
        ("fair_disparity", "Fair: gap\nonly"),
        ("fair_carbon_aware_marlisa", "Fair carbon-aware\nMARLISA"),
        ("fair_worst_case", "Fair: worst-\nbuilding only"),
    ]
    buildings = ["discomfort_building_1", "discomfort_building_2", "discomfort_building_3"]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    style_axes(ax, grid_axis="y")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    width, step = 0.24, 0.27

    for g, (config, _) in enumerate(groups):
        sub = runs[runs["config"] == config]
        for b, column in enumerate(buildings):
            values = sub[column] * 100
            x = g + (b - 1) * step
            rounded_bar(ax, x, values.mean(), width, BUILDING_COLOURS[b])
            ax.errorbar(x, values.mean(), yerr=values.std(), fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=0, zorder=3)

    ax.axvline(1.5, color=AXIS, linewidth=0.7, zorder=1)
    ax.text(0.5, 78, "Same reward, with or\nwithout coordination", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.text(3.5, 78, "Fair reward: same carbon and comfort weights,\ndifferent fairness terms", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([label for _, label in groups])
    ax.set_ylabel("Discomfort (% of occupied hours)")
    handles = [Patch(color=c, label=f"Building {i + 1}") for i, c in enumerate(BUILDING_COLOURS)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, labelcolor=INK_2)
    save(fig, output_dir, "fig_buildings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "report_figures")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(PER_RUN)
    tradeoff_figure(runs, output_dir)
    buildings_figure(runs, output_dir)
    print(f"written to {output_dir}")


if __name__ == "__main__":
    main()
