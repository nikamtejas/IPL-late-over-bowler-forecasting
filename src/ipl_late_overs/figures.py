"""Stage: figures. Static report figures (PNG) in reports/figures/."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config, schemas  # noqa: E402

log = logging.getLogger(__name__)

# Report styling (reference palette: one series hue, text in ink tokens, recessive grid).
SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
SERIES_2 = "#eb6834"  # categorical slot 2


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def runs_by_over() -> None:
    """Small multiples: distribution of runs_in_over for each late over (15-20)."""
    late = schemas.read_table(config.path("late_overs"), schemas.LATE_OVERS)
    overs = sorted(late["over_number"].unique())
    cap = int(np.ceil(late["runs_in_over"].quantile(0.995)))
    bins = np.arange(0, cap + 2)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, n in zip(axes.flat, overs):
        r = late.loc[late["over_number"] == n, "runs_in_over"].clip(upper=cap).to_numpy()
        share = np.histogram(r, bins=bins)[0] / len(r) * 100
        ax.bar(bins[:-1], share, width=0.8, color=SERIES, linewidth=0)
        _style(ax)
        ax.set_title(f"Over {n}", loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.text(0.98, 0.95, f"n = {len(r):,}\nmean {r.mean():.1f}  median {np.median(r):.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
    for ax in axes[-1]:
        ax.set_xlabel(f"runs in the over ({cap}+ grouped at {cap})", fontsize=8, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of overs", fontsize=8, color=INK_2)
    fig.suptitle("Runs scored in late overs of IPL chases, by over", x=0.06, ha="left", fontsize=12, color=INK)
    fig.text(0.06, 0.925, f"{len(late):,} started overs from {late['match_id'].nunique():,} unrevised chases, "
             f"{late['season'].min()}-{late['season'].max()}. Later overs have fewer rows because chases end.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0.04, 0, 1, 0.91))
    p = config.path("fig_runs_by_over")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", p.relative_to(config.REPO_ROOT))


def validation_baselines() -> None:
    """Grouped bars: validation MAE of baselines A and B per fold, plus the mean over folds."""
    res = schemas.read_json(config.path("validation_baselines"), schemas.VALIDATION_BASELINES_KEYS)
    labels = [f"Fold {f['fold']}\nvalidate {f['validation_season']}" for f in res["folds"]] + ["Mean of folds"]
    a = [f["mae_baseline_a"] for f in res["folds"]] + [res["summary"]["mean_fold_mae_baseline_a"]]
    b = [f["mae_baseline_b"] for f in res["folds"]] + [res["summary"]["mean_fold_mae_baseline_b"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for off, vals, color, name in ((-0.2, a, SERIES, "A: bowler's historical late-over mean"),
                                   (0.2, b, SERIES_2, "B: scoreboard Poisson regression")):
        bars = ax.bar(x + off, vals, width=0.38, color=color, label=name, linewidth=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    _style(ax)
    ax.set_xticks(x, labels, fontsize=8, color=INK_2)
    ax.set_ylabel("MAE (runs per over)", fontsize=8, color=INK_2)
    ax.set_ylim(0, max(a + b) * 1.15)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)
    fig.suptitle("Validation MAE of the two baselines (lower is better)", x=0.08, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    p = config.path("fig_validation_baselines")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", p.relative_to(config.REPO_ROOT))


def main() -> None:
    runs_by_over()
    validation_baselines()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
