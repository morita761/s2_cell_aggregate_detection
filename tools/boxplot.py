#!/usr/bin/env python3
"""
tools/boxplot.py — Boxplot with strip overlay for S2 aggregation area data.

Usage:
    python3 tools/boxplot.py config.txt folder1/ folder2/ [... up to 6]

Each folder is treated as one condition.
All *_aggregations.csv files found in a folder are merged into one group.
The area_um2 column is plotted.

Config file: INI format (see config_example.txt)
"""

from __future__ import annotations

import argparse
import configparser
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MAX_GROUPS = 6


# ─── Helpers ─────────────────────────────────────────────────────────────────


def get_asterisk(p: float) -> str:
    if p < 0.00001:
        return "*****"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def load_area_data(folder: Path) -> list[float]:
    """Collect area_um2 from all *_aggregations.csv files in folder."""
    if not folder.exists():
        log.error(f"Folder not found: {folder}")
        return []

    csvs = sorted(folder.glob("*_aggregations.csv"))
    if not csvs:
        log.warning(f"No *_aggregations.csv found in {folder}")
        return []

    values: list[float] = []
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            log.error(f"Cannot read {csv_path}: {exc}")
            continue

        if "area_um2" not in df.columns:
            log.warning(f"'area_um2' column missing in {csv_path} — skipped")
            continue

        clean = df["area_um2"].dropna().tolist()
        if not clean:
            log.warning(f"No valid area_um2 values in {csv_path}")
        values.extend(clean)

    return values


def parse_comparisons(
    cfg: configparser.ConfigParser,
    n_groups: int,
) -> list[tuple[int, int]]:
    """Return list of (x1, x2) pairs (1-indexed) to compare.

    Reads [comparisons] section; falls back to consecutive pairs if absent.
    Each entry: pair1 = 1,2  or  pair1 = 1-2
    """
    if "comparisons" not in cfg:
        return [(i + 1, i + 2) for i in range(n_groups - 1)]

    pairs: list[tuple[int, int]] = []
    for key, val in cfg["comparisons"].items():
        val = val.strip()
        # Accept "1,2" or "1-2"
        sep = "," if "," in val else "-"
        parts = val.split(sep)
        if len(parts) != 2:
            log.warning(f"[comparisons] {key} = '{val}' is not a valid pair — skipped")
            continue
        try:
            a, b = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            log.warning(f"[comparisons] {key} = '{val}' contains non-integer — skipped")
            continue
        if not (1 <= a <= n_groups and 1 <= b <= n_groups):
            log.warning(
                f"[comparisons] {key} = '{val}' out of range (1–{n_groups}) — skipped"
            )
            continue
        if a == b:
            log.warning(f"[comparisons] {key} = '{val}' compares a group to itself — skipped")
            continue
        pairs.append((min(a, b), max(a, b)))

    if not pairs:
        log.warning("[comparisons] section found but no valid pairs; falling back to consecutive.")
        return [(i + 1, i + 2) for i in range(n_groups - 1)]

    # Deduplicate while preserving order
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def assign_bar_heights(
    pairs: list[tuple[int, int]],
    y_base: float,
    y_step: float,
    dy: float,
) -> list[float]:
    """Assign y heights so that bars with overlapping x-ranges don't collide.

    Strategy: sort by span width (narrower first) → assign heights in order,
    bumping up whenever the new bar would overlap an already-placed one.
    """
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1] - pairs[i][0])
    heights = [0.0] * len(pairs)
    placed: list[tuple[int, int, float]] = []  # (x1, x2, y)

    for idx in order:
        x1, x2 = pairs[idx]
        y = y_base
        while True:
            # A collision: x-ranges overlap AND bars are within dy of each other
            collision = any(
                not (x2 < px1 or x1 > px2) and abs(y - py) < dy
                for px1, px2, py in placed
            )
            if not collision:
                break
            y += dy
        heights[idx] = y
        placed.append((x1, x2, y))

    return heights


def draw_sig_bar(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y_base: float,
    dy: float,
    label: str,
) -> None:
    ax.plot(
        [x1, x1, x2, x2],
        [y_base, y_base + dy, y_base + dy, y_base],
        lw=1.5,
        color="black",
    )
    ax.text(
        (x1 + x2) / 2, y_base + dy,
        label,
        ha="center", va="bottom", fontsize=14,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boxplot of S2 aggregation area_um2 across conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="INI config file")
    parser.add_argument(
        "folders", type=Path, nargs="+", metavar="FOLDER",
        help=f"Output folders (up to {MAX_GROUPS}), one per condition",
    )
    args = parser.parse_args()

    if len(args.folders) > MAX_GROUPS:
        log.warning(f"More than {MAX_GROUPS} folders given; only the first {MAX_GROUPS} are used.")
    folders = args.folders[:MAX_GROUPS]

    # ── Config ────────────────────────────────────────────────────────────────
    if not args.config.exists():
        log.error(f"Config file not found: {args.config}")
        sys.exit(1)

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding="utf-8")

    labels_sec = dict(cfg["labels"]) if "labels" in cfg else {}
    plot_sec   = dict(cfg["plot"])   if "plot"   in cfg else {}

    title       = plot_sec.get("title",  "S2 Cell Aggregation Size")
    output_file = plot_sec.get("output", None)

    # ── Data loading ──────────────────────────────────────────────────────────
    group_labels: list[str]         = []
    group_data:   list[list[float]] = []

    for i, folder in enumerate(folders):
        label = labels_sec.get(f"label{i + 1}", folder.name)
        data  = load_area_data(folder)
        if not data:
            log.warning(f"Group '{label}' ({folder}) has no data — skipped.")
            continue
        log.info(f"Group '{label}': {len(data)} aggregations")
        group_labels.append(label)
        group_data.append(data)

    if not group_data:
        log.error("No data to plot. Exiting.")
        sys.exit(1)

    n = len(group_data)

    # ── Comparison pairs ──────────────────────────────────────────────────────
    pairs = parse_comparisons(cfg, n)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, n * 2 + 2), 6))

    hatch_patterns = ["", "///", "...", "xxx", "+++", r"\\\\"]

    bp = ax.boxplot(
        group_data,
        labels=group_labels,
        patch_artist=True,
        widths=0.5,
        zorder=2,
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)
        patch.set_hatch(hatch_patterns[i % len(hatch_patterns)])

    for part in ["whiskers", "caps", "medians"]:
        for elem in bp[part]:
            elem.set_color("black")
            elem.set_linewidth(1.5)

    for flier in bp["fliers"]:
        flier.set(marker="o", color="black", markersize=5, alpha=0.5)

    # Strip plot — jittered raw data points
    rng = np.random.default_rng(seed=42)
    for i, data in enumerate(group_data):
        jitter = rng.normal(0, 0.07, size=len(data))
        ax.scatter(
            np.full(len(data), i + 1) + jitter,
            data,
            color="black", alpha=0.35, s=18, zorder=3,
        )

    # n= labels
    all_vals   = [v for d in group_data for v in d]
    y_data_max = max(all_vals)
    y_n_label  = y_data_max * 1.04

    for i, data in enumerate(group_data):
        ax.text(i + 1, y_n_label, f"n={len(data)}", ha="center", fontsize=11)

    # ── Significance bars ─────────────────────────────────────────────────────
    y_sig_base = y_data_max * 1.15
    dy         = y_data_max * 0.10   # bar height
    y_step     = dy * 1.4            # collision detection granularity

    heights = assign_bar_heights(pairs, y_sig_base, y_step, dy)

    for (x1, x2), y_bar in zip(pairs, heights):
        # x1/x2 are label numbers (1-indexed); map back to group_data indices
        idx1, idx2 = x1 - 1, x2 - 1
        if idx1 >= n or idx2 >= n:
            log.warning(f"Pair ({x1},{x2}) references a group that was skipped — ignored.")
            continue

        d1, d2 = group_data[idx1], group_data[idx2]
        l1, l2 = group_labels[idx1], group_labels[idx2]

        if len(d1) < 2 or len(d2) < 2:
            log.warning(
                f"t-test '{l1}' vs '{l2}' skipped: "
                f"need ≥2 points each (got {len(d1)}, {len(d2)})."
            )
            continue

        try:
            _, p = ttest_ind(d1, d2)
            asters = get_asterisk(p)
            log.info(f"t-test '{l1}' vs '{l2}': p={p:.4g}  {asters}")
            draw_sig_bar(ax, x1, x2, y_bar, dy, asters)
        except Exception as exc:
            log.error(f"t-test '{l1}' vs '{l2}' failed: {exc}")

    # Axis styling
    y_top = max(heights, default=y_sig_base) + dy * 2.5
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_ylabel("Aggregation area (µm²)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_file:
        out = Path(output_file)
        try:
            fig.savefig(out, dpi=150, bbox_inches="tight")
            log.info(f"Saved → {out}")
        except Exception as exc:
            log.error(f"Failed to save figure: {exc}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
