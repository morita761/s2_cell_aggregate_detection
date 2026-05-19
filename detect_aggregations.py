#!/usr/bin/env python3
"""
S2 cell aggregation detection from nd2 fluorescence images.

Segmentation strategy: cell occupancy estimation (not boundary detection).
Pipeline per field: MIP → normalize → Gaussian blur → Otsu threshold
→ binary_fill_holes → morphology close → connected components → area filter.

Usage:
    python detect_aggregations.py input.nd2 output_dir/ [options]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Generator, NamedTuple

import cv2
import nd2
import numpy as np
from scipy.ndimage import binary_fill_holes


# ─── Configuration ────────────────────────────────────────────────────────────

_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red":     (1, 0, 0),
    "green":   (0, 1, 0),
    "blue":    (0, 0, 1),
    "cyan":    (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow":  (1, 1, 0),
    "white":   (1, 1, 1),
}

CHANNEL_COLORS: dict[int, str] = {
    0: "green",
    1: "red",
    2: "blue",
}


class Config(NamedTuple):
    s2_diameter_um: float = 10.0      # S2 cell diameter in µm (typical 8–12 µm)
    aggregation_min_cells: int = 3    # aggregation threshold = this many S2 cells
    gaussian_ksize: int = 25          # Gaussian blur kernel size (forced odd)
    morph_close_radius: int = 5       # disk radius for morphology close (pixels)
    min_active_channels: int = 2           # reject regions with fewer active channels
    channel_pixel_threshold: float = 10.0  # per-pixel intensity (0–255) to call a pixel "positive"
    channel_occupancy_fraction: float = 0.3  # minimum channel occupancy as fraction of single S2 cell area
    channel_colors: dict[int, str] = CHANNEL_COLORS


def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def compute_area_threshold_px(
    pixel_size_um: float,
    s2_diameter_um: float,
    min_cells: int,
) -> float:
    """Convert S2-cell-count threshold to pixel area."""
    cell_area_um2 = np.pi * (s2_diameter_um / 2.0) ** 2
    threshold_um2 = cell_area_um2 * min_cells
    return threshold_um2 / (pixel_size_um ** 2)


def compute_channel_occupancy_threshold_px(
    pixel_size_um: float,
    s2_diameter_um: float,
    occupancy_fraction: float,
) -> float:
    """Minimum positive-pixel count within a component for a channel to be considered active.

    Derived from occupancy_fraction × single-S2-cell area.  Using a fraction
    rather than the full cell area tolerates expression heterogeneity and
    partial focal-plane coverage.
    """
    single_cell_area_um2 = np.pi * (s2_diameter_um / 2.0) ** 2
    min_occupancy_um2 = single_cell_area_um2 * occupancy_fraction
    return min_occupancy_um2 / (pixel_size_um ** 2)


# ─── nd2 loading ──────────────────────────────────────────────────────────────


def load_nd2(path: Path) -> tuple[np.ndarray, dict[str, int], float]:
    """
    Load an nd2 file.

    Returns
    -------
    data : np.ndarray
    sizes : dict[str, int]  e.g. {'P': 3, 'Z': 10, 'C': 3, 'Y': 512, 'X': 512}
    pixel_size_um : float
    """
    with nd2.ND2File(path) as f:
        data = np.asarray(f.asarray())
        sizes = dict(f.sizes)
        voxel = f.voxel_size()
        pixel_size_um = float(voxel.x)
    return data, sizes, pixel_size_um


def iter_fields(
    data: np.ndarray, sizes: dict[str, int]
) -> Generator[tuple[int, np.ndarray, dict[str, int]], None, None]:
    """Yield (field_id, field_array, field_sizes) for each position."""
    dim_keys = list(sizes.keys())
    if "P" in dim_keys:
        p_axis = dim_keys.index("P")
        n_fields = sizes["P"]
        field_sizes = {k: v for k, v in sizes.items() if k != "P"}
        for fid in range(n_fields):
            yield fid, np.take(data, fid, axis=p_axis), field_sizes
    else:
        yield 0, data, sizes


# ─── Image processing helpers ─────────────────────────────────────────────────


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to uint8."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def max_projection(stack: np.ndarray) -> np.ndarray:
    """Max-intensity projection along axis 0.  Input: (Z, Y, X)."""
    return stack.max(axis=0)


def extract_channel_projections(
    field_data: np.ndarray, sizes: dict[str, int]
) -> list[np.ndarray]:
    """Return a list of uint8 MIP images, one per channel."""
    dim_keys = list(sizes.keys())
    n_channels = sizes.get("C", 1)
    projections: list[np.ndarray] = []

    for ch in range(n_channels):
        arr = field_data

        if "C" in dim_keys:
            c_axis = dim_keys.index("C")
            arr = np.take(arr, ch, axis=c_axis)
            remaining = [k for k in dim_keys if k != "C"]
        else:
            remaining = list(dim_keys)

        if "Z" in remaining:
            z_axis = remaining.index("Z")
            arr = arr.max(axis=z_axis)

        projections.append(to_uint8(arr))

    return projections


def merge_channels(channel_projections: list[np.ndarray]) -> np.ndarray:
    """Per-pixel maximum across channels → grayscale uint8 for segmentation."""
    return np.maximum.reduce(channel_projections)


def apply_pseudo_color(gray: np.ndarray, color: str) -> np.ndarray:
    """Map uint8 grayscale to pseudo-colored RGB (H, W, 3)."""
    weights = np.array(_COLOR_RGB[color], dtype=np.uint8)
    return gray[..., np.newaxis] * weights


def pseudo_color_merge(
    channel_projections: list[np.ndarray],
    channel_colors: dict[int, str],
) -> np.ndarray:
    """Assign pseudo-colors and merge by per-pixel maximum → uint8 RGB."""
    h, w = channel_projections[0].shape
    merged_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, proj in enumerate(channel_projections):
        color = channel_colors.get(i, "white")
        merged_rgb = np.maximum(merged_rgb, apply_pseudo_color(proj, color))
    return merged_rgb


# ─── Occupancy-based segmentation ────────────────────────────────────────────


def segment_occupancy(
    img: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cell occupancy estimation pipeline.

    Steps
    -----
    1. Gaussian blur   — suppress shot noise and point-like membrane fluorescence
    2. Otsu threshold  — global, data-driven binarization
    3. binary_fill_holes — fill dark nuclei so donut → disk
    4. morphology close  — bridge sub-pixel gaps without merging distant cells

    Returns
    -------
    mask_otsu   : uint8 binary after Otsu
    mask_filled : uint8 binary after hole filling
    mask_final  : uint8 binary after morphology close (used for detection)
    """
    gk = _ensure_odd(cfg.gaussian_ksize)
    blurred = cv2.GaussianBlur(img, (gk, gk), 0)

    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill enclosed dark regions (e.g. nucleus inside membrane ring)
    mask_filled = binary_fill_holes(mask_otsu > 0).astype(np.uint8) * 255

    r = cfg.morph_close_radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    mask_final = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)

    return mask_otsu, mask_filled, mask_final


# ─── Connected components & area filtering ────────────────────────────────────


class RegionRecord(NamedTuple):
    field_id: int
    aggregation_id: int
    area_px: int
    area_um2: float
    active_channels: int


def count_active_channels(
    component_mask: np.ndarray,
    channel_projections: list[np.ndarray],
    pixel_threshold: float,
    min_occupancy_px: float,
) -> int:
    """Count channels with biologically meaningful occupancy within the component.

    A channel is active when the number of positive pixels (intensity > pixel_threshold)
    inside the component reaches min_occupancy_px.  This rejects hot pixels and
    sub-cellular noise that would pass a pure intensity test.
    """
    return sum(
        1 for ch in channel_projections
        if int(((ch > pixel_threshold) & component_mask).sum()) >= min_occupancy_px
    )


def detect_aggregations(
    mask: np.ndarray,
    channel_projections: list[np.ndarray],
    pixel_size_um: float,
    field_id: int,
    area_threshold_px: float,
    min_active_channels: int,
    channel_pixel_threshold: float,
    min_channel_occupancy_px: float,
) -> tuple[list[RegionRecord], np.ndarray, np.ndarray]:
    """
    Label connected components; filter by area then by active-channel occupancy.

    A region is accepted only when:
      area_px >= area_threshold_px
      AND active_channels >= min_active_channels
      where active = positive-pixel count inside component >= min_channel_occupancy_px

    Returns
    -------
    records       : RegionRecord for each accepted region
    valid_mask    : uint8 binary mask of accepted regions
    rejected_mask : uint8 binary mask of area-passing but insufficient-channel regions
    """
    n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    valid_mask = np.zeros_like(mask)
    rejected_mask = np.zeros_like(mask)
    records: list[RegionRecord] = []
    agg_id = 0
    pixel_area_um2 = pixel_size_um ** 2

    for label_id in range(1, n_labels):  # 0 is background
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < area_threshold_px:
            continue

        component_mask = labeled == label_id
        active_ch = count_active_channels(
            component_mask, channel_projections,
            channel_pixel_threshold, min_channel_occupancy_px,
        )

        if active_ch >= min_active_channels:
            valid_mask[component_mask] = 255
            records.append(RegionRecord(
                field_id=field_id,
                aggregation_id=agg_id,
                area_px=area_px,
                area_um2=round(area_px * pixel_area_um2, 4),
                active_channels=active_ch,
            ))
            agg_id += 1
        else:
            rejected_mask[component_mask] = 255

    return records, valid_mask, rejected_mask


# ─── Visualization ────────────────────────────────────────────────────────────


def _contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def draw_contour_overlay_gray(
    gray: np.ndarray, valid_mask: np.ndarray, rejected_mask: np.ndarray
) -> np.ndarray:
    """Red = valid aggregation, blue = single-channel rejected. Returns BGR."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bgr, _contours_from_mask(rejected_mask), -1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.drawContours(bgr, _contours_from_mask(valid_mask), -1, (0, 0, 255), 2, cv2.LINE_AA)
    return bgr


def draw_contour_overlay_color(
    merged_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw contours of mask on pseudo-color RGB base with the given BGR color. Returns BGR."""
    bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, _contours_from_mask(mask), -1, color, 2, cv2.LINE_AA)
    return bgr


# ─── Debug image saving ───────────────────────────────────────────────────────


def _imwrite(path: Path, img: np.ndarray) -> None:
    """Write an image and print an error if it fails."""
    try:
        ok = cv2.imwrite(str(path), img)
    except Exception as e:
        print(f"[ERROR] Failed to save debug image {path}: {e}")
        return
    if not ok:
        print(f"[ERROR] cv2.imwrite returned False for {path} — image may be empty or path invalid")


def save_debug_images(
    debug_dir: Path,
    prefix: str,
    channel_projections: list[np.ndarray],
    merged_rgb: np.ndarray,
    mask_otsu: np.ndarray,
    mask_filled: np.ndarray,
    mask_final: np.ndarray,
    overlay_gray: np.ndarray,
    overlay_color_valid: np.ndarray,
    overlay_color_rejected: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    for i, proj in enumerate(channel_projections):
        _imwrite(debug_dir / f"{prefix}_ch{i}_projection.png", proj)

    _imwrite(debug_dir / f"{prefix}_merge.png", cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR))
    _imwrite(debug_dir / f"{prefix}_mask_otsu.png", mask_otsu)
    _imwrite(debug_dir / f"{prefix}_mask_filled.png", mask_filled)
    _imwrite(debug_dir / f"{prefix}_mask_final.png", mask_final)
    _imwrite(debug_dir / f"{prefix}_contour_overlay_gray.png", overlay_gray)
    _imwrite(debug_dir / f"{prefix}_contour_overlay_color_valid.png", overlay_color_valid)
    _imwrite(debug_dir / f"{prefix}_contour_overlay_color_rejected.png", overlay_color_rejected)


# ─── Per-field pipeline ───────────────────────────────────────────────────────


def process_field(
    field_data: np.ndarray,
    field_sizes: dict[str, int],
    pixel_size_um: float,
    field_id: int,
    cfg: Config,
    debug_dir: Path,
) -> list[RegionRecord]:
    """Full occupancy-based pipeline for one field of view."""
    channel_projections = extract_channel_projections(field_data, field_sizes)
    gray_merged = merge_channels(channel_projections)
    rgb_merged = pseudo_color_merge(channel_projections, cfg.channel_colors)

    mask_otsu, mask_filled, mask_final = segment_occupancy(gray_merged, cfg)

    area_threshold_px = compute_area_threshold_px(
        pixel_size_um, cfg.s2_diameter_um, cfg.aggregation_min_cells
    )

    min_channel_occupancy_px = compute_channel_occupancy_threshold_px(
        pixel_size_um, cfg.s2_diameter_um, cfg.channel_occupancy_fraction,
    )

    records, valid_mask, rejected_mask = detect_aggregations(
        mask_final, channel_projections, pixel_size_um, field_id,
        area_threshold_px, cfg.min_active_channels,
        cfg.channel_pixel_threshold, min_channel_occupancy_px,
    )

    overlay_gray = draw_contour_overlay_gray(gray_merged, valid_mask, rejected_mask)
    # (0, 0, 255) = red in BGR;  (255, 0, 0) = blue in BGR
    overlay_color_valid = draw_contour_overlay_color(rgb_merged, valid_mask, (0, 0, 255))
    overlay_color_rejected = draw_contour_overlay_color(rgb_merged, rejected_mask, (255, 0, 0))

    prefix = f"field{field_id:02d}"
    save_debug_images(
        debug_dir, prefix, channel_projections, rgb_merged,
        mask_otsu, mask_filled, mask_final, overlay_gray,
        overlay_color_valid, overlay_color_rejected,
    )

    n_rejected = len(_contours_from_mask(rejected_mask))
    print(
        f"  Field {field_id:3d}: {len(records)} accepted, {n_rejected} rejected (single-channel)"
        f"  [area threshold {area_threshold_px:.0f} px"
        f" = {cfg.aggregation_min_cells}× S2 @ {pixel_size_um:.4f} µm/px]"
    )
    return records


# ─── CSV output ───────────────────────────────────────────────────────────────


def save_csv(records: list[RegionRecord], output_path: Path) -> None:
    if not records:
        print("No aggregations detected — CSV not written.")
        return
    fieldnames = ["field_id", "aggregation_id", "area_px", "area_um2", "active_channels"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([r._asdict() for r in records])
    print(f"Results saved → {output_path}")


# ─── Top-level pipeline ───────────────────────────────────────────────────────


def process_nd2(nd2_path: Path, output_dir: Path, cfg: Config) -> None:
    """Load an nd2 file and run the aggregation detection pipeline on all fields."""
    print(f"Loading: {nd2_path}")
    data, sizes, pixel_size_um = load_nd2(nd2_path)
    print(f"  Dimensions : {sizes}")
    print(f"  Pixel size : {pixel_size_um:.4f} µm/pixel")

    s2_area_um2 = np.pi * (cfg.s2_diameter_um / 2.0) ** 2
    print(
        f"  Area threshold: {cfg.aggregation_min_cells} S2 cells"
        f" × {s2_area_um2:.1f} µm² = {s2_area_um2 * cfg.aggregation_min_cells:.1f} µm²"
    )

    debug_dir = output_dir / "debug" / nd2_path.stem
    all_records: list[RegionRecord] = []

    for field_id, field_data, field_sizes in iter_fields(data, sizes):
        records = process_field(
            field_data, field_sizes, pixel_size_um,
            field_id, cfg, debug_dir,
        )
        all_records.extend(records)

    csv_path = output_dir / f"{nd2_path.stem}_aggregations.csv"
    save_csv(all_records, csv_path)


# ─── CLI entry point ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect and quantify S2 cell aggregations in nd2 fluorescence images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("nd2_path", type=Path, help="Input .nd2 file")
    p.add_argument("output_dir", type=Path, help="Directory for CSV and debug images")
    p.add_argument(
        "--s2-diameter", type=float, default=10.0,
        help="S2 cell diameter in µm (used to compute aggregation area threshold)",
    )
    p.add_argument(
        "--min-cells", type=int, default=3,
        help="Minimum number of S2 cells to call a region an aggregation",
    )
    p.add_argument(
        "--gaussian-ksize", type=int, default=25,
        help="Gaussian blur kernel size (forced to odd)",
    )
    p.add_argument(
        "--morph-close-radius", type=int, default=5,
        help="Disk radius (pixels) for morphology close after hole filling",
    )
    p.add_argument(
        "--min-active-channels", type=int, default=2,
        help="Minimum number of fluorescence channels active within a region to accept it",
    )
    p.add_argument(
        "--channel-pixel-threshold", type=float, default=10.0,
        help="Per-pixel intensity (0–255) above which a pixel counts as positive for a channel",
    )
    p.add_argument(
        "--channel-occupancy-fraction", type=float, default=0.3,
        help="Fraction of single-S2-cell area a channel must occupy to be considered active",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        s2_diameter_um=args.s2_diameter,
        aggregation_min_cells=args.min_cells,
        gaussian_ksize=args.gaussian_ksize,
        morph_close_radius=args.morph_close_radius,
        min_active_channels=args.min_active_channels,
        channel_pixel_threshold=args.channel_pixel_threshold,
        channel_occupancy_fraction=args.channel_occupancy_fraction,
    )
    process_nd2(args.nd2_path, args.output_dir, cfg)


if __name__ == "__main__":
    main()
