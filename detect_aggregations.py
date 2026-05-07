#!/usr/bin/env python3
"""
S2 cell aggregation detection from nd2 fluorescence images.

Processes multiple fields of view with multi-channel z-stack images.
Outputs per-aggregation area measurements (pixels and µm²) to CSV,
and saves debug images for each processing stage.

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


# ─── Configuration ────────────────────────────────────────────────────────────

# RGB weights (0 or 1) for each named pseudo-color.
_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red":     (1, 0, 0),
    "green":   (0, 1, 0),
    "blue":    (0, 0, 1),
    "cyan":    (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow":  (1, 1, 0),
    "white":   (1, 1, 1),
}

# Default pseudo-color assignment per channel index.
CHANNEL_COLORS: dict[int, str] = {
    0: "green",
    1: "red",
    2: "blue",
}


class Config(NamedTuple):
    threshold: int = 50           # binarization threshold applied to 0-255 normalized merged image
    area_threshold_px: int = 400  # minimum aggregation area to keep (pixels)
    gaussian_ksize: int = 25      # Gaussian blur kernel size (must be odd)
    median_ksize: int = 25        # median blur kernel size (must be odd)
    channel_colors: dict[int, str] = CHANNEL_COLORS


def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


# ─── nd2 loading ──────────────────────────────────────────────────────────────


def load_nd2(path: Path) -> tuple[np.ndarray, dict[str, int], float]:
    """
    Load an nd2 file.

    Returns
    -------
    data : np.ndarray
        Full image array. Axis order matches the keys of `sizes`.
    sizes : dict[str, int]
        Ordered dimension sizes, e.g. {'P': 3, 'Z': 10, 'C': 3, 'Y': 512, 'X': 512}.
    pixel_size_um : float
        XY pixel size in micrometres.
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
    """
    Yield (field_id, field_array, field_sizes) for each position.

    If there is no 'P' dimension the whole array is yielded as field 0.
    """
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
    """Normalize any numeric array to uint8 via min-max scaling."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def max_projection(stack: np.ndarray) -> np.ndarray:
    """Max-intensity projection along axis 0.  Input shape: (Z, Y, X)."""
    return stack.max(axis=0)


def extract_channel_projections(
    field_data: np.ndarray, sizes: dict[str, int]
) -> list[np.ndarray]:
    """
    Return a list of uint8 max-projected images, one per channel.

    Handles fields with or without Z and/or C dimensions.
    """
    dim_keys = list(sizes.keys())
    n_channels = sizes.get("C", 1)
    projections: list[np.ndarray] = []

    for ch in range(n_channels):
        arr = field_data

        # Select channel axis first (keeps Z intact for projection)
        if "C" in dim_keys:
            c_axis = dim_keys.index("C")
            arr = np.take(arr, ch, axis=c_axis)
            remaining = [k for k in dim_keys if k != "C"]
        else:
            remaining = list(dim_keys)

        # Z-stack max projection
        if "Z" in remaining:
            z_axis = remaining.index("Z")
            arr = arr.max(axis=z_axis)

        projections.append(to_uint8(arr))

    return projections


def merge_channels(channel_projections: list[np.ndarray]) -> np.ndarray:
    """Merge uint8 single-channel images by per-pixel maximum (used for segmentation)."""
    return np.maximum.reduce(channel_projections)


def apply_pseudo_color(gray: np.ndarray, color: str) -> np.ndarray:
    """
    Map a uint8 grayscale image to a pseudo-colored RGB image.

    Each output channel is either the input intensity or zero,
    depending on the RGB weight of `color`.  No loops; fully vectorized.

    Returns uint8 RGB array of shape (H, W, 3).
    """
    weights = np.array(_COLOR_RGB[color], dtype=np.uint8)  # shape (3,)
    return gray[..., np.newaxis] * weights                  # (H, W, 1) * (3,) → (H, W, 3)


def pseudo_color_merge(
    channel_projections: list[np.ndarray],
    channel_colors: dict[int, str],
) -> np.ndarray:
    """
    Assign a pseudo-color to each channel and merge by per-pixel maximum.

    Channels not listed in `channel_colors` fall back to "white".
    Returns uint8 RGB array of shape (H, W, 3).
    """
    h, w = channel_projections[0].shape
    merged_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, proj in enumerate(channel_projections):
        color = channel_colors.get(i, "white")
        colored = apply_pseudo_color(proj, color)
        merged_rgb = np.maximum(merged_rgb, colored)
    return merged_rgb


def preprocess(img: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Apply Gaussian blur → median blur → fixed threshold.

    Parameters
    ----------
    img : uint8 grayscale image
    cfg : Config

    Returns
    -------
    Binary mask (uint8, 0 or 255).
    """
    gk = _ensure_odd(cfg.gaussian_ksize)
    mk = _ensure_odd(cfg.median_ksize)
    blurred = cv2.GaussianBlur(img, (gk, gk), 2)
    blurred = cv2.medianBlur(blurred, mk)
    _, mask = cv2.threshold(blurred, cfg.threshold, 255, cv2.THRESH_BINARY)
    return mask


def detect_contours(mask: np.ndarray, area_threshold_px: int) -> list[np.ndarray]:
    """Find external contours in a binary mask, keeping only those ≥ area_threshold_px."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= area_threshold_px]


def draw_contour_overlay_gray(gray: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    """Red contours on grayscale base. Returns BGR (ready for cv2.imwrite)."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bgr, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
    return bgr


def draw_contour_overlay_color(merged_rgb: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    """Red contours on pseudo-color RGB base. Returns BGR (ready for cv2.imwrite)."""
    bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
    return bgr


# ─── Debug image saving ───────────────────────────────────────────────────────


def save_debug_images(
    debug_dir: Path,
    prefix: str,
    channel_projections: list[np.ndarray],
    merged_rgb: np.ndarray,
    mask: np.ndarray,
    overlay_gray: np.ndarray,
    overlay_color: np.ndarray,
) -> None:
    """
    Save all intermediate debug images for one field of view (all PNG).

    channel_projections : grayscale uint8 — saved as-is (OpenCV writes gray correctly)
    merged_rgb          : RGB uint8      — converted to BGR before imwrite
    mask                : grayscale uint8
    overlay_gray/color  : BGR uint8      — written as-is
    """
    debug_dir.mkdir(parents=True, exist_ok=True)

    for i, proj in enumerate(channel_projections):
        cv2.imwrite(str(debug_dir / f"{prefix}_ch{i}_projection.png"), proj)

    # merged_rgb is in RGB order; convert to BGR so cv2.imwrite produces correct colors
    cv2.imwrite(str(debug_dir / f"{prefix}_merge.png"), cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / f"{prefix}_mask.png"), mask)
    cv2.imwrite(str(debug_dir / f"{prefix}_contour_overlay_gray.png"), overlay_gray)
    cv2.imwrite(str(debug_dir / f"{prefix}_contour_overlay_color.png"), overlay_color)


# ─── Per-field pipeline ───────────────────────────────────────────────────────


def process_field(
    field_data: np.ndarray,
    field_sizes: dict[str, int],
    pixel_size_um: float,
    field_id: int,
    cfg: Config,
    debug_dir: Path,
) -> list[dict]:
    """
    Full processing pipeline for one field of view.

    Returns a list of dicts, one per detected aggregation, with keys:
    field_id, aggregation_id, area_px, area_um2.
    """
    channel_projections = extract_channel_projections(field_data, field_sizes)

    # Grayscale max-merge for segmentation
    gray_merged = merge_channels(channel_projections)
    # Pseudo-color RGB merge for visualization
    rgb_merged = pseudo_color_merge(channel_projections, cfg.channel_colors)

    mask = preprocess(gray_merged, cfg)
    contours = detect_contours(mask, cfg.area_threshold_px)
    overlay_gray = draw_contour_overlay_gray(gray_merged, contours)
    overlay_color = draw_contour_overlay_color(rgb_merged, contours)

    prefix = f"field{field_id:02d}"
    save_debug_images(debug_dir, prefix, channel_projections, rgb_merged, mask, overlay_gray, overlay_color)

    pixel_area_um2 = pixel_size_um ** 2
    records = [
        {
            "field_id": field_id,
            "aggregation_id": i,
            "area_px": cv2.contourArea(c),
            "area_um2": round(cv2.contourArea(c) * pixel_area_um2, 4),
        }
        for i, c in enumerate(contours)
    ]

    print(f"  Field {field_id:3d}: {len(contours)} aggregation(s) detected")
    return records


# ─── CSV output ───────────────────────────────────────────────────────────────


def save_csv(records: list[dict], output_path: Path) -> None:
    if not records:
        print("No aggregations detected — CSV not written.")
        return
    fieldnames = ["field_id", "aggregation_id", "area_px", "area_um2"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Results saved → {output_path}")


# ─── Top-level pipeline ───────────────────────────────────────────────────────


def process_nd2(nd2_path: Path, output_dir: Path, cfg: Config) -> None:
    """Load an nd2 file and run the aggregation detection pipeline on all fields."""
    print(f"Loading: {nd2_path}")
    data, sizes, pixel_size_um = load_nd2(nd2_path)
    print(f"  Dimensions : {sizes}")
    print(f"  Pixel size : {pixel_size_um:.4f} µm/pixel")

    debug_dir = output_dir / "debug" / nd2_path.stem
    all_records: list[dict] = []

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
        "--threshold", type=int, default=50,
        help="Binarization threshold (0–255) applied to the normalised merged image",
    )
    p.add_argument(
        "--area-threshold", type=int, default=400,
        help="Minimum aggregation area in pixels",
    )
    p.add_argument(
        "--gaussian-ksize", type=int, default=25,
        help="Gaussian blur kernel size (forced to odd)",
    )
    p.add_argument(
        "--median-ksize", type=int, default=25,
        help="Median blur kernel size (forced to odd)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        threshold=args.threshold,
        area_threshold_px=args.area_threshold,
        gaussian_ksize=args.gaussian_ksize,
        median_ksize=args.median_ksize,
    )
    process_nd2(args.nd2_path, args.output_dir, cfg)


if __name__ == "__main__":
    main()
