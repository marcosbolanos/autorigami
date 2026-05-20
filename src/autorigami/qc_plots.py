from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_obj_polyline(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError(f"Expected at least two OBJ vertices in {path}")
    return points


def _resolve_selected_dir(path: Path | None) -> Path:
    if path is not None:
        selected_dir = path / "selected_candidate" if (path / "selected_candidate").exists() else path
        if not (selected_dir / "run_info.json").exists():
            raise ValueError(f"Could not find selected candidate run_info.json under {path}")
        return selected_dir

    candidates = sorted(
        Path("outputs").glob("**/selected_candidate/run_info.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise ValueError("No selected candidate found under outputs/")
    return candidates[0].parent


def _polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    small = 1e-12
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    s_num = 0.0
    s_den = denom
    t_num = 0.0
    t_den = denom

    if denom < small:
        s_num = 0.0
        s_den = 1.0
        t_num = e
        t_den = c
    else:
        s_num = b * e - c * d
        t_num = a * e - b * d
        if s_num < 0.0:
            s_num = 0.0
            t_num = e
            t_den = c
        elif s_num > s_den:
            s_num = s_den
            t_num = e + b
            t_den = c

    if t_num < 0.0:
        t_num = 0.0
        if -d < 0.0:
            s_num = 0.0
        elif -d > a:
            s_num = s_den
        else:
            s_num = -d
            s_den = a
    elif t_num > t_den:
        t_num = t_den
        if -d + b < 0.0:
            s_num = 0.0
        elif -d + b > a:
            s_num = s_den
        else:
            s_num = -d + b
            s_den = a

    s = 0.0 if abs(s_num) < small else s_num / s_den
    t = 0.0 if abs(t_num) < small else t_num / t_den
    delta = w + s * u - t * v
    return float(np.linalg.norm(delta))


def _sampled_nonlocal_segment_distances(
    points: np.ndarray,
    *,
    nonlocal_window_world: float,
    max_segments: int,
) -> np.ndarray:
    if points.shape[0] - 1 > max_segments:
        point_indices = np.linspace(0, points.shape[0] - 1, max_segments + 1, dtype=np.int64)
        sampled = points[point_indices]
    else:
        sampled = points

    segment_lengths = np.linalg.norm(sampled[1:] - sampled[:-1], axis=1)
    cumulative = np.concatenate([np.array([0.0]), np.cumsum(segment_lengths)])
    distances: list[float] = []
    for i in range(sampled.shape[0] - 1):
        a0 = cumulative[i]
        a1 = cumulative[i + 1]
        for j in range(i):
            b0 = cumulative[j]
            b1 = cumulative[j + 1]
            arc_gap = b0 - a1 if a1 <= b0 else a0 - b1 if b1 <= a0 else 0.0
            if arc_gap < nonlocal_window_world:
                continue
            distances.append(_segment_distance(sampled[i], sampled[i + 1], sampled[j], sampled[j + 1]))
    if not distances:
        return np.array([], dtype=np.float64)
    return np.asarray(distances, dtype=np.float64)


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_histogram(
    *,
    distances_nm: np.ndarray,
    run_info: dict[str, object],
    output_path: Path,
    spacing_nm: float,
    nonlocal_window_nm: float,
    world_to_nm: float,
    length_nm: float,
) -> None:
    width = 1600
    height = 1000
    margin_left = 130
    margin_right = 70
    margin_top = 170
    margin_bottom = 145
    image = Image.new("RGB", (width, height), "#fbfaf5")
    draw = ImageDraw.Draw(image)
    title_font = _font(38)
    subtitle_font = _font(22)
    label_font = _font(24)
    small_font = _font(19)
    mono_font = _font(18)

    candidate_name = str(run_info.get("candidate_name", "unknown candidate"))
    generator = str(run_info.get("generator", "unknown generator"))
    draw.text((margin_left, 34), "How close does the selected spiral come to itself?", fill="#141414", font=title_font)
    draw.text(
        (margin_left, 86),
        (
            f"Candidate {candidate_name}, generated by {generator}. "
            "Distances are between non-neighboring rod segments in 3D Euclidean space."
        ),
        fill="#4d4a43",
        font=subtitle_font,
    )

    if distances_nm.size == 0:
        draw.text((margin_left, 180), "No nonlocal segment pairs were available for histogramming.", fill="#8a1f11", font=label_font)
        image.save(output_path)
        return

    max_x = max(float(np.percentile(distances_nm, 99.5)), spacing_nm * 4.0)
    max_x = min(max_x, float(np.max(distances_nm))) if np.max(distances_nm) > max_x else max_x
    bins = np.linspace(0.0, max_x, 80)
    counts, edges = np.histogram(np.clip(distances_nm, 0.0, max_x), bins=bins)
    max_count = max(int(np.max(counts)), 1)
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    threshold_x = plot_left + (spacing_nm / max_x) * plot_width
    draw.rectangle((plot_left, plot_top, threshold_x, plot_bottom), fill="#f6d7d2")
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#222222", width=2)
    draw.text(
        (plot_left + 16, plot_top + 16),
        "invalid region:\nself-contact violation",
        fill="#8a1f11",
        font=small_font,
    )
    for grid_index in range(6):
        y = plot_bottom - grid_index * plot_height / 5
        draw.line((plot_left, y, plot_right, y), fill="#ded9cb", width=1)
        count_label = f"{int(max_count * grid_index / 5)}"
        draw.text((48, y - 10), count_label, fill="#555555", font=small_font)

    bar_color = "#315f72"
    for count, left_edge, right_edge in zip(counts, edges[:-1], edges[1:]):
        x0 = plot_left + (left_edge / max_x) * plot_width
        x1 = plot_left + (right_edge / max_x) * plot_width
        y = plot_bottom - (float(count) / max_count) * plot_height
        draw.rectangle((x0 + 1, y, x1 - 1, plot_bottom), fill=bar_color)

    draw.line((threshold_x, plot_top, threshold_x, plot_bottom), fill="#b3261e", width=5)
    draw.rounded_rectangle(
        (threshold_x + 12, plot_top + 14, threshold_x + 455, plot_top + 78),
        radius=10,
        fill="#fff7f5",
        outline="#b3261e",
    )
    draw.text(
        (threshold_x + 26, plot_top + 24),
        f"hard cutoff: distances below {spacing_nm:.3g} nm are violations",
        fill="#8a1f11",
        font=small_font,
    )

    for tick_index in range(6):
        x = plot_left + tick_index * plot_width / 5
        value = tick_index * max_x / 5
        draw.line((x, plot_bottom, x, plot_bottom + 8), fill="#222222", width=2)
        draw.text((x - 24, plot_bottom + 14), f"{value:.1f}", fill="#333333", font=small_font)
    draw.text(
        (plot_left + plot_width / 2 - 340, height - 82),
        "3D distance between non-neighboring spiral segments (nanometers)",
        fill="#222222",
        font=label_font,
    )
    draw.text((18, plot_top + plot_height / 2 - 34), "number of\nsampled pairs", fill="#222222", font=label_font)

    stats_rows = [
        ("Spiral length", f"{length_nm:,.1f} nm"),
        ("Polyline points", f"{int(run_info.get('external_toolpath_stats', {}).get('acap', {}).get('point_count', 0)):,}"),
        ("Sampled segment pairs", f"{distances_nm.size:,}"),
        ("Closest sampled approach", f"{float(np.min(distances_nm)):.4f} nm"),
        ("5th percentile distance", f"{float(np.percentile(distances_nm, 5.0)):.3f} nm"),
        ("Median distance", f"{float(np.percentile(distances_nm, 50.0)):.3f} nm"),
        ("95th percentile distance", f"{float(np.percentile(distances_nm, 95.0)):.3f} nm"),
        ("Violation count in sample", f"{int(np.count_nonzero(distances_nm < spacing_nm)):,}"),
        ("Nonlocal exclusion window", f"{nonlocal_window_nm:.3g} nm along curve"),
        ("Scale", f"{world_to_nm:.3g} nm per world unit"),
    ]
    text_x = 930
    text_y = 35
    draw.rounded_rectangle((text_x - 24, text_y - 14, width - 62, 350), radius=18, fill="#f0eadc", outline="#d2c6ad")
    draw.text((text_x, text_y), "Selected candidate results", fill="#141414", font=label_font)
    for index, (label, value) in enumerate(stats_rows):
        y = text_y + 42 + index * 27
        draw.text((text_x, y), f"{label}:", fill="#4d4a43", font=mono_font)
        draw.text((text_x + 300, y), value, fill="#141414", font=mono_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def export_selected_candidate_qc_plots(
    *,
    selected_dir: Path,
    output_path: Path | None = None,
    max_segments: int = 1600,
) -> dict[str, object]:
    run_info_path = selected_dir / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    points = _load_obj_polyline(selected_dir / "spiral_polyline_raw.obj")

    constraints = run_info["generator_constraints"]
    spacing_nm = float(constraints["self_avoidance_min_distance_nm"])
    nonlocal_window_nm = float(constraints["self_avoidance_nonlocal_window_nm"])
    world_to_nm = spacing_nm / float(
        run_info["external_toolpath_stats"]["acap"]["minimum_separation_world"]
    )
    nonlocal_window_world = nonlocal_window_nm / world_to_nm

    distances_world = _sampled_nonlocal_segment_distances(
        points,
        nonlocal_window_world=nonlocal_window_world,
        max_segments=max_segments,
    )
    distances_nm = distances_world * world_to_nm
    resolved_output_path = output_path if output_path is not None else selected_dir / "separation_histogram.png"
    _draw_histogram(
        distances_nm=distances_nm,
        run_info=run_info,
        output_path=resolved_output_path,
        spacing_nm=spacing_nm,
        nonlocal_window_nm=nonlocal_window_nm,
        world_to_nm=world_to_nm,
        length_nm=_polyline_length(points) * world_to_nm,
    )
    summary_path = resolved_output_path.with_suffix(".json")
    summary = {
        "selected_candidate_dir": str(selected_dir),
        "output_png": str(resolved_output_path),
        "sampled_pair_count": int(distances_nm.size),
        "minimum_sampled_separation_nm": float(np.min(distances_nm)) if distances_nm.size else math.inf,
        "p05_separation_nm": float(np.percentile(distances_nm, 5.0)) if distances_nm.size else math.inf,
        "median_separation_nm": float(np.percentile(distances_nm, 50.0)) if distances_nm.size else math.inf,
        "p95_separation_nm": float(np.percentile(distances_nm, 95.0)) if distances_nm.size else math.inf,
        "sampled_violation_count": int(np.count_nonzero(distances_nm < spacing_nm)),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected-candidate nonlocal separation histogram.")
    parser.add_argument("--output", type=Path, help="Output PNG path. Defaults next to selected run_info.json.")
    parser.add_argument("--selected", type=Path, help="Selected candidate directory or parent run directory.")
    parser.add_argument("--max-segments", type=int, default=1600, help="Maximum sampled segments for pairwise histogram.")
    args = parser.parse_args()

    selected_dir = _resolve_selected_dir(args.selected)
    summary = export_selected_candidate_qc_plots(
        selected_dir=selected_dir,
        output_path=args.output,
        max_segments=args.max_segments,
    )
    print(f"Histogram: {summary['output_png']}")
    print(f"Summary: {Path(str(summary['output_png'])).with_suffix('.json')}")


if __name__ == "__main__":
    main()
