from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import TypeGuard

from PIL import Image, ImageDraw, ImageFont
from svgpathtools import Path as SvgPath
from svgpathtools import svg2paths2
from tqdm import tqdm  # type: ignore[reportMissingModuleSource]

DEFAULT_BG = (255, 255, 255)
DEFAULT_STROKE = (0, 0, 0)
FontType = ImageFont.FreeTypeFont | ImageFont.ImageFont
Svg2PathsResult = (
    tuple[list[SvgPath], list[dict[str, str]], dict[str, str]]
    | tuple[list[SvgPath], list[dict[str, str]]]
)


def _parse_step(path: Path) -> int | None:
    match = re.search(r"step_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def _parse_color(
    value: str | None, default: tuple[int, int, int]
) -> tuple[int, int, int]:
    if value is None:
        return default
    text = value.strip().lower()
    if text.startswith("#"):
        hex_value = text[1:]
        if len(hex_value) == 3:
            hex_value = "".join(ch * 2 for ch in hex_value)
        if len(hex_value) == 6:
            try:
                r = int(hex_value[0:2], 16)
                g = int(hex_value[2:4], 16)
                b = int(hex_value[4:6], 16)
                return (r, g, b)
            except ValueError:
                return default
    if text.startswith("rgb"):
        nums = re.findall(r"\d+", text)
        if len(nums) >= 3:
            r = int(max(0, min(255, int(nums[0]))))
            g = int(max(0, min(255, int(nums[1]))))
            b = int(max(0, min(255, int(nums[2]))))
            return (r, g, b)
    return default


def _read_viewbox(svg_attributes: dict[str, str]) -> tuple[float, float, float, float]:
    viewbox_raw = svg_attributes.get("viewBox") or svg_attributes.get("viewbox")
    if not viewbox_raw:
        raise ValueError("SVG missing viewBox; cannot determine bounds.")
    parts = re.split(r"[ ,]+", viewbox_raw.strip())
    if len(parts) != 4:
        raise ValueError(f"Invalid viewBox: {viewbox_raw}")
    try:
        minx, miny, width, height = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Invalid viewBox values: {viewbox_raw}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid viewBox size: {viewbox_raw}")
    return minx, miny, width, height


def _has_svg_attributes(
    result: Svg2PathsResult,
) -> TypeGuard[tuple[list[SvgPath], list[dict[str, str]], dict[str, str]]]:
    return len(result) == 3


def _viewbox_close(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return all(abs(x - y) <= 1e-3 for x, y in zip(a, b))


def _load_svg(
    svg_path: Path,
) -> tuple[list[SvgPath], list[dict[str, str]], tuple[float, float, float, float]]:
    result = svg2paths2(str(svg_path))
    if not _has_svg_attributes(result):
        raise ValueError(f"svg2paths2 did not return SVG attributes for {svg_path}")
    paths, attributes, svg_attributes = result
    viewbox = _read_viewbox(svg_attributes)
    return paths, attributes, viewbox


def _sample_path(path: SvgPath, sample_step: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for segment in path:
        seg_len = max(float(segment.length(error=1e-4)), 1e-6)
        n = max(2, int(math.ceil(seg_len / sample_step)))
        for i in range(n):
            t = 0.0 if n == 1 else i / (n - 1)
            p = segment.point(t)
            if points and i == 0:
                continue
            points.append((p.real, p.imag))
    return points


def _to_pixel_points(
    points: list[tuple[float, float]],
    minx: float,
    miny: float,
    scale: float,
) -> list[tuple[int, int]]:
    return [
        (int(round((x - minx) * scale)), int(round((y - miny) * scale)))
        for x, y in points
    ]


def _render_paths(
    paths: list[SvgPath],
    attributes: list[dict[str, str]],
    viewbox: tuple[float, float, float, float],
    out_size: tuple[int, int],
    scale: float,
    sample_step: float,
    bg_color: tuple[int, int, int],
) -> Image.Image:
    minx, miny, _width, _height = viewbox
    image = Image.new("RGB", out_size, color=bg_color)
    draw = ImageDraw.Draw(image)

    for path, attrs in zip(paths, attributes):
        stroke = attrs.get("stroke")
        if stroke is None or stroke.strip().lower() == "none":
            continue
        color = _parse_color(stroke, DEFAULT_STROKE)
        stroke_width = _parse_float(attrs.get("stroke-width"))
        if stroke_width is None:
            stroke_width = _parse_float(attrs.get("stroke_width"))
        if stroke_width is None:
            stroke_width = 1.0
        width_px = max(1, int(round(stroke_width * scale)))

        points = _sample_path(path, sample_step)
        if len(points) < 2:
            continue
        pixel_points = _to_pixel_points(points, minx, miny, scale)
        draw.line(pixel_points, fill=color, width=width_px)

    return image


def _load_font(font_path: str | None, size: int) -> FontType:
    if font_path is not None:
        return ImageFont.truetype(font_path, size=size)
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_text_outline(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: FontType,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    outline_width: int = 2,
) -> None:
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(position, text, font=font, fill=fill)


def _draw_step_label(
    image: Image.Image,
    step: int,
    font: FontType,
    padding: int,
) -> None:
    draw = ImageDraw.Draw(image)
    text = f"step {step}"
    text_w, text_h = _text_size(draw, text, font)
    x = max(0, image.width - padding - text_w)
    y = max(0, image.height - padding - text_h)
    _draw_text_outline(
        draw,
        (x, y),
        text,
        font,
        fill=(0, 0, 0),
        outline=(255, 255, 255),
        outline_width=2,
    )


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: FontType,
) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except AttributeError:
        try:
            bbox = font.getbbox(text)
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        except AttributeError:
            mask = font.getmask(text)
            return int(mask.size[0]), int(mask.size[1])


def _collect_svgs(
    input_dir: Path,
    pattern: str,
    start_step: int | None,
    end_step: int | None,
) -> list[tuple[int, Path]]:
    candidates = list(input_dir.glob(pattern))
    if not candidates:
        raise ValueError(f"No SVGs found in {input_dir} matching {pattern}")
    items: list[tuple[int, Path]] = []
    for path in candidates:
        step = _parse_step(path)
        if step is None:
            continue
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        items.append((step, path))
    if not items:
        raise ValueError(
            "No SVGs with step numbers found. Expected filenames like step_000010.svg."
        )
    items.sort(key=lambda item: item[0])
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir", required=True, help="Directory containing checkpoint SVGs"
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output GIF path (defaults to <input-dir>/checkpoints.gif)",
    )
    ap.add_argument(
        "--pattern",
        default="step_*.svg",
        help="Glob pattern for checkpoint SVGs",
    )
    ap.add_argument(
        "--max-size",
        type=int,
        default=480,
        help="Max width or height for output frames",
    )
    ap.add_argument("--fps", type=float, default=12.0, help="Frames per second")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    ap.add_argument(
        "--start-step", type=int, default=None, help="Minimum step to include"
    )
    ap.add_argument(
        "--end-step", type=int, default=None, help="Maximum step to include"
    )
    ap.add_argument(
        "--label-size",
        type=int,
        default=None,
        help="Pixel size for step label text",
    )
    ap.add_argument(
        "--font",
        default=None,
        help="Path to a .ttf font for step labels",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input dir does not exist: {input_dir}")
    if args.max_size <= 0:
        raise ValueError("--max-size must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    svg_items = _collect_svgs(input_dir, args.pattern, args.start_step, args.end_step)
    if args.stride > 1:
        svg_items = svg_items[:: args.stride]
    if not svg_items:
        raise ValueError("No SVGs remain after applying stride/filter settings")

    _first_step, first_svg = svg_items[0]
    _first_paths, _first_attrs, viewbox = _load_svg(first_svg)
    _minx, _miny, width, height = viewbox
    scale = float(args.max_size) / float(max(width, height))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    sample_step = max(1e-6, 1.0 / scale)

    label_size = args.label_size
    if label_size is None:
        label_size = max(12, int(round(min(out_w, out_h) * 0.04)))
    padding = max(6, int(round(min(out_w, out_h) * 0.02)))
    font = _load_font(args.font, label_size)

    frames: list[Image.Image] = []
    for step, svg_path in tqdm(svg_items, desc="Rendering frames", unit="frame"):
        paths, attrs, current_viewbox = _load_svg(svg_path)
        if not _viewbox_close(viewbox, current_viewbox):
            raise ValueError(
                f"ViewBox changed between frames. First={viewbox} current={current_viewbox}"
            )
        image = _render_paths(
            paths,
            attrs,
            viewbox,
            (out_w, out_h),
            scale,
            sample_step,
            DEFAULT_BG,
        )
        _draw_step_label(image, step, font, padding)
        frames.append(image)

    out_path = (
        Path(args.output) if args.output is not None else input_dir / "checkpoints.gif"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / args.fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    print(f"Saved GIF with {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
