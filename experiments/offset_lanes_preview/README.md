Offset Lanes Preview

This folder contains a self-contained preview script for generating boundary-parallel
"lanes" inside a polygon via inward offsets (Shapely buffer) and exporting them to SVG.

Safe to delete: nothing in the main package imports from here.

Usage

  uv run python experiments/offset_lanes_preview/preview_offset_lanes.py \
    --input data/raw/blob1.svg \
    --output data/processed/blob1_offset_lanes_preview.svg

Production-style relaxation (separate script)

  uv run python experiments/offset_lanes_preview/relax_production.py \
    --input data/raw/blob1.svg \
    --output data/processed/blob1_offset_lanes_prodrelax.svg

Step-by-step visualization (folder of SVGs)

  uv run python experiments/offset_lanes_preview/viz_steps.py \
    --input data/raw/blob1.svg \
    --out_dir data/processed/blob1_lane_steps
