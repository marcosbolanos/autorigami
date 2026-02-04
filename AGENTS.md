# AGENTS

This file guides agentic coding in this repository.
Follow it when writing code, tests, or docs.

## Repository layout
- `src/` is the Python package root (import as `src.*`).
- `src/curvepack/` holds geometry and optimization code.
- `src/run_optimize.py` is the main CLI entrypoint.
- `src/utils/` contains helper scripts (data download).
- `data/` stores local datasets and outputs.
- `tests/` is the pytest test root.

## Environment
- Dependencies are listed in `pyproject.toml`.

## Running code
- Use module mode from repo root so imports resolve.

## Build / install
- Use uv
- If you add dependencies, update `pyproject.toml`.

## Lint / format
- No lint or format tools are configured yet.
- Do not introduce new tooling without discussion.
- Keep formatting consistent with existing files.

## Tests (pytest)
- use pytest with uv

## Import conventions
- Keep `from __future__ import annotations` as the first import.
- Order imports: standard library, third-party, local.
- Use explicit relative imports inside `src`.
- Examples: `from .curvepack import optimize` or `from .. import PROJECT_ROOT`.
- Avoid circular imports; move helpers into shared modules.
- Do not add runtime `sys.path` hacks.

## Formatting
- 4-space indentation; no tabs.
- Keep lines ~100 chars or less; wrap with parentheses.
- Prefer f-strings for logging/prints.
- Use blank lines between logical blocks.
- Keep trailing whitespace out of files.

## Types and typing
- Add type hints for public functions and non-trivial helpers.
- Use `np.ndarray` and `jnp.ndarray` for arrays.
- Prefer `float | None` union syntax (Py 3.10+).
- Use explicit dtypes (`np.float32`) when creating arrays.
- Include shapes in docstrings when relevant.
- Keep annotations simple; avoid complex generics.

## Naming
- `snake_case` for functions and variables.
- `CapWords` for classes.
- `UPPER_CASE` for constants and module-level settings.
- Preserve math-style names (Q, C, M, Rmin) when they represent standard symbols.
- Document math-style parameters in docstrings or argparse help.

## Error handling
- Validate user inputs early (shape, dtype, empty arrays).
- Raise `ValueError` with actionable messages.
- Avoid silent fallbacks; document any heuristics.
- Keep numerical safeguards (eps, clamps) explicit.
- Do not swallow exceptions unless you add context.

## Documentation
- Use concise docstrings for public functions.
- Document input shapes, units, and coordinate frames.
- Keep comments focused on non-obvious logic.
- Avoid verbose inline commentary for straightforward code.

## Array / shape conventions
- Points are typically `(N, 2)` in world coordinates.
- Curves are typically `(C, M, 2)` samples.
- Control points are typically `(C, n_ctrl, 2)`.
- SDF grids are `(H, W)` with an `(x, y)` origin.
- Keep axis order consistent; do not transpose unless documented.
- Prefer explicit `axis=` arguments when reducing.

## JAX / NumPy practices
- Convert numpy -> jax at function boundaries with `jnp.asarray`.
- Keep pure JAX inside `jax.jit` and `jax.grad`.
- Do host-side preprocessing (hash grids, sampling) with numpy.
- Avoid Python-side mutation of JAX arrays.
- Use `np.random.default_rng(seed)` for deterministic sampling.
- Thread RNGs explicitly; avoid global state.

## Numerical stability
- Clamp denominators and add small eps where needed.
- Use `jnp.clip` / `np.clip` for bounded values.
- Avoid repeated sqrt when squared distances suffice.
- Prefer stable formulations (soft-min, softplus) when used.

## Geometry / SVG specifics
- SVG parsing uses `svgpathtools` with flatten tolerance in world units.
- `polygon_to_mask` treats y as up in world coords and down in pixel coords.
- Keep viewBox consistent with polygon bounds on export.
- Preserve unit consistency across SDF and SVG steps.
- When adding exporters, keep coordinates in world space.

## Optimization loop behavior
- Keep logging cadence modest (`log_every` style).
- Avoid printing inside tight loops unless gated.
- Keep optimizer state local; avoid globals.
- Preserve reproducibility when changing randomness.

## CLI behavior
- Use argparse with explicit defaults and help text.
- Keep CLI flags stable; add new flags rather than repurposing.
- Print concise progress (loss, step) at reasonable intervals.
- Do not write outputs outside the repo unless user-specified.

## Testing conventions
- Tests go under `tests/` with `test_*.py` names.
- Prefer small, deterministic tests with fixed seeds.
- Use pytest fixtures for shared setup.
- Avoid GPU-only assumptions; keep CPU-compatible tests.
- Validate shapes and dtypes in tests where possible.

## Data paths
- Use `PROJECT_ROOT` from `src/__init__.py` for repo-relative paths.
- Place downloaded data under `data/` (see `src/utils/download_data.py`).
- Keep large artifacts out of version control unless required.

## Cursor/Copilot rules
- No `.cursor/rules`, `.cursorrules`, or `.github/copilot-instructions.md` found.
