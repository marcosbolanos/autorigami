# ODE Spiral Findings

## Final Result

Using the default ODE CLI on `assets/ellipsoid.obj`:

```bash
PYTHONPATH=src .venv/bin/python main.py --generator ode --input assets/ellipsoid.obj
```

Final verified low-density check from `outputs/20260416_122722/run_info.json`:

- Validation separation compliance: `1000 / 1000`
- Validation curvature compliance: `1000 / 1000`
- Polyline length: `611.30 nm`
- Axis coverage span: `80.8%`
- Axis start / end coverage: `9.5% -> 90.4%`
- Nearest nonlocal separation stats:
  - min: `3.192 nm`
  - mean: `6.193 nm`
  - q25: `6.983 nm`
  - q75: `7.141 nm`
  - max: `7.167 nm`

## What Changed

- Added post-validation metrics in `src/autorigami/metrics.py` and surfaced them in `main.py`.
- Kept `src/autorigami/validation.py` unchanged.
- Reworked the ODE generator so it:
  - enforces forward progress along the chosen axis
  - stops before the top pole
  - starts above the bottom pole instead of at the exact tip
- Added CLI controls for:
  - `--min-progress-fraction`
  - `--bottom-clearance-nm`
  - `--top-clearance-nm`
- Tuned ODE defaults to the best verified regime for the ellipsoid:
  - `repulsion-strength = 2.5`
  - `tangential-speed-nm = 12.0`
  - `step-size-nm = 0.8`
  - `bottom-clearance-nm = 5.8`
  - `top-clearance-nm = 5.8`
- I tested `1000` validation samples as a practical operating point, but that does not provide the same guarantee as `20000`.

## What Worked

- The main overlap bug was caused by the spiral backtracking and then re-entering the top cap. A progress guard plus pole clearances removed the self-overlap failure.
- Bottom-pole clearance mattered as much as top-pole clearance. Starting at the exact pole created persistent curvature failures even when separation was fixed.
- The best tradeoff I found was symmetric `5.8 nm` pole clearances with the tuned ODE defaults above.

## What Did Not Work

- Keeping the old exact-pole start: separation could be fixed, but curvature stayed bad.
- Small step sizes with the old sample budget: those runs often collapsed into much shorter traces.
- Very conservative pole clearances like `10.4 nm`: they were compliant, but coverage dropped too much.
- Extremely dense resampling (`10000` to `40000` validation samples): the same spiral stayed separation-compliant, but a tiny number of curvature misses reappeared due to dense Bezier resampling, not top-overlap.

## Resampling Note

At `1000` validation samples, the tuned spiral is fully compliant.

That is not sufficient to claim compliance at `20000`.

For the same long candidate, increasing Bezier resampling density produces a few curvature misses while keeping separation clean. Example from the tuned high-coverage family:

- `1000` samples: `1000 / 1000` curvature compliant
- `2000` samples: `1998 / 2000`
- `5000` samples: `4981 / 5000`
- `10000` samples: `9948 / 10000`

That behavior indicates the old top-overlap problem is solved, but it does not justify lowering the validation target. To claim compliance at `20000`, the spiral must be tuned and checked at `20000`.
