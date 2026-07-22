# Temporary quasistatic gravity handoff

> **DELETE THIS FILE BEFORE MERGING THIS BRANCH INTO `main`.**
>
> This is working-session state for continuation on another machine, not
> permanent project documentation. The durable requirements and audit belong
> in `quasistatic_gravity_spec.md` and `quasistatic_gravity_audit.md`.

## Resume instructions

1. Read `AGENTS.md`.
2. Read `docs/quasistatic_gravity_spec.md` completely.
3. Read `docs/quasistatic_gravity_audit.md` completely.
4. Run the verification commands below before changing defaults.
5. Continue the open acceptance gates in the audit. Do not call the workload
   complete until V1 and V2 are green after final reparametrization and the
   rendered result is visually useful.
6. Delete this handoff file before merging to `main`.

Suggested first prompt in a new Codex session:

```text
Read AGENTS.md, docs/quasistatic_gravity_spec.md,
docs/quasistatic_gravity_audit.md, and
docs/quasistatic_gravity_handoff.md completely. Continue the open acceptance
gates without weakening curvature, separation, material-edge, floor, or final
reparametrization validation. Keep the implementation modular and explicitly
audit any deviation from the specification.
```

## Branch state

The branch implements a non-temporal quasistatic gravity workload in
`src/autorigami/gravity/`:

- minimize `sum(points[:, 2])` with a proximal sparse OSQP problem;
- preserve every reference material-edge length as a linearized equality;
- enforce curvature, exact segment separation, floor support, and a per-vertex
  trust region as inequalities;
- project nonlinear candidates through a sparse active-set correction coupling
  material-edge equalities with only currently blocking inequalities;
- accept only strictly improving, standard-validator-green material iterates;
- reparametrize at 0.34 nm and run the standard validator again.

`src/autorigami/optimization/constraints.py` now vectorizes the existing exact
contact and curvature Jacobian assembly. The formulas and row semantics are
unchanged.

The branch intentionally excludes the uncommitted temporal gravity simulation,
contact-objective experiments, and other unrelated working-tree files.

## Physical defaults that must not drift

- edge sampling: 0.34 nm;
- maximum turning angle: 3.25 degrees in radians;
- minimum nonlocal segment separation: 2.6 nm;
- tube radius: 1.3 nm;
- valid-curvature local exclusion: 100 edges;
- projection correction limit: provisionally 20;
- the standard validator is the final authority after reparametrization.

Do not use the 100-edge exclusion to justify a red-curvature final design. The
standard validator automatically falls back to excluding only immediate
neighbors when curvature is invalid.

## Required input files

The `outputs/` directory is not transferred by Git. Copy these corrected green
inputs to the same relative paths on the remote machine:

```text
outputs/full_double_spiral_20260707_125413/v1_validated_workflow/06_length_restored_feasible.npy
outputs/full_double_spiral_20260711_125622/base_pair_centers_alternating_feasibility_cycle_7_length_restored.npy
```

Example from the original machine, replacing the remote host and checkout:

```bash
rsync -av --progress \
  outputs/full_double_spiral_20260707_125413/v1_validated_workflow/06_length_restored_feasible.npy \
  REMOTE:CHECKOUT/outputs/full_double_spiral_20260707_125413/v1_validated_workflow/

rsync -av --progress \
  outputs/full_double_spiral_20260711_125622/base_pair_centers_alternating_feasibility_cycle_7_length_restored.npy \
  REMOTE:CHECKOUT/outputs/full_double_spiral_20260711_125622/
```

## Last verified state

Automated checks:

```text
60 tests passed from a fresh clone of this branch
Ruff passed for src, tests, and scripts
Pyright: 0 errors, 9 existing native-extension source-resolution warnings
git diff --check passed
```

The repository-wide Ruff command also inspects notebooks and is independently
blocked by a pre-existing unused import in `notebooks/demo1.ipynb`. Do not hide
or conflate that unrelated issue with this workload.

The original dirty working tree reported 73 tests and no Pyright warnings
because it contained thirteen excluded experiment tests and a locally built
native-extension source state. The fresh branch verification above is the
portable result that should be used on the remote machine.

Full-design observations:

- First active-set step:
  - V1: 12.0 s before Jacobian vectorization, height 121.480 -> 121.452 nm;
  - V2: 9.4 s before Jacobian vectorization, height 128.921 -> 128.893 nm;
  - nonlinear projection activated four inequalities and took one correction.
- V1, active set, five steps, projection limit 20:
  - 189.5 s;
  - all five steps accepted at trust radius 0.05 nm;
  - projection corrections per step: 1, 3, 5, 4, 8;
  - height 121.480 -> 121.336 nm;
  - material result green.
- V2, active set, ten steps, projection limit 40:
  - 479.6 s;
  - ten accepted steps in eleven attempts;
  - one 40-correction failure caused trust shrink from 0.05 to 0.025 nm;
  - height 128.921 -> 128.677 nm;
  - final reparametrized result had zero curvature and separation violations.
- V1 ten-step active-set run with projection limit 40 was stopped after
  fourteen minutes. This was a transparent performance stop, not convergence.
- A previous staged projector was faster but did not correctly couple blocking
  separation contacts to edge-length restoration. It is obsolete and must not
  be restored.

Current complete active-set V2 diagnostics, if copied separately, are:

```text
outputs/quasistatic_gravity_trials/v2_active_set_default_10.json
outputs/quasistatic_gravity_trials/v2_active_set_default_10_material.npy
outputs/quasistatic_gravity_trials/v2_active_set_default_10_reparametrized.npy
```

## Verification commands

```bash
uv sync
uv run pytest -q
uv run ruff check src tests scripts --exclude '*.ipynb'
uv run pyright
git diff --check
```

## Next work, in order

1. Add a reproducible command-line experiment runner that writes configuration,
   timings, validation, and every accepted checkpoint incrementally. It must be
   resumable so an interrupted multi-hour run does not lose accepted iterates.
2. Complete V1 for ten accepted steps with the 20-correction limit.
3. Run the specification's parameter screen on both designs. Do not select
   defaults from V2 alone.
4. Benchmark OSQP `builtin`, `mkl`, and `cuda` backends on an identical
   one-step V1 problem. Record total, setup, solve, projection, and validation
   time as well as the final constraints. CUDA uses an indirect solve, so do
   not assume it wins without measurements.
5. Select defaults only from configurations green after every accepted step
   and final reparametrization on both designs.
6. Generate light-tube and DNA comparisons and inspect them visually.
7. Update the durable audit with final measurements.
8. Remove obsolete trial outputs and code paths, rerun every check, and delete
   this handoff document before merging to `main`.

For long remote runs, use `tmux` or the machine's batch scheduler in addition
to incremental checkpoints. Process persistence is not a substitute for
checkpointing.

## Codex conversation continuity

Codex interactive transcripts are local state under `CODEX_HOME` (normally
`~/.codex`) and are not part of this Git branch. If the same Codex state is
available on the remote machine, use `codex resume SESSION_ID -C CHECKOUT` or
`codex fork SESSION_ID -C CHECKOUT`. Do not put Codex authentication state or
raw session files into this repository. This document is the portable,
reviewable fallback when the original transcript is unavailable.
