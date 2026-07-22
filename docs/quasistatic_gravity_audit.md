# Quasistatic gravity implementation audit

This audit compares the implementation directly with
`quasistatic_gravity_spec.md`. A checked implementation item means the code is
present and exercised. It does not mean the workload has passed the full
experimental acceptance gate.

## Optimization and constraints

- [x] The objective is `sum(points[:, 2])` with a strictly convex proximal
  Hessian.
- [x] One global sparse OSQP step contains edge, curvature, contact, floor, and
  trust rows.
- [x] Edge rows are equalities with the current nonlinear residual on the
  right-hand side.
- [x] Curvature, contact, and floor rows are one-sided inequalities.
- [x] Contact candidates use `minimum_distance + margin + 2 * trust_radius`.
- [x] Coordinate trust bounds imply a per-vertex Euclidean trust bound.
- [x] Only OSQP `solved` status is accepted.
- [x] Failed QP or projection attempts shrink the trust radius and rebuild the
  QP; a failed displacement is never merely scaled.
- [x] Accepted iterates are standard-validator green, material-edge valid, and
  strictly lower in gravitational energy.
- [x] Input must be standard-validator green.
- [x] No Sobolev, dynamic simulation, velocity, damping, or timestep module is
  imported by `autorigami.gravity`.

## Nonlinear projection

- [x] One sparse active-set minimum-norm QP couples all material-edge
  equalities to currently violated curvature, contact, and floor inequalities.
- [x] Curvature and separation rows reuse the repository analytical Jacobian.
- [x] Contact candidates are rebuilt after each complete correction pass.
- [x] The projector restores neither barycenter nor orientation.
- [x] A float32 validation reserve is enforced before standard validation.
- [x] Actual projected displacement, rather than only the QP displacement, is
  used by the convergence test.

The active set includes only physically violated inequalities plus the float32
validation reserve. It deliberately excludes nearby nonblocking contacts;
those contacts are allowed to open, and newly blocking contacts enter when the
active set is rebuilt after the correction.

## Finalization and automated checks

- [x] The public workload retains material-chain and reparametrized points.
- [x] It reparametrizes at the configured interval (0.34 nm by default).
- [x] It checks arc-length sampling and runs the standard validator after
  reparametrization.
- [x] Eighteen focused quasistatic tests pass.
- [x] The full Python test suite passes: 73 tests.
- [x] Ruff passes for `src`, `tests`, and `scripts` (the repository-wide command
  is currently blocked by a pre-existing unused import in `notebooks/demo1`).
- [x] Pyright passes with no errors or warnings.

## Full-design results at current defaults

Inputs:

- V1: `v1_validated_workflow/06_length_restored_feasible.npy`
- V2: `base_pair_centers_alternating_feasibility_cycle_7_length_restored.npy`

| Design / trial | Runtime | Accepted / attempts | Height before | Height after | Final validation |
|---|---:|---:|---:|---:|---|
| V1, staged-projector baseline | 306.3 s | 3 / 16 | 121.480 nm | 121.394 nm | 0 curvature, 0 separation |
| V2, staged-projector baseline | 186.7 s | 10 / 16 | 128.921 nm | 128.846 nm | 0 curvature, 0 separation |
| V1, active set, 5 steps / limit 20 | 189.5 s | 5 / 5 | 121.480 nm | 121.336 nm | material result green |
| V2, active set, 10 steps / limit 40 | 479.6 s | 10 / 11 | 128.921 nm | 128.677 nm | 0 curvature, 0 separation |

The staged baseline exposed incomplete coupling between edge restoration and
local separation correction and is no longer the implementation. The active
set produces a larger decrease and passes a coupled blocking-contact fixture,
but later nonlinear projections become expensive. The V1 ten-step active-set
run with a 40-correction limit was stopped transparently after 14 minutes. A
five-step V1 trial with the 20-correction limit accepted every step; the
20-correction limit is therefore the current provisional default. V2 reached
the ten-iteration limit but did not meet the three-step stationarity test.

Saved diagnostic outputs are under `outputs/quasistatic_gravity_trials/`. The
current complete active-set result is `v2_active_set_default_10_*`; earlier
`v1_default_10_*` and `v2_default_10_*` files are staged-projector baselines.

## Open acceptance gates

- [x] Add coupled supported-chain and nonlinear blocking-contact behavioral
  fixtures.
- [ ] Complete a V1 ten-step active-set run at the 20-correction limit.
- [ ] Run the specified parameter screen rather than declaring the provisional
  values final.
- [ ] Record per-configuration setup time, solve time, projection time, peak
  memory, quantiles, and displacements in a reproducible experiment artifact.
- [ ] Select defaults only from configurations green after every accepted step
  and final reparametrization on both designs.
- [ ] Generate light-tube and DNA comparisons and perform visual inspection.
- [ ] Run native checks after separating this Python-only workload from the
  unrelated uncommitted native simulation/contact-objective work.

The workload is therefore implemented and testable, but it is not yet accepted
as the new packing method.
