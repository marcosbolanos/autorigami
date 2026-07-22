# Quasistatic gravity minimization specification

## Goal

Find a lower gravitational-energy configuration of a thick open centerline
without simulating time. The method must move all vertices through one global
sparse solve per iteration, preserve the material chain, enforce the rigid
floor and geometric bounds, reparametrize the final centerline, and use the
standard validator as the final authority.

This is a separate gravity workload. It must not import the fractional Sobolev
metric, tangent-point objective, dynamic PBD state, velocities, damping, or
timesteps.

## Optimization problem

For centerline vertices `x_i = (x_i, y_i, z_i)`, minimize gravitational
potential

```text
G(x) = sum_i z_i
```

subject to

```text
|x_(i+1) - x_i| = reference_length_i
turning_angle_i <= maximum_angle
segment_distance(edge_i, edge_j) >= minimum_distance
z_i >= floor_height + tube_radius
```

The defaults remain physical repository defaults:

- reference sampling interval: 0.34 nm;
- maximum angle: 3.25 degrees in radians;
- minimum nonlocal segment distance: 2.6 nm;
- tube radius: 1.3 nm;
- valid-curvature local exclusion: 100 edges;
- floor placement:
  `min(input[:, 2]) - tube_radius - initial_floor_clearance`.

The input must be green under `validate_polyline`. A red input is rejected.

## Algorithm: proximal sequential convex programming

At feasible iterate `x`, solve for one displacement `d`. The convex subproblem
is

```text
minimize_d  sum_i d_(i,z) + (1 / (2 * proximal_step)) * ||d||^2
```

under linearized constraints and a trust region. Without constraints, every
vertex moves downward by exactly `proximal_step`. The positive quadratic term
makes the QP strictly convex and selects a unique minimum-norm horizontal
motion.

### Linearized constraints

Material edges are equalities:

```text
J_edge d = -edge_residual
```

Curvature, contact, and floor constraints use slack form:

```text
slack + J d >= safety_margin
```

They remain inequalities. A contact is therefore allowed to open when the
proposed displacement increases its distance; it is never frozen merely
because it is close.

Every curvature and floor row is included. Contact rows are included for all
segment pairs within

```text
minimum_distance + contact_safety_margin + 2 * trust_radius
```

because moving each segment endpoint by at most `trust_radius` can reduce the
distance between two segments by at most `2 * trust_radius`. This candidate
radius is a correctness rule, not a tuning heuristic.

The componentwise QP bound is

```text
-trust_radius / sqrt(3) <= d_coordinate <= trust_radius / sqrt(3)
```

and therefore guarantees `||d_i|| <= trust_radius` for every vertex.

### Sparse QP representation

Use OSQP with

```text
P = identity / proximal_step
q_(i,z) = 1; q_(i,x) = q_(i,y) = 0
A = stack(edge, curvature, contact, floor, identity)
```

and OSQP bounds `lower <= A d <= upper`:

- edge rows: equal lower and upper bounds;
- geometric rows: finite lower bound and `+infinity` upper bound;
- trust rows: symmetric finite bounds.

The first implementation rebuilds the OSQP problem whenever the contact-pair
set changes. It may reuse the solver and automatically warm-start only when
the entire sparsity pattern is unchanged. No fragile row remapping is required
in version one.

Accept only OSQP's `solved` status. `solved inaccurate`, maximum-iteration,
primal-infeasible, and dual-infeasible statuses are recorded and trigger a
smaller trust radius or a transparent failure. Polishing is enabled for the
accepted solve.

## Nonlinear projection and acceptance

The QP satisfies only tangent-plane constraints. Apply the complete QP step,
then project `x + d` onto the nonlinear constraints:

1. Rebuild exact segment contacts and curvature and floor slacks.
2. Select only inequalities that violate the physical constraint plus the
   small float32 validation reserve. Nearby but nonblocking contacts remain
   inactive and are not frozen.
3. Solve one sparse minimum-norm correction containing every material-edge
   equality and every selected curvature, contact, and floor inequality.
4. Repeat until edge and inequality tolerances are met or the projection limit
   is reached, rebuilding the active set after every correction.

This active-set correction keeps material edges and blocking geometric
constraints coupled without putting every nearby contact into the projection
solve. The standard validator remains the final acceptance authority.

The gravity projection must not preserve or restore the barycenter. Vertical
translation is part of the objective. It also must not align the candidate
back to the input orientation.

Accept a projected candidate only when:

- nonlinear projection converged;
- floor clearance is nonnegative within tolerance;
- the standard validator is green;
- every material-edge error is within tolerance;
- gravitational energy strictly decreased by the configured minimum amount.

If projection fails, recompute the QP with half the trust radius. Do not merely
scale the already-computed displacement and do not use objective backtracking
along an unprojected path. After an accepted iteration, grow the trust radius
by at most 20 percent, capped by its configured maximum.

## Termination

The workload converges when all of the following hold for three consecutive
accepted iterations:

- maximum vertex displacement is below `displacement_tolerance`;
- relative gravitational-energy decrease is below `energy_tolerance`;
- QP primal and dual residuals meet configured tolerances;
- nonlinear edge, floor, curvature, and separation constraints are green.

Reaching the iteration limit is not convergence. Return the best accepted green
iterate with `converged=False` and an explicit message.

## Finalization

1. Reparametrize the best accepted centerline at 0.34 nm arc-length intervals.
2. Run `validate_polyline` with 3.25 degrees, 2.6 nm, and the normal
   curvature-dependent exclusion logic.
3. Check arc-length sampling explicitly.
4. Never label a red reparametrized result successful.

The pre-reparametrized material-chain result and final reparametrized result
are both retained in the result object so the reparametrization boundary is
observable.

## Code structure and functions

### `src/autorigami/gravity/constraints.py`

- `GravityConstraintConfiguration`
  - Physical bounds, safety margins, and local exclusion.
- `GravityConstraintLinearization`
  - Sparse matrix, lower/upper bounds, row counts, contact pairs, and exact
    nonlinear slacks.
- `floor_slacks(points, floor_height, tube_radius) -> FloatArray`
- `floor_constraint_jacobian(vertex_count) -> csr_matrix`
- `contact_candidate_radius(configuration, trust_radius) -> float`
- `linearize_gravity_constraints(points, reference_lengths, floor_height,
  trust_radius, configuration) -> GravityConstraintLinearization`
  - Builds edge, every curvature, exact nearby contact, every floor, and trust
    row in documented order.
- `evaluate_gravity_constraint_residuals(...) -> GravityConstraintResiduals`
  - Reports maximum edge error, maximum angle, minimum exact contact slack,
    minimum floor slack, and counts.

Existing analytical edge, curvature, and segment-distance derivatives are
reused. Their formulas are not copied into a second implementation.

### `src/autorigami/gravity/quadratic_step.py`

- `GravityQuadraticStepConfiguration`
  - Proximal step and OSQP tolerances/iteration limit.
- `GravityQuadraticStepResult`
  - Displacement, status, objective, primal/dual residuals, solve time, QP
    iteration count, and whether a warm start was used.
- `gravity_linear_cost(vertex_count) -> FloatArray`
- `proximal_hessian(vertex_count, proximal_step) -> csc_matrix`
- `solve_gravity_quadratic_step(linearization, configuration,
  warm_start=None) -> GravityQuadraticStepResult`
  - The only module importing `osqp`.
  - Rejects every non-`solved` status through a typed result, not an assertion.

### `src/autorigami/gravity/projection.py`

- `GravityProjectionConfiguration`
- `GravityProjectionResult`
- `project_gravity_constraints(candidate, reference_lengths, floor_height,
  constraint_configuration, projection_configuration)
  -> GravityProjectionResult`
  - Nonlinear SQP projection with no barycenter restoration.
  - Rebuilds contact candidates after every correction.
  - Reports actual nonlinear residuals at return.

The implementation may extract general code from the current nonlinear
projector, but Sobolev behavior and rigid-orientation alignment must not enter
this module.

### `src/autorigami/gravity/quasistatic.py`

- `QuasistaticGravityConfiguration`
  - Constraint, QP, projection, trust-region, acceptance, and convergence
    controls.
- `QuasistaticGravityIteration`
  - Energy before/after, trust radius, QP diagnostics, projection diagnostics,
    contact count, height, support count, and accepted/rejected state.
- `QuasistaticGravityResult`
  - Best material-chain points, floor metadata, convergence state, message,
    and complete iteration history.
- `gravitational_energy(points) -> float`
  - Returns `sum(points[:, 2])` in float64.
- `minimize_gravitational_energy(polyline, configuration)
  -> QuasistaticGravityResult`
  - Validates the input, places the floor, owns the trust-region loop, calls
    the QP and nonlinear projection, and retains only green improving iterates.

### `src/autorigami/gravity/workloads.py`

- `QuasistaticGravityWorkloadResult`
  - Material-chain result, reparametrized result, standard validation, and
    optimization metadata.
- `pack_under_gravity(polyline, edge_length=0.34, configuration=...)
  -> QuasistaticGravityWorkloadResult`
  - Public workflow performing optimization, reparametrization, arc-sampling
    validation, and standard final validation.

### `src/autorigami/gravity/__init__.py`

Export only the public configuration/result types,
`minimize_gravitational_energy`, and `pack_under_gravity`.

## Dependencies

Install OSQP explicitly with `uv add osqp`. Do not implement a dependency
fallback. SciPy sparse matrices remain the interchange representation.

## Required derivative and assembly tests

1. Floor slack and Jacobian match central finite differences.
2. Every assembled row has the documented bound and row category.
3. Edge equalities use current nonlinear residuals, not a hardcoded zero RHS.
4. Curvature rows include every inner vertex.
5. Contact candidates include every pair that could cross 2.6 nm within the
   trust radius; randomized bounded-displacement tests verify the `2r` bound.
6. Trust coordinate bounds imply the requested per-vertex Euclidean bound.
7. QP matrices have correct symmetry, positive definiteness, types, and sparse
   shapes for a 60,000-variable problem.

## Required behavioral tests

1. An unconstrained test cloud takes the exact proximal downward step.
2. A straight chain above the floor translates down without changing edges.
3. Once part of a chain is supported, the global solve moves unsupported
   vertices downward while keeping supported floor inequalities.
4. A nearby contact that would separate is allowed to separate and is not
   activated as an equality.
5. A blocking contact does not cross 2.6 nm after nonlinear projection.
6. Curvature, contact, floor, and edge constraints remain green in one coupled
   case.
7. Nonlinear projection does not restore the input barycenter.
8. Accepted gravitational energy is strictly monotone decreasing.
9. Projection failure causes QP recomputation at a smaller trust radius.
10. Invalid input and invalid reparametrized output are reported explicitly.
11. Identical inputs and configuration produce identical accepted iterates.

## Full-design experiments before choosing defaults

Run V1 and V2 from their corrected, green, arc-length-parametrized inputs.
Screen at least:

- proximal step: 0.02, 0.05, 0.1 nm;
- initial trust radius: 0.02, 0.05, 0.1 nm;
- contact safety margin: 0.01, 0.02 nm;
- projection limit: 20 and 40 corrections.

First run one iteration on each full design and record QP setup time, solve
time, peak memory, row/nonzero counts, projection time, and validation. Stop
the experiment if the sparse factorization exceeds available memory; do not
hide it by downsampling.

Then run the viable configurations for 10 accepted iterations. Choose defaults
only from configurations that are green after every accepted projection and
after final reparametrization. Compare against the 100-step PBD baseline:

- gravitational-energy decrease;
- total height and 75th/90th Z quantiles;
- RMS and maximum displacement;
- supported-vertex count;
- runtime and peak memory;
- curvature and separation violations.

Generate side-by-side light-tube and DNA renders for visual inspection. A
numerically green result is not sufficient evidence of useful packing.

## Audit gates

The implementation is complete only if:

- no temporal state or Sobolev module is imported;
- the QP contains all five row groups: edge, curvature, contact, floor, trust;
- contacts are inequalities and separating contacts are not frozen;
- contact candidate completeness follows the trust-radius bound;
- nonlinear projection does not restore barycenter or orientation;
- every accepted iterate is green and lowers gravitational energy;
- no downsampling or weaker collision check is introduced;
- final reparametrized V1 and V2 outputs are green under standard validation;
- unit tests, full repository tests, Ruff, Pyright, and native checks pass;
- full-design runtime, memory, visual outcome, and every unmet requirement are
  reported transparently.
