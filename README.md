# autorigami

Design curved DNA nanostructures as spirals, and make them obey the curvature and separation limits of DNA.

![A branching tube before and after gravity packing](assets/gravity_packing.png)

*A branching tube as designed (left) and after gravity packing (right).*

## Install

```
uv sync
```

This compiles the C++ code.

## Start here

1. `notebooks/1_spiral_generation.ipynb`: build a spiral from segments.
2. `notebooks/2_gravity_based_packing.ipynb`: pack it and validate it.

## Layout

| Path | Contents |
|---|---|
| `src/autorigami/spiral_generation` | Spiral design |
| `src/autorigami/geometry` | Validation: curvature and separation |
| `src/autorigami/packing` | Gravity packing |
| `src/autorigami/viz` | 3D views for notebooks |
| `src/autorigami/mesh_io` | Mesh export and renders |
| `cpp` | Native code: distances and PhysX simulation |
| `tests` | Tests |

See [CONTRIBUTING.md](CONTRIBUTING.md) to develop.
