# Contributing

## Setup

```bash
uv sync
```

This installs the dev tools and compiles the C++ code. Changes to Python files need no reinstall.

## Native C++ development

The native code is in `cpp/include/autorigami` and `cpp/src`. Rebuild it when you change a file under `cpp/`:

```bash
uv sync --reinstall-package autorigami
```

The build tree stays in `.cmake-build`, so PhysX compiles only one time.

If you change a binding, update `src/autorigami/_native.pyi` too.

## Testing

```bash
uv run pytest
```

## Linting

```bash
uv run ruff format
uv run ruff check
uv run pyright
```

`clang-tidy` runs during the C++ build when the CMake option `AUTORIGAMI_DEV_LINT` is `ON`. It is `OFF` by default.
