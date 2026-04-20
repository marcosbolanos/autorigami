# Contributing

## Setup

Install the dev toolchain once:

```bash
uv sync --group dev
```

Refresh the editable install whenever you add, delete, or rename Python package files:

```bash
uv pip install --python .venv/bin/python -e . --reinstall
```

## Native C++ development

Rebuild the native extension when you change files under `cpp/`:

```bash
./scripts/build_native.sh
```

That script configures CMake in `.cmake-build/native`, builds the `_native` module, installs it in-place to `src/autorigami/`, and refreshes `compile_commands.json` for `clangd`.

Native code is organized under:

```text
cpp/include/autorigami/...
cpp/src/...
cpp/tests/...
```

## Testing

We use pytest for python test and ctest for C++ tests

```bash
./scripts/build_native.sh
ctest --test-dir .cmake-build/native --output-on-failure
./.venv/bin/python -m pytest
```

If you only changed Python code, skip the native rebuild and just run pytest.

## Linting

```bash
uv run ruff format
./scripts/build_native.sh
```

C/C++ linting is handled by `clang-tidy`, and it runs automatically during CMake builds.
This is enabled by default in the dev build script (`./scripts/build_native.sh`).
