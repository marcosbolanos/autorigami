# Contributing

## Native development

Install the dev toolchain once:

```bash
uv sync --group dev
```

After that, rebuild only the native extension when you change files under `cpp/`:

```bash
./scripts/build_native.sh
```

That script configures CMake in `.cmake-build/native`, builds the `_native` module, and installs it in-place to `src/autorigami/`.
