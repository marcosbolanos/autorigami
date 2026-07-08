#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-"$ROOT_DIR/.venv/bin/python"}"
VENV_BIN_DIR="$(dirname "$VENV_PYTHON")"
VENV_CMAKE="${VENV_CMAKE:-"$VENV_BIN_DIR/cmake"}"
VENV_NINJA="${VENV_NINJA:-"$VENV_BIN_DIR/ninja"}"
C_COMPILER="${CC:-/usr/bin/gcc}"
CXX_COMPILER="${CXX:-/usr/bin/g++}"
BUILD_DIR="${BUILD_DIR:-"$ROOT_DIR/.cmake-build/native"}"
DEFAULT_INSTALL_PREFIX="$("$VENV_PYTHON" - <<'PY'
import sysconfig
print(sysconfig.get_path("platlib"))
PY
)"
INSTALL_PREFIX="${INSTALL_PREFIX:-"$DEFAULT_INSTALL_PREFIX"}"
COMPILE_COMMANDS_LINK="${COMPILE_COMMANDS_LINK:-"$ROOT_DIR/compile_commands.json"}"
PYTHON_INCLUDE_DIR="$("$VENV_PYTHON" - <<'PY'
import sysconfig
print(sysconfig.get_path("include"))
PY
)"
PYTHON_LIBRARY="$("$VENV_PYTHON" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LIBDIR") + "/" + sysconfig.get_config_var("LDLIBRARY"))
PY
)"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "expected Python interpreter at $VENV_PYTHON" >&2
  echo "run 'uv sync --group dev' first or set VENV_PYTHON explicitly" >&2
  exit 1
fi

if [[ ! -x "$VENV_CMAKE" || ! -x "$VENV_NINJA" ]]; then
  echo "expected cmake and ninja in $VENV_BIN_DIR" >&2
  echo "run 'uv sync --group dev' first" >&2
  exit 1
fi

if [[ ! -x "$C_COMPILER" || ! -x "$CXX_COMPILER" ]]; then
  echo "expected C and C++ compilers at $C_COMPILER and $CXX_COMPILER" >&2
  echo "set CC and CXX explicitly if your compilers live elsewhere" >&2
  exit 1
fi

export PATH="$VENV_BIN_DIR:$PATH"

PYBIND11_CMAKE_DIR="$("$VENV_PYTHON" -m pybind11 --cmakedir)"

"$VENV_CMAKE" \
  -S "$ROOT_DIR" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_MAKE_PROGRAM="$VENV_NINJA" \
  -DCMAKE_C_COMPILER="$C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DAUTORIGAMI_DEV_LINT=ON \
  -DPython_EXECUTABLE="$VENV_PYTHON" \
  -DPYTHON_EXECUTABLE="$VENV_PYTHON" \
  -DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
  -DPYTHON_LIBRARY="$PYTHON_LIBRARY" \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"

"$VENV_CMAKE" --build "$BUILD_DIR"
"$VENV_CMAKE" --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX" --component autorigami-python

ln -sf "$BUILD_DIR/compile_commands.json" "$COMPILE_COMMANDS_LINK"
