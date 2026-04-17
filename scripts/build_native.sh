#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-"$ROOT_DIR/.venv/bin/python"}"
VENV_BIN_DIR="$(dirname "$VENV_PYTHON")"
VENV_CMAKE="${VENV_CMAKE:-"$VENV_BIN_DIR/cmake"}"
VENV_NINJA="${VENV_NINJA:-"$VENV_BIN_DIR/ninja"}"
BUILD_DIR="${BUILD_DIR:-"$ROOT_DIR/.cmake-build/native"}"
INSTALL_PREFIX="${INSTALL_PREFIX:-"$ROOT_DIR/src"}"
COMPILE_COMMANDS_LINK="${COMPILE_COMMANDS_LINK:-"$ROOT_DIR/compile_commands.json"}"

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

export PATH="$VENV_BIN_DIR:$PATH"

PYBIND11_CMAKE_DIR="$("$VENV_PYTHON" -m pybind11 --cmakedir)"

"$VENV_CMAKE" \
  -S "$ROOT_DIR" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_MAKE_PROGRAM="$VENV_NINJA" \
  -DPython_EXECUTABLE="$VENV_PYTHON" \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"

"$VENV_CMAKE" --build "$BUILD_DIR"
"$VENV_CMAKE" --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX" --component autorigami-python

ln -sf "$BUILD_DIR/compile_commands.json" "$COMPILE_COMMANDS_LINK"
