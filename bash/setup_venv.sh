#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — one-shot project setup: PETSc → venv → petsc4py → requirements
#
# Usage: bash/setup.sh [--no-petsc]
#   --no-petsc   skip PETSc build and petsc4py install
# ─────────────────────────────────────────────────────────────────────────────

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd -P)"

PETSC_PREFIX="$REPO_ROOT/.local/petsc"
VENV_DIR="$REPO_ROOT/.venv"
MODULE_FILE="$REPO_ROOT/bash/module_loads.sh"
PETSC_ENV="$REPO_ROOT/bash/petsc.sh"
INSTALL_PETSC="$REPO_ROOT/bash/build_petsc.sh"
REQUIREMENTS="$REPO_ROOT/requirements.txt"

_have_cmd() { command -v "$1" >/dev/null 2>&1; }

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────────────

skip_petsc=0
for arg in "$@"; do
    case "$arg" in
        --no-petsc) skip_petsc=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# 0) Deactivate any active venv
# ─────────────────────────────────────────────────────────────────────────────

if _have_cmd deactivate; then deactivate || true; fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Source module loads (HPC environments)
# ─────────────────────────────────────────────────────────────────────────────

if [[ -f "$MODULE_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$MODULE_FILE"
else
    echo "Note: $MODULE_FILE not found — continuing without module loads."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2) Build PETSc (unless --no-petsc or already installed)
# ─────────────────────────────────────────────────────────────────────────────

if [[ $skip_petsc -eq 1 ]]; then
    echo "Skipping PETSc build (--no-petsc)."
elif [[ -d "$PETSC_PREFIX" ]]; then
    echo "PETSc already installed at $PETSC_PREFIX — skipping build."
else
    [[ -f "$INSTALL_PETSC" ]] || { echo "Error: $INSTALL_PETSC not found." >&2; exit 1; }
    echo "Building PETSc..."
    bash "$INSTALL_PETSC"
    echo "PETSc build complete."
fi

# Determine whether PETSc is available
has_petsc=0
if [[ -f "$PETSC_ENV" ]]; then
    # shellcheck source=/dev/null
    source "$PETSC_ENV"
    if [[ -d "${PETSC_DIR:-}" ]]; then
        has_petsc=1
    fi
fi

if [[ $has_petsc -eq 0 && $skip_petsc -eq 0 ]]; then
    echo "Error: $PETSC_ENV not found — was PETSc installed?" >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3) Create venv
#    Prefer uv for speed; fall back to python -m venv if uv is absent.
# ─────────────────────────────────────────────────────────────────────────────

if [[ -d "$VENV_DIR" ]]; then
    echo "Venv already exists at $VENV_DIR — skipping creation."
else
    if _have_cmd uv; then
        echo "Creating venv at $VENV_DIR (via uv)..."
        uv venv "$VENV_DIR" --python=3.11
    else
        echo "uv not found — bootstrapping venv via python -m venv..."
        if _have_cmd python3.11; then
            python3.11 -m venv "$VENV_DIR"
        elif _have_cmd python3; then
            python3 -m venv "$VENV_DIR"
        else
            echo "Error: no suitable Python interpreter (3.11 or python3) found." >&2
            exit 1
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4) Activate venv and ensure uv is available inside it
# ─────────────────────────────────────────────────────────────────────────────

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

VENV_PY="$VENV_DIR/bin/python"

if ! _have_cmd uv; then
    echo "Installing uv into venv..."
    "$VENV_PY" -m pip install --quiet uv
    if ! _have_cmd uv; then
        echo "Error: failed to install uv into venv." >&2
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5) Install petsc4py (only if PETSc is available)
#    Note: petsc.sh (sourced above) sets OMPI_CC/MPICH_CC on macOS so that
#    mpicc uses Homebrew GCC — required for compiling against GCC-built PETSc.
# ─────────────────────────────────────────────────────────────────────────────

if [[ $has_petsc -eq 1 ]]; then
    PETSC_MM=$(
        awk '
            /#define[ \t]+PETSC_VERSION_MAJOR/ { maj = $3 }
            /#define[ \t]+PETSC_VERSION_MINOR/ { min = $3 }
            END { print maj "." min }
        ' "$PETSC_DIR/include/petscversion.h"
    )
    echo "Detected PETSc version: $PETSC_MM"

    # Build deps (setuptools <74 avoids dry_run breakage with petsc4py)
    echo "Installing build dependencies..."
    uv pip install --python "$VENV_PY" "setuptools<74" wheel Cython numpy

    # petsc4py from source — must link against our PETSc
    echo "Installing petsc4py ${PETSC_MM}.* from source..."
    uv pip install --python "$VENV_PY" \
        --no-binary=:all: --no-build-isolation "petsc4py==${PETSC_MM}.*"

    # Verify
    python3 - <<'PY'
from petsc4py import PETSc
print("petsc4py ok — version:", PETSc.Sys.getVersion())
PY
else
    echo "Skipping petsc4py (no PETSc installation found)."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6) Project requirements
# ─────────────────────────────────────────────────────────────────────────────
if [[ -f "$REQUIREMENTS" ]]; then
    echo "Installing project requirements..."
    uv pip install --python "$VENV_PY" -r "$REQUIREMENTS"
else
    echo "Warning: $REQUIREMENTS not found — skipping."
fi

deactivate
echo
echo "Setup complete. Activate with: source $REPO_ROOT/initialise.sh"
