#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# install_petsc.sh — build and install PETSc into the repo-local prefix
#
# Environment variables (all optional):
#   PETSC_TAG   git ref to check out        (default: "release")
#   BUILD       "opt" or "dbg"              (default: "opt")
#   IND64       "0" or "1" for 64-bit idx   (default: "1")
#   COMPLEX     "0" or "1" for complex       (default: "0")
#   MPI_IMPL    "mpich" or "openmpi"         (default: "mpich")
#   JOBS        parallel make jobs           (default: auto-detected)
# ─────────────────────────────────────────────────────────────────────────────

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PETSC_SRC_DIR="$REPO_ROOT/third_party/petsc-src"
PETSC_PREFIX="$REPO_ROOT/.local/petsc"
ENV_DIR="$REPO_ROOT/bash"
ENV_FILE="$ENV_DIR/petsc.sh"

mkdir -p "$REPO_ROOT/third_party" "$REPO_ROOT/.local"

# On macOS, ensure Homebrew-installed tools (including keg-only packages
# like bison) are visible. brew shellenv sets PATH, LIBRARY_PATH, etc.
if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    eval "$(brew shellenv)"
    # Keg-only formulae (not linked into the main prefix)
    for keg in bison flex; do
        keg_dir="$(brew --prefix "$keg" 2>/dev/null)" || continue
        [[ -d "$keg_dir/bin" ]] && export PATH="$keg_dir/bin:$PATH"
    done
fi

# Defaults
PETSC_TAG="${PETSC_TAG:-release}"
BUILD="${BUILD:-opt}"
IND64="${IND64:-1}"
COMPLEX="${COMPLEX:-0}"
MPI_IMPL="${MPI_IMPL:-mpich}"

# Auto-detect parallelism
if [[ -n "${JOBS:-}" ]]; then
    J="$JOBS"
elif command -v nproc >/dev/null 2>&1; then
    J="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    J="$(sysctl -n hw.ncpu)"
else
    J=2
fi

echo "Repo root:      $REPO_ROOT"
echo "PETSc source:   $PETSC_SRC_DIR"
echo "Install prefix: $PETSC_PREFIX"
echo "PETSc tag:      $PETSC_TAG"
echo "Build:          $BUILD"
echo "64-bit idx:     $IND64"
echo "Complex:        $COMPLEX"
echo "MPI:            $MPI_IMPL"
echo "Jobs:           $J"
echo

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — prefer system packages, download only as fallback
# ─────────────────────────────────────────────────────────────────────────────

# _have_cmd — true if a command exists on PATH
_have_cmd() { command -v "$1" >/dev/null 2>&1; }

# _have_lib — true if pkg-config knows about a library
_have_lib() { pkg-config --exists "$1" 2>/dev/null; }

# _prefer_system — add --with-<pkg>-dir or --download-<pkg>=1
# Usage: _prefer_system <petsc-flag-name> <test-command-or-pkg-config-name> [dir-hint]
# If the test succeeds, uses the system version; otherwise downloads.
_add_pkg() {
    local name="$1" found="$2" dir="${3:-}"
    if [[ "$found" == "1" && -n "$dir" ]]; then
        echo "  $name: using system ($dir)"
        CONFIG_OPTS+=("--with-${name}-dir=$dir")
    elif [[ "$found" == "1" ]]; then
        echo "  $name: using system"
        # Let PETSc auto-detect (no flag needed, or explicit enable)
    else
        echo "  $name: not found — will download"
        CONFIG_OPTS+=("--download-${name}=1")
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Clone / update PETSc source
# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -d "$PETSC_SRC_DIR/.git" ]]; then
    git clone https://gitlab.com/petsc/petsc.git "$PETSC_SRC_DIR"
fi
cd "$PETSC_SRC_DIR"
git fetch --tags --prune
git checkout "$PETSC_TAG"

# ─────────────────────────────────────────────────────────────────────────────
# Build configure options
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_OPTS=(
    "--prefix=$PETSC_PREFIX"
    "--with-shared-libraries=1"
    "--with-debugging=$([[ "$BUILD" == "dbg" ]] && echo 1 || echo 0)"
)

# On macOS, Apple Clang has no OpenMP support. If Homebrew GCC is available,
# use it so OpenMP (needed by MUMPS etc.) works.
#
# We set OMPI_CC/MPICH_CC etc. so the system MPI wrappers (mpicc, mpicxx)
# invoke GCC instead of Clang. This avoids the conflict between --with-cc
# and --with-mpi-dir that PETSc doesn't allow.
# When no system MPI is present (download path), we pass --with-cc directly.
USE_BREW_GCC=0
if [[ "$(uname)" == "Darwin" ]]; then
    GCC_VER=$(ls /opt/homebrew/bin/gcc-* /usr/local/bin/gcc-* 2>/dev/null \
              | grep -oE '[0-9]+$' | sort -n | tail -1 || true)
    if [[ -n "$GCC_VER" ]]; then
        echo "Compilers:      Homebrew GCC $GCC_VER"
        USE_BREW_GCC=1
        # OpenMPI respects OMPI_CC/CXX/FC; MPICH respects MPICH_CC/CXX/FC
        export OMPI_CC="gcc-$GCC_VER"  OMPI_CXX="g++-$GCC_VER"  OMPI_FC="gfortran-$GCC_VER"
        export MPICH_CC="gcc-$GCC_VER" MPICH_CXX="g++-$GCC_VER" MPICH_FC="gfortran-$GCC_VER"
    else
        echo "Warning: Homebrew GCC not found — using Apple Clang (no OpenMP)" >&2
    fi
fi

# OpenMP: test whether the selected compiler actually supports it
_test_openmp() {
    local cc="${CC:-cc}"
    [[ $USE_BREW_GCC -eq 1 ]] && cc="gcc-$GCC_VER"
    echo '#include <omp.h>
int main(void){return omp_get_num_threads();}' | \
        "$cc" -fopenmp -x c - -o /dev/null 2>/dev/null
}

if _test_openmp; then
    echo "OpenMP:         supported"
    CONFIG_OPTS+=("--with-openmp=1")
else
    echo "OpenMP:         not supported by compiler — disabled"
fi

# Scalar and index types
[[ "$COMPLEX" == "1" ]] && CONFIG_OPTS+=("--with-scalar-type=complex")
[[ "$IND64"   == "1" ]] && CONFIG_OPTS+=("--with-64-bit-indices=1")

echo "Detecting system packages..."

# --- MPI ---
# Auto-detect which MPI flavour (if any) is on the system, then reconcile
# with the requested MPI_IMPL. If the system has *any* MPI, use it rather
# than downloading — even if it's a different flavour than requested.
mpi_found=0
mpi_system=""
if _have_cmd mpicc; then
    mpi_id=$(mpicc --showme:version 2>&1 || mpicc -v 2>&1 || true)
    if echo "$mpi_id" | grep -qi "open.mpi"; then
        mpi_system="openmpi"
    elif echo "$mpi_id" | grep -qi "mpich"; then
        mpi_system="mpich"
    fi
fi

if [[ -n "$mpi_system" ]]; then
    # Resolve MPI prefix so PETSc finds it even with non-default compilers
    mpi_dir=""
    if _have_cmd mpicc; then
        # OpenMPI: mpicc --showme:home; MPICH: mpicc -show then infer prefix
        mpi_dir=$(mpicc --showme:home 2>/dev/null || true)
        if [[ -z "$mpi_dir" ]]; then
            # Fallback: derive from mpicc location (e.g. /opt/homebrew/bin/mpicc -> /opt/homebrew)
            mpi_dir=$(dirname "$(dirname "$(command -v mpicc)")")
        fi
    fi

    if [[ "$mpi_system" != "$MPI_IMPL" ]]; then
        echo "  MPI: found $mpi_system (requested $MPI_IMPL) — using system $mpi_system"
    else
        echo "  MPI ($mpi_system): using system"
    fi
    CONFIG_OPTS+=("--with-mpi=1")
    [[ -n "$mpi_dir" ]] && CONFIG_OPTS+=("--with-mpi-dir=$mpi_dir")
else
    echo "  MPI ($MPI_IMPL): not found — will download"
    CONFIG_OPTS+=("--with-mpi=1" "--download-${MPI_IMPL}=1")
    # No --with-mpi-dir, so we can pass compilers directly
    if [[ $USE_BREW_GCC -eq 1 ]]; then
        CONFIG_OPTS+=(
            "--with-cc=gcc-$GCC_VER"
            "--with-cxx=g++-$GCC_VER"
            "--with-fc=gfortran-$GCC_VER"
        )
    fi
fi

# --- BLAS/LAPACK ---
# Prefer system OpenBLAS or Apple Accelerate; download only if nothing works.
blas_resolved=0
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: Accelerate is always available and well-optimised
    echo "  BLAS/LAPACK: using Apple Accelerate"
    CONFIG_OPTS+=("--with-blaslapack-lib=-framework Accelerate")
    blas_resolved=1
fi
if [[ $blas_resolved -eq 0 ]] && _have_lib openblas; then
    echo "  BLAS/LAPACK: using system OpenBLAS (pkg-config)"
    # PETSc will find it via pkg-config automatically
    blas_resolved=1
fi
if [[ $blas_resolved -eq 0 ]] && _have_cmd gfortran; then
    echo "  BLAS/LAPACK: gfortran available — will download OpenBLAS"
    CONFIG_OPTS+=("--download-openblas=1")
    blas_resolved=1
fi
if [[ $blas_resolved -eq 0 ]]; then
    echo "  BLAS/LAPACK: no Fortran — will download f2cblaslapack"
    CONFIG_OPTS+=("--download-f2cblaslapack=1" "--with-fc=0")
fi

# --- Other dependencies ---
# For each: check system first, download if missing.

if _have_lib hwloc; then
    _add_pkg hwloc 1 "$(pkg-config --variable=prefix hwloc 2>/dev/null)"
else
    _add_pkg hwloc 0
fi

# These are specialised solver libraries — unlikely to be on system, so
# just download them. PETSc builds them quickly since they're small.
for pkg in hypre mumps scalapack parmetis metis ptscotch superlu_dist; do
    CONFIG_OPTS+=("--download-${pkg}=1")
done

# Bison: PTScotch requires >= 3.0; macOS ships 2.3 but Homebrew's is on PATH now
bison_ok=0
if _have_cmd bison; then
    bison_ver=$(bison --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    bison_major="${bison_ver%%.*}"
    if [[ -n "$bison_major" && "$bison_major" -ge 3 ]]; then
        echo "  bison: using system ($bison_ver)"
        bison_ok=1
    fi
fi
if [[ $bison_ok -eq 0 ]]; then
    echo "  bison: not found or too old (need >= 3.0) — will download"
    CONFIG_OPTS+=("--download-bison=1")
fi

echo

# ─────────────────────────────────────────────────────────────────────────────
# Configure, build, install
# ─────────────────────────────────────────────────────────────────────────────

export PETSC_DIR="$PETSC_SRC_DIR"
unset PETSC_ARCH

echo "=== Running configure ==="
python3 ./configure "${CONFIG_OPTS[@]}"

echo
echo "=== Building and installing (jobs: $J) ==="
make PETSC_DIR="$PETSC_SRC_DIR" -j"$J" all
make PETSC_DIR="$PETSC_SRC_DIR" install

# ─────────────────────────────────────────────────────────────────────────────
# Write env file (repo-local, relocatable)
# ─────────────────────────────────────────────────────────────────────────────

cat > "$ENV_FILE" <<'EOF'
# Source this to use the repo-local PETSc build: source bash/petsc.sh
_detect_script_path() {
    if [ -n "${BASH_SOURCE:-}" ] && [ -n "${BASH_SOURCE[0]:-}" ]; then
        printf '%s\n' "${BASH_SOURCE[0]}"; return
    fi
    if [ -n "${ZSH_VERSION:-}" ]; then
        # shellcheck disable=SC2296
        printf '%s\n' "${(%):-%x}"; return
    fi
    printf '%s\n' "$0"
}
_this_file="$(_detect_script_path)"
_env_dir="$(cd -- "$(dirname -- "$_this_file")" >/dev/null 2>&1 && pwd -P)"
_repo_root="$(cd "$_env_dir/.." >/dev/null 2>&1 && pwd -P)"

export PETSC_DIR="$_repo_root/.local/petsc"
unset PETSC_ARCH
export PATH="$PETSC_DIR/bin:$PATH"
export PKG_CONFIG_PATH="$PETSC_DIR/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$PETSC_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$PETSC_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

# On macOS with Homebrew GCC: ensure MPI wrappers use GCC (not Apple Clang)
# so that extensions (petsc4py etc.) compile against GCC-built PETSc correctly.
if [ "$(uname)" = "Darwin" ]; then
    _gcc_ver=$(ls /opt/homebrew/bin/gcc-* /usr/local/bin/gcc-* 2>/dev/null \
               | grep -oE '[0-9]+$' | sort -n | tail -1) 2>/dev/null || true
    if [ -n "${_gcc_ver:-}" ]; then
        export OMPI_CC="gcc-$_gcc_ver"   OMPI_CXX="g++-$_gcc_ver"   OMPI_FC="gfortran-$_gcc_ver"
        export MPICH_CC="gcc-$_gcc_ver"  MPICH_CXX="g++-$_gcc_ver"  MPICH_FC="gfortran-$_gcc_ver"
    fi
    unset _gcc_ver
fi
EOF

echo
echo "PETSc install complete at $PETSC_PREFIX."

# Clean up source tree (non-fatal — macOS .git locks can cause issues)
rm -rf "$REPO_ROOT/third_party" 2>/dev/null || true
