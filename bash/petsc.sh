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
