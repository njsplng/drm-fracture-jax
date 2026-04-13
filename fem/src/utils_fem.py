"""Additional utility functions for FEM simulations."""

import os
import subprocess
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import scipy.sparse as sp
from jaxtyping import Array
from scipy.sparse import csr_matrix

from utils import timed_jit

_SOLVER_TYPE = "scipy"  # Default solver type
_PARALLEL_SOLVER = None


def reorder_qoi_dict(
    qoi_dict: Dict[str, List[Array]],
    inv_perm_2d: Array,
    inv_perm_1d: Array,
    dofs_size: int,
) -> None:
    """Postprocess the qoi dict by extrapolating to nodes from integration points.

    Parameters
    ----------
    qoi_dict : Dict[str, List[Array]]
        Dictionary of quantities of interest to reorder.
    inv_perm_2d : Array
        Inverse permutation array for 2D reordering.
    inv_perm_1d : Array
        Inverse permutation array for 1D reordering.
    dofs_size : int
        Size of the degrees of freedom.
    """
    # Make a copy to not
    for key, value in qoi_dict.items():
        if not key.startswith("ip"):
            values = []
            for entry in value:
                if entry.shape[0] == dofs_size:
                    # 1D case
                    values.append(entry[inv_perm_2d.astype(int)])
                elif entry.shape[0] == dofs_size // 2:
                    # 2D case
                    values.append(entry[inv_perm_1d.astype(int)])

            qoi_dict[key] = values


def solve_sparse_scipy(A: Array, b: Array) -> Array:
    """Solve a sparse linear system using SciPy."""
    A = csr_matrix(A)
    return sp.linalg.spsolve(A, b)


@timed_jit
def solve_sparse_jax(
    A: Array,
    b: Array,
) -> Array:
    """Solve a sparse linear system using JAX's pure callback.

    Directs the sparse solver to use either SciPy or PETSc based on
    the global settings.

    Parameters
    ----------
    A : Array
        Sparse matrix representing the linear system.
    b : Array
        Right-hand side vector.

    Returns
    -------
    Array
        Solution vector to the linear system.
    """
    out_shape = b.squeeze().shape
    out_aval = jax.ShapeDtypeStruct(out_shape, jnp.float64)
    if _SOLVER_TYPE == "scipy":
        return jax.pure_callback(solve_sparse_scipy, out_aval, A, b)
    elif _SOLVER_TYPE == "petsc":
        global _PARALLEL_SOLVER
        return jax.pure_callback(
            _PARALLEL_SOLVER,
            out_aval,
            A,
            b,
            int(_NUM_PROCS),
            int(_TERMINATION_THRESHOLD),
        )


def set_up_sparse_solver(
    solver_type: str,
    num_procs: int,
    termination_threshold: int = 500,
) -> None:
    """Set up the sparse solver type and number of processes.

    Parameters
    ----------
    solver_type : str
        Type of solver to use ("scipy" or "petsc").
    num_procs : int
        Number of processes for parallel computation.
    termination_threshold : int, optional
        Termination threshold for the solver. Default is 500.
    """
    global _SOLVER_TYPE
    global _NUM_PROCS
    global _TERMINATION_THRESHOLD
    _SOLVER_TYPE = solver_type
    _NUM_PROCS = num_procs
    _TERMINATION_THRESHOLD = termination_threshold

    if solver_type == "petsc":
        numa_nodes, cores_per_node = detect_numa_topology()
        num_mpi_procs = numa_nodes
        omp_threads = num_procs // num_mpi_procs

        _NUM_PROCS = num_mpi_procs

        # Must be set BEFORE PETSc import
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        os.environ["OMP_PROC_BIND"] = "close"
        os.environ["OMP_PLACES"] = "cores"

        global _PARALLEL_SOLVER
        _PARALLEL_SOLVER = lazy_load_parallel_sparse_solver()


def detect_numa_topology() -> Tuple[int, int]:
    """Detect the NUMA topology of the system.

    Returns
    -------
    Tuple[int, int]
        A tuple containing (numa_nodes, cores_per_node).
    """
    total_cores = os.cpu_count() or 1

    try:
        result = subprocess.run(["lscpu", "-p=NODE"], capture_output=True, text=True)
        # Count unique NUMA node IDs (skip comment lines)
        nodes = set(
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("#")
        )
        num_nodes = len(nodes) if nodes else 1
    except FileNotFoundError:
        num_nodes = 1

    cores_per_node = total_cores // num_nodes
    return num_nodes, cores_per_node


def cleanup_solver() -> None:
    """Sends a termination signal if PETSc is used, otherwise does nothing."""
    global _SOLVER_TYPE
    if _SOLVER_TYPE == "petsc":
        # Get termination definition
        from sparse_solver import terminate_solver

        # Send termination signal
        terminate_solver()


def lazy_load_parallel_sparse_solver() -> Callable[..., Array]:
    """Lazy initialization of the parallel sparse solver."""
    from sparse_solver import run_parallel_sparse_solver

    return run_parallel_sparse_solver
