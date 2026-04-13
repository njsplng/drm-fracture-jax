#!/usr/bin/env python3

"""PETSc sparse solver for JAX arrays using MPI.

Provides parallel sparse linear system solving using PETSc with MUMPS
for efficient solution of large-scale systems in FEM simulations.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jaxtyping import Array
from mpi4py import MPI
from petsc4py import PETSc

# Global intercommunicator
_intercomm = None
_solver_iters = 0

# Cached sparsity patterns per system shape
_cached_systems = {}


def run_parallel_sparse_solver(
    A_jax: Array,
    b_jax: Array,
    num_procs: int = 2,
    termination_threshold: int = 500,
) -> Array:
    """Solve sparse linear system Ax=b using PETSc with MUMPS.

    Broadcasts the sparse matrix in CSR format and right-hand side to a
    persistent MPI worker pool, solves with MUMPS direct solver, and
    gathers the solution. Maintains separate cached sparsity patterns
    for different system sizes.

    Parameters
    ----------
    A_jax : Array
        Sparse coefficient matrix as JAX array.
    b_jax : Array
        Right-hand side vector as JAX array.
    num_procs : int, optional
        Number of worker processes to spawn. Default is 2.
    termination_threshold : int, optional
        Iteration count at which to terminate and recreate the solver
        pool. Default is 500.

    Returns
    -------
    x : Array
        Solution vector as JAX array.
    """
    global _intercomm, _solver_iters, _cached_systems

    parent_comm = MPI.COMM_SELF

    # Convert to CSR — cache the sparsity pattern per system shape
    A_csr: sp.csr_matrix = sp.csr_matrix(A_jax)
    shape: Tuple[int, int] = A_csr.shape
    data: np.ndarray = A_csr.data

    # Check if sparsity changed for this particular system
    cached: Dict = _cached_systems.get(shape)
    sparsity_changed = (
        cached is None
        or cached["indptr"].shape != A_csr.indptr.shape
        or not np.array_equal(cached["indptr"], A_csr.indptr)
    )

    if sparsity_changed:
        _cached_systems[shape] = {
            "indptr": A_csr.indptr,
            "indices": A_csr.indices,
        }
        cached: Dict = _cached_systems[shape]

    # Get the rhs
    b: np.ndarray = jax.device_get(b_jax)

    # Spawn worker pool once
    if _intercomm is None:
        script = str(Path(__file__).parent / "sparse_solver.py")
        _intercomm = parent_comm.Spawn(
            sys.executable, args=[script], maxprocs=num_procs
        )

    # Broadcast CSR + sparsity_changed flag
    _intercomm.bcast(
        (cached["indptr"], cached["indices"], data, shape, sparsity_changed),
        root=MPI.ROOT,
    )

    # Prepare RHS buffer
    rhs_buf: np.ndarray = np.copy(b)

    # Broadcast RHS in-place
    _intercomm.Bcast([rhs_buf, MPI.DOUBLE], root=MPI.ROOT)

    # Gather solution slices
    n: int = shape[0]
    sol_np: np.ndarray = np.empty(n, dtype=b.dtype)
    for rank in range(_intercomm.Get_remote_size()):
        r0, r1 = compute_slice_indices(rank, n, _intercomm.Get_remote_size())
        _intercomm.Recv([sol_np[r0:r1], MPI.DOUBLE], source=rank, tag=rank)

    # Free memory every threshold iterations
    _solver_iters += 1
    if _solver_iters % termination_threshold == 0:
        terminate_solver()

    return jnp.asarray(sol_np)


def compute_slice_indices(rank: int, M: int, size: int) -> Tuple[int, int]:
    """Compute start and end indices for a rank in distributed setup.

    Parameters
    ----------
    rank : int
        Rank of the process.
    M : int
        Total number of rows to distribute.
    size : int
        Total number of processes.

    Returns
    -------
    tuple[int, int]
        Start and end indices for the given rank.
    """
    rows_per = M // size
    rem = M % size
    if rank < rem:
        r0 = rank * (rows_per + 1)
        r1 = r0 + (rows_per + 1)
    else:
        r0 = rem * (rows_per + 1) + (rank - rem) * rows_per
        r1 = r0 + rows_per
    return r0, r1


def child_service_loop() -> None:
    """Run persistent worker service for solving linear systems.

    Receives CSR matrix and RHS from parent, solves with MUMPS via PETSc,
    and sends back the solution. Maintains separate KSP/matrix objects
    per system shape so that switching between displacement and
    phase-field solves preserves each solver's symbolic factorization.
    """
    parent = MPI.Comm.Get_parent()
    if parent == MPI.COMM_NULL:
        return

    # Initialise PETSc
    comm = PETSc.COMM_WORLD
    size = comm.getSize()
    rank = comm.getRank()

    # Cache of solver objects per system shape
    # Each entry: { "A_p", "ksp", "x_vec", "b_vec", "global_rhs", "rhs_buf", "sol_buf" }
    solver_cache: Dict[Tuple, Dict] = {}

    while True:
        # Get CSR + RHS from parent
        msg: object = parent.bcast(None, root=0)

        # Check for termination signal
        if msg is None:
            break

        # Unpack CSR + sparsity change flag
        indptr, indices, data, shape, sparsity_changed = msg  # type: ignore

        # Manual ownership range matching PETSc defaults
        M: int = shape[0]
        r0, r1 = compute_slice_indices(rank, M, size)  # type: ignore
        start, end = int(indptr[r0]), int(indptr[r1])  # type: ignore
        loc_indptr: np.ndarray = indptr[r0 : r1 + 1] - indptr[r0]
        loc_indices: np.ndarray = indices[start:end]
        loc_data: np.ndarray = data[start:end]

        # Get or create solver objects for this system shape
        ctx: object = solver_cache.get(shape)

        if ctx is None or sparsity_changed:
            # Build matrix and solver from scratch
            A_p: PETSc.Mat = PETSc.Mat().createAIJ(
                size=shape, csr=(loc_indptr, loc_indices, loc_data), comm=comm
            )
            A_p.setUp()
            A_p.assemblyBegin()
            A_p.assemblyEnd()

            # Direct solver with MUMPS
            ksp: PETSc.KSP = PETSc.KSP().create(comm=comm)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")

            # Tell MUMPS to save the symbolic factorization for reuse
            opts = PETSc.Options()
            opts["mat_mumps_icntl_33"] = 1

            ksp.setOperators(A_p)
            ksp.setFromOptions()
            ksp.setUp()

            # Create PETSc Vecs for RHS and solution
            x_vec, b_vec = A_p.getVecs()  # type: ignore

            # Prepare buffers
            global_rhs: np.ndarray = np.empty(M, dtype=np.float64)
            rhs_buf: np.ndarray = global_rhs[r0:r1]
            sol_buf: np.ndarray = x_vec.getArray()

            # Store in cache
            solver_cache[shape] = {
                "A_p": A_p,
                "ksp": ksp,
                "x_vec": x_vec,
                "b_vec": b_vec,
                "global_rhs": global_rhs,
                "rhs_buf": rhs_buf,
                "sol_buf": sol_buf,
            }
            ctx = solver_cache[shape]

        else:
            # Same sparsity — update values only, MUMPS reuses symbolic factorization
            ctx["A_p"].setValuesCSR(loc_indptr, loc_indices, loc_data)
            ctx["A_p"].assemblyBegin()
            ctx["A_p"].assemblyEnd()
            ctx["ksp"].setOperators(ctx["A_p"])

        # Broadcast the global RHS
        parent.Bcast([ctx["global_rhs"], MPI.DOUBLE], root=0)

        # Solve for this RHS
        ctx["b_vec"].setArray(ctx["rhs_buf"])
        ctx["ksp"].solve(ctx["b_vec"], ctx["x_vec"])

        # Send local solution
        parent.Send([ctx["sol_buf"], MPI.DOUBLE], dest=0, tag=rank)

    # Upon termination, clean up
    parent.Disconnect()
    sys.exit(0)


def terminate_solver() -> None:
    """Terminate the persistent worker service."""
    global _intercomm
    if _intercomm is not None:
        _intercomm.bcast(None, root=MPI.ROOT)
        _intercomm.Disconnect()
        _intercomm = None
        time.sleep(1)


if __name__ == "__main__":
    child_service_loop()
