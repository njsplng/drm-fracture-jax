"""FEM nodal computations for solving the system of equations.

This module provides functions for assembling stiffness matrices,
computing residuals, and solving displacement and phasefield
systems using Newton-Raphson iterations.
"""

from functools import partial
from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array
from utils_fem import solve_sparse_jax

from utils import elemental_to_global, timed_jit


@partial(timed_jit, static_argnames=("lagrange_multiplier_enabled"))
def pre_newton_raphson_load(
    F_incremental: Array,
    displacement_incremental: Array,
    B_full: Array,
    V: Array,
    lagrange_multiplier_enabled: bool = False,
) -> Array:
    """Set the initial residual for the Newton-Raphson method.

    Parameters
    ----------
    F_incremental : Array
        Incremental external force vector.
    displacement_incremental : Array
        Incremental displacement vector.
    B_full : Array
        Full constraint matrix for kinematic constraints.
    V : Array
        Constraint volume/target values.
    lagrange_multiplier_enabled : bool, optional
        Whether to use Lagrange multipliers. Default is False.

    Returns
    -------
    residual_iterative : Array
        Initial residual vector for the iterative solver.
    """
    if lagrange_multiplier_enabled:
        residual_iterative_upper = -F_incremental
        residual_iterative_lower = B_full @ displacement_incremental.reshape(-1, 1) - V
        residual_iterative = jnp.vstack(
            (residual_iterative_upper.reshape(-1, 1), residual_iterative_lower)
        )
    else:
        residual_iterative = -F_incremental

    return residual_iterative


@timed_jit
def pointwise_D_to_global_stiffness_coeffs(
    ip_D: Array,
    ip_volumes: Array,
    ip_B: Array,
    dofs: Array,
) -> Tuple[Array, Array, Array]:
    """Convert pointwise constitutive matrices to global stiffness coefficients.

    Parameters
    ----------
    ip_D : Array
        Pointwise constitutive matrices of shape (E, G, 3, 3).
    ip_volumes : Array
        Integration point volumes of shape (E, G).
    ip_B : Array
        Pointwise strain-displacement matrices of shape (E, G, 3, n).
    dofs : Array
        Degrees of freedom for each element of shape (E, n).

    Returns
    -------
    row_idx : Array
        Row indices for sparse matrix assembly.
    col_idx : Array
        Column indices for sparse matrix assembly.
    values : Array
        Stiffness values for sparse matrix assembly.
    """
    # Need B.T @ D @ B * point_volume
    # B.T @ D
    pointwise_stiffness = jnp.einsum(
        "egji, egki -> egjk", ip_B.transpose(0, 1, 3, 2), ip_D
    )
    # B.T @ D @ B
    pointwise_stiffness = jnp.einsum("egji, egik -> egjk", pointwise_stiffness, ip_B)
    # B.T @ D @ B * point_volume
    pointwise_stiffness = jnp.einsum(
        "egjk, eg -> egjk", pointwise_stiffness, ip_volumes
    )

    # ip_D [E, G, 3, 3]
    elemental_stiffness = jnp.sum(pointwise_stiffness, axis=(1))
    # [E, n_dofs, n_dofs]

    # Get the number of local dofs
    E, n_local, _ = elemental_stiffness.shape
    # Create index pairs for each element
    row_idx = jnp.repeat(dofs[:, :, None], n_local, axis=2)
    col_idx = jnp.repeat(dofs[:, None, :], n_local, axis=1)
    row_idx_flat = row_idx.flatten()
    col_idx_flat = col_idx.flatten()
    values_flat = elemental_stiffness.flatten()

    return (row_idx_flat, col_idx_flat, values_flat)


@partial(
    timed_jit,
    static_argnames=(
        "dofs_size",
        "lagrange_multiplier_enabled",
        "mask",
        "fixed_dofs",
        "penalty_method_enabled",
        "penalty_value",
        "penalty_master_dof",
    ),
)
def pointwise_D_to_global_stiffness_sparse_matrix(
    ip_D: Array,
    ip_volumes: Array,
    ip_B: Array,
    dofs: Array,
    dofs_size: int,
    fixed_dofs: Array,
    lagrange_multiplier_enabled: bool,
    B_full: Array,
    mask: Array,
    lagrange_mask: Array,
    penalty_method_enabled: bool,
    penalty_value: float,
    penalty_master_dof: int,
    penalty_tied_dofs: Array,
) -> Tuple[Array, Tuple[Array, Array]]:
    """Convert pointwise constitutive matrices to global stiffness sparse matrix.

    Parameters
    ----------
    ip_D : Array
        Pointwise constitutive matrices of shape (E, G, 3, 3).
    ip_volumes : Array
        Integration point volumes of shape (E, G).
    ip_B : Array
        Pointwise strain-displacement matrices.
    dofs : Array
        Degrees of freedom for each element.
    dofs_size : int
        Total number of degrees of freedom in the system.
    fixed_dofs : Array
        Indices of fixed degrees of freedom.
    lagrange_multiplier_enabled : bool
        Whether Lagrange multipliers are used for constraints.
    B_full : Array
        Full constraint matrix for Lagrange multipliers.
    mask : Array
        Boolean mask for active constraints.
    lagrange_mask : Array
        Indices for extracting Lagrange multiplier contributions.
    penalty_method_enabled : bool
        Whether penalty method is used for tying constraints.
    penalty_value : float
        Penalty value for constraint enforcement.
    penalty_master_dof : int
        Master degree of freedom for penalty constraints.
    penalty_tied_dofs : Array
        Degrees of freedom to tie to the master dof.

    Returns
    -------
    values : Array
        Flattened stiffness values.
    indices : Tuple[Array, Array]
        Row and column indices for sparse matrix.
    """
    # Get the pointwise stiffness
    row_idx, col_idx, values = pointwise_D_to_global_stiffness_coeffs(
        ip_D=ip_D,
        ip_volumes=ip_volumes,
        ip_B=ip_B,
        dofs=dofs,
    )

    # Slice according to the mask
    row_free = row_idx[jnp.array(mask)]
    col_free = col_idx[jnp.array(mask)]
    values_free = values[jnp.array(mask)]

    if lagrange_multiplier_enabled:
        # Extract the B matrix indices
        i_B, j_B = lagrange_mask
        i_B = jnp.array(i_B)
        j_B = jnp.array(j_B)

        # Get the values of the B matrix
        values_B = B_full[i_B, j_B]

        # Form the lagrange multiplier indices for non-transpose B matrix
        row_B = dofs_size + i_B
        col_B = j_B

        # Form the lagrange multiplier indices for transpose B matrix
        row_BT = j_B
        col_BT = dofs_size + i_B

        # Re-form the free indices and values
        row_free = jnp.concatenate([row_free, row_B, row_BT], dtype=int)
        col_free = jnp.concatenate([col_free, col_B, col_BT], dtype=int)
        values_free = jnp.concatenate([values_free, values_B, values_B])

    if penalty_method_enabled:
        # Tie constraints: for each i in bound_dofs (except master),
        # enforce u_i - u_master = 0 via penalty.
        master = int(penalty_master_dof)

        tied = penalty_tied_dofs
        # If nothing to tie, do nothing
        if tied.size > 0:
            n = tied.shape[0]
            m = jnp.full((n,), master, dtype=int)

            # COO entries for sum_i p * (e_i - e_m)(e_i - e_m)^T
            row_pen = jnp.concatenate([tied, m, tied, m], dtype=int)
            col_pen = jnp.concatenate([tied, m, m, tied], dtype=int)

            ones = jnp.ones((n,), dtype=float)
            vals_pen = penalty_value * jnp.concatenate([ones, ones, -ones, -ones])

            row_free = jnp.concatenate([row_free, row_pen], dtype=int)
            col_free = jnp.concatenate([col_free, col_pen], dtype=int)
            values_free = jnp.concatenate([values_free, vals_pen])

    # Get the fixed dofs
    values_fixed = jnp.ones(len(fixed_dofs))
    row_fixed = jnp.array(fixed_dofs)
    col_fixed = jnp.array(fixed_dofs)

    # Assemble the full vectors now
    row_full = jnp.concatenate([row_free, row_fixed], dtype=int)
    col_full = jnp.concatenate([col_free, col_fixed], dtype=int)
    values_full = jnp.concatenate([values_free, values_fixed])

    return (values_full, (row_full, col_full))


@partial(timed_jit, static_argnames=("dofs_size"))
def pointwise_internal_forces_to_global(
    ip_internal_forces: Array,
    dofs: Array,
    dofs_size: int,
) -> Array:
    """Convert elemental internal forces to global internal forces.

    Parameters
    ----------
    ip_internal_forces : Array
        Pointwise internal forces of shape (E, G, n).
    dofs : Array
        Degrees of freedom for each element of shape (E, n).
    dofs_size : int
        Total number of degrees of freedom in the system.

    Returns
    -------
    global_forces : Array
        Global internal forces of shape (dofs_size,).
    """
    # Get the elemental forces first
    elemental_internal_forces = jnp.sum(ip_internal_forces, axis=1)
    # [E, n_dofs]

    return elemental_to_global(elemental_internal_forces, dofs, dofs_size)


def stiffness_to_global_stiffness(
    ip_D: Array,
    dofs: Array,
) -> Tuple[Array, Tuple[Array, Array]]:
    """Convert pointwise constitutive matrices to global stiffness.

    Parameters
    ----------
    ip_D : Array
        Pointwise constitutive matrices of shape (E, G, n, m).
    dofs : Array
        Degrees of freedom for each element of shape (E, n).

    Returns
    -------
    values : Array
        Flattened stiffness values.
    indices : Tuple[Array, Array]
        Row and column indices for sparse matrix.
    """
    elemental_stiffness = jnp.sum(ip_D, axis=(1))
    # Get the number of local dofs
    E, n_local, _ = elemental_stiffness.shape
    # Create index pairs for each element
    row_idx = jnp.repeat(dofs[:, :, None], n_local, axis=2)
    col_idx = jnp.repeat(dofs[:, None, :], n_local, axis=1)
    row_idx_flat = row_idx.flatten()
    col_idx_flat = col_idx.flatten()
    values_flat = elemental_stiffness.flatten()

    return (values_flat, (row_idx_flat, col_idx_flat))


@partial(timed_jit, static_argnames=("dofs_size"))
def pointwise_qoi_to_global_pf_force(
    ip_N: Array,
    ip_volumes: Array,
    dofs_size: int,
    dofs: Array,
) -> Array:
    """Convert pointwise quantities to global phase field force.

    Parameters
    ----------
    ip_N : Array
        Pointwise shape function values of shape (E, G, n).
    ip_volumes : Array
        Integration point volumes of shape (E, G).
    dofs_size : int
        Total number of degrees of freedom in the system.
    dofs : Array
        Degrees of freedom for each element of shape (E, n).

    Returns
    -------
    global_force : Array
        Global phase field force vector of shape (dofs_size,).
    """
    # Calculate the forcing at the integration points
    ip_forcing = jnp.einsum("eg, egn -> egn", ip_volumes, ip_N)
    elemental_forcing = jnp.sum(ip_forcing, axis=1)

    return elemental_to_global(elemental_forcing, dofs, dofs_size)


@partial(
    timed_jit,
    static_argnames=("lagrange_slice", "lagrange_multiplier_enabled"),
)
def solve_displacement_load_control(
    lagrange_multiplier_enabled: bool,
    K: Tuple[Array, Tuple[Array, Array]],
    displacement_iterative: Array,
    residual_iterative: Array,
    lagrange_slice: slice,
) -> Tuple[Array, Optional[Array]]:
    """Solve for the displacement in the case of load control.

    Parameters
    ----------
    lagrange_multiplier_enabled : bool
        Whether Lagrange multipliers are used.
    K : Tuple[Array, Tuple[Array, Array]]
        Stiffness matrix in sparse format (values, (rows, cols)).
    displacement_iterative : Array
        Current displacement estimate.
    residual_iterative : Array
        Current residual vector.
    lagrange_slice : slice
        Slice for separating displacement and multiplier components.

    Returns
    -------
    displacement : Array
        Updated displacement vector.
    multipliers : Optional[Array]
        Lagrange multipliers if enabled, otherwise None.
    """
    # Solve the system
    solution = solve_sparse_jax(K, -residual_iterative).squeeze()

    # In the case of lagrange multipliers
    if lagrange_multiplier_enabled:
        # Displacement calculation with Lagrange multipliers
        displacement_iterative = solution[:lagrange_slice]
        lagrange_multipliers_iterative = solution[lagrange_slice:]
        return displacement_iterative, lagrange_multipliers_iterative

    # Otherwise
    else:
        # Default displacement calculation
        displacement_iterative = solution
        return displacement_iterative, None


@partial(
    timed_jit,
    static_argnames=(
        "newton_raphson_tolerance",
        "lagrange_multiplier_enabled",
        "penalty_method_enabled",
        "penalty_value",
        "penalty_master_dof",
    ),
)
def hyplas_residual(
    displacement_incremental: Array,
    F_incremental: Array,
    internal_forces_incremental: Array,
    free_dofs: Array,
    boundary_conditions: Array,
    B_full: Array,
    V: Array,
    lagrange_multipliers_incremental: Array,
    newton_raphson_tolerance: float,
    lagrange_multiplier_enabled: bool,
    penalty_method_enabled: bool,
    penalty_value: float,
    penalty_master_dof: int,
    penalty_tied_dofs: Array,
) -> Tuple[float, Array, bool]:
    """Calculate the hyplas-style residual.

    Parameters
    ----------
    displacement_incremental : Array
        Incremental displacement vector.
    F_incremental : Array
        Incremental external force vector.
    internal_forces_incremental : Array
        Incremental internal forces.
    free_dofs : Array
        Indices of free degrees of freedom.
    boundary_conditions : Array
        Indices of constrained degrees of freedom.
    B_full : Array
        Constraint matrix for Lagrange multipliers.
    V : Array
        Constraint target values.
    lagrange_multipliers_incremental : Array
        Lagrange multiplier values.
    newton_raphson_tolerance : float
        Convergence tolerance for Newton-Raphson.
    lagrange_multiplier_enabled : bool
        Whether Lagrange multipliers are used.
    penalty_method_enabled : bool
        Whether penalty method is used.
    penalty_value : float
        Penalty value for constraint enforcement.
    penalty_master_dof : int
        Master degree of freedom for penalty constraints.
    penalty_tied_dofs : Array
        Degrees of freedom to tie to master.

    Returns
    -------
    ratio : float
        Relative residual ratio as percentage.
    residual : Array
        Residual vector.
    converged : bool
        Whether Newton-Raphson has converged.
    """
    # Residual calculation
    if lagrange_multiplier_enabled:
        # Get the upper residual
        residual_iterative_upper = (
            internal_forces_incremental
            + B_full.T @ lagrange_multipliers_incremental
            - F_incremental.squeeze()
        )
        # Get the lower residual
        residual_iterative_lower = B_full @ displacement_incremental.reshape(-1, 1) - V
        # Form the residual
        residual_iterative = jnp.vstack(
            (residual_iterative_upper.reshape(-1, 1), residual_iterative_lower)
        )
        # No residual force at Dirichlet boundary conditions
        residual_iterative = residual_iterative.at[boundary_conditions].set(0)

    # If penalty method is used
    elif penalty_method_enabled:
        # Base mechanical residual
        residual_iterative = internal_forces_incremental.reshape(-1, 1) - F_incremental
        residual_iterative = residual_iterative.at[boundary_conditions].set(0)

        # Tie constraints: u_i - u_m = 0 for all tied dofs i
        master = int(penalty_master_dof)
        tied = penalty_tied_dofs

        if tied.size > 0:
            delta = displacement_incremental[tied] - displacement_incremental[master]
            # (n,)
            # Add +p*(u_i-u_m) on each tied dof row
            residual_iterative = residual_iterative.at[tied].add(
                penalty_value * delta[:, None]
            )
            # Add -p*sum(u_i-u_m) on the master row (accumulates all ties)
            residual_iterative = residual_iterative.at[master].add(
                -penalty_value * jnp.sum(delta)
            )

        # Re-zero any strong Dirichlet rows (safety; master should not be in BCs)
        residual_iterative = residual_iterative.at[boundary_conditions].set(0)
        # If no kinematic constraints
    else:
        residual_iterative = internal_forces_incremental.reshape(-1, 1) - F_incremental
        residual_iterative = residual_iterative.at[boundary_conditions].set(0)

    # Get the hyplas-style residual
    reactive_forces = jnp.zeros_like(displacement_incremental).reshape(-1, 1)
    reactive_forces = reactive_forces.at[free_dofs].set(F_incremental[free_dofs])
    reactive_forces = reactive_forces.at[boundary_conditions].set(
        internal_forces_incremental[boundary_conditions.reshape(-1, 1)]
    )

    # Get the ratios
    residual_ratio = jnp.sum(residual_iterative[free_dofs] ** 2)
    reactive_forces_ratio = jnp.sum(reactive_forces[free_dofs] ** 2)
    maximum_residual = jnp.max(jnp.abs(residual_iterative[free_dofs]))
    residual_norm = jnp.sqrt(residual_ratio)
    reactive_forces_norm = jnp.sqrt(reactive_forces_ratio)

    # Calculate the ratio
    ratio = jnp.where(
        reactive_forces_norm == 0,
        0.0,
        100 * residual_norm / reactive_forces_norm,
    )

    # Determine whether the newton raphson has converged
    converged = jnp.logical_or(
        ratio < newton_raphson_tolerance,
        jnp.abs(maximum_residual) < newton_raphson_tolerance * 1e-3,
    )

    # If the ratio is not satisfied, return the ratio and False
    return ratio, residual_iterative, converged


@partial(
    timed_jit,
    static_argnames=(
        "lagrange_slice",
        "lagrange_multiplier_enabled",
        "target_dof",
    ),
)
def solve_displacement_disp_control_initial(
    lagrange_multiplier_enabled: bool,
    F: Array,
    V: Array,
    lagrange_slice: slice,
    K: Tuple[Array, Tuple[Array, Array]],
    displacement_target_incremental: Union[float, Array],
    target_dof: int,
    load_factor_incremental: float,
) -> Tuple[Array, Array, Optional[Array], float]:
    """Solve the first step in displacement control.

    Parameters
    ----------
    lagrange_multiplier_enabled : bool
        Whether Lagrange multipliers are used.
    F : Array
        External force vector.
    V : Array
        Constraint target values.
    lagrange_slice : slice
        Slice for separating displacement and multiplier components.
    K : Tuple[Array, Tuple[Array, Array]]
        Stiffness matrix in sparse format.
    displacement_target_incremental : Union[float, Array]
        Target displacement increment.
    target_dof : int
        Target degree of freedom index.
    load_factor_incremental : float
        Initial load factor.

    Returns
    -------
    displacement : Array
        Computed displacement.
    F_incremental : Array
        Applied force.
    multipliers : Optional[Array]
        Lagrange multipliers if enabled, otherwise None.
    load_factor : float
        Updated load factor.
    """
    displacement_iterative_unfactored, lagrange_multipliers_iterative_unfactored = (
        solve_displacement_unfactored_disp_control(
            K=K,
            F=F,
            lagrange_multiplier_enabled=lagrange_multiplier_enabled,
            lagrange_slice=lagrange_slice,
            V=V,
        )
    )

    # Calculate the load factor
    load_factor_iterative = (
        displacement_target_incremental / displacement_iterative_unfactored[target_dof]
    )

    # Increment the load factor
    load_factor_incremental = load_factor_incremental + load_factor_iterative

    # Factor the displacement and applied force
    displacement_iterative = load_factor_incremental * displacement_iterative_unfactored
    F_incremental = load_factor_incremental * F

    if lagrange_multiplier_enabled:
        # Increment the lagrange incremental multiplier
        lagrange_multipliers_iterative = (
            load_factor_incremental * lagrange_multipliers_iterative_unfactored
        )
        return (
            displacement_iterative,
            F_incremental,
            lagrange_multipliers_iterative,
            load_factor_incremental,
        )

    # If not kinematic constraints
    return displacement_iterative, F_incremental, None, load_factor_incremental


@partial(
    timed_jit,
    static_argnames=(
        "lagrange_slice",
        "lagrange_multiplier_enabled",
        "target_dof",
    ),
)
def solve_displacement_disp_control_subsequent(
    lagrange_multiplier_enabled: bool,
    F: Array,
    V: Array,
    lagrange_slice: slice,
    K: Tuple[Array, Tuple[Array, Array]],
    residual_iterative: Array,
    target_dof: int,
    load_factor_incremental: float,
) -> Tuple[Array, Array, Optional[Array], float]:
    """Solve subsequent steps in displacement control.

    Parameters
    ----------
    lagrange_multiplier_enabled : bool
        Whether Lagrange multipliers are used.
    F : Array
        External force vector.
    V : Array
        Constraint target values.
    lagrange_slice : slice
        Slice for separating displacement and multiplier components.
    K : Tuple[Array, Tuple[Array, Array]]
        Stiffness matrix in sparse format.
    residual_iterative : Array
        Current residual vector.
    target_dof : int
        Target degree of freedom index.
    load_factor_incremental : float
        Current load factor.

    Returns
    -------
    displacement : Array
        Computed displacement.
    F_incremental : Array
        Applied force.
    multipliers : Optional[Array]
        Lagrange multipliers if enabled, otherwise None.
    load_factor : float
        Updated load factor.
    """
    displacement_iterative_unfactored, _ = solve_displacement_unfactored_disp_control(
        K=K,
        F=F,
        lagrange_multiplier_enabled=lagrange_multiplier_enabled,
        lagrange_slice=lagrange_slice,
        V=V,
    )
    if lagrange_multiplier_enabled:
        # Solve for the displacements with Lagrange multipliers
        solution = solve_sparse_jax(K, -residual_iterative).squeeze()
        displacement_iterative = solution[:lagrange_slice]
        lagrange_multipliers_iterative = solution[lagrange_slice:]

    else:
        # Solve for the displacements in a default manner
        displacement_iterative = solve_sparse_jax(K, -residual_iterative).squeeze()

    # Calculate the load factor due to the residual displacement
    load_factor_iterative = (
        -displacement_iterative[target_dof]
        / displacement_iterative_unfactored[target_dof]
    )

    # Increment the load factor
    load_factor_incremental = load_factor_incremental + load_factor_iterative

    # Factor the incremental and iterative forces
    F_incremental = load_factor_incremental * F
    F_iterative = load_factor_iterative * F

    # Solve for the displacements with the adjusted forcing
    if lagrange_multiplier_enabled:
        # Solve for the displacements with Lagrange multipliers
        F_solve = -residual_iterative
        F_solve = F_solve.at[: len(F_iterative)].add(F_iterative)

        # Compute the solution
        solution = solve_sparse_jax(K, F_solve).squeeze()

        # Distribute the solution terms
        displacement_iterative = solution[:lagrange_slice]
        lagrange_multipliers_iterative = solution[lagrange_slice:]
        return (
            displacement_iterative,
            F_incremental,
            lagrange_multipliers_iterative,
            load_factor_incremental,
        )

    else:
        # Solve for the displacements in a default manner
        displacement_iterative = solve_sparse_jax(
            K, -residual_iterative + F_iterative
        ).squeeze()

    return displacement_iterative, F_incremental, None, load_factor_incremental


@partial(
    timed_jit,
    static_argnames=(
        "lagrange_slice",
        "lagrange_multiplier_enabled",
    ),
)
def solve_displacement_unfactored_disp_control(
    K: Tuple[Array, Tuple[Array, Array]],
    F: Array,
    lagrange_multiplier_enabled: bool,
    lagrange_slice: int,
    V: Array,
) -> Tuple[Array, Optional[Array]]:
    """Solve for the unfactored displacement.

    Parameters
    ----------
    K : Tuple[Array, Tuple[Array, Array]]
        Stiffness matrix in sparse format.
    F : Array
        External force vector.
    lagrange_multiplier_enabled : bool
        Whether Lagrange multipliers are used.
    lagrange_slice : int
        Slice for separating displacement and multiplier components.
    V : Array
        Constraint target values.

    Returns
    -------
    displacement : Array
        Unfactored displacement vector.
    multipliers : Optional[Array]
        Lagrange multipliers if enabled, otherwise None.
    """
    lagrange_multipliers_iterative_unfactored = None
    # Check for Lagrange multipliers
    if lagrange_multiplier_enabled:
        # Displacement calculation with Lagrange multipliers
        F_solve = jnp.vstack((F.reshape(-1, 1), V))
        solution_unfactored = solve_sparse_jax(K, F_solve).squeeze()
        displacement_iterative_unfactored = solution_unfactored[:lagrange_slice]
        lagrange_multipliers_iterative_unfactored = solution_unfactored[lagrange_slice:]

    # Otherwise, don't apply any physical constraints
    else:
        # Solve the system
        displacement_iterative_unfactored = solve_sparse_jax(K, F).squeeze()

    return displacement_iterative_unfactored, lagrange_multipliers_iterative_unfactored


def solve_phasefield(
    previous_phasefield: Array,
    phasefield_model: object,
    ip_data: object,
    mesh_data: object,
    ip_history_field: Array,
) -> Array:
    """Construct and solve the phasefield system of equations.

    Parameters
    ----------
    previous_phasefield : Array
        Previous phasefield solution.
    phasefield_model : object
        Phasefield material model with tangent method.
    ip_data : object
        Integration point data container.
    mesh_data : object
        Mesh data container with connectivities.
    ip_history_field : Array
        History field at integration points.

    Returns
    -------
    phasefield : Array
        Solved phasefield variable.
    """
    # Form the by-element phasefield
    c_elemental = previous_phasefield[mesh_data.connectivities]
    c_elemental = jnp.repeat(c_elemental[:, None, :], ip_data.N.shape[1], axis=1)

    # Get the pointwise constitutive matrices
    ip_D = phasefield_model.tangent(
        c_elemental=c_elemental,
        N=ip_data.N,
        dNdx=(
            ip_data.physical_derivatives_rot
            if ip_data.physical_derivatives_rot is not None
            else ip_data.physical_derivatives
        ),
        history_field=ip_history_field,
        d2Ndx2=(
            ip_data.physical_derivatives_2_rot
            if ip_data.physical_derivatives_2_rot is not None
            else ip_data.physical_derivatives_2
        ),
        gamma=ip_data.gamma_matrix,
    )

    # Weight by the volumes
    ip_D = jnp.einsum("egnm, eg->egnm", ip_D, ip_data.volumes)

    # Assemble the global stiffness and force
    K_pf = stiffness_to_global_stiffness(ip_D, mesh_data.connectivities)
    F_pf = pointwise_qoi_to_global_pf_force(
        ip_data.N,
        ip_data.volumes,
        mesh_data.nodal_coordinates.shape[0],
        mesh_data.connectivities,
    )

    # Solve for the phasefield
    phasefield = solve_sparse_jax(K_pf, F_pf)

    return phasefield
