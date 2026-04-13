"""Finite element method setup functions.

Provide utilities for configuring load factors, displacement factors,
concentrated forces, window-based forcing, and boundary conditions
for finite element simulations.
"""

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array

from distance_functions import CompositeDistanceFunction
from utils import set_up_window_generic


def set_up_load_factor(config: dict) -> Array:
    """Set up the load factor for load control.

    Compute the load factor array based on the number of timesteps
    and loading directions specified in the configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing loop and solution parameters.

    Returns
    -------
    load_factor : Array
        Array of load factors with shape (timesteps + 1,) for single
        loading direction, or varying length for multiple directions.
    """
    timesteps = config.loop_parameters.timesteps
    loading_directions = (
        config.solution_parameters.load_control_parameters.loading_directions
    )

    # In case of a single loading direction, return the full load
    if len(loading_directions) == 1:
        return jnp.linspace(0, 1, timesteps + 1)

    # Start from 0 to get the required differences between the load areas
    loading_directions = [0] + loading_directions
    loading_directions = jnp.array(loading_directions)

    # Find the overall loading "magnitude"
    overall_loading = float(jnp.sum(jnp.abs(jnp.diff(loading_directions))))
    # Determine the number of points per unit of loading
    points_per_unit = timesteps / overall_loading

    # Initialise the empty segments and the current value
    segments = []
    current_val = 0
    for i in range(len(loading_directions) - 1):
        # Find the current step
        step = loading_directions[i + 1] - loading_directions[i]
        # Determine the number of points in the segment
        n_points = int(round(abs(step) * points_per_unit)) + 1
        # Create the segment
        segment = jnp.linspace(current_val, current_val + step, n_points)
        # Save the segment
        segments.append(segment)
        # Save the current value
        current_val += step

    # Concatenate the segments
    full_load = segments[0]
    for segment in segments[1:]:
        full_load = jnp.concatenate((full_load, segment[1:]))

    return full_load


def set_up_disp_factor(config: dict) -> Array:
    """Set up the displacement factor for displacement control.

    Compute the displacement factor array based on the number of
    timesteps, maximum controlled displacement, and loading directions.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing loop and solution parameters.

    Returns
    -------
    disp_factor : Array
        Array of displacement factors with shape depending on loading
        directions.
    """
    # Extract values for easier access
    timesteps = config.loop_parameters.timesteps
    total_control_displacement = (
        config.solution_parameters.displacement_control_parameters.maximum_controlled_displacement
    )
    loading_directions = (
        config.solution_parameters.displacement_control_parameters.loading_directions
    )

    # In case of a single loading direction, return the full displacement
    if len(loading_directions) == 1:
        displacement = jnp.linspace(0, total_control_displacement, timesteps + 1)
        displacement = [0] + displacement.tolist()
        return jnp.array(displacement)

    # Start from 0 to get the required differences between the load areas
    loading_directions = [0] + loading_directions
    loading_directions = jnp.array(loading_directions)

    # Find the overall loading "magnitude"
    overall_loading = float(jnp.sum(jnp.abs(jnp.diff(loading_directions))))
    # Determine the number of points per unit of loading
    points_per_unit = timesteps / overall_loading

    # Initialise the empty segments and the current value
    segments = []
    current_val = 0
    for i in range(len(loading_directions) - 1):
        # Find the current step
        step = loading_directions[i + 1] - loading_directions[i]
        # Determine the number of points in the segment
        n_points = int(round(abs(step) * points_per_unit)) + 1
        # Create the segment
        segment = jnp.linspace(
            current_val, current_val + step * total_control_displacement, n_points
        )
        # Save the segment
        segments.append(segment)
        # Save the current value
        current_val += step * total_control_displacement

    # Concatenate the segments
    full_displacement = segments[0]
    for segment in segments[1:]:
        full_displacement = jnp.concatenate((full_displacement, segment[1:]))

    full_displacement = [0] + full_displacement.tolist()
    full_displacement = jnp.array(full_displacement)

    return full_displacement


def set_up_concentrated_F(config: dict, dofs_size: int, target_dof: int) -> Array:
    """Set up concentrated forcing.

    Create a force vector with concentrated loads applied at specified
    degrees of freedom.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing forcing parameters.
    dofs_size : int
        Total number of degrees of freedom in the system.
    target_dof : int
        Target degree of freedom to apply force if no forced dofs
        are specified.

    Returns
    -------
    F : Array
        Force vector of shape (dofs_size, 1) with concentrated loads.
    """
    forced_degrees_of_freedom = jnp.array(
        config.forcing_parameters.concentrated_parameters.forced_degrees_of_freedom
    )
    # In case the forced dof list is empty, use the target dof
    if list(forced_degrees_of_freedom) == []:
        forced_degrees_of_freedom = jnp.array([target_dof])

    magnitude = config.forcing_parameters.concentrated_parameters.magnitude
    # Extract the required values
    F = jnp.zeros((dofs_size, 1))
    F = F.at[forced_degrees_of_freedom].set(magnitude)
    return F


def set_up_window_F(config: dict, nodal_coordinates: Array) -> Array:
    """Set up the forcing vector from the load window.

    Compute a spatially distributed forcing vector based on the load
    window geometry and magnitude.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing boundary conditions and
        forcing parameters.
    nodal_coordinates : Array
        Array of nodal coordinates for evaluating the load window.

    Returns
    -------
    F : Array
        Forcing vector of shape (n_nodes * 2, 1) with loads oriented
        according to the target degree of freedom.
    """
    load_distance_fn = CompositeDistanceFunction(
        config.boundary_conditions.load_window_parameters
    )
    load_distance_points = load_distance_fn(nodal_coordinates)
    # Filter non unity and non zero values
    load_distance_points = load_distance_points.at[
        jnp.where(
            load_distance_points
            < config.forcing_parameters.window_parameters.filter_threshold
        )
    ].set(0)
    forcing_vector = (
        load_distance_points * config.forcing_parameters.window_parameters.magnitude
    )
    tracked_dof = config.solution_parameters.displacement_control_parameters[
        "target_dof"
    ]

    if tracked_dof % 2 == 0:
        F = jnp.stack([forcing_vector, jnp.zeros_like(forcing_vector)])
        return F.T.flatten()[:, None]

    F = jnp.stack([jnp.zeros_like(forcing_vector), forcing_vector])
    return F.T.flatten()[:, None]


def set_up_window_boundary(config: dict, nodal_coordinates: Array) -> Array:
    """Return the indices of the nodes that are on the fixed window boundary.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing boundary conditions.
    nodal_coordinates : Array
        Array of nodal coordinates for evaluating the fixed window.

    Returns
    -------
    bound_indices : Array
        Indices of nodes on the fixed window boundary.
    """
    return set_up_window_generic(
        nodal_coordinates,
        config.boundary_conditions_parameters.window_value_threshold,
        config.boundary_conditions.fixed_window_parameters,
        config.boundary_conditions_parameters.constrain_directions,
    )


def set_up_load_window_lagrange(config: dict, nodal_coordinates: Array) -> Array:
    """Return the indices of the nodes that are on the load window boundary.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing boundary conditions.
    nodal_coordinates : Array
        Array of nodal coordinates for evaluating the load window.

    Returns
    -------
    bound_indices : Array
        Indices of nodes on the load window boundary for Lagrange
        multiplier constraints.
    """
    return set_up_window_generic(
        nodal_coordinates,
        config.lagrange_multiplier_parameters.window_value_threshold,
        config.boundary_conditions.load_window_parameters,
        config.lagrange_multiplier_parameters.constrain_directions,
    )


def set_up_load_window_penalty(config: dict, nodal_coordinates: Array) -> Array:
    """Return the indices of the nodes that are on the load window boundary.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing boundary conditions.
    nodal_coordinates : Array
        Array of nodal coordinates for evaluating the load window.

    Returns
    -------
    bound_indices : Array
        Indices of nodes on the load window boundary for penalty
        method constraints.
    """
    return set_up_window_generic(
        nodal_coordinates,
        config.penalty_method_parameters.window_value_threshold,
        config.boundary_conditions.load_window_parameters,
        config.penalty_method_parameters.constrain_directions,
    )


def set_up_window_dofs(
    dof_selection_mode: str,
    bound_dofs: Array,
    nodal_coordinates: Array,
    window_function: Callable,
    config: dict,
) -> Array:
    """Given the dof selection mode, return the bound dofs.

    Select and combine degrees of freedom based on the specified mode:
    explicit list, window-based selection, or hybrid of both.

    Parameters
    ----------
    dof_selection_mode : str
        Selection mode: 'explicit', 'window', or 'hybrid'.
    bound_dofs : Array
        Explicit list of bound degrees of freedom (used for 'explicit'
        and 'hybrid' modes).
    nodal_coordinates : Array
        Array of nodal coordinates for window-based selection.
    window_function : Callable
        Function that takes config and nodal_coordinates and returns
        window-selected dofs.
    config : dict
        Configuration dictionary passed to the window function.

    Returns
    -------
    bound_dofs : Array
        Combined array of bound degrees of freedom.

    Raises
    ------
    ValueError
        If dof_selection_mode is not one of 'explicit', 'window', or
        'hybrid'.
    """
    match dof_selection_mode:
        case "explicit":
            bound_dofs = jnp.array(bound_dofs)
        case "window":
            bound_dofs = window_function(config, nodal_coordinates)
        case "hybrid":
            bound_dofs_window = window_function(config, nodal_coordinates)
            bound_dofs_explicit = jnp.array(bound_dofs)
            bound_dofs = jnp.concatenate((bound_dofs_window, bound_dofs_explicit))
        case _:
            raise ValueError(
                "Invalid condition mode. Accepted values are 'explicit' (provide bound dofs), 'window' \
                    (provide parameters for dofs to be bound) or 'hybrid' (both)"
            )
    return bound_dofs


def set_up_body_F(
    config: dict,
    dofs_size: int,
    dofs: Array,
    point_volumes: Array,
    N: Array,
) -> Array:
    """Set up body force loading.

    Compute the distributed body force vector from volumetric forces
    in x and y directions.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing forcing parameters.
    dofs_size : int
        Total number of degrees of freedom in the system.
    dofs : Array
        Array of degrees of freedom for element connectivity.
    point_volumes : Array
        Array of point volumes for numerical integration.
    N : Array
        Shape function matrix for interpolation.

    Returns
    -------
    F : Array
        Body force vector of shape (dofs_size, 1).
    """
    # Extract the required values
    body_force_x = config.forcing_parameters.body_parameters.magnitude_x
    body_force_y = config.forcing_parameters.body_parameters.magnitude_y

    # Initialise the force vector
    F = jnp.zeros((dofs_size, 1))
    force_component_x = body_force_x * jnp.einsum("eng,eg->en", N, point_volumes)
    force_component_y = body_force_y * jnp.einsum("eng,eg->en", N, point_volumes)
    body_force_elemental = jnp.stack((force_component_x, force_component_y), axis=-1)
    body_force_elemental = body_force_elemental.reshape(-1, 1)

    flat_elemental_dofs = dofs.flatten()
    F = F.at[flat_elemental_dofs].add(body_force_elemental)
    return F


def set_up_lagrange_parameters(
    bound_dofs: Array,
    free_dofs: Array,
    dofs_size: int,
) -> tuple[Array, Array, Array, Array]:
    """Set up the Lagrange multiplier parameters for the constrained dofs.

    Construct the constraint matrices and vectors for Lagrange
    multiplier enforcement of degree of freedom constraints.

    Parameters
    ----------
    bound_dofs : Array
        Array of bound (constrained) degrees of freedom.
    free_dofs : Array
        Array of free (unconstrained) degrees of freedom.
    dofs_size : int
        Total number of degrees of freedom in the system.

    Returns
    -------
    B_full : Array
        Full constraint matrix of shape (n_constraints, dofs_size).
    B : Array
        Reduced constraint matrix of shape (n_constraints, n_free_dofs).
    V : Array
        Constraint values vector of shape (n_constraints, 1).
    lagrange_multipliers : Array
        Initial Lagrange multiplier array of shape (n_constraints,).
    """
    # Form the bound dofs array (in pairs)
    index_offset = jnp.array(list(range(1, len(bound_dofs))))
    bound_dofs_offset = jnp.zeros(len(bound_dofs), dtype=int)
    bound_dofs_offset = bound_dofs_offset.at[index_offset].set(bound_dofs[:-1])
    bound_dofs_array = jnp.array([bound_dofs_offset, bound_dofs]).T[1:]
    B_full = jnp.zeros((bound_dofs_array.shape[0], dofs_size), dtype=int)
    for i in range(bound_dofs_array.shape[0]):
        B_full = B_full.at[i, bound_dofs_array[i][0]].set(1)
        B_full = B_full.at[i, bound_dofs_array[i][1]].set(-1)

    # Cut the constrained dofs
    B = B_full[:, free_dofs]

    V = jnp.zeros((bound_dofs_array.shape[0], 1), dtype=int)

    lagrange_multipliers = jnp.zeros(len(bound_dofs) - 1)

    return B_full, B, V, lagrange_multipliers
