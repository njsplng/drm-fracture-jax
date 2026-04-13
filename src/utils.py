"""Miscellaneous utility functions used in the models."""

import inspect
import logging
import os
import subprocess
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, wraps
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from addict import Dict as AddictDict
from equinox import filter_jit
from jaxtyping import Array
from tabulate import tabulate

from distance_functions import CompositeDistanceFunction

# Global flag for timing
ENABLE_TIMING = False

# Global flag for timing aggregation
AGGREGATE_TIMINGS = False
timing_aggregator = defaultdict(list)


def set_enable_timing(flag: bool) -> None:
    """Set the global flag for enabling timing of functions.

    Parameters
    ----------
    flag : bool
        Whether to enable timing.
    """
    global ENABLE_TIMING
    ENABLE_TIMING = flag


def set_aggregate_timings(flag: bool) -> None:
    """Set the global flag for enabling timing aggregation.

    Parameters
    ----------
    flag : bool
        Whether to enable timing aggregation.
    """
    global AGGREGATE_TIMINGS
    AGGREGATE_TIMINGS = flag


def timed_jit(
    func: Callable[..., object], **jit_kwargs: Dict[str, object]
) -> Callable[..., object]:
    """Apply JAX's jit to a function and measure its execution time.

    Parameters
    ----------
    func : Callable
        The function to jit-compile.
    **jit_kwargs : Dict[str, object]
        Additional keyword arguments passed to jax.jit.

    Returns
    -------
    Callable
        The jit-compiled wrapper function.
    """
    jitted = jax.jit(func, **jit_kwargs)

    def wrapper(*args, **kwargs) -> object:
        if not ENABLE_TIMING:
            return jitted(*args, **kwargs)
        start = time.perf_counter()
        result = jitted(*args, **kwargs)
        # Block until computation is done
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result,
        )
        elapsed = time.perf_counter() - start
        if AGGREGATE_TIMINGS:
            timing_aggregator[func.__name__].append(elapsed)
        else:
            logger = logging.getLogger(__name__)
            logger.timing(f"{func.__name__} took {elapsed:.6f} seconds")
        return result

    return wrapper


def timed_filter_jit(
    func: Callable[..., object], **jit_kwargs: Dict[str, object]
) -> Callable[..., object]:
    """Apply equinox's filter_jit to a function and measure execution time.

    Parameters
    ----------
    func : Callable
        The function to jit-compile.
    **jit_kwargs : Dict[str, object]
        Additional keyword arguments passed to filter_jit.

    Returns
    -------
    Callable
        The jit-compiled wrapper function.
    """
    jitted = filter_jit(func, **jit_kwargs)

    def wrapper(*args, **kwargs) -> object:
        if not ENABLE_TIMING:
            return jitted(*args, **kwargs)
        start = time.perf_counter()
        result = jitted(*args, **kwargs)
        # Block until computation is done
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result,
        )
        elapsed = time.perf_counter() - start
        if AGGREGATE_TIMINGS:
            timing_aggregator[func.__name__].append(elapsed)
        else:
            logger = logging.getLogger(__name__)
            logger.timing(f"{func.__name__} took {elapsed:.6f} seconds")
        return result

    return wrapper


@contextmanager
def _timed_block(name: str) -> object:
    """Context manager to record wall time for a named block.

    Records timing into timing_aggregator. Respects ENABLE_TIMING and
    AGGREGATE_TIMINGS flags. No-op if ENABLE_TIMING is False.

    Parameters
    ----------
    name : str
        Name of the code block to time.
    """
    if not ENABLE_TIMING:
        yield
        return
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if AGGREGATE_TIMINGS:
        timing_aggregator[name].append(elapsed)
    else:
        logger = logging.getLogger(__name__)
        logger.timing(f"{name} took {elapsed:.6f} seconds")


def timed_func(func: Callable[..., object]) -> Callable[..., object]:
    """Decorator for plain functions to record wall time.

    Records wall time into timing_aggregator under the function name.
    Respects ENABLE_TIMING and AGGREGATE_TIMINGS flags.

    Parameters
    ----------
    func : Callable
        The function to time.

    Returns
    -------
    Callable
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> object:
        with _timed_block(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def handle_errors(func: Callable[..., object]) -> Callable[..., object]:
    """Decorator to log exceptions with full traceback and continue execution.

    Parameters
    ----------
    func : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function that catches and logs exceptions.
    """
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def aw(*args, **kwargs) -> object:
            """Async wrapper to handle exceptions."""
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logging.error("Exception in %s: %s", func.__qualname__, e)
                traceback.print_exception(type(e), e, e.__traceback__)
                return None

        return aw

    @wraps(func)
    def w(*args, **kwargs) -> object:
        """Sync wrapper to handle exceptions."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error("Exception in %s: %s", func.__qualname__, e)
            traceback.print_exception(type(e), e, e.__traceback__)
            return None

    return w


def handle_errors_class(cls: object) -> object:
    """Class decorator to wrap all methods starting with '_' using handle_errors.

    Parameters
    ----------
    cls : type
        The class whose methods to wrap.

    Returns
    -------
    type
        The decorated class.
    """
    for name, attr in vars(cls).items():
        if name.startswith("_") and callable(attr):
            setattr(cls, name, handle_errors(attr))
    return cls


# skylos: ignore-start
def get_notebook_sourcecode(
    filename: str = "export.txt",
) -> Optional[str]:
    """Extract the source code of imported functions or classes.

    Parses the notebook history to retrieve source code of imported
    objects and writes them to a file.

    Parameters
    ----------
    filename : str, optional
        Output filename. Default is "export.txt".

    Returns
    -------
    None
        This function prints a message but returns None.
    """
    if not filename.endswith(".txt"):
        filename += ".txt"

    # Dynamically get the name of the current function.
    current_function_name = inspect.currentframe().f_code.co_name

    # Get the current IPython shell so we can access its user namespace.
    ip = get_ipython()

    # Run the %history magic command programmatically.
    ip.run_line_magic("history", "-t -f history.txt")

    # Read and then remove the temporary history file.
    with open("history.txt", "r") as f:
        readlines = f.readlines()
    os.remove("history.txt")

    # Parse the history lines to extract imported object names.
    imported_lines = []
    for line in readlines:
        if line.startswith("from") and "import" in line:
            # Skip any line that might reference the current function.
            if current_function_name in line:
                continue
            # Split on "import" and then split the imported names by comma.
            imported_names = line.split("import", 1)[1].strip().split(",")
            # Strip extra whitespace from each imported name.
            imported_lines.extend([name.strip() for name in imported_names])

    # Now, look up each object in the notebook's user namespace.
    function_definitions = []
    user_ns = ip.user_ns  # The notebook's global namespace
    for obj_name in imported_lines:
        if obj_name in user_ns:
            try:
                source = inspect.getsource(user_ns[obj_name])
            except Exception as e:
                source = f"Error retrieving source for {obj_name}: {e}"
        else:
            source = f"Object {obj_name} not found in the notebook's namespace."
        function_definitions.append(source + "\n\n")

    # Write both the history and the collected source codes to the output file.
    with open(filename, "w") as f:
        f.writelines(readlines)
        f.write("\n\n")
        f.writelines(function_definitions)

    print(f"Source code for {len(imported_lines)} objects written to {filename}.")


# skylos: ignore-end


def deep_update(orig: Dict[str, object], updates: Dict[str, object]) -> None:
    """Recursively update a dictionary with values from another dictionary.

    Merges nested dictionaries in-place.

    Parameters
    ----------
    orig : Dict[str, object]
        The original dictionary to update.
    updates : Dict[str, object]
        The dictionary with updates to apply.

    Returns
    -------
    None
        Modifies `orig` in-place.
    """
    for key, val in updates.items():
        if key in orig and isinstance(orig[key], dict) and isinstance(val, dict):
            # Both are dicts → recurse
            deep_update(orig[key], val)
        else:
            # Otherwise replace
            orig[key] = val


@partial(timed_jit, static_argnames=("double_off_diagonal"))
def matrix_to_voigt_single(matrix: Array, double_off_diagonal: bool = False) -> Array:
    """Convert a 3x3 matrix to Voigt notation.

    Parameters
    ----------
    matrix : Array
        The 3x3 matrix to convert.
    double_off_diagonal : bool, optional
        Whether to double the off-diagonal components. Default is False.

    Returns
    -------
    Array
        Voigt notation array with 6 components.
    """
    matrix_voigt = jnp.zeros((6,))
    matrix_voigt = matrix_voigt.at[0].set(matrix[0, 0])
    matrix_voigt = matrix_voigt.at[1].set(matrix[1, 1])
    matrix_voigt = matrix_voigt.at[2].set(matrix[2, 2])
    matrix_voigt = matrix_voigt.at[3].set(matrix[2, 1])
    matrix_voigt = matrix_voigt.at[4].set(matrix[2, 0])
    matrix_voigt = matrix_voigt.at[5].set(matrix[1, 0])
    if double_off_diagonal:
        matrix_voigt = matrix_voigt.at[3:].set(matrix_voigt[3:] * 2)
    return matrix_voigt


@partial(timed_jit, static_argnames=("double_off_diagonal"))
def matrix_to_voigt(matrix: Array, double_off_diagonal: bool = False) -> Array:
    """Vectorized conversion from matrix notation to Voigt notation.

    Parameters
    ----------
    matrix : Array
        The matrix or batch of matrices to convert.
    double_off_diagonal : bool, optional
        Whether to double the off-diagonal components. Default is False.

    Returns
    -------
    Array
        Voigt notation array(s) with 6 components.
    """
    return jax.vmap(
        jax.vmap(matrix_to_voigt_single, in_axes=(0, None)), in_axes=(0, None)
    )(matrix, double_off_diagonal)


@partial(timed_jit, static_argnames=("conns_size"))
def elemental_to_global(
    elemental_values: Array,
    connectivities: Array,
    conns_size: int,
) -> Array:
    """Convert elemental values to global values.

    Parameters
    ----------
    elemental_values : Array
        Values at the element level.
    connectivities : Array
        Element connectivity array mapping elements to global DOFs.
    conns_size : int
        Total number of global degrees of freedom.

    Returns
    -------
    Array
        Global values assembled from elemental contributions.
    """
    # Determine extra field dimensions (it will be an empty tuple if scalar)
    field_shape = elemental_values.shape[2:]
    # Global output shape is [conns_size, *field]
    out_shape = (conns_size,) + field_shape

    # Create a global accumulator initialized to zero
    qoi = jnp.zeros(out_shape)

    # Flatten the element/node dimensions while preserving extra field dimensions
    elemental_values_flat = elemental_values.reshape(-1, *field_shape)  # [E*N, *field]
    conns_flat = connectivities.reshape(-1)  # [E*N]

    # Scatter-add: For each flattened index in dofs_flat, add the corresponding value
    qoi = qoi.at[conns_flat].add(elemental_values_flat)
    return qoi


@partial(timed_jit, static_argnames=("conns_size"))
def pointwise_to_nodes(
    ip_values: Array,
    connectivities: Array,
    conns_size: int,
    extrapolation_matrices: Array,
) -> Array:
    """Extrapolate values from Gauss points to mesh nodes.

    Parameters
    ----------
    ip_values : Array
        Values at integration points.
    connectivities : Array
        Element connectivity array.
    conns_size : int
        Total number of global degrees of freedom.
    extrapolation_matrices : Array
        Matrices for extrapolating from integration points to nodes.

    Returns
    -------
    Array
        Values extrapolated to mesh nodes.
    """
    elemental_values = jnp.einsum(
        "eng, eg... -> en...", extrapolation_matrices, ip_values
    )
    qoi = elemental_to_global(elemental_values, connectivities, conns_size)
    weights = jnp.zeros(conns_size)
    weights = weights.at[connectivities.flatten()].add(
        jnp.ones_like(connectivities.flatten())
    )
    weights_reshaped = weights.reshape(weights.shape[0], *([1] * (qoi.ndim - 1)))
    qoi = qoi / weights_reshaped
    return qoi


def check_input_placeholders(obj: object) -> bool:
    """Recursively check for the presence of "placeholder" in strings.

    Parameters
    ----------
    obj : object
        Object to check. Can be a string, dict, or list.

    Returns
    -------
    bool
        True if no placeholders are found, False otherwise.
    """
    # Base case for strings
    if isinstance(obj, str):
        return "placeholder" not in obj.lower()

    # Recursive case for dictionaries
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not check_input_placeholders(value):
                return False
        return True

    # Recursive case for lists
    if isinstance(obj, list):
        for item in obj:
            if not check_input_placeholders(item):
                return False
        return True

    # For all other types (int, float, bool, None, etc.), no placeholder possible
    return True


def split_displacement(displacement: Array) -> Array:
    """Split a 2D displacement vector into x, y, and z components.

    Uses alternating-dof convention where even indices are x and odd
    indices are y.

    Parameters
    ----------
    displacement : Array
        1D displacement array in alternating-dof convention.

    Returns
    -------
    Array
        Stacked array with x, y, z components.
    """
    return jnp.stack(
        (
            displacement[::2],
            displacement[1::2],
            jnp.zeros_like(displacement[::2]),
        ),
        axis=1,
    )


def postprocess_qoi_dict(
    qoi_dict: Dict[str, object],
    connectivities: Array,
    conns_size: int,
    extrapolation_matrices: Array,
    keys_to_split: List[str] = ["displacement"],
    sanitise_keys: bool = False,
) -> Dict[str, object]:
    """Postprocess the QoI dict by extrapolating to nodes from integration points.

    Parameters
    ----------
    qoi_dict : Dict[str, object]
        Dictionary of quantities of interest.
    connectivities : Array
        Element connectivity array.
    conns_size : int
        Total number of global degrees of freedom.
    extrapolation_matrices : Array
        Matrices for extrapolating from integration points to nodes.
    keys_to_split : List[str], optional
        Keys whose values should be split into components. Default is
        ["displacement"].
    sanitise_keys : bool, optional
        Whether to remove empty lists and None values. Default is False.

    Returns
    -------
    Dict[str, object]
        Postprocessed QoI dictionary.
    """

    def split_dict_qois_2d(
        qoi_dict: Dict[str, object], keys_to_split: List[str]
    ) -> Dict[str, object]:
        """Split provided keys into x and y components."""
        for key in keys_to_split:
            new_entries = []
            if key not in qoi_dict:
                continue
            for entry in qoi_dict[key]:
                new_entries.append(split_displacement(entry))
            qoi_dict[key] = new_entries

        return qoi_dict

    # Clone the input dict to avoid mutating it
    qoi_clone = qoi_dict.copy()

    # If keys are to be sanitised, remove empty lists and elements with None values
    if sanitise_keys:
        keys_to_remove = []
        # Iterate over the keys and values in the QoI dict
        for key in list(qoi_clone.keys()):
            # Clear any empty lists
            if isinstance(qoi_clone[key], list) and len(qoi_clone[key]) == 0:
                keys_to_remove.append(key)
            # Remove any lists containing None values (something went wrong)
            if any(value is None for value in qoi_clone[key]):
                keys_to_remove.append(key)
                logging.warning(
                    f"Removing key '{key}' from QoI dict due to None values in its list."
                )
        # Remove the identified keys from the QoI dict
        for key in keys_to_remove:
            qoi_clone.pop(key)

    # Postprocess the keys
    for key in list(qoi_clone.keys()):
        if key.startswith("ip"):
            new_key = key + "_nodes"
            new_entries = []
            # Extrapolate each entry from integration points to nodes
            for entry in qoi_clone[key]:
                new_entries.append(
                    pointwise_to_nodes(
                        ip_values=entry,
                        connectivities=connectivities,
                        conns_size=conns_size,
                        extrapolation_matrices=extrapolation_matrices,
                    )
                )
            qoi_clone[new_key] = new_entries

    # Split specified keys into x and y components
    qoi_clone = split_dict_qois_2d(qoi_dict=qoi_clone, keys_to_split=keys_to_split)

    return qoi_clone


def prune_qoi_dict(
    qoi_dict: Dict[str, object], keys_to_keep: List[str]
) -> Dict[str, object]:
    """Prune a dictionary to only keep specified keys.

    Parameters
    ----------
    qoi_dict : Dict[str, object]
        The dictionary to prune.
    keys_to_keep : List[str]
        List of keys to retain.

    Returns
    -------
    Dict[str, object]
        Dictionary containing only the specified keys.
    """
    return {key: qoi_dict[key] for key in keys_to_keep if key in qoi_dict}


def rescale_qoi_dict(
    qoi_dict: Dict[str, object],
    displacement_scaling: float,
    energy_scaling: float,
    force_scaling: float,
) -> None:
    """Rescale the quantities of interest in the dictionary.

    Parameters
    ----------
    qoi_dict : Dict[str, object]
        Dictionary of quantities of interest to rescale.
    displacement_scaling : float
        Scaling factor for displacement-related quantities.
    energy_scaling : float
        Scaling factor for energy-related quantities.
    force_scaling : float
        Scaling factor for force and stress-related quantities.

    Returns
    -------
    None
        Modifies qoi_dict in-place.
    """
    inv_energy = np.float64(1.0 / energy_scaling)
    inv_force = np.float64(1.0 / force_scaling)
    inv_displacement = np.float64(1.0 / displacement_scaling)

    for key, entries in qoi_dict.items():
        if "energy" in key:
            scale = inv_energy
        elif "stress" in key or "force" in key or "F_" in key:
            scale = inv_force
        elif "strain" in key or "displacement" in key:
            scale = inv_displacement
        else:
            continue  # No scaling for this key

        for i, arr in enumerate(entries):
            if isinstance(arr, np.ndarray) and arr.flags.writeable:
                np.multiply(arr, scale, out=arr)
            else:
                # Read-only or JAX array — allocate and write back
                entries[i] = arr * scale


def build_quad4_connectivity(info: Dict[str, object]) -> Array:
    """Build linear quad connectivity for a regular grid of control points.

    Returns an (n_elems, 4) array of connectivity indices ordered
    row-major in Paraview style.

    Parameters
    ----------
    info : Dict[str, object]
        Dictionary containing "control_points" key with the 2D grid.

    Returns
    -------
    Array
        Connectivity array of shape (n_elems, 4).
    """
    ctrlpts2d = info["control_points"]
    n_ctrl_v = len(ctrlpts2d)
    n_ctrl_u = len(ctrlpts2d[0])
    conn = []
    # Loop over each cell in the uv-grid
    for j in range(n_ctrl_v - 1):
        for i in range(n_ctrl_u - 1):
            n0 = j * n_ctrl_u + i
            n1 = n0 + 1
            n3 = (j + 1) * n_ctrl_u + i
            n2 = n3 + 1
            # [bottom-left, bottom-right, top-right, top-left]
            conn.append([n0, n1, n2, n3])
    return jnp.array(conn, dtype=int)


def output_timing_table(timing_aggregator: Dict[str, List[float]]) -> None:
    """Output a table of function timings.

    Parameters
    ----------
    timing_aggregator : Dict[str, List[float]]
        Dictionary mapping function names to lists of timing values.

    Returns
    -------
    None
        Logs a formatted table of timing information.
    """
    logging.info("Aggregating and printing function timings...")
    table_data = []
    for func_name, times in timing_aggregator.items():
        total_time = jnp.sum(jnp.array(times))
        median_time = jnp.median(jnp.array(times))
        table_data.append([func_name, total_time, median_time, len(times)])

    headers = ["Function name", "Total time", "Median time", "Number of calls"]
    logging.info(
        "Aggregated function timings: \n"
        + tabulate(table_data, headers=headers, floatfmt=".3e"),
    )


def raise_osx_notification(title: str) -> None:
    """Display a macOS notification indicating completion of a run.

    Parameters
    ----------
    title : str
        The title to display in the notification.

    Returns
    -------
    None
        Logs an error if notification fails.
    """
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "Complete" with title "{title}"',
            ]
        )
    # If the notification fails, log the error (likely file not found for osascript)
    except Exception as e:
        logging.error(f"Error displaying notification: {e}")


def set_up_rotation_array(
    nodal_coordinates: Array,
    connectivities: Array,
    slices: int,
    rotation_angle: float,
    direction: str = "x",
) -> Array:
    """Set up a rotation array for a sliced domain.

    Parameters
    ----------
    nodal_coordinates : Array
        Nodal coordinates of the mesh.
    connectivities : Array
        Element connectivity array.
    slices : int
        Number of slices in the domain.
    rotation_angle : float
        Rotation angle in degrees.
    direction : str, optional
        Direction for slicing ("x" or "y"). Default is "x".

    Returns
    -------
    Array
        Rotation array with alternating angles per slice.
    """
    assert direction in ["x", "y"], "Direction must be either 'x' or 'y'."
    dir_ix = 0 if direction == "x" else 1

    # Compute the centroids of each element
    centroids = jnp.mean(nodal_coordinates[connectivities], axis=1)

    # Build slice boundaries
    domain_min, domain_max = (
        nodal_coordinates[:, dir_ix].min(),
        nodal_coordinates[:, dir_ix].max(),
    )
    edges = jnp.linspace(domain_min, domain_max, slices + 1)

    # Map each centroid into a slice
    slice_ids = jnp.clip(
        jnp.searchsorted(edges, centroids[:, dir_ix], side="right") - 1, 0, slices - 1
    )

    # Convert theta to radians
    theta = jnp.radians(rotation_angle)
    # Create the rotation array
    return jnp.where((slice_ids % 2) == 0, theta, -theta)


def set_up_window_generic(
    nodal_coordinates: Array,
    window_value_threshold: float,
    function_dict: Dict[str, object],
    effective_directions: Union[str, List[str]],
) -> Array:
    """Return the indices of nodes on the fixed window boundary.

    Parameters
    ----------
    nodal_coordinates : Array
        Nodal coordinates of the mesh.
    window_value_threshold : float
        Threshold value for the window.
    function_dict : Dict[str, object]
        Dictionary describing the distance function.
    effective_directions : Union[str, List[str]]
        Directions to constrain ("x", "y", or "xy").

    Returns
    -------
    Array
        Array of constrained degree of freedom indices.
    """
    assert effective_directions in [
        "x",
        "y",
        "xy",
    ] or isinstance(
        effective_directions, list
    ), "Invalid constrain directions, accepted values are 'x', 'y' or 'both"

    aggregate = False
    if isinstance(effective_directions, list):
        aggregate = True

    # Form the distance function
    distance_fns = CompositeDistanceFunction(function_dict)

    # Evaluate the distance function
    distance_fns_points = distance_fns(nodal_coordinates, aggregate=aggregate)

    # If the distance function was not a list, convert it to a list
    if not aggregate:
        distance_fns_points = [distance_fns_points]
        effective_directions = [effective_directions]

    captured_nodes = []
    for i, distance_fn_points in enumerate(distance_fns_points):
        # Filter values below threshold
        distance_fn_points = distance_fn_points.at[
            jnp.where(distance_fn_points < window_value_threshold)
        ].set(0)

        # Extract the constrained nodes
        targeted_nodes = jnp.where(distance_fn_points > 0)[0].flatten()

        match effective_directions[i]:
            case "x":
                targeted_dofs = targeted_nodes * 2
            case "y":
                targeted_dofs = targeted_nodes * 2 + 1
            case "xy":
                targeted_dofs = jnp.stack(
                    (targeted_nodes * 2, targeted_nodes * 2 + 1), axis=1
                ).reshape(-1)

        # Append the captured nodes
        captured_nodes.extend(targeted_dofs)

    # Return the formed dofs list
    return jnp.array(captured_nodes)


def set_up_target_dof(mesh_data: object, target_dof_config: Dict[str, object]) -> int:
    """Return the index of the target degree of freedom.

    Parameters
    ----------
    mesh_data : object
        Mesh data object containing nodal coordinates.
    target_dof_config : Dict[str, object]
        Configuration dictionary with "coordinates" and "direction".

    Returns
    -------
    int
        Index of the target degree of freedom.
    """
    # Extract the nodal coordinates and target coordinates
    coordinates = mesh_data.nodal_coordinates
    target_coordinates = jnp.array(target_dof_config["coordinates"])

    # Find the node closest to the target coordinates
    distances = jnp.linalg.norm(coordinates - target_coordinates, axis=1)
    target_element = jnp.argmin(distances)

    # Return the target dof index
    return int(target_element * 2 + (0 if target_dof_config["direction"] == "x" else 1))


class StrictDict(AddictDict):
    """Wrapper around addict's Dict that disables auto-vivification.

    Raises KeyError or AttributeError on missing keys instead of
    creating new nested dictionaries.
    """

    def __missing__(self, key: str) -> object:
        """Raise error on missing key."""
        raise KeyError(key)

    # Enforce KeyError on item access for missing keys
    def __getitem__(self, key: str) -> object:
        """Get item by key, raising KeyError if missing."""
        if key not in self:
            raise KeyError(key)
        return super().__getitem__(key)

    # Enforce AttributeError on missing attribute access
    def __getattr__(self, name: str) -> object:
        """Get attribute, raising AttributeError if missing."""
        # Avoid Addict's default behavior that creates nodes
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Missing key: {name}") from None
