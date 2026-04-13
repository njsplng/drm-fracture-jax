"""Mesh loading and processing module.

Provides functions for loading, parsing, and rescaling meshes from
various sources including GMSH and NURBS formats.
"""

import json
import logging
import pathlib
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import gmsh
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from iga import generate_knot_vector, greville_abscissae

QUAD_TYPES = ["quad4", "quad8", "quad9"]
TRI_TYPES = ["tri3", "tri6", "tri7"]


@contextmanager
def gmsh_session() -> None:
    """Context manager for initializing and finalizing GMSH.

    Ensures proper cleanup of GMSH resources after mesh operations
    are complete.

    Yields
    ------
    None
        Yielded value is None; context manager provides GMSH session.

    Examples
    --------
    >>> with gmsh_session():
    ...     gmsh.open("mesh.msh")
    """
    gmsh.initialize()
    try:
        yield None
    finally:
        gmsh.finalize()


def load_and_rescale_mesh(
    mesh_dict: Dict[str, Any], problem_domain_dict: Dict[str, Tuple[float, float]]
) -> Tuple[Array, Array, Array, Dict[str, Any]]:
    """Load mesh from dictionary and rescale to match problem domain.

    Parses the mesh file specified in the dictionary and rescales
    nodal coordinates to fit the specified problem domain.

    Parameters
    ----------
    mesh_dict : Dict[str, Any]
        Dictionary containing mesh metadata with 'filename' and 'type' keys.
    problem_domain_dict : Dict[str, Tuple[float, float]]
        Dictionary with 'x' and 'y' keys mapping to domain bounds.

    Returns
    -------
    Tuple[Array, Array, Array, Dict[str, Any]]
        Nodal coordinates, connectivities, degrees of freedom, and
        auxiliary NURBS data.
    """
    logging.info(
        f"Loading mesh from {mesh_dict['filename']} of type {mesh_dict['type']}..."
    )
    with gmsh_session():
        nodal_coordinates, connectivities, dofs, aux_nurbs = parse_mesh(
            mesh_dict["filename"], mesh_dict["type"]
        )
    logging.info("Mesh parsed successfully.")
    nodal_coordinates, rescale_executed = rescale_mesh(
        nodal_coordinates=nodal_coordinates,
        x_domain=problem_domain_dict["x"],
        y_domain=problem_domain_dict["y"],
    )

    # In case of rescaling, inform the user
    if rescale_executed:
        logging.info(
            f"Mesh rescaled to match the problem domain: {problem_domain_dict}"
        )

    return nodal_coordinates, connectivities, dofs, aux_nurbs


def parse_mesh(
    mesh_filename: str, mesh_type: str = "quad"
) -> Tuple[Array, Array, Array, Any]:
    """Extract mesh data from file based on mesh type.

    Routes to the appropriate parser based on mesh_type: NURBS,
    T-spline, or GMSH format.

    Parameters
    ----------
    mesh_filename : str
        Name of the mesh file to parse.
    mesh_type : str, optional
        Type of mesh: 'nurbs', 't-spline', or element type (default 'quad').

    Returns
    -------
    Tuple[Array, Array, Array, Any]
        Nodal coordinates, connectivities, degrees of freedom, and
        auxiliary data.
    """
    # Get the current path and the mesh file path
    current_path = pathlib.Path(__file__).parent.resolve()
    project_root = current_path.parent
    mesh_file_path = project_root / "mesh" / mesh_type / mesh_filename

    if mesh_type == "nurbs":
        return parse_nurbs_mesh(mesh_file_path=mesh_file_path)

    if mesh_type == "t-spline":
        return parse_t_spline_mesh(mesh_file_path=mesh_file_path)

    return parse_gmsh_mesh(
        mesh_file_path=mesh_file_path,
    )


def build_structured_adjacency(
    nx: int,
    ny: int,
) -> Array:
    """Build element-to-element adjacency matrix for structured grid.

    Creates an adjacency matrix where entry (i, j) is True if elements
    i and j share an edge in a structured grid layout.

    Parameters
    ----------
    nx : int
        Number of elements in x direction.
    ny : int
        Number of elements in y direction.

    Returns
    -------
    adj : Array
        Adjacency matrix of shape (n_elements, n_elements) where
        n_elements = nx * ny.
    """
    num_elements = nx * ny
    adj = jnp.zeros((num_elements, num_elements), dtype=bool)

    # Reshape to 2D grid indices
    indices = jnp.arange(num_elements).reshape((ny, nx))

    # Horizontal neighbors (left-right)
    left = indices[:, :-1].flatten()
    right = indices[:, 1:].flatten()
    adj = adj.at[left, right].set(True)
    adj = adj.at[right, left].set(True)

    # Vertical neighbors (up-down)
    bottom = indices[:-1, :].flatten()
    top = indices[1:, :].flatten()
    adj = adj.at[bottom, top].set(True)
    adj = adj.at[top, bottom].set(True)

    return adj.astype(jnp.float32)


def parse_nurbs_mesh(
    mesh_file_path: pathlib.Path,
) -> Tuple[Array, Array, Array, Dict[str, Any]]:
    """Construct and parse a NURBS mesh from JSON file.

    Reads mesh parameters from JSON file and generates control points,
    knot vectors, and element connectivity for a NURBS basis.

    Parameters
    ----------
    mesh_file_path : pathlib.Path
        Path to the JSON file containing mesh parameters.

    Returns
    -------
    Tuple[Array, Array, Array, Dict[str, Any]]
        Nodal coordinates, connectivities, degrees of freedom, and
        auxiliary NURBS data including knot vectors and control points.
    """
    with open(str(mesh_file_path) + ".json", "r") as file:
        params = json.load(file)

    # Extract the problem parameters
    xmin, xmax = params["problem_domain"]["x"]
    ymin, ymax = params["problem_domain"]["y"]
    nx = params["elements_x"]
    ny = params["elements_y"]
    p = params["order_x"]
    q = params["order_y"]

    # Number of control points in each direction
    n_ctrl_u = nx + p
    n_ctrl_v = ny + q

    # Build the knot vectors up‐front
    U = generate_knot_vector(p, n_ctrl_u)
    V = generate_knot_vector(q, n_ctrl_v)

    # Generate the Greville abscissae
    grev_u = greville_abscissae(U, p)
    grev_v = greville_abscissae(V, q)

    # Map from parametric [0,1] to physical [xmin,xmax], [ymin,ymax]
    u_coords = xmin + grev_u * (xmax - xmin)
    v_coords = ymin + grev_v * (ymax - ymin)

    # Generate the control points in 3D
    ctrlpts2d = [[(u, v, 0.0) for u in u_coords] for v in v_coords]
    flat_ctrlpts = [pt for row in ctrlpts2d for pt in row]

    # Determine unique parametric spans (elements)
    uu = np.unique(U)
    spans_u = np.stack([uu[:-1], uu[1:]], axis=1).tolist()
    uv = np.unique(V)
    spans_v = np.stack([uv[:-1], uv[1:]], axis=1).tolist()

    # Build element connectivity: for each span pair, find control point indices
    elem_conn = []
    for j in range(len(spans_v)):
        for i in range(len(spans_u)):
            # Local u-indices: from i to i+p
            u_idx = list(range(i, i + p + 1))
            # Local v-indices: from j to j+q
            v_idx = list(range(j, j + q + 1))
            # Flatten 2D indices
            conn = []
            for vv in v_idx:
                for uu in u_idx:
                    conn.append(vv * n_ctrl_u + uu)
            elem_conn.append(conn)

    # Prepare output
    info = {
        "degrees": {"u": p, "v": q},
        "nx": nx,
        "ny": ny,
        "knotvector_u": U,
        "knotvector_v": V,
        "control_points": ctrlpts2d,
        "weights": [[1.0] * n_ctrl_u for _ in range(n_ctrl_v)],
        "parametric_spans_u": spans_u,
        "parametric_spans_v": spans_v,
        "element_connectivity": elem_conn,
        "control_points_flat": flat_ctrlpts,
        "gauss_integration_order": params.get("gauss_integration_order"),
    }

    # Return list
    nodal_coordinates = np.array(flat_ctrlpts)[:, :2]
    connectivities = np.array(elem_conn)
    dofs = np.array([convert_ids_to_dofs(entry) for entry in connectivities])

    return nodal_coordinates, connectivities, dofs, info


def convert_ids_to_dofs(ids: Array) -> Array:
    """Convert element node IDs to degrees of freedom in 2D.

    Each node has two degrees of freedom (x and y displacements).

    Parameters
    ----------
    ids : Array
        Array of node IDs to convert.

    Returns
    -------
    dofs : Array
        Flattened array of degree of freedom indices, where each
        node ID produces two DOF indices.
    """
    dofs = np.array(ids) * 2
    dofs = np.dstack((dofs, dofs + 1))
    dofs = dofs.reshape(-1)
    return dofs


def legendre_gauss_quadrature_2d(order: int) -> Tuple[Array, Array]:
    """Generate Legendre-Gauss quadrature points and weights for 2D domain.

    Creates tensor product quadrature rule from 1D Legendre-Gauss
    points and weights.

    Parameters
    ----------
    order : int
        Number of quadrature points in each direction.

    Returns
    -------
    xi_2d : Array
        Quadrature points of shape (order**2, 2).
    w_2d : Array
        Quadrature weights of shape (order**2,).
    """
    xi_1d, w_1d = np.polynomial.legendre.leggauss(order)
    xi_2d = np.array(np.meshgrid(xi_1d, xi_1d)).T.reshape(-1, 2)
    w_2d = np.outer(w_1d, w_1d).flatten()
    return jnp.array(xi_2d), jnp.array(w_2d)


def parse_gmsh_mesh(
    mesh_file_path: pathlib.Path,
) -> Tuple[Array, Array, Array, None]:
    """Parse GMSH mesh file and extract mesh data.

    Reads a GMSH .msh file and extracts node coordinates, element
    connectivity, and degrees of freedom. Supports quad4, quad8,
    quad9, tri3, and tri6 element types.

    Parameters
    ----------
    mesh_file_path : pathlib.Path
        Path to the GMSH mesh file (.msh).

    Returns
    -------
    Tuple[Array, Array, Array, None]
        Nodal coordinates, connectivities, degrees of freedom, and
        None for auxiliary data.

    Raises
    ------
    ValueError
        If mesh type is not one of the supported element types.
    """
    gmsh.option.setNumber("General.Verbosity", 0)  # Suppress GMSH output

    # Open the mesh file
    gmsh.open(str(mesh_file_path) + ".msh")

    # Extract node data: get the original node tags and the coordinates
    all_node_tags, coords, _ = gmsh.model.mesh.getNodes(dim=-1, tag=-1)
    all_node_tags = np.array(all_node_tags)
    node_coordinates = coords.reshape(-1, 3)[:, 0:2]

    # Extract the mesh type from the file path
    mesh_type = mesh_file_path.parent.stem

    # Special case for automated tests
    if mesh_type == "tests":
        mesh_type = mesh_file_path.parent.parent.stem

    # Set the element type based on the mesh type
    match mesh_type:
        case "quad4":
            element_type = 3
        case "quad8":
            element_type = 16
        case "quad9":
            element_type = 10
        case "tri3":
            element_type = 2
        case "tri6":
            element_type = 9
        case _:
            raise ValueError(
                f"Mesh type {mesh_type} not supported. Supported types are: {QUAD_TYPES + TRI_TYPES}"
            )

    # Extract element data based on the element type set
    _, element_nodeTags = gmsh.model.mesh.getElementsByType(
        elementType=element_type, tag=-1
    )
    element_nodeTags = element_nodeTags.reshape(-1, int(mesh_type[-1]))

    # Determine which node tags are used in the connectivity
    used_tags = np.unique(element_nodeTags)
    sorted_used_tags = np.sort(used_tags)
    # Build a mapping from the original tag to a new, contiguous index
    tag_to_index = {tag: i for i, tag in enumerate(sorted_used_tags)}

    # Update the connectivity: remap each node tag in element_nodeTags to its new index
    connectivities = np.vectorize(lambda tag: tag_to_index[tag])(element_nodeTags)

    # Filter and re-order the node coordinates
    # Keep only nodes that are used, and sort them according to the sorted used tags
    mask = np.isin(all_node_tags, sorted_used_tags)
    node_tags_used = all_node_tags[mask]
    node_coordinates_used = node_coordinates[mask]
    order = np.argsort(node_tags_used)  # order so that node_tags_used becomes sorted
    node_tags_used = node_tags_used[order]
    node_coordinates_used = node_coordinates_used[order]

    # Generate the degrees of freedom for the elements using the updated connectivity
    dofs = np.array([convert_ids_to_dofs(entry) for entry in connectivities])

    return [node_coordinates_used, connectivities, dofs, None]


def rescale_mesh(
    nodal_coordinates: Array,
    x_domain: Tuple[float, float],
    y_domain: Tuple[float, float],
) -> Tuple[Array, bool]:
    """Rescale mesh nodal coordinates to match specified domain.

    Applies affine transformation to map mesh extents to the target
    x and y domain bounds.

    Parameters
    ----------
    nodal_coordinates : Array
        Array of nodal coordinates of shape (n_nodes, 2).
    x_domain : Tuple[float, float]
        Target x domain bounds as (min, max).
    y_domain : Tuple[float, float]
        Target y domain bounds as (min, max).

    Returns
    -------
    Tuple[Array, bool]
        Rescaled nodal coordinates and boolean indicating whether
        rescaling was actually applied.
    """
    nodal_coordinates = np.array(nodal_coordinates)

    # Find the extremities of the mesh
    mesh_x_lower = nodal_coordinates[:, 0].min()
    mesh_x_upper = nodal_coordinates[:, 0].max()
    mesh_y_lower = nodal_coordinates[:, 1].min()
    mesh_y_upper = nodal_coordinates[:, 1].max()

    # Extract the domain data
    x_lower, x_upper = x_domain
    y_lower, y_upper = y_domain

    # Find if it is necessary to rescale the mesh first
    if (
        mesh_x_lower == x_lower
        and mesh_x_upper == x_upper
        and mesh_y_lower == y_lower
        and mesh_y_upper == y_upper
    ):
        return jnp.array(nodal_coordinates), False

    # Find the scale factor for the nodal coordinates
    x_scale = (x_upper - x_lower) / (mesh_x_upper - mesh_x_lower)
    y_scale = (y_upper - y_lower) / (mesh_y_upper - mesh_y_lower)

    # Rescale the nodal coordinates
    nodal_coordinates[:, 0] = x_lower + x_scale * (
        nodal_coordinates[:, 0] - mesh_x_lower
    )
    nodal_coordinates[:, 1] = y_lower + y_scale * (
        nodal_coordinates[:, 1] - mesh_y_lower
    )
    return jnp.array(nodal_coordinates), True
