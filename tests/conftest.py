"""Testing configuration for unit tests.

Sets up sys.path for importing from src/, fem/src/, and nn/src/,
enables JAX float64 mode, and provides shared fixtures for mesh
and network parameter generation.
"""

import pathlib
import sys
from typing import Any, Dict

import gmsh
import jax
import pytest
from jaxtyping import Array

# Enable JAX float64 mode
jax.config.update("jax_enable_x64", True)

# Set up the necessary paths
current_path = pathlib.Path(__file__).parent.resolve()
project_root = current_path.parent

# Link the generic source
generic_source_path = current_path.parent / "src"
if str(generic_source_path) not in sys.path:
    sys.path.insert(0, str(generic_source_path))

# Link the fem source
fem_source_path = current_path.parent / "fem" / "src"
if fem_source_path.exists() and str(fem_source_path) not in sys.path:
    sys.path.insert(0, str(fem_source_path))

# Link the nn source
nn_source_path = current_path.parent / "nn" / "src"
if nn_source_path.exists() and str(nn_source_path) not in sys.path:
    sys.path.insert(0, str(nn_source_path))

# Import after path setup
from mesh import parse_gmsh_mesh, parse_nurbs_mesh


@pytest.fixture(scope="session")
def project_root_path() -> pathlib.Path:
    """Return the project root path."""
    return current_path.parent


def _generate_gmsh_mesh(
    tmp_path: pathlib.Path,
    mesh_type_dir: str,
    nx: int,
    ny: int,
    element_order: int = 1,
    use_triangles: bool = False,
) -> tuple[Array, Array, Array]:
    """Generate a gmsh mesh programmatically.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for mesh file.
    mesh_type_dir : str
        Directory name for mesh type (e.g., "quad4", "tri3").
    nx : int
        Number of elements in x direction.
    ny : int
        Number of elements in y direction.
    element_order : int, optional
        Element order (1 for linear, 2 for quadratic). Default is 1.
    use_triangles : bool, optional
        Whether to use triangles instead of quads. Default is False.

    Returns
    -------
    tuple[Array, Array, Array]
        (coords, connectivity, dofs) from parse_gmsh_mesh.
    """
    # Initialize gmsh if not already initialized
    if not gmsh.is_initialized():
        gmsh.initialize()

    # Suppress gmsh output
    gmsh.option.setNumber("General.Verbosity", 0)

    # Clear any existing model by removing the current one
    try:
        gmsh.model.remove()
    except RuntimeError:
        pass

    # Create a new model
    gmsh.model.add("test_mesh")
    gmsh.model.setCurrent("test_mesh")

    # Create a rectangle
    p1 = gmsh.model.geo.add_point(0, 0, 0, 1.0)
    p2 = gmsh.model.geo.add_point(1, 0, 0, 1.0)
    p3 = gmsh.model.geo.add_point(1, 1, 0, 1.0)
    p4 = gmsh.model.geo.add_point(0, 1, 0, 1.0)

    l1 = gmsh.model.geo.add_line(p1, p2)
    l2 = gmsh.model.geo.add_line(p2, p3)
    l3 = gmsh.model.geo.add_line(p3, p4)
    l4 = gmsh.model.geo.add_line(p4, p1)

    cl = gmsh.model.geo.add_curve_loop([l1, l2, l3, l4])
    ps = gmsh.model.geo.add_plane_surface([cl])

    # Transfinite meshing for structured mesh
    gmsh.model.geo.synchronize()

    if use_triangles:
        # Use Delaunay triangulation with MathEval field for mesh size control
        field_tag = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.set_string(field_tag, "F", str(max(1.0 / nx, 1.0 / ny)))
        gmsh.model.mesh.field.set_as_background_mesh(field_tag)
    else:
        # Transfinite lines and surface for quad mesh
        gmsh.model.mesh.set_transfinite_curve(l1, nx + 1, "Progression", 1.0)
        gmsh.model.mesh.set_transfinite_curve(l3, nx + 1, "Progression", 1.0)
        gmsh.model.mesh.set_transfinite_curve(l2, ny + 1, "Progression", 1.0)
        gmsh.model.mesh.set_transfinite_curve(l4, ny + 1, "Progression", 1.0)
        gmsh.model.mesh.set_transfinite_surface(ps, "Left")
        gmsh.model.mesh.set_recombine(2, ps)

    # Set element order
    gmsh.model.mesh.set_order(element_order)

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Create output directory and save mesh
    mesh_dir = tmp_path / mesh_type_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_file = mesh_dir / "test_mesh"  # parse_gmsh_mesh adds .msh
    gmsh.write(str(mesh_file) + ".msh")

    # Parse the mesh
    coords, connectivity, dofs, _ = parse_gmsh_mesh(mesh_file_path=mesh_file)

    return coords, connectivity, dofs


@pytest.fixture
def make_quad4_mesh(tmp_path: pathlib.Path) -> tuple[Array, Array, Array]:
    """Generate a quad4 mesh and return parsed data.

    Creates a unit square with a 2x2 quad4 mesh using gmsh API,
    saves to tmp_path/quad4/test_mesh.msh, and returns parsed data.

    Returns
    -------
    tuple[Array, Array, Array]
        (coords, connectivity, dofs) from parse_gmsh_mesh.
    """
    return _generate_gmsh_mesh(
        tmp_path, "quad4", nx=2, ny=2, element_order=1, use_triangles=False
    )


@pytest.fixture
def make_tri3_mesh(tmp_path: pathlib.Path) -> tuple[Array, Array, Array]:
    """Generate a tri3 mesh and return parsed data.

    Creates a unit square with a triangular mesh using gmsh API,
    saves to tmp_path/tri3/test_mesh.msh, and returns parsed data.

    Returns
    -------
    tuple[Array, Array, Array]
        (coords, connectivity, dofs) from parse_gmsh_mesh.
    """
    return _generate_gmsh_mesh(
        tmp_path, "tri3", nx=4, ny=4, element_order=1, use_triangles=True
    )


@pytest.fixture
def make_quad8_mesh(tmp_path: pathlib.Path) -> tuple[Array, Array, Array]:
    """Generate a quad8 mesh and return parsed data.

    Creates a unit square with a 2x2 quad8 mesh (second-order) using gmsh API.
    Note: gmsh generates element type 15 (incomplete quad8) for order 2 recombined quads.

    Returns
    -------
    tuple[Array, Array, Array]
        (coords, connectivity, dofs) from parse_gmsh_mesh.
    """
    # gmsh generates element type 15 (incomplete quad8) for order 2 recombined quads
    # We need to manually construct the quad8 connectivity from quad4 connectivity
    # by adding midside nodes
    coords, conn4, dofs4 = _generate_gmsh_mesh(
        tmp_path, "quad4", nx=2, ny=2, element_order=1, use_triangles=False
    )

    # Add midside nodes for quad8 elements
    import numpy as np

    n_elems = conn4.shape[0]
    n_nodes = conn4.shape[1]  # 4 for quad4

    # Get unique edge midpoints
    edge_nodes = set()
    for e in range(n_elems):
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            edge = tuple(sorted([conn4[e, i], conn4[e, j]]))
            edge_nodes.add(edge)

    # Create midside nodes
    midside_coords = []
    edge_to_midside = {}
    for idx, (n1, n2) in enumerate(edge_nodes):
        c1 = coords[n1]
        c2 = coords[n2]
        mid = (c1 + c2) / 2
        midside_coords.append(mid)
        edge_to_midside[(n1, n2)] = len(coords) + idx
        edge_to_midside[(n2, n1)] = len(coords) + idx

    midside_coords = np.array(midside_coords)
    coords8 = np.vstack([coords, midside_coords])

    # Build quad8 connectivity
    conn8 = np.zeros((n_elems, 8), dtype=int)
    for e in range(n_elems):
        conn8[e, :4] = conn4[e]
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            edge = tuple(sorted([conn4[e, i], conn4[e, j]]))
            conn8[e, 4 + i] = edge_to_midside[edge]

    # Build dofs for quad8
    from mesh import convert_ids_to_dofs

    dofs8 = np.array([convert_ids_to_dofs(entry) for entry in conn8])

    return coords8, conn8, dofs8


@pytest.fixture
def make_tri6_mesh(tmp_path: pathlib.Path) -> tuple[Array, Array, Array]:
    """Generate a tri6 mesh and return parsed data.

    Creates a unit square with a triangular mesh (second-order) using gmsh API.

    Returns
    -------
    tuple[Array, Array, Array]
        (coords, connectivity, dofs) from parse_gmsh_mesh.
    """
    # Generate tri3 mesh first, then add midside nodes
    coords, conn3, dofs3 = _generate_gmsh_mesh(
        tmp_path, "tri3", nx=4, ny=4, element_order=1, use_triangles=True
    )

    # Add midside nodes for tri6 elements
    import numpy as np

    n_elems = conn3.shape[0]
    n_nodes = conn3.shape[1]  # 3 for tri3

    # Get unique edge midpoints
    edge_nodes = set()
    for e in range(n_elems):
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            edge = tuple(sorted([conn3[e, i], conn3[e, j]]))
            edge_nodes.add(edge)

    # Create midside nodes
    midside_coords = []
    edge_to_midside = {}
    for idx, (n1, n2) in enumerate(edge_nodes):
        c1 = coords[n1]
        c2 = coords[n2]
        mid = (c1 + c2) / 2
        midside_coords.append(mid)
        edge_to_midside[(n1, n2)] = len(coords) + idx
        edge_to_midside[(n2, n1)] = len(coords) + idx

    midside_coords = np.array(midside_coords)
    coords6 = np.vstack([coords, midside_coords])

    # Build tri6 connectivity
    conn6 = np.zeros((n_elems, 6), dtype=int)
    for e in range(n_elems):
        conn6[e, :3] = conn3[e]
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            edge = tuple(sorted([conn3[e, i], conn3[e, j]]))
            conn6[e, 3 + i] = edge_to_midside[edge]

    # Build dofs for tri6
    from mesh import convert_ids_to_dofs

    dofs6 = np.array([convert_ids_to_dofs(entry) for entry in conn6])

    return coords6, conn6, dofs6


@pytest.fixture
def make_tiny_nurbs_mesh(
    project_root_path: pathlib.Path,
) -> tuple[Array, Array, Array, Dict[str, Any]]:
    """Load and parse the smallest NURBS mesh.

    Loads mesh/nurbs/square_4.json and parses it via parse_nurbs_mesh.

    Returns
    -------
    tuple[Array, Array, Array, Dict[str, Any]]
        (coords, connectivity, dofs, aux_nurbs) from parse_nurbs_mesh.
    """
    mesh_file_path = project_root_path / "mesh" / "nurbs" / "tests" / "square_4"
    return parse_nurbs_mesh(mesh_file_path=mesh_file_path)


def make_network_params(
    network_type: str = "resnet2",
    hidden_size: int = 16,
    n_hidden: int = 4,
) -> Dict[str, Any]:
    """Create a network parameters dictionary matching the JSON schema.

    Parameters
    ----------
    network_type : str, optional
        Network type ("fnn", "resnet2", "resnet3"). Default is "resnet2".
    hidden_size : int, optional
        Hidden layer size. Default is 16.
    n_hidden : int, optional
        Number of hidden layers. Default is 4.

    Returns
    -------
    Dict[str, Any]
        Dictionary matching the network parameters schema.
    """
    return {
        "type": network_type,
        "input_size": 2,
        "output_size": 3,
        "hidden_size": hidden_size,
        "hidden_count": n_hidden,
        "activation": {
            "function": "tanh",
            "initial_coefficient": 1.0,
            "trainable": True,
            "trainable_global": True,
        },
        "rff": {
            "enabled": False,
            "features": 64,
            "scale": 1.0,
        },
        "weight_initialisation_type": "xavier_normal",
    }


@pytest.fixture
def make_network_params_fixture() -> callable:
    """Fixture that returns the make_network_params function.

    Returns
    -------
    callable
        The make_network_params function for creating network parameter dicts.
    """
    return make_network_params


@pytest.fixture(autouse=True)
def cleanup_gmsh() -> None:
    """Auto-cleanup fixture for gmsh between tests."""
    yield
    # Clean up any remaining models
    if gmsh.is_initialized():
        try:
            gmsh.model.remove()
        except RuntimeError:
            # gmsh error, that's fine
            pass
