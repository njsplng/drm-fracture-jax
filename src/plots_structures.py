"""All plots associated with structure visualisation."""

import os
import pathlib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array

from distance_functions import CompositeDistanceFunction
from io_handlers import (
    generate_paraview_fig,
    generate_paraview_fig_local,
    output_paraview,
    parse_parent_input_json,
)
from mesh import gmsh_session, parse_mesh, rescale_mesh


# skylos: ignore-start
def visualise_structure_windows(
    input_file: str,
    mesh_outline: bool = False,
) -> None:
    """Generate paraview files for load and boundary windows.

    Parse the parent input file and generate visualization files for the
    load window, boundary window, and crack window.

    Parameters
    ----------
    input_file : str
        Path to the parent input JSON file.
    mesh_outline : bool, optional
        Whether to include mesh outlines in the figures. Default is False.

    """
    # Parse the input dict
    input_dict = parse_parent_input_json(input_file)

    # Parse the mesh
    with gmsh_session():
        nodal_coordinates, connectivities, _, aux_nurbs = parse_mesh(
            mesh_filename=input_dict["mesh"]["filename"],
            mesh_type=input_dict["mesh"]["type"],
        )

    # Rescale the mesh according to specification
    nodal_coordinates, _ = rescale_mesh(
        nodal_coordinates=nodal_coordinates,
        x_domain=input_dict["problem_domain"]["x"],
        y_domain=input_dict["problem_domain"]["y"],
    )

    # Generate the distance functions according to the input dict specification
    bound_distance_fn = CompositeDistanceFunction(
        input_dict["boundary_conditions"]["fixed_window_parameters"]
    )
    load_distance_fn = CompositeDistanceFunction(
        input_dict["boundary_conditions"]["load_window_parameters"]
    )
    crack_distance_fn = CompositeDistanceFunction(
        input_dict["phasefield_parameters"]["initial_crack_parameters"]
    )

    # Apply the distance functions to the nodes
    bound_distance_points = bound_distance_fn(nodal_coordinates)
    load_distance_points = load_distance_fn(nodal_coordinates)
    crack_distance_points = crack_distance_fn(nodal_coordinates)
    zeroes = jnp.zeros_like(nodal_coordinates[:, 0])

    # Format the dict for dumping
    out_dict = {
        "load_window": [load_distance_points],
        "bound_window": [bound_distance_points],
        "crack_window": [crack_distance_points],
        "displacement": [jnp.stack((zeroes, zeroes, zeroes), axis=1)],
    }

    # Generate the paraview file
    filename = output_paraview(
        file_name=input_dict["mesh"]["filename"],
        increment_list=[0, 0],
        qoi_dict=out_dict,
        coordinates=nodal_coordinates,
        connectivity=connectivities,
        mesh_type=input_dict["mesh"]["type"],
        aux_nurbs=aux_nurbs,
    )

    fig_generation_function = generate_paraview_fig
    pvpython_path = os.environ.get("PARAVIEW_PYTHON")
    if pvpython_path is not None:
        fig_generation_function = generate_paraview_fig_local

    # Generate the figs
    fig_generation_function(
        filename, "load_window", mesh_outline=mesh_outline, axis_limits=(0, 1)
    )
    fig_generation_function(
        filename, "bound_window", mesh_outline=mesh_outline, axis_limits=(0, 1)
    )
    fig_generation_function(
        filename, "crack_window", mesh_outline=mesh_outline, axis_limits=(0, 1)
    )

    # Clean up the paraview dump files
    current_path = pathlib.Path(__file__).parent.resolve()
    project_root = current_path.parent
    output_path = project_root / "output" / "paraview"
    output_path_folder = output_path / filename
    output_path_file = output_path / f"{filename}.pvd"

    for file in os.listdir(output_path_folder):
        os.remove(os.path.join(output_path_folder, file))
    os.rmdir(output_path_folder)
    os.remove(output_path_file)


# skylos: ignore-end


def plot_structure(
    coordinates: Array,
    connectivities: Array,
    figsize: tuple[int, int] = (8, 8),
    dpi: int = 100,
    annotate_nodes: bool = True,
    coordinate_offset: float = 0.01,
    mesh_type: str = "quad",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the structure mesh with optional node annotations.

    Display the mesh connectivity and optionally annotate node indices
    for visualization and debugging purposes.

    Parameters
    ----------
    coordinates : Array
        Node coordinates of shape (N, D) where N is number of nodes
        and D is dimensionality.
    connectivities : Array
        Element connectivity array defining which nodes form each
        element.
    figsize : tuple[int, int], optional
        Figure size as (width, height) in inches. Default is (8, 8).
    dpi : int, optional
        Figure resolution in dots per inch. Default is 100.
    annotate_nodes : bool, optional
        Whether to display node indices on the plot. Default is True.
    coordinate_offset : float, optional
        Offset for annotation text position. Default is 0.01.
    mesh_type : str, optional
        Type of mesh elements. Options are "quad", "quad8", "quad9",
        or "tri6". Default is "quad".

    Returns
    -------
    fig : matplotlib.Figure
        The figure object.
    ax : matplotlib.Axes
        The axes object containing the plot.

    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Plot the nodes
    ax.plot(coordinates[:, 0], coordinates[:, 1], "sk")

    if mesh_type == "quad8" or mesh_type == "quad9":
        connectivities = connectivities[:, :4]
    if mesh_type == "tri6":
        connectivities = connectivities[:, :3]

    # Plot the element outlines
    for i in range(len(connectivities)):
        ax.fill(
            coordinates[connectivities[i]][:, 0],
            coordinates[connectivities[i]][:, 1],
            edgecolor="k",
            fill=False,
        )

    if annotate_nodes:
        # Annotate the nodes
        for i in range(len(coordinates)):
            ax.text(
                coordinates[i][0] + coordinate_offset,
                coordinates[i][1] + coordinate_offset,
                f"{i}",
                fontsize=8,
            )

    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_title("Structure", fontsize=16)
    ax.set_aspect("equal")
    plt.tight_layout()

    return fig, ax


def plot_structure_nurbs(
    control_points: Array,
    connectivities: Array,
    degree: int,
    figsize: tuple[int, int] = (8, 8),
    dpi: int = 100,
    annotate_nodes: bool = True,
    coordinate_offset: float = 0.01,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a NURBS control net with patch boundaries.

    Display the control point network and patch boundaries for a
    NURBS surface. Control point indices can be annotated for
    reference.

    Parameters
    ----------
    control_points : Array
        Control points of shape (N, 2) or (N, 3). Z-coordinates are
        ignored if present.
    connectivities : Array
        Connectivity array of shape (M, (p+1)*(q+1)), where each row
        contains control point indices for one patch.
    degree : int
        Polynomial degree of the NURBS surface.
    figsize : tuple[int, int], optional
        Figure size as (width, height). Default is (8, 8).
    dpi : int, optional
        Figure resolution in dots per inch. Default is 100.
    annotate_nodes : bool, optional
        Whether to label control point indices. Default is True.
    coordinate_offset : float, optional
        Offset for annotation text placement. Default is 0.01.

    Returns
    -------
    fig : matplotlib.Figure
        The figure object.
    ax : matplotlib.Axes
        The axes object containing the plot.

    """
    pts = np.asarray(control_points)
    # drop z if present
    if pts.ndim == 2 and pts.shape[1] > 2:
        pts2d = pts[:, :2]
    else:
        pts2d = pts

    # unpack degrees
    p = degree

    conn = np.asarray(connectivities, dtype=int)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # plot control points
    ax.plot(pts2d[:, 0], pts2d[:, 1], "o", color="black", markersize=4)

    # draw each patch outline using only the 4 corner control points
    # local ordering of conn: grouped by rows of length (p+1)
    for elem in conn:
        # southwest corner
        sw = elem[0]
        # southeast corner
        se = elem[p]
        # northeast corner
        ne = elem[-1]
        # northwest corner
        nw = elem[-(p + 1)]
        # assemble closed rectangle
        rect = np.array([pts2d[sw], pts2d[se], pts2d[ne], pts2d[nw], pts2d[sw]])
        ax.plot(rect[:, 0], rect[:, 1], "-", color="black", linewidth=1)

    # annotate control-point indices
    if annotate_nodes:
        for idx, (x, y) in enumerate(pts2d):
            ax.text(
                x + coordinate_offset,
                y + coordinate_offset,
                str(idx),
                fontsize=8,
                color="black",
            )

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("NURBS Control Net & Patches")
    plt.tight_layout()
    return fig, ax
