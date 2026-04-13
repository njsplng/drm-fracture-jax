#!/usr/bin/env python3

"""View the mesh/generate a .png using Matplotlib."""

import pathlib
import sys

import matplotlib.pyplot as plt

# Link the necessary libraries
current_path = pathlib.Path(__file__).parent.resolve()

generic_source_path = current_path.parent / "src"
if generic_source_path not in sys.path:
    sys.path.append(str(generic_source_path))

from mesh import gmsh_session, parse_mesh
from plots_structures import plot_structure, plot_structure_nurbs

# Extract the necessary arguments
if len(sys.argv) < 4:
    print(
        "Usage: python view_mesh.py <mesh_filename (no .msh)> <mesh_type> <save/plot> <optional: coordinate_offset> <optional: custom figsize>"
    )
    sys.exit(1)

mesh_filename = sys.argv[1]
mesh_type = sys.argv[2]
save_plot = sys.argv[3]
if len(sys.argv) >= 5:
    coordinate_offset = float(sys.argv[4])
else:
    coordinate_offset = 0.01

if len(sys.argv) == 6:
    figsize = (float(sys.argv[5]), float(sys.argv[5]))
else:
    figsize = (60, 60)

# Get the mesh
with gmsh_session():
    [node_coordinates, connectivities, dofs, aux_nurbs] = parse_mesh(
        mesh_filename, mesh_type
    )

if mesh_type == "nurbs":
    fig, ax = plot_structure_nurbs(
        node_coordinates,
        connectivities,
        aux_nurbs["degrees"]["u"],
        figsize=figsize,
        coordinate_offset=coordinate_offset,
    )
else:
    # Plot the mesh
    fig, ax = plot_structure(
        node_coordinates,
        connectivities,
        figsize=figsize,
        coordinate_offset=coordinate_offset,
        mesh_type=mesh_type,
        dpi=100,
    )

if save_plot == "save":
    fig.savefig(
        str(current_path) + f"/visualisations/{mesh_filename}_{mesh_type}.png",
        dpi=300,
        bbox_inches="tight",
    )

if save_plot == "plot":
    window_size = 2000
    # force the window to be 800×800 pixels on-screen
    manager = plt.get_current_fig_manager()
    try:
        # Qt5Agg
        manager.window.resize(window_size, window_size)
    except AttributeError:
        try:
            # older Matplotlib / other backends
            manager.resize(window_size, window_size)
        except AttributeError:
            # TkAgg
            manager.window.wm_geometry(f"{window_size}x{window_size}")

    plt.show()
