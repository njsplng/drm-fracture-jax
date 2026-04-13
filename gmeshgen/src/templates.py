"""Mesh generation utilities using GMSH."""

import inspect
import pathlib
import sys
from typing import Callable

import gmsh


def generic_mesh(
    geometry_function: Callable[[], None],
    filename: str | None = None,
    mesh_algorithm: int = 8,
    recombination_algorithm: int = 2,
    element_order: int = 1,
) -> None:
    """
    Mesh generation function using GMSH.
    """
    # Find where the function was called from
    if filename is None:
        frame = inspect.stack()[1]
        filename = frame.filename

    # Extract the filename from the path
    filename = filename.split("/")[-1].split(".")[0]

    # Get the system arguments for mesh type
    if "-tri" in sys.argv:
        mesh_type = "tri"
    elif "-quad" in sys.argv:
        mesh_type = "quad"
    else:
        raise ValueError("Please specify whether the mesh is -tri or -quad")

    # Check for order specification in command line arguments
    if "-order1" in sys.argv:
        element_order = 1
    elif "-order2" in sys.argv:
        element_order = 2

    complete_quad = False
    if "-centroid" in sys.argv:
        complete_quad = True

    # Initialise the gmsh API
    gmsh.initialize()

    # Map frequently used API functions to shorter names
    options = gmsh.option

    # Name the model
    gmsh.model.add(filename)

    # Call the mesh generation function
    geometry_function()

    # Push to model
    gmsh.model.geo.synchronize()

    # Configure options
    options.setNumber("Mesh.Algorithm", mesh_algorithm)
    # Set the element order
    options.setNumber("Mesh.ElementOrder", element_order)

    match mesh_type:
        case "tri":
            options.setNumber("Mesh.RecombineAll", 0)
            node_number = 3 * element_order
        case "quad":
            options.setNumber("Mesh.RecombineAll", 1)
            options.setNumber("Mesh.RecombinationAlgorithm", recombination_algorithm)
            node_number = 4 * element_order + 1
            if not complete_quad:
                options.setNumber("Mesh.SecondOrderIncomplete", 1)
                node_number = 4 * element_order

    # Set the mesh type based on the element order
    mesh_type = mesh_type + str(node_number)

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Optionally, run the GUI
    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    # Check if a file with the same name already exists in the FEM directory
    current_dir = pathlib.Path(__file__).parent.resolve()
    mesh_dir = current_dir.parent.parent / "mesh"
    mesh_dir_specific = mesh_dir / mesh_type
    mesh_dir_specific.mkdir(parents=True, exist_ok=True)
    filename_path = mesh_dir_specific / (filename + ".msh")

    # If so, save the file in the meshgen directory
    if filename_path.exists():
        print(
            "File with the same name already exists... Saving the file to meshgen/output"
        )
        mesh_dir = current_dir.parent / "output" / mesh_type
        mesh_dir.mkdir(parents=True, exist_ok=True)
        filename_path = mesh_dir / (filename + ".msh")

    if input("save mesh? 'y' to save, anything else to abort\n") == "y":
        # Save the file
        gmsh.write(str(filename_path))

    # Close the gmsh API
    gmsh.finalize()
