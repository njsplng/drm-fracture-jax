#!/usr/bin/env python3

"""Compute the difference between two nodes in a mesh."""

import pathlib
import sys

# Link the necessary libraries
current_path = pathlib.Path(__file__).parent.resolve()

generic_source_path = current_path.parent / "src"
if generic_source_path not in sys.path:
    sys.path.append(str(generic_source_path))

from mesh import gmsh_session, parse_mesh

# Extract the necessary arguments
if len(sys.argv) < 4:
    print(
        "Usage: python view_mesh.py <mesh_filename (no .msh)> <mesh_type> <node1> <node2>"
    )
    sys.exit(1)

mesh_filename = sys.argv[1]
mesh_type = sys.argv[2]
node1 = int(sys.argv[3])
node2 = int(sys.argv[4])

# Get the mesh
with gmsh_session():
    [node_coordinates, connectivities, dofs, aux_nurbs] = parse_mesh(
        mesh_filename, mesh_type
    )

# Print the difference between the two nodes
diff = node_coordinates[node1] - node_coordinates[node2]
print(f"Difference between node {node1} and node {node2}: {diff}")
