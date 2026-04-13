"""Generate a 2D geometry mesh using GMSH."""

import gmsh

from src.templates import generic_mesh


# Define the geometry
def generate_mesh() -> None:
    """Generate a 2D geometry mesh using GMSH."""
    # These are formed at runtime, need to be defined here instead of importing
    geo = gmsh.model.geo
    field = gmsh.model.mesh.field

    # Set the parameters of interest
    # 5.5e-2 for tri, 3.3e-2 for quad
    lengthscale_parameter = 3.3e-2
    len_refinement_y = 0.2

    # Create a 2D geometry
    p1 = geo.addPoint(-0.5, -0.5, 0, lengthscale_parameter)
    p2 = geo.addPoint(0.0, -0.5, 0, lengthscale_parameter)
    p3 = geo.addPoint(0.0, 0.0, 0, lengthscale_parameter)
    p4 = geo.addPoint(0.5, 0.0, 0, lengthscale_parameter)
    p5 = geo.addPoint(0.5, 0.5, 0, lengthscale_parameter)
    p6 = geo.addPoint(-0.5, 0.5, 0, lengthscale_parameter)

    # Create lines
    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p5)
    l5 = geo.addLine(p5, p6)
    l6 = geo.addLine(p6, p1)

    # Create a line loop
    cl1 = geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
    geo.addPlaneSurface([cl1])

    # Create a box field for refinement
    f1 = field.add("Box")
    # Min mesh size
    field.setNumber(f1, "VIn", lengthscale_parameter / 10)
    # Max mesh size
    field.setNumber(f1, "VOut", lengthscale_parameter)
    field.setNumber(f1, "XMin", -0.5)
    field.setNumber(f1, "XMax", 0.1)
    field.setNumber(f1, "YMin", -len_refinement_y / 2)
    field.setNumber(f1, "YMax", len_refinement_y)

    field.setAsBackgroundMesh(f1)


# Run the generation
if __name__ == "__main__":
    generic_mesh(generate_mesh, mesh_algorithm=8, recombination_algorithm=3)
    # Mesh algorithm 1 for triangles, 8 for quads
    # Recombination 3 for quads
