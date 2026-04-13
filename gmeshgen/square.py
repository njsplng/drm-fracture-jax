"""Generate a 2D geometry mesh using GMSH."""

import gmsh

from src.templates import generic_mesh


# Define the geometry
def generate_mesh() -> None:
    """Generate a 2D geometry mesh using GMSH.

    Creates a 2D square mesh with refinement in the center using GMSH.
    """
    # These are formed at runtime, need to be defined here instead of importing
    geo = gmsh.model.geo
    field = gmsh.model.mesh.field

    # Set the parameters of interest
    lengthscale_parameter = 1 / 50

    # Create a 2D geometry
    p1 = geo.addPoint(-0.5, -0.5, 0, lengthscale_parameter)
    p2 = geo.addPoint(0.5, -0.5, 0, lengthscale_parameter)
    p5 = geo.addPoint(0.5, 0.0, 0, lengthscale_parameter)
    p3 = geo.addPoint(0.5, 0.5, 0, lengthscale_parameter)
    p4 = geo.addPoint(-0.5, 0.5, 0, lengthscale_parameter)
    p6 = geo.addPoint(0.0, 0.5, 0, lengthscale_parameter)
    p7 = geo.addPoint(-0.5, 0.0, 0, lengthscale_parameter)
    p8 = geo.addPoint(0.0, -0.5, 0, lengthscale_parameter)

    # Create lines
    # l1 = geo.addLine(p1, p2)
    # l2 = geo.addLine(p2, p5)
    # l5 = geo.addLine(p5, p3)
    # l3 = geo.addLine(p3, p4)
    # l4 = geo.addLine(p4, p1)

    l1 = geo.addLine(p1, p8)
    l2 = geo.addLine(p8, p2)
    l3 = geo.addLine(p2, p5)
    l4 = geo.addLine(p5, p3)
    l5 = geo.addLine(p3, p6)
    l6 = geo.addLine(p6, p4)
    l7 = geo.addLine(p4, p7)
    l8 = geo.addLine(p7, p1)

    # l1 = geo.addLine(p1, p2)
    # l2 = geo.addLine(p2, p3)
    # l3 = geo.addLine(p3, p4)
    # l4 = geo.addLine(p4, p1)

    # Create a line loop
    # cl1 = geo.addCurveLoop([l1, l2, l5, l3, l4])
    # cl1 = geo.addCurveLoop([l1, l2, l3, l4])
    cl1 = geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])
    geo.addPlaneSurface([cl1])

    # Create a box field for refinement
    f1 = field.add("Box")
    field.setNumber(f1, "VIn", lengthscale_parameter / 5)  # Min mesh size
    field.setNumber(f1, "VOut", lengthscale_parameter)  # Max mesh size
    field.setNumber(f1, "XMin", -0.5)
    field.setNumber(f1, "XMax", 0.5)
    field.setNumber(f1, "YMin", -0.2)
    field.setNumber(f1, "YMax", 0.2)

    field.setAsBackgroundMesh(f1)


# Run the generation
if __name__ == "__main__":
    generic_mesh(generate_mesh, mesh_algorithm=8, recombination_algorithm=3)
    # Mesh algorithm 1 for triangles, 8 for quads
    # Recombination 3 for quads
