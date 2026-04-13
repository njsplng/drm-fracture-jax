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
    crack_width = 1e-4
    crack_length = 0.5
    lengthscale_parameter = 4e-2
    len_refinement_y = 0.1

    # Create a 2D geometry
    p1 = geo.addPoint(-0.5, -0.5, 0, lengthscale_parameter)
    p2 = geo.addPoint(0.5, -0.5, 0, lengthscale_parameter)
    p3 = geo.addPoint(0.5, 0.5, 0, lengthscale_parameter)
    p4 = geo.addPoint(-0.5, 0.5, 0, lengthscale_parameter)
    p5 = geo.addPoint(-0.5, crack_width / 2, 0, lengthscale_parameter)
    p6 = geo.addPoint(-0.5, -crack_width / 2, 0, lengthscale_parameter)
    p7 = geo.addPoint(-0.5 + crack_length, 0, 0, lengthscale_parameter)

    # Create lines
    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p5)
    l5 = geo.addLine(p5, p7)
    l6 = geo.addLine(p7, p6)
    l7 = geo.addLine(p6, p1)

    # Create a line loop
    cl1 = geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7])
    geo.addPlaneSurface([cl1])

    # Create a box field for refinement
    f1 = field.add("Box")
    # Min mesh size
    field.setNumber(f1, "VIn", lengthscale_parameter / 7.5)
    # Max mesh size
    field.setNumber(f1, "VOut", lengthscale_parameter)
    field.setNumber(f1, "XMin", -0.1)
    field.setNumber(f1, "XMax", 0.5)
    field.setNumber(f1, "YMin", -0.5)
    field.setNumber(f1, "YMax", len_refinement_y)

    field.setAsBackgroundMesh(f1)


# Run the generation
if __name__ == "__main__":
    generic_mesh(generate_mesh)
