"""Generate a 2D geometry mesh using GMSH."""

import gmsh

from src.templates import generic_mesh


# Define the geometry
def generate_mesh() -> None:
    """Generate a 2D geometry mesh using GMSH."""
    # These are formed at runtime, need to be defined here
    geo = gmsh.model.geo
    gmsh.model.mesh.field

    # Set the parameters of interest
    lengthscale_parameter = 0.5  # 0.5 for quad, 0.5 for tri

    # Create a 2D geometry
    p1 = geo.addPoint(0.0, 0.0, 0, lengthscale_parameter)
    p2 = geo.addPoint(1.0, 0.0, 0, lengthscale_parameter)
    p3 = geo.addPoint(2.0, 0.0, 0, lengthscale_parameter)
    p4 = geo.addPoint(3.0, 0.0, 0, lengthscale_parameter)
    p5 = geo.addPoint(3.0, 1.0, 0, lengthscale_parameter)
    p6 = geo.addPoint(2.0, 1.0, 0, lengthscale_parameter)
    p7 = geo.addPoint(1.0, 1.0, 0, lengthscale_parameter)
    p8 = geo.addPoint(0.0, 1.0, 0, lengthscale_parameter)

    # Create lines
    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p5)
    l5 = geo.addLine(p5, p6)
    l6 = geo.addLine(p6, p7)
    l7 = geo.addLine(p7, p8)
    l8 = geo.addLine(p8, p1)

    # Create a line loop
    cl1 = geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])
    geo.addPlaneSurface([cl1])


# Run the generation
if __name__ == "__main__":
    generic_mesh(generate_mesh, mesh_algorithm=1)
    # Mesh algo 1 for tri, 8 for quad
