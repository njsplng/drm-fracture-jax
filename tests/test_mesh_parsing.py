"""Unit tests for mesh parsing functions.

Tests the parse_gmsh_mesh, parse_nurbs_mesh, convert_ids_to_dofs,
rescale_mesh, and legendre_gauss_quadrature_2d functions.
"""

import numpy as np

from mesh import (
    convert_ids_to_dofs,
    legendre_gauss_quadrature_2d,
    rescale_mesh,
)


class TestParseNurbsMesh:
    """Test NURBS mesh parsing."""

    def test_parse_nurbs_mesh_shapes(self, make_tiny_nurbs_mesh) -> None:
        """Test 2.1: parse_nurbs_mesh returns arrays with correct shapes for a known NURBS configuration.

        Use square_4.json (smallest NURBS mesh).
        Verify nodal_coordinates shape is (n_ctrl_u * n_ctrl_v, 2).
        Verify connectivities shape is (n_elems, (p+1)*(q+1)).
        Verify dofs shape is (E, 2*(p+1)*(q+1)).
        Verify aux_nurbs dict contains all expected keys.
        """
        coords, connectivity, dofs, aux_nurbs = make_tiny_nurbs_mesh

        # square_4.json has 11×11 elements with order 2×2
        # Control points: (11+2) × (11+2) = 169
        n_ctrl_u = 13  # elements_x + order_x
        n_ctrl_v = 13  # elements_y + order_y
        n_elems = 11 * 11  # 121 elements
        nodes_per_elem = 3 * 3  # (p+1) × (q+1) for order 2

        # Verify nodal_coordinates shape
        expected_coords_shape = (n_ctrl_u * n_ctrl_v, 2)
        assert (
            coords.shape == expected_coords_shape
        ), f"Expected {expected_coords_shape}, got {coords.shape}"

        # Verify connectivities shape
        expected_conn_shape = (n_elems, nodes_per_elem)
        assert (
            connectivity.shape == expected_conn_shape
        ), f"Expected {expected_conn_shape}, got {connectivity.shape}"

        # Verify dofs shape
        expected_dofs_shape = (n_elems, 2 * nodes_per_elem)
        assert (
            dofs.shape == expected_dofs_shape
        ), f"Expected {expected_dofs_shape}, got {dofs.shape}"

        # Verify aux_nurbs keys
        assert "control_points" in aux_nurbs
        assert "knotvector_u" in aux_nurbs or "knots_u" in aux_nurbs
        assert "knotvector_v" in aux_nurbs or "knots_v" in aux_nurbs

    def test_parse_nurbs_mesh_coordinates_in_domain(self, make_tiny_nurbs_mesh) -> None:
        """Test 2.2: NURBS control point coordinates lie within the specified problem domain.

        Load the NURBS mesh, extract problem_domain from the JSON.
        Assert all coordinates are within bounds.
        """
        coords, connectivity, dofs, aux_nurbs = make_tiny_nurbs_mesh

        # square_4.json has domain [-0.5, 0.5] × [-0.5, 0.5]
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.5, 0.5

        # Verify all x coordinates are within bounds
        assert np.all(coords[:, 0] >= x_min - 1e-10)
        assert np.all(coords[:, 0] <= x_max + 1e-10)

        # Verify all y coordinates are within bounds
        assert np.all(coords[:, 1] >= y_min - 1e-10)
        assert np.all(coords[:, 1] <= y_max + 1e-10)


class TestParseGmshMesh:
    """Test gmsh mesh parsing."""

    def test_parse_gmsh_mesh_quad4(self, make_quad4_mesh) -> None:
        """Test 2.3: parse_gmsh_mesh correctly handles quad4 meshes.

        Generate a small quad4 mesh programmatically using gmsh API, save to temp .msh file.
        Parse it via parse_gmsh_mesh.
        Verify: node coordinates shape (N, 2), connectivity has 4 nodes per element,
        dofs shape (E, 8), connectivity is 0-based and contiguous.
        """
        coords, connectivity, dofs = make_quad4_mesh

        # 2×2 quad4 mesh: 9 nodes, 4 elements
        assert coords.shape[1] == 2, "Coordinates should be 2D"
        assert connectivity.shape[1] == 4, "Quad4 elements should have 4 nodes"
        assert dofs.shape == (connectivity.shape[0], 8), "Dofs should be (E, 8)"

        # Verify connectivity is 0-based
        assert connectivity.min() >= 0
        assert connectivity.max() < coords.shape[0]

    def test_parse_gmsh_mesh_tri3(self, make_tri3_mesh) -> None:
        """Test 2.4: parse_gmsh_mesh correctly handles tri3 meshes.

        Generate a small tri3 mesh, parse it.
        Verify: connectivity has 3 nodes per element, dofs shape (E, 6).
        """
        coords, connectivity, dofs = make_tri3_mesh

        assert coords.shape[1] == 2, "Coordinates should be 2D"
        assert connectivity.shape[1] == 3, "Tri3 elements should have 3 nodes"
        assert dofs.shape == (connectivity.shape[0], 6), "Dofs should be (E, 6)"

        # Verify connectivity is 0-based
        assert connectivity.min() >= 0
        assert connectivity.max() < coords.shape[0]

    def test_parse_gmsh_mesh_quad8(self, make_quad8_mesh) -> None:
        """Test 2.5: Higher-order quad8 elements parse correctly.

        Generate a second-order quad mesh (incomplete — 8 nodes per element).
        Verify connectivity shape (E, 8), dofs shape (E, 16).
        """
        coords, connectivity, dofs = make_quad8_mesh

        assert coords.shape[1] == 2, "Coordinates should be 2D"
        assert connectivity.shape[1] == 8, "Quad8 elements should have 8 nodes"
        assert dofs.shape == (connectivity.shape[0], 16), "Dofs should be (E, 16)"

        # Verify connectivity is 0-based
        assert connectivity.min() >= 0
        assert connectivity.max() < coords.shape[0]

    def test_parse_gmsh_mesh_tri6(self, make_tri6_mesh) -> None:
        """Test 2.6: Higher-order tri6 elements parse correctly.

        Generate a second-order triangle mesh.
        Verify connectivity shape (E, 6), dofs shape (E, 12).
        """
        coords, connectivity, dofs = make_tri6_mesh

        assert coords.shape[1] == 2, "Coordinates should be 2D"
        assert connectivity.shape[1] == 6, "Tri6 elements should have 6 nodes"
        assert dofs.shape == (connectivity.shape[0], 12), "Dofs should be (E, 12)"

        # Verify connectivity is 0-based
        assert connectivity.min() >= 0
        assert connectivity.max() < coords.shape[0]


class TestConvertIdsToDofs:
    """Test node ID to DOF conversion."""

    def test_convert_ids_to_dofs(self) -> None:
        """Test 2.7: convert_ids_to_dofs maps node IDs to interleaved dof pairs.

        Input: [0, 3, 5] → Expected: [0, 1, 6, 7, 10, 11].
        Input: [1] → Expected: [2, 3].
        """
        # Test case 1
        input_ids = np.array([0, 3, 5])
        expected_dofs = np.array([0, 1, 6, 7, 10, 11])
        result = convert_ids_to_dofs(input_ids)
        assert np.array_equal(
            result, expected_dofs
        ), f"Expected {expected_dofs}, got {result}"

        # Test case 2
        input_ids = np.array([1])
        expected_dofs = np.array([2, 3])
        result = convert_ids_to_dofs(input_ids)
        assert np.array_equal(
            result, expected_dofs
        ), f"Expected {expected_dofs}, got {result}"

        # Test case 3: empty array
        input_ids = np.array([])
        result = convert_ids_to_dofs(input_ids)
        assert result.shape == (0,), "Empty input should produce empty output"


class TestRescaleMesh:
    """Test mesh rescaling."""

    def test_rescale_mesh(self) -> None:
        """Test 2.8: rescale_mesh correctly maps coordinates to a new domain.

        Create a mesh in [0, 1] × [0, 1], rescale to [0, 3] × [0, 2].
        Verify extremities match the target domain.
        Test the no-op case: verify rescale_executed=False when mesh already matches.
        """
        # Create a simple mesh in [0, 1] × [0, 1]
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )

        # Rescale to [0, 3] × [0, 2] - note: function takes separate x_domain and y_domain
        rescaled_coords, rescale_executed = rescale_mesh(coords, (0.0, 3.0), (0.0, 2.0))

        assert rescale_executed, "Rescaling should have been executed"
        assert np.isclose(rescaled_coords[:, 0].min(), 0.0, atol=1e-10)
        assert np.isclose(rescaled_coords[:, 0].max(), 3.0, atol=1e-10)
        assert np.isclose(rescaled_coords[:, 1].min(), 0.0, atol=1e-10)
        assert np.isclose(rescaled_coords[:, 1].max(), 2.0, atol=1e-10)

        # Test no-op case
        already_scaled = np.array(
            [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]], dtype=np.float64
        )
        rescaled_coords2, rescale_executed2 = rescale_mesh(
            already_scaled, (0.0, 3.0), (0.0, 2.0)
        )
        assert not rescale_executed2, "Rescaling should not have been executed"


class TestLegendreGaussQuadrature:
    """Test Gauss quadrature point generation."""

    def test_legendre_gauss_quadrature_2d(self) -> None:
        """Test 2.9: legendre_gauss_quadrature_2d returns correct point/weight counts.

        For orders 1, 2, 3: verify correct counts, points in [-1, 1]², weights sum to 4.0.
        """
        for order in [1, 2, 3]:
            points, weights = legendre_gauss_quadrature_2d(order)

            # Verify point count
            expected_points = order * order
            assert points.shape == (
                expected_points,
                2,
            ), f"Order {order}: expected {expected_points} points"

            # Verify points are in [-1, 1]²
            assert np.all(points >= -1.0 - 1e-10)
            assert np.all(points <= 1.0 + 1e-10)

            # Verify weights sum to 4.0 (area of [-1, 1]²)
            weight_sum = np.sum(weights)
            assert np.isclose(
                weight_sum, 4.0
            ), f"Order {order}: weights sum to {weight_sum}, expected 4.0"

            # Verify all weights are positive
            assert np.all(weights > 0), f"Order {order}: all weights should be positive"
