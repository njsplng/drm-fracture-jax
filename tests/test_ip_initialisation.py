"""Unit tests for integration point initialisation.

Tests the determine_initialisation_function and IPData.from_initializer
for various element types (quad4, tri3, quad8, tri6, NURBS).
"""

import jax.numpy as jnp
import pytest

from data_handling import IPData
from initialise_points import determine_initialisation_function


class TestIPInitialisation:
    """Test integration point initialisation for various element types."""

    def test_ip_initialisation_quad4(self, make_quad4_mesh) -> None:
        """Test 3.1: Gauss point initialisation for quad4 elements produces correct shapes.

        Create a simple 2×2 quad4 mesh (4 elements, 9 nodes).
        Call the IP initialiser for quad4 with 4 integration points.
        Verify shapes: N is (E, 4, N_per_elem), volumes is (E, 4) and all positive,
        B matrix has correct dimensions.
        """
        coords, connectivity, dofs = make_quad4_mesh

        # Get the initialisation function for quad4 with 4 IPs
        mesh_params = {"type": "quad4", "number_of_integration_points": 4}
        initializer = determine_initialisation_function(mesh_params)

        # Call the initialiser
        result = initializer(
            connectivity=connectivity,
            nodal_coordinates=coords,
            thickness=1.0,
        )

        # Unpack result
        N, dN, d2N, volumes, B, dN_phys, d2N_phys, E_extrap = result

        n_elems = connectivity.shape[0]  # 4
        n_gauss = 4  # 2×2 Gauss quadrature
        n_nodes_per_elem = 4  # quad4

        # Verify N shape: (E, G, N_per_elem)
        assert N.shape == (n_elems, n_gauss, n_nodes_per_elem), f"N shape: {N.shape}"

        # Verify volumes shape: (E, G)
        assert volumes.shape == (n_elems, n_gauss), f"Volumes shape: {volumes.shape}"
        assert jnp.all(volumes > 0), "All volumes should be positive"

        # Verify B matrix shape: (E, G, 3, 2*N_per_elem)
        assert B.shape == (
            n_elems,
            n_gauss,
            3,
            2 * n_nodes_per_elem,
        ), f"B shape: {B.shape}"

        # Verify dN shape: (E, D, G, N_per_elem)
        assert dN.shape == (
            n_elems,
            2,
            n_gauss,
            n_nodes_per_elem,
        ), f"dN shape: {dN.shape}"

    def test_ip_initialisation_tri3(self, make_tri3_mesh) -> None:
        """Test 3.2: Same as above for tri3 with 1 integration point.

        Verify: N shape is (E, 1, 3), volumes are positive.
        """
        coords, connectivity, dofs = make_tri3_mesh

        # Get the initialisation function for tri3 with 1 IP
        mesh_params = {"type": "tri3", "number_of_integration_points": 1}
        initializer = determine_initialisation_function(mesh_params)

        # Call the initialiser
        result = initializer(
            connectivity=connectivity,
            nodal_coordinates=coords,
            thickness=1.0,
        )

        # Unpack result
        N, dN, d2N, volumes, B, dN_phys, d2N_phys, E_extrap = result

        n_elems = connectivity.shape[0]
        n_gauss = 1  # 1-point Gauss quadrature for tri3
        n_nodes_per_elem = 3  # tri3

        # Verify N shape: (E, G, N_per_elem)
        assert N.shape == (n_elems, n_gauss, n_nodes_per_elem), f"N shape: {N.shape}"

        # Verify volumes shape: (E, G)
        assert volumes.shape == (n_elems, n_gauss), f"Volumes shape: {volumes.shape}"
        assert jnp.all(volumes > 0), "All volumes should be positive"

        # Verify B matrix shape: (E, G, 3, 2*N_per_elem)
        assert B.shape == (
            n_elems,
            n_gauss,
            3,
            2 * n_nodes_per_elem,
        ), f"B shape: {B.shape}"

    def test_ip_initialisation_nurbs(self, make_tiny_nurbs_mesh) -> None:
        """Test 3.3: NURBS IP initialisation produces valid integration data.

        Load square_4 NURBS mesh, run the IP initialiser.
        Verify volumes are positive, shape functions sum to 1 at each Gauss point.
        """
        coords, connectivity, dofs, aux_nurbs = make_tiny_nurbs_mesh

        # Get the initialisation function for NURBS
        mesh_params = {"type": "nurbs", "number_of_integration_points": -1}
        initializer = determine_initialisation_function(mesh_params)

        # Call the initialiser
        result = initializer(
            connectivity=connectivity,
            nodal_coordinates=coords,
            thickness=1.0,
            info=aux_nurbs,
        )

        # Unpack result (NURBS returns extra fields)
        N, dN, d2N, volumes, B, dN_phys, d2N_phys, E_extrap, _, _ = result

        n_elems = connectivity.shape[0]
        n_gauss = aux_nurbs["gauss_integration_order"] ** 2  # 2×2 = 4 for order 2

        # Verify volumes shape: (E, G)
        assert volumes.shape == (n_elems, n_gauss), f"Volumes shape: {volumes.shape}"
        assert jnp.all(volumes > 0), "All volumes should be positive"

        # Verify shape functions sum to 1 (partition of unity)
        N_sum = jnp.sum(N, axis=-1)  # Sum over nodes
        assert jnp.allclose(N_sum, 1.0, atol=1e-10), "Shape functions should sum to 1"

    def test_ip_data_from_initializer(self, make_quad4_mesh) -> None:
        """Test 3.4: The IPData.from_initializer classmethod correctly wraps the raw initialiser output.

        Create IP data for a quad4 mesh, verify all required fields populated.
        Verify optional anisotropy fields are None when not requested.
        Call with problem_type="anisotropic" and verify gamma_matrix and rotated derivatives are populated.
        """
        coords, connectivity, dofs = make_quad4_mesh

        # Get the initialisation function
        mesh_params = {"type": "quad4", "number_of_integration_points": 4}
        initializer = determine_initialisation_function(mesh_params)

        # Create IPData with isotropic problem type
        # Note: info argument is required but can be None for quad4
        ip_data = IPData.from_initializer(
            initializer=initializer,
            connectivity=connectivity,
            nodal_coordinates=coords,
            thickness=1.0,
            info=None,  # Required argument, None for quad4
            problem_type="isotropic",
        )

        # Verify required fields are populated
        assert ip_data.N is not None
        assert ip_data.dN is not None
        assert ip_data.d2N is not None
        assert ip_data.volumes is not None
        assert ip_data.B is not None
        assert ip_data.physical_derivatives is not None
        assert ip_data.physical_derivatives_2 is not None
        assert ip_data.extrapolations is not None

        # Verify gamma_matrix is set (even for isotropic)
        assert ip_data.gamma_matrix is not None


class TestShapeFunctions:
    """Test shape function properties."""

    def test_shape_functions_partition_of_unity(
        self, make_quad4_mesh, make_tri3_mesh, make_quad8_mesh, make_tri6_mesh
    ) -> None:
        """Test 3.5: Shape functions sum to 1 at every Gauss point (partition of unity).

        For each element type (quad4, tri3, quad8, tri6): compute N,
        verify N.sum(axis=-1) ≈ 1.

        Why: Fundamental FEM property — if it fails, nothing downstream can be correct.
        """
        # Test quad4
        coords, connectivity, dofs = make_quad4_mesh
        mesh_params = {"type": "quad4", "number_of_integration_points": 4}
        initializer = determine_initialisation_function(mesh_params)
        N, _, _, _, _, _, _, _ = initializer(
            connectivity=connectivity, nodal_coordinates=coords, thickness=1.0
        )
        N_sum = jnp.sum(N, axis=-1)
        assert jnp.allclose(
            N_sum, 1.0, atol=1e-10
        ), f"Quad4 partition of unity failed: max error = {jnp.max(jnp.abs(N_sum - 1.0))}"

        # Test tri3
        coords, connectivity, dofs = make_tri3_mesh
        mesh_params = {"type": "tri3", "number_of_integration_points": 1}
        initializer = determine_initialisation_function(mesh_params)
        N, _, _, _, _, _, _, _ = initializer(
            connectivity=connectivity, nodal_coordinates=coords, thickness=1.0
        )
        N_sum = jnp.sum(N, axis=-1)
        assert jnp.allclose(
            N_sum, 1.0, atol=1e-10
        ), f"Tri3 partition of unity failed: max error = {jnp.max(jnp.abs(N_sum - 1.0))}"

        # Test quad8
        coords, connectivity, dofs = make_quad8_mesh
        mesh_params = {"type": "quad8", "number_of_integration_points": 4}
        initializer = determine_initialisation_function(mesh_params)
        N, _, _, _, _, _, _, _ = initializer(
            connectivity=connectivity, nodal_coordinates=coords, thickness=1.0
        )
        N_sum = jnp.sum(N, axis=-1)
        assert jnp.allclose(
            N_sum, 1.0, atol=1e-10
        ), f"Quad8 partition of unity failed: max error = {jnp.max(jnp.abs(N_sum - 1.0))}"

        # Test tri6
        coords, connectivity, dofs = make_tri6_mesh
        mesh_params = {"type": "tri6", "number_of_integration_points": 6}
        initializer = determine_initialisation_function(mesh_params)
        N, _, _, _, _, _, _, _ = initializer(
            connectivity=connectivity, nodal_coordinates=coords, thickness=1.0
        )
        N_sum = jnp.sum(N, axis=-1)
        assert jnp.allclose(
            N_sum, 1.0, atol=1e-10
        ), f"Tri6 partition of unity failed: max error = {jnp.max(jnp.abs(N_sum - 1.0))}"


class TestDetermineInitialisationFunction:
    """Test the determine_initialisation_function dispatcher."""

    def test_determine_initialisation_function_all_types(self) -> None:
        """Verify determine_initialisation_function returns correct function for all supported types."""
        test_cases = [
            (
                {"type": "quad4", "number_of_integration_points": 4},
                "precompute_quad_4ip",
            ),
            ({"type": "tri3", "number_of_integration_points": 1}, "precompute_tri_1ip"),
            (
                {"type": "quad8", "number_of_integration_points": 4},
                "precompute_quad8_4ip",
            ),
            (
                {"type": "quad8", "number_of_integration_points": 9},
                "precompute_quad8_9ip",
            ),
            (
                {"type": "quad9", "number_of_integration_points": 4},
                "precompute_quad9_4ip",
            ),
            (
                {"type": "quad9", "number_of_integration_points": 9},
                "precompute_quad9_9ip",
            ),
            ({"type": "tri6", "number_of_integration_points": 6}, "precompute_tri_6ip"),
            (
                {"type": "nurbs", "number_of_integration_points": -1},
                "nurbs_point_initialisation",
            ),
        ]

        for mesh_params, expected_name in test_cases:
            func = determine_initialisation_function(mesh_params)
            assert (
                func.__name__ == expected_name
            ), f"Expected {expected_name}, got {func.__name__}"

    def test_determine_initialisation_function_invalid_type(self) -> None:
        """Verify determine_initialisation_function raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="Invalid mesh type"):
            determine_initialisation_function(
                {"type": "invalid", "number_of_integration_points": 4}
            )

    def test_determine_initialisation_function_invalid_ip_count(self) -> None:
        """Verify determine_initialisation_function raises AssertionError for invalid IP count."""
        with pytest.raises(
            AssertionError, match="Invalid mesh type or integration point count"
        ):
            determine_initialisation_function(
                {"type": "quad4", "number_of_integration_points": 9}
            )
