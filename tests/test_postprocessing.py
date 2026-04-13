"""Unit tests for postprocessing functions.

Tests the postprocess_qoi_dict, split_displacement, prune_qoi_dict,
rescale_qoi_dict, elemental_to_global, and pointwise_to_nodes functions.
"""

import jax.numpy as jnp

from utils import (
    elemental_to_global,
    pointwise_to_nodes,
    postprocess_qoi_dict,
    prune_qoi_dict,
    rescale_qoi_dict,
    split_displacement,
)


class TestPostprocessQoiDict:
    """Test postprocess_qoi_dict function."""

    def test_postprocess_qoi_dict_ip_to_nodes(self, make_quad4_mesh) -> None:
        """Test 4.1: postprocess_qoi_dict correctly extrapolates IP values to nodes.

        Create a simple mesh with constant IP values (1.0).
        Postprocess with correct extrapolation matrices.
        Verify the resulting nodal values are approximately 1.0 everywhere.
        """
        coords, connectivity, dofs = make_quad4_mesh
        n_elems = connectivity.shape[0]
        n_gauss = 4  # 2x2 Gauss quadrature for quad4
        n_nodes = coords.shape[0]
        n_nodes_per_elem = 4  # quad4 has 4 nodes

        # Create constant IP values (1.0 at all Gauss points)
        ip_values = jnp.ones((n_elems, n_gauss))

        # Create extrapolation matrices for quad4 (E, n_gauss, n_nodes_per_elem)
        # For bilinear quad, the extrapolation matrix maps Gauss values to corner nodes
        # Using simple averaging: each node gets average of all Gauss points
        E_extrap = jnp.ones((n_elems, n_gauss, n_nodes_per_elem)) / n_gauss

        # Create QoI dict with IP values
        qoi_dict = {"ip_test": [ip_values]}

        # Postprocess - use connectivity (node indices) and conns_size = n_nodes
        result = postprocess_qoi_dict(
            qoi_dict=qoi_dict,
            connectivities=connectivity,
            conns_size=n_nodes,  # Number of nodes (not DOFs)
            extrapolation_matrices=E_extrap,
        )

        # Verify the _nodes key was created
        assert "ip_test_nodes" in result

        # Verify the extrapolated values are approximately 1.0
        nodal_values = result["ip_test_nodes"][0]
        assert jnp.allclose(
            nodal_values, 1.0, atol=1e-6
        ), "Extrapolated values should be ~1.0"

    def test_postprocess_qoi_dict_displacement_split(self, make_quad4_mesh) -> None:
        """Test 4.2: postprocess_qoi_dict splits displacement into x, y, z components.

        Create a QoI dict with "displacement" as a flat interleaved array.
        Postprocess with keys_to_split=["displacement"].
        Verify the result has shape (N, 3) with correct x, y columns and z=0.
        """
        coords, connectivity, dofs = make_quad4_mesh
        n_nodes = coords.shape[0]

        # Create displacement as flat interleaved array [u0, v0, u1, v1, ...]
        displacement_flat = jnp.array([1.0, 2.0, 3.0, 4.0] + [0.0] * (2 * n_nodes - 4))

        # Create QoI dict with displacement
        qoi_dict = {"displacement": [displacement_flat]}

        # Postprocess with displacement splitting
        result = postprocess_qoi_dict(
            qoi_dict=qoi_dict,
            connectivities=connectivity,
            conns_size=n_nodes * 2,
            extrapolation_matrices=jnp.eye(4),
            keys_to_split=["displacement"],
        )

        # Verify displacement was split into (N, 3) array
        assert "displacement" in result
        split_disp = result["displacement"][0]
        assert split_disp.shape == (
            n_nodes,
            3,
        ), f"Expected shape ({n_nodes}, 3), got {split_disp.shape}"

        # Verify x and y components are correct
        assert jnp.allclose(
            split_disp[:2, 0], jnp.array([1.0, 3.0])
        ), "X components incorrect"
        assert jnp.allclose(
            split_disp[:2, 1], jnp.array([2.0, 4.0])
        ), "Y components incorrect"

        # Verify z component is zero
        assert jnp.allclose(split_disp[:, 2], 0.0), "Z component should be zero"


class TestSplitDisplacement:
    """Test split_displacement function."""

    def test_split_displacement(self) -> None:
        """Test 4.3: split_displacement correctly deinterleaves a flat dof vector.

        Input: [u0, v0, u1, v1, u2, v2] -> Expected shape (3, 3) with correct column layout.
        """
        # Input: flat interleaved displacement [u0, v0, u1, v1, u2, v2]
        displacement = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Split into components
        result = split_displacement(displacement)

        # Verify shape is (3, 3)
        assert result.shape == (3, 3), f"Expected shape (3, 3), got {result.shape}"

        # Verify x component (column 0)
        assert jnp.allclose(
            result[:, 0], jnp.array([1.0, 3.0, 5.0])
        ), "X component incorrect"

        # Verify y component (column 1)
        assert jnp.allclose(
            result[:, 1], jnp.array([2.0, 4.0, 6.0])
        ), "Y component incorrect"

        # Verify z component (column 2) is zero
        assert jnp.allclose(result[:, 2], jnp.zeros(3)), "Z component should be zero"


class TestPruneQoiDict:
    """Test prune_qoi_dict function."""

    def test_prune_qoi_dict(self) -> None:
        """Test 4.4: prune_qoi_dict keeps only specified keys."""
        # Create a QoI dict with multiple keys
        qoi_dict = {
            "displacement": [jnp.ones(10)],
            "energy": [jnp.array([1.0, 2.0])],
            "stress": [jnp.ones((5, 6))],
            "phasefield": [jnp.ones(10)],
        }

        # Prune to keep only displacement and energy
        result = prune_qoi_dict(qoi_dict, keys_to_keep=["displacement", "energy"])

        # Verify only specified keys are present
        assert set(result.keys()) == {"displacement", "energy"}

        # Verify values are preserved
        assert jnp.allclose(result["displacement"][0], jnp.ones(10))
        assert jnp.allclose(result["energy"][0], jnp.array([1.0, 2.0]))

        # Test with non-existent key (should be silently ignored)
        result = prune_qoi_dict(qoi_dict, keys_to_keep=["displacement", "nonexistent"])
        assert set(result.keys()) == {"displacement"}


class TestRescaleQoiDict:
    """Test rescale_qoi_dict function."""

    def test_rescale_qoi_dict(self) -> None:
        """Test 4.5: rescale_qoi_dict applies correct scaling factors by key name.

        Verify "displacement" scaled by 1/displacement_scaling, "energy" by 1/energy_scaling,
        "stress" by 1/force_scaling, "phasefield" untouched.
        """
        # Create a QoI dict with various quantities
        qoi_dict = {
            "displacement": [jnp.array([1.0, 2.0, 3.0])],
            "ip_displacement_nodes": [jnp.array([4.0, 5.0])],
            "energy": [jnp.array([10.0, 20.0])],
            "stress": [jnp.array([100.0, 200.0])],
            "force_reaction": [jnp.array([50.0])],
            "phasefield": [jnp.array([0.5, 1.0])],
        }

        # Define scaling factors
        displacement_scaling = 2.0
        energy_scaling = 10.0
        force_scaling = 100.0

        # Apply rescaling
        rescale_qoi_dict(qoi_dict, displacement_scaling, energy_scaling, force_scaling)

        # Verify displacement scaled by 1/displacement_scaling = 0.5
        assert jnp.allclose(qoi_dict["displacement"][0], jnp.array([0.5, 1.0, 1.5]))
        assert jnp.allclose(qoi_dict["ip_displacement_nodes"][0], jnp.array([2.0, 2.5]))

        # Verify energy scaled by 1/energy_scaling = 0.1
        assert jnp.allclose(qoi_dict["energy"][0], jnp.array([1.0, 2.0]))

        # Verify stress scaled by 1/force_scaling = 0.01
        assert jnp.allclose(qoi_dict["stress"][0], jnp.array([1.0, 2.0]))

        # Verify force scaled by 1/force_scaling = 0.01
        assert jnp.allclose(qoi_dict["force_reaction"][0], jnp.array([0.5]))

        # Verify phasefield unchanged (no scaling rule matches)
        assert jnp.allclose(qoi_dict["phasefield"][0], jnp.array([0.5, 1.0]))


class TestElementalToGlobal:
    """Test elemental_to_global function."""

    def test_elemental_to_global_scatter(self, make_quad4_mesh) -> None:
        """Test 4.6: elemental_to_global correctly scatter-adds elemental values to global nodes.

        Create a 2-element mesh sharing a common edge, verify accumulation at shared nodes.
        """
        coords, connectivity, dofs = make_quad4_mesh
        n_elems = connectivity.shape[0]
        n_nodes_per_elem = 4

        # Create elemental values (scalar per node per element)
        # Each element contributes 1.0 to each of its nodes
        elemental_values = jnp.ones((n_elems, n_nodes_per_elem))

        # Scatter to global - use connectivity (node indices), not dofs
        result = elemental_to_global(
            elemental_values=elemental_values,
            connectivities=connectivity,
            conns_size=coords.shape[0],  # Total nodes
        )

        # Verify shape
        assert (
            result.shape[0] == coords.shape[0]
        ), "Result should have one entry per node"

        # Corner nodes belong to 1 element, edge nodes to 2, center node to 4
        # For a 2x2 quad4 mesh:
        # - 4 corner nodes: each in 1 element -> value = 1.0
        # - 4 edge mid nodes: each in 2 elements -> value = 2.0
        # - 1 center node: in 4 elements -> value = 4.0
        unique_values = jnp.unique(result)
        assert set(unique_values.tolist()) == {
            1.0,
            2.0,
            4.0,
        }, f"Expected {{1, 2, 4}}, got {set(unique_values.tolist())}"


class TestPointwiseToNodes:
    """Test pointwise_to_nodes function."""

    def test_pointwise_to_nodes_extrapolation(self, make_quad4_mesh) -> None:
        """Test 4.7: pointwise_to_nodes correctly extrapolates from Gauss points and averages at shared nodes.

        Create constant values at Gauss points, verify nodal extrapolation produces
        correct values with proper averaging at shared nodes.
        """
        coords, connectivity, dofs = make_quad4_mesh
        n_elems = connectivity.shape[0]
        n_gauss = 4
        n_nodes = coords.shape[0]
        n_nodes_per_elem = 4

        # Create constant IP values (1.0 at all Gauss points)
        ip_values = jnp.ones((n_elems, n_gauss))

        # Create extrapolation matrices for quad4 (E, n_gauss, n_nodes_per_elem)
        # Simple: each node gets average of all Gauss points in element
        E_extrap = jnp.ones((n_elems, n_gauss, n_nodes_per_elem)) / n_gauss

        # Extrapolate to nodes - use connectivity (node indices), not dofs
        result = pointwise_to_nodes(
            ip_values=ip_values,
            connectivities=connectivity,
            conns_size=n_nodes,
            extrapolation_matrices=E_extrap,
        )

        # Verify all nodal values are 1.0 (constant field)
        assert jnp.allclose(
            result, 1.0, atol=1e-6
        ), "Constant field should extrapolate to 1.0 everywhere"

        # Test with linearly varying field
        # Set IP values to element index
        ip_values_linear = jnp.arange(n_elems, dtype=jnp.float64).reshape(
            -1, 1
        ) * jnp.ones(n_gauss)
        result_linear = pointwise_to_nodes(
            ip_values=ip_values_linear,
            connectivities=connectivity,
            conns_size=n_nodes,
            extrapolation_matrices=E_extrap,
        )

        # Shared nodes should have averaged values
        # Center node is shared by all 4 elements -> value = (0+1+2+3)/4 = 1.5
        center_node_idx = n_nodes - 1  # Last node is center in 2x2 mesh
        assert jnp.isclose(
            result_linear[center_node_idx], 1.5, atol=1e-6
        ), "Center node should have averaged value 1.5"
