"""Unit tests for neural network setup utilities.

Tests for construct_network, construct_displacement, and NonsmoothSigmoid.
"""

import jax
import jax.numpy as jnp
import pytest
from setup_nn import (
    NonsmoothSigmoid,
    construct_displacement,
    construct_network,
    construct_pf_constraint,
)

# Enable JAX float64 mode
jax.config.update("jax_enable_x64", True)


class TestConstructNetwork:
    """Tests for construct_network function."""

    def test_construct_network_from_params(self) -> None:
        """Test 6.15: Test construct_network with type='fnn', 'resnet3'. Verify valid output on forward pass."""
        # Test FNN
        fnn_params = {
            "type": "fnn",
            "input_size": 2,
            "output_size": 3,
            "hidden_size": 16,
            "hidden_count": 4,
            "activation": {
                "function": "tanh",
                "initial_coefficient": 1.0,
                "trainable": True,
                "trainable_global": True,
            },
            "rff": {"enabled": False, "features": 64, "scale": 1.0},
            "weight_initialisation_type": "xavier_normal",
        }
        fnn = construct_network(fnn_params, seed=0)
        x = jnp.ones((5, 2))
        out = jax.vmap(fnn)(x)  # Use vmap for batched input
        assert out.shape == (5, 3), f"FNN expected shape (5, 3), got {out.shape}"
        assert jnp.all(jnp.isfinite(out)), "FNN output contains non-finite values"

        # Test ResNet3
        resnet3_params = fnn_params.copy()
        resnet3_params["type"] = "resnet3"
        resnet3_params["hidden_count"] = 6  # Must be divisible by 3
        resnet3 = construct_network(resnet3_params, seed=0)
        out = jax.vmap(resnet3)(x)  # Use vmap for batched input
        assert out.shape == (5, 3), f"ResNet3 expected shape (5, 3), got {out.shape}"
        assert jnp.all(jnp.isfinite(out)), "ResNet3 output contains non-finite values"


class TestConstructDisplacement:
    """Tests for construct_displacement function."""

    def test_construct_displacement_schedule(self) -> None:
        """Test 6.16: Verify starts near 0 (1e-12), correct total length, monotonically increasing within segments."""
        displacement_params = {
            "start": 0.0,
            "end": 1.0,
            "coarse_end": 0.5,
            "increments_coarse": 4,
            "increments_fine": 4,
        }

        displacements = construct_displacement(displacement_params)

        # First value should be near 0 (1e-12)
        assert jnp.isclose(
            displacements[0], 1e-12
        ), f"First displacement should be 1e-12, got {displacements[0]}"

        # Total length should be increments_coarse + increments_fine + 1 (includes starting point)
        expected_length = (
            displacement_params["increments_coarse"]
            + displacement_params["increments_fine"]
            + 1
        )
        assert (
            len(displacements) == expected_length
        ), f"Expected length {expected_length}, got {len(displacements)}"

        # Monotonically increasing within coarse segment
        coarse_segment = displacements[: displacement_params["increments_coarse"] + 1]
        assert jnp.all(
            jnp.diff(coarse_segment) > 0
        ), "Coarse segment should be monotonically increasing"

        # Monotonically increasing within fine segment
        fine_segment = displacements[displacement_params["increments_coarse"] :]
        assert jnp.all(
            jnp.diff(fine_segment) > 0
        ), "Fine segment should be monotonically increasing"


class TestNonsmoothSigmoid:
    """Tests for NonsmoothSigmoid function."""

    def test_nonsmooth_sigmoid(self) -> None:
        """Test 6.17: Verify 0.5 at x=0, 0 at x=-support, 1 at x=+support, correct slope outside, monotonic."""
        support = 1.0
        coeff = 0.5

        sigmoid = NonsmoothSigmoid(coeff=coeff, support=support, offset=0.0)

        # At x = 0, should be 0.5
        assert jnp.isclose(
            sigmoid(0.0), 0.5
        ), f"Expected 0.5 at x=0, got {sigmoid(0.0)}"

        # At x = -support, should be 0
        assert jnp.isclose(
            sigmoid(-support), 0.0
        ), f"Expected 0 at x=-support, got {sigmoid(-support)}"

        # At x = +support, should be 1
        assert jnp.isclose(
            sigmoid(support), 1.0
        ), f"Expected 1 at x=+support, got {sigmoid(support)}"

        # Outside the support, slope should be coeff
        # At x = -2*support, value should be coeff * (-2*support + support) = -coeff * support
        x_outside_neg = -2 * support
        expected_neg = coeff * (x_outside_neg + support)
        assert jnp.isclose(
            sigmoid(x_outside_neg), expected_neg
        ), f"Expected {expected_neg} at x={x_outside_neg}, got {sigmoid(x_outside_neg)}"

        # At x = 2*support, value should be coeff * (2*support - support) + 1 = coeff * support + 1
        x_outside_pos = 2 * support
        expected_pos = coeff * (x_outside_pos - support) + 1.0
        assert jnp.isclose(
            sigmoid(x_outside_pos), expected_pos
        ), f"Expected {expected_pos} at x={x_outside_pos}, got {sigmoid(x_outside_pos)}"

        # Monotonic: derivative should be positive everywhere (check by sampling)
        x_samples = jnp.linspace(-3 * support, 3 * support, 100)
        y_samples = sigmoid(x_samples)
        diffs = jnp.diff(y_samples)
        assert jnp.all(
            diffs >= 0
        ), "NonsmoothSigmoid should be monotonically increasing"


class TestConstructPfConstraint:
    """Tests for construct_pf_constraint function."""

    def test_construct_pf_constraint_numerical(self) -> None:
        """Test construct_pf_constraint with type='numerical'."""
        constraint_params = {
            "type": "numerical",
            "numerical_coefficient": 0.5,
            "numerical_support": 1.0,
            "numerical_offset": 0.0,
        }

        constraint = construct_pf_constraint(constraint_params)
        assert isinstance(
            constraint, NonsmoothSigmoid
        ), "Should return NonsmoothSigmoid instance"

    def test_construct_pf_constraint_analytical(self) -> None:
        """Test construct_pf_constraint with type='analytical'."""
        import jax.nn as jnn

        constraint_params = {"type": "analytical"}

        constraint = construct_pf_constraint(constraint_params)
        assert constraint == jnn.sigmoid, "Should return jnn.sigmoid"

    def test_construct_pf_constraint_unknown(self) -> None:
        """Test construct_pf_constraint with unknown type raises ValueError."""
        constraint_params = {"type": "unknown"}

        with pytest.raises(ValueError, match="Unknown phasefield constraint type"):
            construct_pf_constraint(constraint_params)
