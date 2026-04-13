"""Unit tests for linear layers.

Tests for Linear layer and weight initialization schemes.
"""

import jax
import jax.numpy as jnp
import pytest
from linears import (
    Linear,
    get_linear_layer,
    he_fan_in,
    he_fan_out,
    xavier_normal,
    xavier_uniform,
)

# Enable JAX float64 mode
jax.config.update("jax_enable_x64", True)


class TestWeightInitialisation:
    """Tests for weight initialization schemes."""

    def test_linear_layer_weight_initialisation(self) -> None:
        """Test 6.12: Verify weight shapes and statistical properties for xavier_uniform, xavier_normal, he_fan_in, he_fan_out."""
        key = jax.random.PRNGKey(0)
        in_features = 10
        out_features = 20

        # Test xavier_uniform
        layer_xu = Linear(
            in_features, out_features, weight_init_function=xavier_uniform, key=key
        )
        assert layer_xu.weight.shape == (
            out_features,
            in_features,
        ), "Weight shape should be (out_features, in_features)"
        # Xavier uniform: weights in [-lim, lim] where lim = sqrt(6 / (in + out))
        lim = jnp.sqrt(6 / (in_features + out_features))
        assert jnp.all(
            jnp.abs(layer_xu.weight) <= lim + 1e-6
        ), "Xavier uniform weights should be within bounds"

        # Test xavier_normal
        layer_xn = Linear(
            in_features, out_features, weight_init_function=xavier_normal, key=key
        )
        assert layer_xn.weight.shape == (
            out_features,
            in_features,
        ), "Weight shape should be (out_features, in_features)"
        # Xavier normal: std = sqrt(2 / (in + out))
        std = jnp.sqrt(2 / (in_features + out_features))
        # Check that weights are roughly within 3 std (statistical property)
        assert jnp.all(
            jnp.abs(layer_xn.weight) <= 3 * std + 0.1
        ), "Xavier normal weights should be within reasonable bounds"

        # Test he_fan_in
        layer_hi = Linear(
            in_features, out_features, weight_init_function=he_fan_in, key=key
        )
        assert layer_hi.weight.shape == (
            out_features,
            in_features,
        ), "Weight shape should be (out_features, in_features)"
        # He fan-in: std = sqrt(2 / in)
        std_hi = jnp.sqrt(2 / in_features)
        assert jnp.all(
            jnp.abs(layer_hi.weight) <= 3 * std_hi + 0.1
        ), "He fan-in weights should be within reasonable bounds"

        # Test he_fan_out
        layer_ho = Linear(
            in_features, out_features, weight_init_function=he_fan_out, key=key
        )
        assert layer_ho.weight.shape == (
            out_features,
            in_features,
        ), "Weight shape should be (out_features, in_features)"
        # He fan-out: std = sqrt(2 / out)
        std_ho = jnp.sqrt(2 / out_features)
        assert jnp.all(
            jnp.abs(layer_ho.weight) <= 3 * std_ho + 0.1
        ), "He fan-out weights should be within reasonable bounds"


class TestGetLinearLayer:
    """Tests for the get_linear_layer factory function."""

    def test_get_linear_layer_factory(self) -> None:
        """Test 6.14: Verify factory returns Linear and raises for unknown scheme."""
        # Test Linear
        linear_factory = get_linear_layer("xavier_uniform")
        layer = linear_factory(
            in_features=10, out_features=20, key=jax.random.PRNGKey(0)
        )
        assert isinstance(layer, Linear), "Should return Linear instance"

        # Test unknown initialization scheme
        with pytest.raises(ValueError, match="not recognized"):
            get_linear_layer("unknown_scheme")
