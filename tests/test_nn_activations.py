"""Unit tests for activation functions.

Tests for static and trainable tanh/relu activations.
"""

import jax
import jax.numpy as jnp
import pytest
from activations import (
    StaticReLU,
    StaticTanh,
    TrainableReLU,
    TrainableTanh,
    get_activation,
)

# Enable JAX float64 mode
jax.config.update("jax_enable_x64", True)


class TestActivationRegistry:
    """Tests for the activation function registry."""

    def test_activations_registry(self) -> None:
        """Test 6.10: Verify get_activation returns correct classes for tanh/relu × static/trainable, raises for unknown."""
        # Test static tanh
        static_tanh = get_activation("tanh", coeff=1.0, trainable=False)
        assert isinstance(static_tanh, StaticTanh), "Should return StaticTanh instance"

        # Test trainable tanh
        trainable_tanh = get_activation("tanh", coeff=1.0, trainable=True)
        assert isinstance(
            trainable_tanh, TrainableTanh
        ), "Should return TrainableTanh instance"

        # Test static relu
        static_relu = get_activation("relu", coeff=1.0, trainable=False)
        assert isinstance(static_relu, StaticReLU), "Should return StaticReLU instance"

        # Test trainable relu
        trainable_relu = get_activation("relu", coeff=1.0, trainable=True)
        assert isinstance(
            trainable_relu, TrainableReLU
        ), "Should return TrainableReLU instance"

        # Test unknown activation
        with pytest.raises(ValueError, match="No activation registered"):
            get_activation("unknown", coeff=1.0)


class TestActivationForwardPass:
    """Tests for activation function forward passes."""

    def test_activation_forward_pass(self) -> None:
        """Test 6.11: Verify each activation produces output matching activation(coeff * x)."""
        import jax.nn as jnn

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        coeff = 2.0

        # Test static tanh
        static_tanh = get_activation("tanh", coeff=coeff, trainable=False)
        out_tanh = static_tanh(x)
        expected_tanh = jnn.tanh(coeff * x)
        assert jnp.allclose(
            out_tanh, expected_tanh
        ), "StaticTanh output should match tanh(coeff * x)"

        # Test trainable tanh
        trainable_tanh = get_activation("tanh", coeff=coeff, trainable=True)
        out_trainable_tanh = trainable_tanh(x)
        assert jnp.allclose(
            out_trainable_tanh, expected_tanh
        ), "TrainableTanh output should match tanh(coeff * x)"

        # Test static relu
        static_relu = get_activation("relu", coeff=coeff, trainable=False)
        out_relu = static_relu(x)
        expected_relu = jnn.relu(coeff * x)
        assert jnp.allclose(
            out_relu, expected_relu
        ), "StaticReLU output should match relu(coeff * x)"

        # Test trainable relu
        trainable_relu = get_activation("relu", coeff=coeff, trainable=True)
        out_trainable_relu = trainable_relu(x)
        assert jnp.allclose(
            out_trainable_relu, expected_relu
        ), "TrainableReLU output should match relu(coeff * x)"
