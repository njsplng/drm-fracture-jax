"""Unit tests for neural network utilities.

Tests for gradient flow, weight decay, predict_model_output, collect_auxiliary_data,
and calculate_weight_sparsity.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from networks import FNN, ResNet
from utils_nn import (
    calculate_weight_sparsity,
    collect_auxiliary_data,
    predict_model_output,
)

# Enable JAX float64 mode
jax.config.update("jax_enable_x64", True)


class TestGradientFlow:
    """Tests for gradient flow through networks."""

    def test_gradient_flow_through_networks(self) -> None:
        """Test 6.20: For each network type: dummy loss, eqx.filter_value_and_grad, verify all gradient leaves finite."""
        key = jax.random.PRNGKey(0)

        # Test FNN
        fnn = FNN(layers=[2, 16, 16, 3], key=key)

        def dummy_loss_fnn(params):
            x = jnp.ones((5, 2))
            out = jax.vmap(params)(x)
            return jnp.mean(out**2)

        loss, grads = eqx.filter_value_and_grad(dummy_loss_fnn)(fnn)
        assert jnp.isfinite(loss), "FNN loss should be finite"
        # Extract all array leaves from gradients
        grad_arrays = jax.tree_util.tree_leaves(grads)
        for arr in grad_arrays:
            if isinstance(arr, jnp.ndarray):
                assert jnp.all(
                    jnp.isfinite(arr)
                ), f"FNN gradient contains non-finite values: {arr}"

        # Test ResNet
        resnet = ResNet(layers=[2, 16, 16, 16, 16, 3], block_depth=2, key=key)

        def dummy_loss_resnet(params):
            x = jnp.ones((5, 2))
            out = jax.vmap(params)(x)
            return jnp.mean(out**2)

        loss, grads = eqx.filter_value_and_grad(dummy_loss_resnet)(resnet)
        assert jnp.isfinite(loss), "ResNet loss should be finite"
        # Extract all array leaves from gradients
        grad_arrays = jax.tree_util.tree_leaves(grads)
        for arr in grad_arrays:
            if isinstance(arr, jnp.ndarray):
                assert jnp.all(
                    jnp.isfinite(arr)
                ), f"ResNet gradient contains non-finite values: {arr}"


class TestWeightDecay:
    """Tests for weight decay loss."""

    def test_weight_decay_loss(self) -> None:
        """Test 6.21: Construct a minimal model, call loss_weight_decay(0.01), verify finite non-negative scalar."""
        # We need a minimal model with a network
        # Create a simple FNN and wrap it in a minimal structure
        key = jax.random.PRNGKey(0)
        network = FNN(layers=[2, 16, 3], key=key)

        # Create a minimal model-like object with loss_weight_decay
        class MinimalModel(eqx.Module):
            network: eqx.Module

            def loss_weight_decay(self, coefficient: float) -> float:
                # Partition the model into trainable parameters vs. static objects.
                params, _ = eqx.partition(self.network, eqx.is_inexact_array)

                # Build the weight decay term
                decay_tree = jax.tree_util.tree_map(
                    lambda p: jnp.sum(jnp.square(p)), params
                )

                # Zero out any activation parameters
                if hasattr(self.network, "activation_list"):
                    for i, act in enumerate(self.network.activation_list):
                        # Zero out the activation parameters
                        zero_subtree = jax.tree_util.tree_map(lambda _: 0.0, act)
                        decay_tree = eqx.tree_at(
                            lambda n, i=i: n.activation_list[i],
                            decay_tree,
                            replace=zero_subtree,
                        )

                loss = sum(jax.tree_util.tree_leaves(decay_tree))
                return coefficient * loss

        model = MinimalModel(network=network)
        weight_decay_coeff = 0.01

        loss = model.loss_weight_decay(weight_decay_coeff)

        assert jnp.isfinite(loss), "Weight decay loss should be finite"
        assert loss >= 0, "Weight decay loss should be non-negative"
        assert loss.shape == (), "Weight decay loss should be a scalar"


class TestPredictModelOutput:
    """Tests for predict_model_output function."""

    def test_predict_model_output_shapes(self) -> None:
        """Test 6.24: Verify displacement shape (2*N,) and phasefield shape (N,)."""
        # Create a minimal model that outputs (u, v, c)
        key = jax.random.PRNGKey(0)
        network = FNN(layers=[2, 16, 3], key=key)

        class MinimalPredictModel(eqx.Module):
            network: eqx.Module

            def __call__(self, x):
                out = jax.vmap(self.network)(x)
                return out[:, 0], out[:, 1], out[:, 2]

        model = MinimalPredictModel(network=network)
        inp = jnp.ones((10, 2))  # N=10 nodes

        displacement, phasefield = predict_model_output(model, inp)

        # Displacement should be shape (2*N,) = (20,)
        assert displacement.shape == (
            20,
        ), f"Expected displacement shape (20,), got {displacement.shape}"

        # Phasefield should be shape (N,) = (10,)
        assert phasefield.shape == (
            10,
        ), f"Expected phasefield shape (10,), got {phasefield.shape}"


class TestCollectAuxiliaryData:
    """Tests for collect_auxiliary_data function."""

    def test_collect_auxiliary_data(self) -> None:
        """Test 6.25: Verify correct dispatch for FNN, ResNet. Raises for unknown."""
        key = jax.random.PRNGKey(0)

        # Test FNN
        fnn = FNN(layers=[2, 16, 3], key=key)
        aux_fnn = collect_auxiliary_data(fnn)
        assert isinstance(aux_fnn, dict), "FNN auxiliary data should be a dict"

        # Test ResNet
        resnet = ResNet(layers=[2, 16, 16, 16, 16, 3], block_depth=2, key=key)
        aux_resnet = collect_auxiliary_data(resnet)
        assert isinstance(aux_resnet, dict), "ResNet auxiliary data should be a dict"
        assert "alpha" in aux_resnet, "ResNet auxiliary data should contain 'alpha'"

        # Test unknown network type
        class UnknownNetwork(eqx.Module):
            pass

        unknown = UnknownNetwork()
        with pytest.raises(NotImplementedError, match="not implemented"):
            collect_auxiliary_data(unknown)


class TestCalculateWeightSparsity:
    """Tests for calculate_weight_sparsity function."""

    def test_calculate_weight_sparsity(self) -> None:
        """Test 6.26: Verify fractions between 0 and 100, random init has near-100% above 1e-10."""
        key = jax.random.PRNGKey(0)
        network = FNN(layers=[2, 16, 16, 3], key=key)

        sparsity = calculate_weight_sparsity(network, eps=(1e-6, 1e-8, 1e-10))

        # Verify structure
        assert "1e-06" in sparsity, "Should have 1e-06 key"
        assert "1e-08" in sparsity, "Should have 1e-08 key"
        assert "1e-10" in sparsity, "Should have 1e-10 key"

        # Verify fractions are between 0 and 100
        for eps_key, fractions in sparsity.items():
            for frac in fractions:
                assert (
                    0 <= frac <= 100
                ), f"Fraction {frac} for {eps_key} should be between 0 and 100"

        # Random initialization should have near-100% weights above 1e-10
        fractions_1e_10 = sparsity["1e-10"]
        assert all(
            f > 90 for f in fractions_1e_10
        ), f"Random init should have >90% weights above 1e-10, got {fractions_1e_10}"
