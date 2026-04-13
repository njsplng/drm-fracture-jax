"""Unit tests for neural network architectures.

Tests for FNN, ResNet, and RandomFourierFeatures.
"""

import jax
import jax.numpy as jnp
import pytest
from networks import FNN, RandomFourierFeatures, ResNet, get_network

# Enable JAX float64 mode
jax.config.update("jax_enable_x64", True)


class TestFNN:
    """Tests for the FNN architecture."""

    def test_fnn_construction_and_forward_pass(self) -> None:
        """Test 6.1: Build FNN with layers [2, 32, 32, 3], verify output shape (N, 3), finite values."""
        key = jax.random.PRNGKey(0)
        layers = [2, 32, 32, 3]

        fnn = FNN(layers=layers, key=key)

        # Forward pass with single input (network expects 1D input, use vmap for batches)
        x = jnp.ones((2,))
        out = fnn(x)

        assert out.shape == (3,), f"Expected shape (3,), got {out.shape}"
        assert jnp.all(jnp.isfinite(out)), "Output contains non-finite values"

        # Test with batched input using vmap
        x_batch = jnp.ones((10, 2))
        out_batch = jax.vmap(fnn)(x_batch)
        assert out_batch.shape == (
            10,
            3,
        ), f"Expected batch shape (10, 3), got {out_batch.shape}"
        assert jnp.all(
            jnp.isfinite(out_batch)
        ), "Batch output contains non-finite values"

    def test_fnn_global_vs_per_layer_activation(self) -> None:
        """Test 6.2: Verify global has len(activation_list) == 1, per-layer has len == N-2."""
        key = jax.random.PRNGKey(0)
        layers = [2, 16, 16, 16, 3]  # 3 hidden layers

        # Global activation
        fnn_global = FNN(layers=layers, activation_function_global=True, key=key)
        assert (
            len(fnn_global.activation_list) == 1
        ), f"Global activation should have 1 activation, got {len(fnn_global.activation_list)}"

        # Per-layer activation
        fnn_per_layer = FNN(layers=layers, activation_function_global=False, key=key)
        expected_len = len(layers) - 2  # N - 2 = 3 hidden layers
        assert (
            len(fnn_per_layer.activation_list) == expected_len
        ), f"Per-layer activation should have {expected_len} activations, got {len(fnn_per_layer.activation_list)}"


class TestResNet:
    """Tests for the ResNet architecture."""

    def test_resnet_construction_and_forward_pass(self) -> None:
        """Test 6.5: Build ResNet with block_depth=2 and block_depth=3, verify both produce correct shapes."""
        key = jax.random.PRNGKey(0)

        # ResNet2: depth must be divisible by 2
        layers_2 = [2, 16, 16, 16, 16, 3]  # 4 hidden layers (divisible by 2)
        resnet2 = ResNet(layers=layers_2, block_depth=2, key=key)
        x = jnp.ones((2,))
        out2 = resnet2(x)
        assert out2.shape == (3,), f"ResNet2 expected shape (3,), got {out2.shape}"
        assert jnp.all(jnp.isfinite(out2)), "ResNet2 output contains non-finite values"

        # Test with batched input using vmap
        x_batch = jnp.ones((5, 2))
        out2_batch = jax.vmap(resnet2)(x_batch)
        assert out2_batch.shape == (
            5,
            3,
        ), f"ResNet2 expected batch shape (5, 3), got {out2_batch.shape}"

        # ResNet3: depth must be divisible by 3
        layers_3 = [2, 16, 16, 16, 16, 16, 16, 3]  # 6 hidden layers (divisible by 3)
        resnet3 = ResNet(layers=layers_3, block_depth=3, key=key)
        out3 = resnet3(x)
        assert out3.shape == (3,), f"ResNet3 expected shape (3,), got {out3.shape}"
        assert jnp.all(jnp.isfinite(out3)), "ResNet3 output contains non-finite values"

        # Test with batched input using vmap
        out3_batch = jax.vmap(resnet3)(x_batch)
        assert out3_batch.shape == (
            5,
            3,
        ), f"ResNet3 expected batch shape (5, 3), got {out3_batch.shape}"

    def test_resnet_depth_validation(self) -> None:
        """Test 6.6: Verify AssertionError for depth not divisible by block_depth."""
        key = jax.random.PRNGKey(0)

        # ResNet2 with 3 hidden layers (not divisible by 2)
        layers = [2, 16, 16, 16, 3]  # 3 hidden layers

        with pytest.raises(AssertionError) as exc_info:
            ResNet(layers=layers, block_depth=2, key=key)
        assert "multiple of 2" in str(exc_info.value)

    def test_resnet_with_rff(self) -> None:
        """Test 6.7: Build with RFF enabled, verify self.rff is not None. Build without, verify None."""
        key = jax.random.PRNGKey(0)
        layers = [2, 16, 16, 16, 16, 3]  # 4 hidden layers

        # With RFF
        rff_dict = {"enabled": True, "features": 32, "scale": 1.0}
        resnet_with_rff = ResNet(
            layers=layers, block_depth=2, rff_dict=rff_dict, key=key
        )
        assert resnet_with_rff.rff is not None, "RFF should not be None when enabled"

        # Without RFF
        resnet_without_rff = ResNet(
            layers=layers, block_depth=2, rff_dict=None, key=key
        )
        assert resnet_without_rff.rff is None, "RFF should be None when not enabled"


class TestRandomFourierFeatures:
    """Tests for RandomFourierFeatures."""

    def test_random_fourier_features(self) -> None:
        """Test 6.8: Input (2,) -> output (128,), bounded in [-1, 1]."""
        key = jax.random.PRNGKey(0)
        input_dim = 2
        num_features = 64  # Output will be 2 * num_features = 128
        scale = 1.0

        rff = RandomFourierFeatures(
            input_dim=input_dim, num_features=num_features, scale=scale, key=key
        )

        # Forward pass with single input
        x = jnp.ones((2,))
        out = rff(x)

        # Output shape should be (2 * num_features,)
        expected_shape = (2 * num_features,)
        assert (
            out.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {out.shape}"

        # Values should be bounded in [-1, 1] (cos and sin outputs)
        assert jnp.all(out >= -1.0), "Output values should be >= -1"
        assert jnp.all(out <= 1.0), "Output values should be <= 1"

        # Test with batched input using vmap
        x_batch = jnp.ones((10, 2))
        out_batch = jax.vmap(rff)(x_batch)
        assert out_batch.shape == (
            10,
            2 * num_features,
        ), f"Expected batch shape (10, {2 * num_features}), got {out_batch.shape}"


class TestGetNetwork:
    """Tests for the get_network registry function."""

    def test_get_network_registry(self) -> None:
        """Test 6.9: Verify get_network resolves 'fnn', 'resnet2', 'resnet3', and raises for unknown."""
        # Test valid network types
        fnn_class = get_network("fnn")
        assert fnn_class == FNN, "get_network('fnn') should return FNN"

        resnet2_class = get_network("resnet2")
        # resnet2 returns a partial with block_depth=2
        assert (
            resnet2_class.keywords.get("block_depth") == 2
        ), "get_network('resnet2') should have block_depth=2"

        resnet3_class = get_network("resnet3")
        assert (
            resnet3_class.keywords.get("block_depth") == 3
        ), "get_network('resnet3') should have block_depth=3"

        # Test unknown network type
        with pytest.raises(ValueError, match="not supported"):
            get_network("unknown_network")
