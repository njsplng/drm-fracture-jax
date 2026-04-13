"""Neural network architectures for scientific machine learning."""

import inspect
import logging
import sys
from functools import partial
from typing import Callable, Dict, List, Optional, Type

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from linears import get_linear_layer

from utils import timed_filter_jit


class FNN(eqx.Module):
    """A simple feedforward neural network (FNN).

    Constructs a fully connected network with configurable activation
    functions and weight initialization schemes.

    Parameters
    ----------
    layers : List[int]
        List of layer sizes, including input and output dimensions.
    activation : Callable[[Array], Array], optional
        Activation function to use. Default is tanh.
    weight_init_type : str, optional
        Weight initialization method. Default is "xavier_uniform".
    activation_function_global : bool, optional
        If True, use a single global activation. Default is False.
    key : PRNGKeyArray
        JAX PRNG key for initialization.
    """

    activation_list: List[Callable[[Array], Array]] = eqx.field()
    linears: List[eqx.Module] = eqx.field()
    weight_init_type: str = eqx.field(static=True, default="xavier_uniform")
    activation_function_global: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        layers: List[int],
        activation: Callable[[Array], Array] = jax.nn.tanh,
        weight_init_type: str = "xavier_uniform",
        activation_function_global: bool = False,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ) -> None:
        """Initialize the FNN."""
        keys = jax.random.split(key, len(layers) - 1)

        # Determine the linear layer type.
        linear_type = get_linear_layer(
            weight_init_type
        )

        # Build the list of linear layers.
        self.linears = [
            linear_type(
                in_features=layers[i],
                out_features=layers[i + 1],
                use_bias=True,
                key=keys[i],
            )
            for i in range(len(layers) - 1)
        ]

        self.activation_function_global = activation_function_global
        self.activation_list = [activation] * (len(layers) - 2)

        if self.activation_function_global:
            self.activation_list = self.activation_list[:1]

    @timed_filter_jit
    def __call__(self, x: Array) -> Array:
        """Compute the forward pass through the network.

        Parameters
        ----------
        x : Array
            Input array of shape (batch_size, input_dim).

        Returns
        -------
        Array
            Output array of shape (batch_size, output_dim).
        """
        for i, layer in enumerate(self.linears[:-1]):
            if self.activation_function_global:
                x = self.activation_list[0](layer(x))
            else:
                x = self.activation_list[i](layer(x))
        return self.linears[-1](x)


class RandomFourierFeatures(eqx.Module):
    """Fixed random Fourier feature embedding Phi(x) = [cos(Bx), sin(Bx)].

    Parameters
    ----------
    input_dim : int
        Dimension of the input space.
    num_features : int
        Number of random features to generate.
    scale : float
        Scale parameter for the Gaussian distribution of B.
    key : PRNGKeyArray
        JAX PRNG key for initialization.
    """

    B: jnp.ndarray = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        scale: float,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Initialize random Fourier features."""
        logging.info(
            f"Initialising RFF with input {input_dim}, "
            f"{num_features} features and scale {scale}."
        )
        self.scale = scale
        # Sample B from N(0, scale^2)
        self.B = jax.random.normal(key, shape=(num_features, input_dim)) * scale

    @timed_filter_jit
    def __call__(self, x: Array) -> Array:
        """Compute the Fourier feature embedding.

        Parameters
        ----------
        x : Array
            Input array of shape (batch_size, input_dim).

        Returns
        -------
        Array
            Embedded array of shape (batch_size, 2 * num_features).
        """
        projection = x @ self.B.T
        return jnp.concatenate([jnp.cos(projection), jnp.sin(projection)], axis=-1)


class ResNetBlock(eqx.Module):
    """A single ResNet block.

    Parameters
    ----------
    width : int
        Hidden layer width.
    key : PRNGKeyArray
        JAX PRNG key for initialization.
    weight_init_type : str, optional
        Weight initialization method. Default is "xavier_uniform".
    init_alpha : float, optional
        Initial value for alpha blending. Default is 0.0.
    block_depth : int, optional
        Number of layers in the block. Default is 2.
    """

    linear_list: List[eqx.Module] = eqx.field()
    alpha: jnp.ndarray

    def __init__(
        self,
        width: int,
        *,
        key: PRNGKeyArray,
        weight_init_type: str = "xavier_uniform",
        init_alpha: float = 0.0,
        block_depth: int = 2,
    ) -> None:
        """Initialize ResNetBlock."""
        # Generate the keys for the whole block here
        block_keys = jax.random.split(key, block_depth)

        # Determine the linear layer type.
        linear_type = get_linear_layer(
            weight_init_type
        )

        # Generate the linear list and the alpha list
        self.linear_list = [
            linear_type(width, width, use_bias=True, key=entry) for entry in block_keys
        ]
        self.alpha = jnp.array(init_alpha, dtype=jnp.float64)

    @timed_filter_jit
    def __call__(self, x: Array, activation: Callable[[Array], Array]) -> Array:
        """Compute the forward pass through the ResNetBlock.

        Parameters
        ----------
        x : Array
            Input array.
        activation : Callable[[Array], Array]
            Activation function to apply.

        Returns
        -------
        Array
            Output array with residual connection applied.
        """
        h = x
        for entry in self.linear_list:
            h = activation(entry(h))
        return h * self.alpha + x


class ResNet(eqx.Module):
    """ResNet implementation.

    Uses ReZero-style learnable alpha blending in each block.

    Parameters
    ----------
    layers : List[int]
        List of layer sizes, including input and output dimensions.
    activation : Callable[[Array], Array], optional
        Activation function. Default is tanh.
    weight_init_type : str, optional
        Weight initialization method. Default is "xavier_uniform".
    key : PRNGKeyArray
        JAX PRNG key for initialization.
    block_depth : int, optional
        Number of layers per block. Default is 2.
    activation_function_global : bool, optional
        If True, use global activation. Default is False.
    init_alpha : float, optional
        Initial value for alpha blending. Default is 0.0.
    rff_dict : Dict[str, object], optional
        RFF configuration dictionary.
    """

    blocks: List[ResNetBlock]
    head_in: eqx.Module
    head_out: eqx.Module
    weight_init_type: str = eqx.field(static=True, default="xavier_uniform")
    activation_function_global: bool = eqx.field(static=True, default=True)
    activation_list: List[Callable[[Array], Array]] = eqx.field()
    rff: Optional[RandomFourierFeatures] = None

    def __init__(
        self,
        layers: List[int],
        activation: Callable[[Array], Array] = jax.nn.tanh,
        weight_init_type: str = "xavier_uniform",
        *,
        key: PRNGKeyArray,
        block_depth: int = 2,
        activation_function_global: bool = False,
        init_alpha: float = 0.0,
        rff_dict: Optional[Dict[str, object]] = None,
        **kwargs,
    ) -> None:
        """Initialize the adaptive net."""
        # Extract the layer info
        in_dim = layers[0]
        out_dim = layers[-1]
        depth = len(layers) - 2
        width = layers[1]

        assert (
            depth % block_depth == 0
        ), f"ResNet{block_depth} requires depth to be multiple of {block_depth}."
        depth //= block_depth

        # Compatibility fields
        self.weight_init_type = weight_init_type
        self.activation_function_global = activation_function_global
        self.activation_list = [activation]
        self.activation_list *= 1 if self.activation_function_global else depth

        # Generate the necessary keys
        k_blocks, k_head_in, k_head_out, k_rff = jax.random.split(key, 4)
        block_keys = jax.random.split(k_blocks, depth)

        # Determine the linear layer type.
        linear_type = get_linear_layer(
            weight_init_type
        )

        # Build the RFF if enabled and the head_in
        if rff_dict is not None and rff_dict.get("enabled"):
            self.rff = RandomFourierFeatures(
                input_dim=in_dim,
                num_features=rff_dict.get("features"),
                scale=rff_dict.get("scale"),
                key=k_rff,
            )
            phi_dim = 2 * rff_dict.get("features")
            self.head_in = linear_type(phi_dim, width, use_bias=True, key=k_head_in)
        else:
            self.rff = None
            # Input head x -> width
            self.head_in = linear_type(in_dim, width, use_bias=True, key=k_head_in)

        # Blocks
        self.blocks = [
            ResNetBlock(
                width=width,
                key=block_keys[i],
                weight_init_type=weight_init_type,
                init_alpha=init_alpha,
                block_depth=block_depth,
            )
            for i in range(depth)
        ]

        # Output head width -> out_dim
        self.head_out = linear_type(width, out_dim, use_bias=True, key=k_head_out)

    @timed_filter_jit
    def __call__(self, x: Array) -> Array:
        """Compute the forward pass through the ResNet.

        Parameters
        ----------
        x : Array
            Input array of shape (batch_size, input_dim).

        Returns
        -------
        Array
            Output array of shape (batch_size, output_dim).
        """
        if self.rff is not None:
            x = self.rff(x)

        h = self.head_in(x)

        # Blocks
        for i, block in enumerate(self.blocks):
            h = (
                block(h, self.activation_list[0])
                if self.activation_function_global
                else block(h, self.activation_list[i])
            )

        return self.head_out(h)


_mod = sys.modules[__name__]

NETWORK_REGISTRY: Dict[str, Type[eqx.Module]] = {}
for name, obj in list(vars(_mod).items()):
    # pick out only classes that subclass eqx.Module
    if inspect.isclass(obj) and issubclass(obj, eqx.Module):
        # drop the "Static"/"Trainable" prefix and lower-case to get the key
        NETWORK_REGISTRY[name.lower()] = obj


def get_network(name: str) -> Optional[Type[eqx.Module]]:
    """Retrieve a network class from the registry.

    Parameters
    ----------
    name : str
        Name of the network (e.g., "fnn", "resnet").
        For ResNet, can specify block depth as "resnet2", "resnet3", etc.

    Returns
    -------
    Optional[Type[eqx.Module]]
        The network class, or a partial with block_depth set for ResNet.

    Raises
    ------
    ValueError
        If the network name is not supported.
    """
    # In case of ResNet, treat it a bit
    blocks = None
    if name.startswith("resnet"):
        blocks = int(name[len("resnet") :])
        name = "resnet"

    # Match the network name
    out_class = NETWORK_REGISTRY.get(name.lower())
    if out_class is None:
        raise ValueError(f'Network type "{name}" is not supported.')

    if blocks is not None:
        return partial(out_class, block_depth=blocks)
    return out_class
