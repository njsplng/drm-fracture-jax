"""Linear layer implementations with various weight initialisation schemes.

This module provides a base linear layer class and a random weight
factorization variant, along with multiple weight initialisation
functions (Xavier uniform/normal, He fan-in/fan-out).
"""

import inspect
import sys
from functools import partial
from typing import Callable, Dict, Literal, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


# Collected via the factory so need to tell the linter to ignore these explicitly
# skylos: ignore-start
def xavier_uniform(in_features: int, out_features: int, key: PRNGKeyArray) -> Array:
    """Initialise weights using Xavier uniform distribution.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    key : PRNGKeyArray
        JAX PRNG key for random number generation.

    Returns
    -------
    weights : Array
        Initialised weight array of shape (out_features, in_features).
    """
    lim = jnp.sqrt(6 / (in_features + out_features))
    weights = jax.random.uniform(
        key=key, shape=(out_features, in_features), minval=-lim, maxval=lim
    )
    return weights


def xavier_normal(in_features: int, out_features: int, key: PRNGKeyArray) -> Array:
    """Initialise weights using Xavier normal distribution.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    key : PRNGKeyArray
        JAX PRNG key for random number generation.

    Returns
    -------
    weights : Array
        Initialised weight array of shape (out_features, in_features).
    """
    std = jnp.sqrt(2 / (in_features + out_features))
    weights = jax.random.normal(key=key, shape=(out_features, in_features)) * std
    return weights


def he_fan_in(in_features: int, out_features: int, key: PRNGKeyArray) -> Array:
    """Initialise weights using He fan-in distribution.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    key : PRNGKeyArray
        JAX PRNG key for random number generation.

    Returns
    -------
    weights : Array
        Initialised weight array of shape (out_features, in_features).
    """
    std = jnp.sqrt(2 / in_features)
    weights = jax.random.normal(key=key, shape=(out_features, in_features)) * std
    return weights


def he_fan_out(in_features: int, out_features: int, key: PRNGKeyArray) -> Array:
    """Initialise weights using He fan-out distribution.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    key : PRNGKeyArray
        JAX PRNG key for random number generation.

    Returns
    -------
    weights : Array
        Initialised weight array of shape (out_features, in_features).
    """
    std = jnp.sqrt(2 / out_features)
    weights = jax.random.normal(key=key, shape=(out_features, in_features)) * std
    return weights


# skylos: ignore-end


class Linear(eqx.nn.Linear):
    """Linear layer with custom weight initialisation function.

    Parameters
    ----------
    in_features : int or "scalar"
        Number of input features.
    out_features : int or "scalar"
        Number of output features.
    use_bias : bool, optional
        Whether to include a bias term. Default is True.
    weight_init_function : Callable
        Function to initialise the weight matrix.
    key : PRNGKeyArray
        JAX PRNG key for random number generation.
    """

    # skylos: ignore-start
    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    # skylos: ignore-end

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        *,
        weight_init_function: Callable[..., Array],
        key: PRNGKeyArray,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, use_bias, key=key)
        self.weight = weight_init_function(
            in_features=in_features, out_features=out_features, key=key
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias


_mod = sys.modules[__name__]
builders = {name: func for name, func in inspect.getmembers(_mod, inspect.isfunction)}


def get_linear_layer(
    init_scheme: str,
) -> Callable[..., Linear]:
    """Factory function to create a linear layer class.

    Returns a partially applied linear layer class with the specified
    weight initialisation scheme.

    Parameters
    ----------
    init_scheme : str
        Name of the initialisation scheme (e.g., "xavier_uniform",
        "he_fan_in").

    Returns
    -------
    layer_factory : Callable
        Partially applied linear layer class.

    Raises
    ------
    ValueError
        If init_scheme is not recognized.
    """
    base_class = Linear
    initialisation = builders.get(init_scheme)
    if initialisation is None:
        raise ValueError(f"Initialisation scheme '{init_scheme}' not recognized.")
    return partial(base_class, weight_init_function=initialisation)
