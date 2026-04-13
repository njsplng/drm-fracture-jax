"""Activation functions with static and trainable coefficients.

Provides tanh and ReLU activation functions with optional trainable
scaling coefficients.
"""

import inspect
import sys
from typing import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array


# Collected via the factory so need to tell the linter to ignore these explicitly
# skylos: ignore-start
class StaticTanh(eqx.Module):
    """Static tanh activation function with a scaling coefficient.

    Parameters
    ----------
    coeff : float
        Scaling coefficient applied to the input before tanh.
    """

    coeff: Array = eqx.field(static=True)
    activation: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(self, coeff: float) -> None:
        self.activation: Callable[[Array], Array] = jnn.tanh
        self.coeff = jnp.array(coeff)

    def __call__(self, x: Array) -> Array:
        """Apply scaled tanh activation to input."""
        return self.activation(self.coeff * x)


class TrainableTanh(StaticTanh):
    """Trainable tanh activation function with a scaling coefficient.

    Parameters
    ----------
    coeff : float
        Initial scaling coefficient (trainable).
    """

    coeff: Array = eqx.field()

    def __init__(self, coeff: float) -> None:
        self.activation: Callable[[Array], Array] = jnn.tanh
        self.coeff = jnp.array(coeff)


class StaticReLU(eqx.Module):
    """Static ReLU activation function with a scaling coefficient.

    Parameters
    ----------
    coeff : float
        Scaling coefficient applied to the input before ReLU.
    """

    coeff: Array = eqx.field(static=True)
    activation: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(self, coeff: float) -> None:
        self.activation: Callable[[Array], Array] = jnn.relu
        self.coeff = jnp.array(coeff)

    def __call__(self, x: Array) -> Array:
        """Apply scaled ReLU activation to input."""
        activation: Callable[[Array], Array] = jnn.relu
        return activation(self.coeff * x)


class TrainableReLU(StaticReLU):
    """Trainable ReLU activation function with a scaling coefficient.

    Parameters
    ----------
    coeff : float
        Initial scaling coefficient (trainable).
    """

    coeff: Array = eqx.field()

    def __init__(self, coeff: float) -> None:
        self.activation: Callable[[Array], Array] = jnn.relu
        self.coeff = jnp.array(coeff)


# skylos: ignore-end

_mod = sys.modules[__name__]

ACTIVATION_REGISTRY: dict[tuple[str, bool], type[eqx.Module]] = {}
for name, obj in list(vars(_mod).items()):
    # pick out only classes that subclass eqx.Module
    if inspect.isclass(obj) and issubclass(obj, eqx.Module):
        # drop the "Static"/"Trainable" prefix and lower-case to get the key
        if name.startswith(("Static", "Trainable")):
            trainable = name.startswith("Trainable")
            base = name[len("Trainable") :] if trainable else name[len("Static") :]
            ACTIVATION_REGISTRY[(base.lower(), trainable)] = obj


def get_activation(
    name: str, coeff: float, trainable: bool = False
) -> type[eqx.Module]:
    """Return an activation function class based on name and trainable status.

    Look up the activation class in the registry using the provided
    name and trainable flag, then instantiate it with the given
    coefficient.

    Parameters
    ----------
    name : str
        Name of the activation function (e.g., 'tanh', 'relu').
    coeff : float
        Scaling coefficient for the activation.
    trainable : bool, optional
        Whether the coefficient should be trainable. Default is False.

    Returns
    -------
    type[eqx.Module]
        Instantiated activation function class.

    Raises
    ------
    ValueError
        If no activation is registered for the given name and trainable
        status.
    RuntimeError
        If the registered class has an unexpected constructor signature.
    """
    key: tuple[str, bool] = (name.lower(), trainable)
    try:
        cls: type[eqx.Module] = ACTIVATION_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"No activation registered for {name!r} with trainable={trainable}"
        )
    # Inspect the signature if you care about extra validation:
    sig = inspect.signature(cls)
    if len(sig.parameters) != 1 or "coeff" not in sig.parameters:
        raise RuntimeError(f"{cls.__name__} has an unexpected constructor")
    return cls(coeff)
