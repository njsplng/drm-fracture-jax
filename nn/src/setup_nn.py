"""Wrappers for neural network setup utilities.

Provide helper functions for constructing neural networks, displacement
vectors, and phasefield constraints.
"""

from typing import Callable, Dict

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from activations import get_activation
from networks import get_network


def construct_displacement(displacement_parameters: Dict[str, object]) -> jnp.ndarray:
    """Construct incremental displacements from parameters.

    Create a concatenated array of coarse and fine displacement values
    using linear spacing.

    Parameters
    ----------
    displacement_parameters : Dict[str, object]
        Dictionary containing displacement configuration including
        start, end, coarse_end, and increment counts.

    Returns
    -------
    incremental_displacements : jnp.ndarray
        Combined displacement array with coarse and fine increments.
    """
    displacements_coarse = jnp.linspace(
        displacement_parameters["start"],
        displacement_parameters["coarse_end"],
        displacement_parameters["increments_coarse"] + 1,
    )
    # Can't train on zero displacement, the energy will diverge.
    displacements_coarse = displacements_coarse.at[0].set(1e-12)
    displacements_fine = jnp.linspace(
        displacement_parameters["coarse_end"],
        displacement_parameters["end"],
        displacement_parameters["increments_fine"] + 1,
    )

    # Form the overall displacements
    incremental_displacements = jnp.concatenate(
        (displacements_coarse, displacements_fine[1:])
    )

    return incremental_displacements


def construct_network(
    network_parameters: Dict[str, object],
    seed: int = 0,
) -> object:
    """Construct a neural network from configuration parameters.

    Initialize a neural network with specified architecture, activation
    function, and weight initialization scheme.

    Parameters
    ----------
    network_parameters : Dict[str, object]
        Dictionary containing network configuration including layer sizes,
        activation settings, and initialization parameters.
    seed : int, optional
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    network : object
        Initialized neural network instance.
    """
    layers = (
        [network_parameters["input_size"]]
        + [network_parameters["hidden_size"]] * network_parameters["hidden_count"]
        + [network_parameters["output_size"]]
    )

    # Get the network initialiser and activation function
    network_initialiser = get_network(network_parameters["type"])

    # Get the activation function
    activation = get_activation(
        name=network_parameters["activation"]["function"],
        coeff=network_parameters["activation"]["initial_coefficient"],
        trainable=network_parameters["activation"]["trainable"],
    )

    # Construct the network
    network = network_initialiser(
        layers=layers,
        activation=activation,
        weight_init_type=network_parameters["weight_initialisation_type"],
        activation_function_global=network_parameters["activation"]["trainable_global"],
        key=jax.random.PRNGKey(seed),
        rff_features=network_parameters["rff"]["features"],
        rff_scale=network_parameters["rff"]["scale"],
        rff_dict=network_parameters["rff"],
    )

    return network


def construct_pf_constraint(
    pf_constraint_parameters: Dict[str, object],
) -> Callable[..., object]:
    """Construct a phasefield constraint function.

    Return either a numerical NonsmoothSigmoid constraint or the
    analytical sigmoid function based on configuration.

    Parameters
    ----------
    pf_constraint_parameters : Dict[str, object]
        Dictionary specifying constraint type and parameters.

    Returns
    -------
    phasefield_constraint : Callable[..., object]
        The constructed constraint function.

    Raises
    ------
    ValueError
        If an unknown constraint type is specified.
    """
    if pf_constraint_parameters["type"] == "numerical":
        phasefield_constraint = NonsmoothSigmoid(
            coeff=pf_constraint_parameters["numerical_coefficient"],
            support=pf_constraint_parameters["numerical_support"],
            offset=pf_constraint_parameters["numerical_offset"],
        )
    # Set analytical phasefield constraint function
    elif pf_constraint_parameters["type"] == "analytical":
        phasefield_constraint = jnn.sigmoid
    # Raise error
    else:
        raise ValueError(
            f"Unknown phasefield constraint type: {pf_constraint_parameters['type']}"
        )
    return phasefield_constraint


class NonsmoothSigmoid(eqx.Module):
    """Numerical sigmoid for capping function values.

    A continuous piecewise linear function transitioning from 0 to 1
    over the interval (-support, support), with linear extrapolation
    outside this region.

    Parameters
    ----------
    coeff : float
        Slope for linear regions outside the support interval.
    support : float
        Half-width of the transition region.
    offset : float, optional
        Horizontal offset applied to input. Default is 0.0.
    """

    support: float = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    offset: float = eqx.field(static=True)

    def __init__(self, coeff: float, support: float, offset: float = 0.0) -> None:
        """Initialize the nonsmooth sigmoid parameters."""
        self.coeff = coeff
        self.support = support
        self.offset = offset

    @jax.jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the nonsmooth sigmoid transformation."""
        s = self.support
        c = self.coeff
        # Add the offset to the input
        x += self.offset
        # Use jnp.where to handle piecewise conditions
        result = jnp.where(
            x < -s,
            c * (x + s),  # For x < -support
            jnp.where(
                x > s,
                c * (x - s) + 1.0,  # For x > support
                (x / (2.0 * s)) + 0.5,  # For -support <= x <= support
            ),
        )
        return result
