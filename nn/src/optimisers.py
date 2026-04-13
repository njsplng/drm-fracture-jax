"""Optimisers module for parsing and building various optimisers."""

import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import optax
from soap_jax import soap
from ss_broyden import OptimistixAsOptax, SSBroydenArmijo

from utils import StrictDict


# -----------------------------------------------------------
# Build and retrieve optimisers
# -----------------------------------------------------------
def build_adam(params: Dict[str, object]) -> Tuple[optax.GradientTransformation, int]:
    """
    Build Adam.
    """
    return generic_optimiser_build(params, optax.adam)


def build_adamw(params: Dict[str, object]) -> Tuple[optax.GradientTransformation, int]:
    """
    Build AdamW.
    """
    return generic_optimiser_build(params, optax.adamw)


def build_radam(params: Dict[str, object]) -> Tuple[optax.GradientTransformation, int]:
    """
    Build RAdam.
    """
    return generic_optimiser_build(params, optax.radam)


def build_sgd(params: Dict[str, object]) -> Tuple[optax.GradientTransformation, int]:
    """
    Build SGD.
    """
    return generic_optimiser_build(params, optax.sgd)


def build_rprop(params: Dict[str, object]) -> Tuple[optax.GradientTransformation, int]:
    """
    Build RPROP.
    """
    return generic_optimiser_build(params, optax.rprop)


def build_soap(params: Dict[str, object]) -> Tuple[optax.GradientTransformation, int]:
    """
    Build SOAP.
    """
    return generic_optimiser_build(params, soap)


def build_lbfgs(params: Dict[str, object]) -> Tuple[optax.GradientTransformation, int]:
    """
    Build LBFGS.
    """
    return generic_optimiser_build(params, optax.lbfgs)


def build_ssbroydenarmijo(
    params: Dict[str, object]
) -> Tuple[optax.GradientTransformation, int]:
    """
    Build a second order Broyden-family optimiser with Armijo backtracking line search.
    """
    solver = SSBroydenArmijo(
        bt_alpha_init=params.get("bt_alpha_init", 1.0),
        bt_c1=params.get("bt_c1", 1e-4),
        bt_beta=params.get("bt_beta", 0.5),
        bt_max_steps=params.get("bt_max_steps", 50),
        memory=params.get("memory_size", 20),
    )
    optimiser = OptimistixAsOptax(solver)
    return optimiser, params["number_of_epochs"]


# -----------------------------------------------------------
# Helpers/public API exposure
# -----------------------------------------------------------
# Consumed by the factory so need to tell linter to not flag as unused
# skylos: ignore-start
def generic_optimiser_build(
    input_dict: Dict[str, object],
    optimiser_constructor: object,
) -> Tuple[optax.GradientTransformation, int]:
    """
    Build a generic optimiser based on the provided input dictionary.
    """

    def preprocess_optimiser_input(
        input_dict: Dict[str, object]
    ) -> Tuple[Dict[str, object], int, Dict[str, object]]:
        """
        Preprocess the input dictionary to separate learning rate parameters and other parameters.
        """
        # Extract and remove the number of epochs
        n_epochs = input_dict["number_of_epochs"]
        input_dict.pop("number_of_epochs")

        # Clear the optimiser type
        optimiser_type = input_dict.get("optimiser", None)
        if optimiser_type is not None:
            input_dict.pop("optimiser")

        # Extract learning rate parameters
        learning_rate_params = {}
        for key, value in list(input_dict.items()):
            if key.startswith("learning_rate_"):
                learning_rate_params[key[len("learning_rate_") :]] = value
                input_dict.pop(key)

        # Check special case of fixed learning rate
        if "learning_rate" in input_dict.keys():
            learning_rate_params["rate"] = input_dict["learning_rate"]
            input_dict.pop("learning_rate", None)

        return learning_rate_params, n_epochs, input_dict

    def form_learning_schedule(
        learning_rate_params: Dict[str, object],
        n_epochs: int,
    ) -> optax.Schedule:
        """
        Form a learning rate schedule based on the provided parameters.
        """
        # Wrap for easier indexing
        params_dict = StrictDict(learning_rate_params)

        # Return scalar learning rate if provided
        if "rate" in learning_rate_params:
            return learning_rate_params["rate"]

        # Test if the learning rate type is provided
        test_type = learning_rate_params.get("type", None)
        if test_type is None:
            logging.warning("No learning rate type provided, defaulting to linear.")

        # Test if the transition is provided
        test_transition = learning_rate_params.get("transition", None)
        if test_transition is None:
            logging.warning(
                "No learning rate transition provided, defaulting to 0 (no transition)."
            )
            params_dict.transition = 0

        # Determine the schedule type and create the corresponding schedule
        match learning_rate_params.get("type", "linear"):
            case "linear":
                return optax.linear_schedule(
                    init_value=params_dict.start,
                    end_value=params_dict.end,
                    transition_steps=n_epochs - params_dict.transition,
                    transition_begin=params_dict.transition,
                )
            case "constant":
                return optax.constant_schedule(params_dict.start)
            case "cosine_decay":
                return optax.cosine_decay_schedule(
                    init_value=params_dict.start,
                    decay_steps=n_epochs,
                    exponent=params_dict.get("exponent", 1.0),
                    alpha=params_dict.get("alpha", 0.0),
                )
            case "exponential_decay":
                return optax.exponential_decay(
                    init_value=params_dict.start,
                    transition_steps=n_epochs - params_dict.transition,
                    transition_begin=params_dict.transition,
                    end_value=params_dict.end,
                    decay_rate=params_dict.get("decay_rate", 0.9),
                    staircase=params_dict.get("staircase", False),
                )

    # Preprocess the input dictionary
    learning_rate_params, n_epochs, optimiser_params = preprocess_optimiser_input(
        input_dict
    )

    # Form the learning rate schedule
    learning_rate_schedule = form_learning_schedule(learning_rate_params, n_epochs)

    # Form the optimiser
    return (
        optimiser_constructor(learning_rate=learning_rate_schedule, **optimiser_params),
        n_epochs,
    )


# skylos: ignore-end


_mod = sys.modules[__name__]
builders = {
    name[len("build_") :]: func
    for name, func in inspect.getmembers(_mod, inspect.isfunction)
    if name.startswith("build_")
}


def get_optimiser(name: str) -> Tuple[optax.GradientTransformation, int]:
    """
    Given an optimiser name, return the corresponding optimiser and number of epochs.
    """
    # Extract the optimiser information
    current_path = Path(__file__).parent.parent.resolve()
    target_path = current_path.parent / "input" / "optimisers" / name
    with open(str(target_path) + ".json", "r") as file:
        parameter_dict = json.load(file)

    # If the name includes a subpath, extract the optimiser name from the last part of the path
    builder_name = name.split("/")[-1].split("_")[0].lower()

    # Try to match the key and return the parameters
    try:
        return builders[builder_name](parameter_dict)
    except KeyError:
        raise ValueError(
            f"Optimiser {name.split('_')[0]} not found. Available optimisers: {list(builders.keys())}"
        )
