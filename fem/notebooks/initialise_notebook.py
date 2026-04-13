"""Utilities for initializing the JAX environment for FEM notebooks."""

import logging
import os

logging.getLogger("jax").setLevel(logging.ERROR)
os.environ["JAX_PLATFORMS"] = "cpu"

# System imports
import pathlib
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=6, linewidth=800, suppress=True)

# Set up the necessary paths
current_path = pathlib.Path(__file__).parent.parent.resolve()
# Link the fem source
fem_source_path = current_path / "src"
if fem_source_path not in sys.path:
    sys.path.append(str(fem_source_path))

# Link the generic source
generic_source_path = current_path.parent / "src"
if generic_source_path not in sys.path:
    sys.path.append(str(generic_source_path))
