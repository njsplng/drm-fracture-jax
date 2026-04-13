"""Initialise the JAX environment and imports for the notebook.

Configure JAX settings and set up import paths for easy importing.
"""

import logging

logging.getLogger("jax").setLevel(logging.ERROR)

# System imports
import pathlib
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=8, linewidth=800, suppress=True)

# Set up the necessary paths for imports
current_path = pathlib.Path(__file__).parent.parent.resolve()

# Get the NN-specific source path and add it to sys.path if not already present
nn_source_path = current_path / "src"
if nn_source_path not in sys.path:
    sys.path.append(str(nn_source_path))

# Get the generic source path and add it to sys.path if not already present
generic_source_path = current_path.parent / "src"
if generic_source_path not in sys.path:
    sys.path.append(str(generic_source_path))


import logging

logging.getLogger("jax").setLevel(logging.ERROR)

import pathlib
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=8, linewidth=800, suppress=True)

# Set up the necessary paths for imports
current_path = pathlib.Path(__file__).parent.resolve()

# Get the NN-specific source path and add it to sys.path if not already present
nn_source_path = current_path / "src"
if nn_source_path not in sys.path:
    sys.path.append(str(nn_source_path))

# Get the generic source path and add it to sys.path if not already present
generic_source_path = current_path.parent / "src"
if generic_source_path not in sys.path:
    sys.path.append(str(generic_source_path))
