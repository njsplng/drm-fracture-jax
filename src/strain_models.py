"""Strain computations using JAX."""

from functools import partial

import jax.numpy as jnp
from jaxtyping import Array

from utils import timed_jit


@timed_jit
def nodal_displacement_to_strain(
    nodal_displacements: Array, connectivity: Array, B: Array
) -> Array:
    """Calculate the split strain components from nodal displacements.

    Parameters
    ----------
    nodal_displacements : Array
        Nodal displacement values.
    connectivity : Array
        Element connectivity array.
    B : Array
        Strain-displacement matrix.

    Returns
    -------
    strains : Array
        Elemental strains at each Gauss point, shape [E, G, 3].
    """
    # Split the displacements
    x_displacement, y_displacement = nodal_displacements

    # Stack into a single array and reshape
    elemental_displacements_x = x_displacement[connectivity]
    elemental_displacements_y = y_displacement[connectivity]
    elemental_displacements_stacked = jnp.stack(
        (elemental_displacements_x, elemental_displacements_y), axis=2
    )
    elemental_displacements = elemental_displacements_stacked.reshape(
        (elemental_displacements_stacked.shape[0], -1)
    )

    # Compute the elemental strains at each Gauss point
    # [E, G, 3]
    strains = jnp.einsum("efgi,ei->efg", B, elemental_displacements)

    return strains


@partial(timed_jit, static_argnames=("nu", "plane_mode"))
def voigt_strain_to_tensor(strains: Array, nu: float, plane_mode: str) -> Array:
    """Convert the Voigt notation strain vector to a full strain tensor.

    Parameters
    ----------
    strains : Array
        Strain components in Voigt notation.
    nu : float
        Poisson's ratio.
    plane_mode : str
        Either "stress" for plane stress or "strain" for plane strain.

    Returns
    -------
    full_strains : Array
        Full strain tensor, shape [E, G, 3, 3].
    """
    # Compute the out-of-plane strain component
    out_of_plane_strain = (
        -(strains[:, :, 0] + strains[:, :, 1]) * (nu / (1 - nu))
        if plane_mode == "stress"
        else jnp.zeros_like(strains[:, :, 0])
    )

    # Convert to full matrix representation
    num_elements, num_gauss_points, _ = strains.shape
    full_strains = jnp.zeros(
        (num_elements, num_gauss_points, 3, 3),
    )
    # Assign in-plane strain components
    # ε_xx
    full_strains = full_strains.at[:, :, 0, 0].set(strains[:, :, 0])
    # ε_yy
    full_strains = full_strains.at[:, :, 1, 1].set(strains[:, :, 1])
    # 0.5 * ε_xy
    full_strains = full_strains.at[:, :, 0, 1].set(0.5 * strains[:, :, 2])
    # 0.5 * ε_xy
    full_strains = full_strains.at[:, :, 1, 0].set(0.5 * strains[:, :, 2])
    # Assign out-of-plane strain
    # ε_zz
    full_strains = full_strains.at[:, :, 2, 2].set(out_of_plane_strain)

    return full_strains
