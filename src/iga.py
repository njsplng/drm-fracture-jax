"""Isogeometric analysis (IGA) related functions."""

from functools import partial

import jax.numpy as jnp
from jaxtyping import Array

from utils import timed_jit


def find_span(
    degree: int,
    knot_vector: Array,
    num_ctrlpts: int,
    u: float,
) -> int:
    """Find the span index for a given parametric coordinate.

    Locate the index i such that knot_vector[i] <= u < knot_vector[i+1].

    Parameters
    ----------
    degree : int
        Polynomial degree of the B-spline basis.
    knot_vector : Array
        Knot vector defining the B-spline basis.
    num_ctrlpts : int
        Number of control points.
    u : float
        Parametric coordinate to locate.

    Returns
    -------
    int
        Span index i such that knot_vector[i] <= u < knot_vector[i+1].
    """
    i = jnp.searchsorted(knot_vector, u, side="right") - 1
    return jnp.clip(i, degree, num_ctrlpts - 1)


@partial(timed_jit, static_argnames=("p"))
def greville_abscissae(
    U: Array,
    p: int,
) -> Array:
    """Compute the Greville abscissae for a knot vector.

    The Greville abscissae are the averages of p consecutive knot values
    and are commonly used as quadrature points in isogeometric analysis.

    Parameters
    ----------
    U : Array
        Knot vector defining the B-spline basis.
    p : int
        Polynomial degree of the B-spline basis.

    Returns
    -------
    Array
        Greville abscissae of shape (n_ctrl,), where n_ctrl is the
        number of control points.
    """
    U = jnp.asarray(U)
    M = U.shape[0]
    n_ctrl = M - p - 1
    prefix = jnp.concatenate(
        [jnp.zeros((1,) + U.shape[1:], dtype=U.dtype), jnp.cumsum(U, axis=0)], axis=0
    )
    idx = jnp.arange(n_ctrl)
    sums = prefix[idx + p + 1] - prefix[idx + 1]
    return sums / p


def generate_knot_vector(
    degree: int,
    num_ctrlpts: int,
) -> Array:
    """Generate an equally spaced knot vector.

    Create a clamped knot vector with degree+1 repetitions at both
    endpoints and equally spaced interior knots.

    Parameters
    ----------
    degree : int
        Polynomial degree of the B-spline basis.
    num_ctrlpts : int
        Number of control points.

    Returns
    -------
    Array
        Clamped knot vector of length num_ctrlpts + degree + 1.
    """
    # Number of repetitions at the start and end of the array
    num_repeat = degree

    # Number of knots in the middle
    num_segments = num_ctrlpts - (degree + 1)

    # Create the knot vector repeat part
    repeat_mask = jnp.arange(0, num_repeat)

    # Create the knot vector main part
    knot_vector_middle = jnp.linspace(0.0, 1.0, num_segments + 2)

    knot_vector = jnp.zeros((num_repeat + num_segments + 2 * num_repeat,))
    knot_vector = knot_vector.at[repeat_mask].set(0.0)
    knot_vector = knot_vector.at[num_repeat:-num_repeat].set(knot_vector_middle)
    knot_vector = knot_vector.at[-num_repeat:].set(1.0)

    # Return knot vector
    return knot_vector
