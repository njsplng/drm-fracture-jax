"""Distance functions for 1D and 2D geometries using JAX."""

from typing import Any, List, Union

import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array


class DistanceFunction1D:
    """Compute a 1D distance function.

    The function equals 1 on the interval [x_init, x_init+L] and decays
    to 0 outside by a distance d0.

    Parameters
    ----------
    x_init : float
        Starting position of the interval.
    L : float
        Length of the interval.
    d0 : float
        Decay distance outside the interval.
    order : int, optional
        Order of the distance function (1 or 2). Default is 2.
    """

    def __init__(
        self,
        x_init: float,
        L: float,
        d0: float,
        order: int = 2,
        **kwargs,
    ) -> None:
        """Initialize the 1D distance function."""
        assert order in [
            1,
            2,
        ], "Only first and second order distance functions are currently supported"
        # Convert inputs to JAX arrays.
        self.x_init = jnp.asarray(x_init)
        self.L = jnp.asarray(L)
        self.d0 = jnp.asarray(d0)
        self.order = jnp.asarray(order)

    def __call__(self, inp: Array) -> Array:
        """Compute the distance function value.

        Parameters
        ----------
        inp : Array
            Input coordinates of shape (N,) or (N, 1).

        Returns
        -------
        Array
            Distance function values.
        """
        if inp.ndim == 1:
            inp = inp[:, None]

        # Shift the x-coordinate.
        x = inp[:, 0] - self.x_init

        # 1. Main Interval: 0 <= x <= L
        mask_p1 = ((x >= 0) & (x <= self.L)).astype(jnp.float32)
        dist_fn_p1 = mask_p1

        # 2. Right Decay: L < x < L + d0
        mask_p2 = (((x > self.L) & (x < self.L + self.d0))).astype(jnp.float32)
        decay_p2 = (1 - (x - self.L) / self.d0) ** self.order
        # Apply ReLU and divide by itself plus epsilon.
        eps = jnp.finfo(jnp.float32).eps
        decay_p2 = (
            jnn.relu(self.d0 - (x - self.L)) / (self.d0 - (x - self.L) + eps) * decay_p2
        )
        dist_fn_p2 = decay_p2 * mask_p2

        # 3. Left Decay: -d0 < x < 0
        mask_p3 = (((x < 0) & (x > -self.d0))).astype(jnp.float32)
        decay_p3 = (1 - (-x) / self.d0) ** self.order
        decay_p3 = jnn.relu(self.d0 + x) / (self.d0 + x + eps) * decay_p3
        dist_fn_p3 = decay_p3 * mask_p3

        return jnp.array(dist_fn_p1 + dist_fn_p2 + dist_fn_p3)


class DistanceFunction2D:
    """Compute a 2D distance function for a line segment.

    The function equals 1 on the line segment and decays to 0 at a distance
    d0 from it. The segment starts at (x_init, y_init), is oriented at angle
    theta from the x-axis, and has length L.

    Parameters
    ----------
    x_init : float
        Starting x-coordinate of the line segment.
    y_init : float
        Starting y-coordinate of the line segment.
    theta : float
        Orientation angle in degrees from the x-axis.
    L : float
        Length of the line segment.
    d0 : float
        Decay distance from the line segment.
    order : int, optional
        Order of the distance function (1 or 2). Default is 2.
    smooth_end : bool, optional
        Whether to include smooth decay at endpoints. Default is True.
    """

    def __init__(
        self,
        x_init: float,
        y_init: float,
        theta: float,
        L: float,
        d0: float,
        order: int = 2,
        smooth_end: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the 2D distance function."""
        self.x_init = x_init
        self.y_init = y_init
        self.theta = jnp.deg2rad(theta)
        self.L = L
        self.d0 = d0
        self.order = order
        self.smooth_end = smooth_end

    def __call__(self, inp: Array) -> Array:
        """Compute the distance function value.

        Parameters
        ----------
        inp : Array
            Input coordinates of shape (N, 2).

        Returns
        -------
        Array
            Distance function values.
        """
        # Convert constants to JAX arrays with float64 dtype.
        L = jnp.array([self.L])
        d0 = jnp.array([self.d0])
        theta = jnp.array([self.theta])
        input_coords = jnp.array(inp)

        # Extract coordinates.
        x_coords = input_coords[:, 0]
        y_coords = input_coords[:, 1]

        # Shift coordinates.
        x_coords_shifted = x_coords - self.x_init
        y_coords_shifted = y_coords - self.y_init

        # Rotation matrix.
        Rt = jnp.array(
            [
                [jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta), jnp.cos(theta)],
            ],
        ).squeeze()
        # Rotate coordinates.
        stacked_coordinates = jnp.stack([x_coords_shifted, y_coords_shifted], axis=1)
        coordinates_rotated = jnp.matmul(stacked_coordinates, Rt)
        x = coordinates_rotated[:, 0]
        y = coordinates_rotated[:, 1]

        eps = jnp.finfo(jnp.float64).eps

        # Region 1: Main interval.
        term1 = jnn.relu(x * (L - x)) / (jnp.abs(x * (L - x)) + eps)
        term2 = jnn.relu(d0 - jnp.abs(y)) / (jnp.abs(d0 - jnp.abs(y)) + eps)
        decay1 = (1 - jnp.abs(y) / d0) ** self.order
        dist_fn_p1 = term1 * term2 * decay1

        # Region 2: Right decay.
        term3 = jnn.relu(x - L) / (jnp.abs(x - L) + eps)
        term4 = jnn.relu(d0**2 - ((x - L) ** 2 + y**2)) / (
            jnp.abs(d0**2 - ((x - L) ** 2 + y**2)) + eps
        )
        decay2 = (1 - jnp.sqrt((x - L) ** 2 + y**2) / d0) ** self.order
        dist_fn_p2 = term3 * term4 * decay2

        # Region 3: Left decay.
        term5 = jnn.relu(-x) / (jnp.abs(x) + eps)
        term6 = jnn.relu(d0**2 - (x**2 + y**2)) / (jnp.abs(d0**2 - (x**2 + y**2)) + eps)
        decay3 = (1 - jnp.sqrt(x**2 + y**2) / d0) ** self.order
        dist_fn_p3 = term5 * term6 * decay3

        # Combine the distance functions.
        dist_fn = dist_fn_p1
        if self.smooth_end:
            dist_fn += dist_fn_p2 + dist_fn_p3
        return jnp.array(dist_fn)


class CompositeDistanceFunction:
    """A composite distance function that handles multiple segments.

    Parameters
    ----------
    input_list : list[dict[str, Any]]
        List of parameter dictionaries for each segment.
    dimension : int, optional
        Dimensionality (1 or 2). Default is 2.
    """

    def __init__(
        self,
        input_list: list[dict[str, Any]],
        dimension: int = 2,
    ) -> None:
        """Initialize the composite distance function."""
        # Cast to list if a single value is provided
        if not isinstance(input_list, list):
            input_list = [input_list]

        self.distance_functions: list[Any] = []
        for entry in input_list:
            if dimension == 1:
                self.distance_functions.append(DistanceFunction1D(**entry))
            elif dimension == 2:
                self.distance_functions.append(DistanceFunction2D(**entry))

    def __call__(
        self,
        inp: Array,
        aggregate: bool = False,
    ) -> Union[Array, List[Array]]:
        """Compute the composite distance function.

        Parameters
        ----------
        inp : Array
            Input coordinates.
        aggregate : bool, optional
            If True, return list of individual outputs. Default is False.

        Returns
        -------
        Union[Array, List[Array]]
            Aggregated or individual distance function values.
        """
        # Initialise the shape
        if aggregate:
            output = []
        else:
            output = jnp.zeros(inp.shape[0])

        # Iterate over the functions and sum their outputs
        for entry in self.distance_functions:
            # Produce the distance function values
            if aggregate:
                output.append(entry(inp))
            else:
                output = output + entry(inp)

        return output
