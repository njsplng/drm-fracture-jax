"""Gauss point initialisation functions for various 2D elements.

This module provides functions for computing shape functions, their
derivatives, and related quantities at Gauss integration points for
various 2D finite element types including quadrilaterals, triangles,
and NURBS elements.
"""

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
from geomdl import helpers
from jax import hessian, jacfwd, jit, lax, vmap
from jaxtyping import Array

from iga import find_span
from mesh import QUAD_TYPES, TRI_TYPES, legendre_gauss_quadrature_2d


def build_J33(jacobian: Array) -> Array:
    """Build the J33 matrix for 2D isoparametric elements.

    Construct the J33 transformation matrix used for computing second
    derivatives in physical coordinates from isoparametric derivatives.

    Parameters
    ----------
    jacobian : Array
        Jacobian matrix of shape (E, G, 2, 2) where E is the number
        of elements and G is the number of Gauss points.

    Returns
    -------
    Array
        J33 matrix of shape (E, G, 3, 3).
    """

    def J33(jac: Array) -> Array:
        dxdxi, dxdet = jac[0, 0], jac[1, 0]
        dydxi, dydet = jac[0, 1], jac[1, 1]
        return jnp.array(
            [
                [dxdxi**2, dydxi**2, 2 * dxdxi * dydxi],
                [dxdet**2, dydet**2, 2 * dxdet * dydet],
                [dxdxi * dxdet, dydxi * dydet, dxdxi * dydet + dxdet * dydxi],
            ]
        )

    return vmap(vmap(J33))(jacobian)


def lagrange_element_point_initialisation(
    shape_functions: Callable[[Tuple[float, float]], Array],
    gauss_points: Array,
    connectivity: Array,
    nodal_coordinates: Array,
    gauss_weights: Array,
    thickness: float = 1.0,
    rotation_angles: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Generic Gauss point initialisation function for 2D elements.

    Compute shape functions, their derivatives, Jacobians, and related
    quantities at Gauss integration points for Lagrange finite elements.

    Parameters
    ----------
    shape_functions : Callable[[Tuple[float, float]], Array]
        Function that computes shape function values at a given
        (xi, eta) point in the reference element.
    gauss_points : Array
        Gauss point coordinates in the reference element of shape
        (n_points, 2).
    connectivity : Array
        Element-node connectivity array of shape (n_elements, n_nodes).
    nodal_coordinates : Array
        Nodal coordinates of shape (n_nodes, 2).
    gauss_weights : Array
        Gauss quadrature weights of shape (n_points,).
    thickness : float, optional
        Element thickness for volume computation. Default is 1.0.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        A tuple containing (N, dN, d2N, point_volumes, B, dN_phys,
        d2N_phys, E_extrap) where:
        - N: Shape function values
        - dN: First derivatives in isoparametric coordinates
        - d2N: Second derivatives in isoparametric coordinates
        - point_volumes: Integration point volumes
        - B: Strain-displacement matrix
        - dN_phys: First derivatives in physical coordinates
        - d2N_phys: Second derivatives in physical coordinates
        - E_extrap: Extrapolation matrix
    """
    # Form the element coordinates
    element_coordinates = nodal_coordinates[connectivity]

    # Given a material orientation, apply it
    if rotation_angles is not None:
        s = jnp.sin(rotation_angles)
        c = jnp.cos(rotation_angles)
        rotation_matrices = jnp.array([[c, -s], [s, c]])
        element_coordinates = jnp.einsum(
            "eij, jke -> eik", element_coordinates, rotation_matrices
        )

    # Form the shape function matrix
    N = vmap(shape_functions)(gauss_points)  # [G, N]
    N = jnp.expand_dims(N, axis=0).repeat(connectivity.shape[0], axis=0)  # [E, G, N]
    E_extrap = jnp.linalg.pinv(N)

    # Get the shape function derivatives
    jacobian_fn_dn = jacfwd(shape_functions, argnums=0)
    jacobian_dn_vectorised = vmap(jacobian_fn_dn)(gauss_points)
    dN = jnp.transpose(jacobian_dn_vectorised, (2, 0, 1))
    dN = jnp.expand_dims(dN, axis=0).repeat(
        connectivity.shape[0], axis=0
    )  # [E, D, G, N]

    # Get the shape function derivatives (2nd order iso)
    hessian_fn_dn = hessian(shape_functions, argnums=0)
    hessian_dn_vectorised = vmap(hessian_fn_dn)(gauss_points)  # [G, N, D, D]
    d2N = jnp.transpose(hessian_dn_vectorised, (2, 3, 0, 1))  # [D, D, G, N]
    d2N = jnp.expand_dims(d2N, axis=0).repeat(
        connectivity.shape[0], axis=0
    )  # [E, D, D, G, N]

    # Create the physical mapping
    def physical_mapping(xi_eta: Tuple[float, float], physical_coords: Array) -> Array:
        dN_vals = jacobian_fn_dn(xi_eta)
        return jnp.dot(dN_vals.T, physical_coords)

    # Vectorise the physical mapping over all gauss points
    def jacobian_for_element(elemental_coords: Array) -> Array:
        return vmap(lambda pt: physical_mapping(pt, elemental_coords))(gauss_points)

    # Vectorise the jacobian over all gauss points and elements
    jacobian_all = vmap(jacobian_for_element)(element_coordinates)

    # Get the point volumes
    J_det = jnp.linalg.det(jacobian_all)  # [E, G]
    point_volumes = J_det * gauss_weights * thickness

    # Get the physical derivatives (1st order)
    J_inv = jnp.linalg.inv(jacobian_all)  # [E, G, D, D]
    dN_phys = jnp.einsum("egij, ejgn-> egin", J_inv, dN)  # [E, G, D, N]

    # Build the J33 matrix and its inverse
    J33_all = build_J33(jacobian_all)  # [E, G, 3, 3]
    J33_inv_all = jnp.linalg.inv(J33_all)  # [E, G, 3, 3]

    # Flatten the second isoparametric derivatives
    d2N_xx = d2N[:, 0, 0, :, :]  # [E, G, N]
    d2N_yy = d2N[:, 1, 1, :, :]  # [E, G, N]
    d2N_xy = d2N[:, 0, 1, :, :]  # [E, G, N]
    d2N_flattened = jnp.stack([d2N_xx, d2N_yy, d2N_xy], axis=2)  # [E, G, 3, N]

    # Compute jcb_isp_2_phys_2 (equivalent to jcb_isp_2_phys_2 = der2_shp_func_isp * a_grd_ctrl_pt_crd in MATLAB)
    jcb_isp_2_phys_2 = jnp.einsum(
        "egkn, end -> egkd", d2N_flattened, element_coordinates
    )  # [E, G, 3, D])

    # Compute Atemp (Atemp = jcb_isp_2_phys_2 * der1_shp_func_phys in MATLAB)
    Atemp = jnp.einsum("egkd, egdn -> egkn", jcb_isp_2_phys_2, dN_phys)  # [E, G, 3, N]

    # Compute A2temp (A2temp = der2_shp_func_isp - Atemp in MATLAB)
    A2temp = d2N_flattened - Atemp  # [E, G, 3, N]

    # Compute the physical second derivatives (der2_shp_func_phys = invJ33 * A2temp in MATLAB)
    d2N_phys = jnp.einsum("egij, egjn -> egin", J33_inv_all, A2temp)  # [E, G, 3, N]

    def assemble_B_matrix(dN_phys: Array) -> Array:
        """Assemble the strain-displacement (B) matrix from physical derivatives."""
        E, G, dimensions, num_nodes = dN_phys.shape

        # Create an empty B matrix of shape (E, G, 3, num_nodes * n_dims)
        B = jnp.zeros((E, G, 3, num_nodes * dimensions))

        # Normal strain in the x direction:
        B = B.at[:, :, 0, 0::2].set(dN_phys[:, :, 0, :])

        # Normal strain in the y direction:
        B = B.at[:, :, 1, 1::2].set(dN_phys[:, :, 1, :])

        # Shear strain (gamma_xy):
        B = B.at[:, :, 2, 0::2].set(dN_phys[:, :, 1, :])
        B = B.at[:, :, 2, 1::2].set(dN_phys[:, :, 0, :])
        return B

    # Assemble the B matrix
    B = assemble_B_matrix(dN_phys)

    return N, dN, d2N, point_volumes, B, dN_phys, d2N_phys, E_extrap


def precompute_quad8_9ip(
    connectivity: Array,
    nodal_coordinates: Array,
    thickness: float = 1.0,
    rotation_angles: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute parameters for 8-node quadrilateral elements with 9 integration points.

    Parameters
    ----------
    connectivity : Array
        Element-node connectivity array.
    nodal_coordinates : Array
        Nodal coordinates.
    thickness : float, optional
        Element thickness. Default is 1.0.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        Initialisation parameters for the element.
    """
    gauss_points, gauss_weights = legendre_gauss_quadrature_2d(3)

    def shape_functions(xi_eta: Tuple[float, float]) -> Array:
        xi, eta = xi_eta
        N5 = 0.50 * (-(xi**2) + 1.0) * (1.0 - eta)
        N6 = 0.50 * (1.0 + xi) * (-(eta**2) + 1.0)
        N7 = 0.50 * (-(xi**2) + 1.0) * (1.0 + eta)
        N8 = 0.50 * (1.0 - xi) * (-(eta**2) + 1.0)
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta) - 0.5 * (N8 + N5)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta) - 0.5 * (N5 + N6)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta) - 0.5 * (N6 + N7)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta) - 0.5 * (N7 + N8)
        return jnp.array([N1, N2, N3, N4, N5, N6, N7, N8])

    return lagrange_element_point_initialisation(
        shape_functions=shape_functions,
        gauss_points=gauss_points,
        connectivity=connectivity,
        nodal_coordinates=nodal_coordinates,
        gauss_weights=gauss_weights,
        thickness=thickness,
        rotation_angles=rotation_angles,
    )


def precompute_quad8_4ip(
    connectivity: Array,
    nodal_coordinates: Array,
    thickness: float = 1.0,
    rotation_angles: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute parameters for 8-node quadrilateral elements with 4 integration points.

    Parameters
    ----------
    connectivity : Array
        Element-node connectivity array.
    nodal_coordinates : Array
        Nodal coordinates.
    thickness : float, optional
        Element thickness. Default is 1.0.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        Initialisation parameters for the element.
    """
    gauss_points, gauss_weights = legendre_gauss_quadrature_2d(2)

    def shape_functions(xi_eta: Tuple[float, float]) -> Array:
        xi, eta = xi_eta
        N5 = 0.50 * (-(xi**2) + 1.0) * (1.0 - eta)
        N6 = 0.50 * (1.0 + xi) * (-(eta**2) + 1.0)
        N7 = 0.50 * (-(xi**2) + 1.0) * (1.0 + eta)
        N8 = 0.50 * (1.0 - xi) * (-(eta**2) + 1.0)
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta) - 0.5 * (N8 + N5)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta) - 0.5 * (N5 + N6)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta) - 0.5 * (N6 + N7)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta) - 0.5 * (N7 + N8)
        return jnp.array([N1, N2, N3, N4, N5, N6, N7, N8])

    return lagrange_element_point_initialisation(
        shape_functions=shape_functions,
        gauss_points=gauss_points,
        connectivity=connectivity,
        nodal_coordinates=nodal_coordinates,
        gauss_weights=gauss_weights,
        thickness=thickness,
        rotation_angles=rotation_angles,
    )


def precompute_quad9_9ip(
    connectivity: Array,
    nodal_coordinates: Array,
    thickness: float = 1.0,
    rotation_angles: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute parameters for 9-node quadrilateral elements with 9 integration points.

    Parameters
    ----------
    connectivity : Array
        Element-node connectivity array.
    nodal_coordinates : Array
        Nodal coordinates.
    thickness : float, optional
        Element thickness. Default is 1.0.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        Initialisation parameters for the element.
    """
    gauss_points, gauss_weights = legendre_gauss_quadrature_2d(3)

    def shape_functions(xi_eta: Tuple[float, float]) -> Array:
        xi, eta = xi_eta
        N1 = 0.25 * xi * eta * (1.0 - xi) * (1.0 - eta)
        N2 = -0.25 * xi * eta * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * xi * eta * (1.0 + xi) * (1.0 + eta)
        N4 = -0.25 * xi * eta * (1.0 - xi) * (1.0 + eta)
        N5 = -0.5 * eta * (1.0 - xi**2) * (1.0 - eta)
        N6 = 0.5 * xi * (1.0 + xi) * (1.0 - eta**2)
        N7 = 0.5 * eta * (1.0 - xi**2) * (1.0 + eta)
        N8 = -0.5 * xi * (1.0 - xi) * (1.0 - eta**2)
        N9 = (1.0 - xi**2) * (1.0 - eta**2)
        return jnp.array([N1, N2, N3, N4, N5, N6, N7, N8, N9])

    return lagrange_element_point_initialisation(
        shape_functions=shape_functions,
        gauss_points=gauss_points,
        connectivity=connectivity,
        nodal_coordinates=nodal_coordinates,
        gauss_weights=gauss_weights,
        thickness=thickness,
        rotation_angles=rotation_angles,
    )


def precompute_quad9_4ip(
    connectivity: Array,
    nodal_coordinates: Array,
    thickness: float = 1.0,
    rotation_angles: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute parameters for 9-node quadrilateral elements with 4 integration points.

    Parameters
    ----------
    connectivity : Array
        Element-node connectivity array.
    nodal_coordinates : Array
        Nodal coordinates.
    thickness : float, optional
        Element thickness. Default is 1.0.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        Initialisation parameters for the element.
    """
    gauss_points, gauss_weights = legendre_gauss_quadrature_2d(2)

    def shape_functions(xi_eta: Tuple[float, float]) -> Array:
        xi, eta = xi_eta
        N1 = 0.25 * xi * eta * (1.0 - xi) * (1.0 - eta)
        N2 = -0.25 * xi * eta * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * xi * eta * (1.0 + xi) * (1.0 + eta)
        N4 = -0.25 * xi * eta * (1.0 - xi) * (1.0 + eta)
        N5 = -0.5 * eta * (1.0 - xi**2) * (1.0 - eta)
        N6 = 0.5 * xi * (1.0 + xi) * (1.0 - eta**2)
        N7 = 0.5 * eta * (1.0 - xi**2) * (1.0 + eta)
        N8 = -0.5 * xi * (1.0 - xi) * (1.0 - eta**2)
        N9 = (1.0 - xi**2) * (1.0 - eta**2)
        return jnp.array([N1, N2, N3, N4, N5, N6, N7, N8, N9])

    return lagrange_element_point_initialisation(
        shape_functions=shape_functions,
        gauss_points=gauss_points,
        connectivity=connectivity,
        nodal_coordinates=nodal_coordinates,
        gauss_weights=gauss_weights,
        thickness=thickness,
        rotation_angles=rotation_angles,
    )


def precompute_tri_6ip(
    connectivity: Array,
    nodal_coordinates: Array,
    thickness: float = 1.0,
    rotation_angles: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute parameters for 6-node triangular elements.

    Parameters
    ----------
    connectivity : Array
        Element-node connectivity array.
    nodal_coordinates : Array
        Nodal coordinates.
    thickness : float, optional
        Element thickness. Default is 1.0.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        Initialisation parameters for the element.
    """

    # Define the Gauss point coordinates for a higher integration rule
    # Using a 3-point integration rule for second-order triangles

    gauss_points = jnp.array(
        [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]]
    )

    # Define the gauss weights corresponding to the gauss points

    gauss_weights = jnp.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

    # Define the shape functions in natural coordinates

    def shape_functions(xi_eta: Tuple[float, float]) -> Array:
        xi, eta = xi_eta
        zeta = 1.0 - xi - eta

        # Corner nodes (vertices)
        N1 = zeta * (2.0 * zeta - 1.0)  # Node at (0,0)
        N2 = xi * (2.0 * xi - 1.0)  # Node at (1,0)
        N3 = eta * (2.0 * eta - 1.0)  # Node at (0,1)

        # Mid-side nodes
        N4 = 4.0 * xi * zeta  # Node at (0.5,0)
        N5 = 4.0 * xi * eta  # Node at (0.5,0.5)
        N6 = 4.0 * eta * zeta  # Node at (0,0.5)

        # Return the shape functions as an array

        return jnp.array([N1, N2, N3, N4, N5, N6])

    # Return the generic point initialisation with problem-specific parameters

    return lagrange_element_point_initialisation(
        shape_functions=shape_functions,
        gauss_points=gauss_points,
        connectivity=connectivity,
        nodal_coordinates=nodal_coordinates,
        gauss_weights=gauss_weights,
        thickness=thickness,
        rotation_angles=rotation_angles,
    )


def precompute_quad_4ip(
    connectivity: Array,
    nodal_coordinates: Array,
    thickness: float = 1.0,
    rotation_angles: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute parameters for 4-node quadrilateral elements with 4 integration points.

    Parameters
    ----------
    connectivity : Array
        Element-node connectivity array.
    nodal_coordinates : Array
        Nodal coordinates.
    thickness : float, optional
        Element thickness. Default is 1.0.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        Initialisation parameters for the element.
    """
    gauss_points, gauss_weights = legendre_gauss_quadrature_2d(2)

    # Define the shape functions in polar coordinates

    def shape_functions(xi_eta: Tuple[float, float]) -> Array:
        xi, eta = xi_eta
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta)

        # Return the shape functions as an array

        return jnp.array([N1, N2, N3, N4])  # (4,4)

    # Return the generic point initialisation with problem-specific parameters

    return lagrange_element_point_initialisation(
        shape_functions=shape_functions,
        gauss_points=gauss_points,
        connectivity=connectivity,
        nodal_coordinates=nodal_coordinates,
        gauss_weights=gauss_weights,
        thickness=thickness,
        rotation_angles=rotation_angles,
    )


def precompute_tri_1ip(
    connectivity: Array,
    nodal_coordinates: Array,
    thickness: float = 1.0,
    rotation_angles: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute parameters for 3-node triangular elements with 1 integration point.

    Parameters
    ----------
    connectivity : Array
        Element-node connectivity array.
    nodal_coordinates : Array
        Nodal coordinates.
    thickness : float, optional
        Element thickness. Default is 1.0.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        Initialisation parameters for the element.
    """

    # Define the Gauss point coordinates

    gauss_points = jnp.array([[1.0 / 3.0, 1.0 / 3.0]])

    # Define the gauss weights corresponding to the gauss points

    gauss_weights = jnp.array([0.5])

    # Define the shape functions in polar coordinates

    def shape_functions(xi_eta: Tuple[float, float]) -> Array:
        xi, eta = xi_eta
        N1 = 1.0 - xi - eta
        N2 = xi
        N3 = eta

        # Return the shape functions as an array

        return jnp.array([N1, N2, N3])  # (1, 3)

    # Return the generic point initialisation with problem-specific parameters

    return lagrange_element_point_initialisation(
        shape_functions=shape_functions,
        gauss_points=gauss_points,
        connectivity=connectivity,
        nodal_coordinates=nodal_coordinates,
        gauss_weights=gauss_weights,
        thickness=thickness,
        rotation_angles=rotation_angles,
    )


def nurbs_point_initialisation(
    connectivity: Array,
    nodal_coordinates: Array,
    thickness: float = 1.0,
    info: Optional[Dict[str, Any]] = None,
    rotation_angles: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute initialisation parameters for NURBS elements.

    Similar to Lagrange element initialisation but uses rational
    NURBS basis functions with knot vectors and weights.

    Parameters
    ----------
    connectivity : Array
        Element-node connectivity array.
    nodal_coordinates : Array
        Nodal coordinates.
    thickness : float, optional
        Element thickness. Default is 1.0.
    info : Dict[str, Any], optional
        Dictionary containing NURBS data including knot vectors,
        degrees, weights, and integration order. Required.
    rotation_angles : Array, optional
        Rotation angles for material orientation. Default is None.

    Returns
    -------
    tuple
        Initialisation parameters for the NURBS element.

    Raises
    ------
    AssertionError
        If info dictionary is not provided.
    """
    assert info is not None, "Please supply the auxiliary NURBS dict!"
    gauss_pts, gauss_wts = legendre_gauss_quadrature_2d(info["gauss_integration_order"])

    @partial(
        jit,
        static_argnames=(
            "p",
            "q",
            "number_control_points_u",
            "number_control_points_v",
            "effective_derivative_order_u",
            "effective_derivative_order_v",
        ),
    )
    def compute_nurbs_base_funs_and_ders(
        u: float,
        v: float,
        p: int,
        q: int,
        U: Array,
        V: Array,
        number_control_points_u: int,
        number_control_points_v: int,
        weights: Array,
        effective_derivative_order_u: int,
        effective_derivative_order_v: int,
    ) -> Tuple[Array, Array, Array]:
        """
        Compute rational NURBS shape functions R, and their derivatives up to 2nd order at a single parametric point (u,v).

        Returns:
        R:    shape (n_ctrl_u*n_ctrl_v,)
        dR:   shape (2, n_ctrl_u*n_ctrl_v)    # dR/du, dR/dv
        d2R:  shape (2,2, n_ctrl_u*n_ctrl_v)  # d²R/du², d²R/dudv, d²R/dv²
        """
        # Find spans
        span_u = find_span(p, U, number_control_points_u, u)
        span_v = find_span(q, V, number_control_points_v, v)

        # Clamp the derivative order to ensure the degree of the basis functions is not exceeded
        du = effective_derivative_order_u
        dv = effective_derivative_order_v

        # Get the univariate B-spline basis functions and their derivatives
        Nu = jnp.array(helpers.basis_function_ders(p, U, span_u, u, du))
        Mv = jnp.array(helpers.basis_function_ders(q, V, span_v, v, dv))

        # Find the weight patch
        w_patch = lax.dynamic_slice(
            weights,
            (span_v - q, span_u - p),  # dynamic start indices
            (q + 1, p + 1),  # static slice sizes
        )

        # Ensure that the derivative orders are respected
        Nu0 = Nu[0]
        Nu1 = Nu[1] if du >= 1 else jnp.zeros_like(Nu0)
        Nu2 = Nu[2] if du >= 2 else jnp.zeros_like(Nu0)

        Mv0 = Mv[0]
        Mv1 = Mv[1] if dv >= 1 else jnp.zeros_like(Mv0)
        Mv2 = Mv[2] if dv >= 2 else jnp.zeros_like(Mv0)

        # Build all numerator matrices
        num0_mat = Mv0[:, None] * Nu0[None, :] * w_patch
        num_u_mat = Mv0[:, None] * Nu1[None, :] * w_patch
        num_v_mat = Mv1[:, None] * Nu0[None, :] * w_patch
        num_uu_mat = Mv0[:, None] * Nu2[None, :] * w_patch
        num_uv_mat = Mv1[:, None] * Nu1[None, :] * w_patch
        num_vv_mat = Mv2[:, None] * Nu0[None, :] * w_patch

        # Collapse the matrices to vectors
        num0, num_u, num_v = map(lambda M: M.ravel(), (num0_mat, num_u_mat, num_v_mat))
        num_uu, num_uv, num_vv = map(
            lambda M: M.ravel(), (num_uu_mat, num_uv_mat, num_vv_mat)
        )

        # Sum the numerators
        W = jnp.sum(num0)
        Wu = jnp.sum(num_u)
        Wv = jnp.sum(num_v)
        Wuu = jnp.sum(num_uu)
        Wuv = jnp.sum(num_uv)
        Wvv = jnp.sum(num_vv)

        # Form rational R
        R = num0 / W

        # Form derivatives using the quotient rule
        dR_u = (num_u * W - num0 * Wu) / (W * W)
        dR_v = (num_v * W - num0 * Wv) / (W * W)
        dR = jnp.vstack([dR_u, dR_v])  # (2, (p+1)*(q+1))

        # Second derivatives
        # d²R/du² = (num_uu W - 2 num_u Wu - num0 Wuu)*W + 2 num0 Wu^2
        d2R_uu = (
            (num_uu * W - 2 * num_u * Wu - num0 * Wuu) * W + 2 * num0 * Wu * Wu
        ) / (W**3)
        # d²R/dudv = (num_uv W - num_u Wv - num_v Wu - num0 Wuv)*W + 2 num0 Wu Wv
        d2R_uv = (
            (num_uv * W - num_u * Wv - num_v * Wu - num0 * Wuv) * W + 2 * num0 * Wu * Wv
        ) / (W**3)
        # d²R/dv² similar
        d2R_vv = (
            (num_vv * W - 2 * num_v * Wv - num0 * Wvv) * W + 2 * num0 * Wv * Wv
        ) / (W**3)
        d2R = jnp.stack(
            (jnp.array([d2R_uu, d2R_uv]), jnp.array([d2R_uv, d2R_vv]))
        )  # (2, 2, (p+1)*(q+1))
        return R, dR, d2R

    # Extract the necessary parameters from the info dictionary
    p = info["degrees"]["u"]
    q = info["degrees"]["v"]
    U = jnp.array(info["knotvector_u"])
    V = jnp.array(info["knotvector_v"])
    w = jnp.array(info["weights"])
    spans_u = jnp.array(info["parametric_spans_u"])
    spans_v = jnp.array(info["parametric_spans_v"])

    # Generate the span indices
    e_idxs = jnp.arange(len(connectivity))
    i_el = e_idxs % spans_u.shape[0]
    j_el = e_idxs // spans_u.shape[0]

    u0 = spans_u[i_el, 0]
    u1 = spans_u[i_el, 1]
    v0 = spans_v[j_el, 0]
    v1 = spans_v[j_el, 1]

    # Extract the gauss point locations for the parametric space
    xi = jnp.array([pt[0] for pt in gauss_pts])
    eta = jnp.array([pt[1] for pt in gauss_pts])
    u_full = 0.5 * (u1[:, None] - u0[:, None]) * xi[None, :] + 0.5 * (
        u1[:, None] + u0[:, None]
    )
    v_full = 0.5 * (v1[:, None] - v0[:, None]) * eta[None, :] + 0.5 * (
        v1[:, None] + v0[:, None]
    )

    # vmap the computation of shape functions and their derivatives over the gauss points
    shape_funcs_gp_vmap = vmap(
        compute_nurbs_base_funs_and_ders,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None),
    )

    # vmap the computation of shape functions and their derivatives over the elements
    shape_funcs_el_vmap = vmap(
        lambda u_row, v_row: shape_funcs_gp_vmap(
            u_row,
            v_row,
            p,
            q,
            U,
            V,
            len(info["knotvector_u"]) - p - 1,
            len(info["knotvector_v"]) - q - 1,
            w,
            min(p, 2),
            min(q, 2),
        ),
        in_axes=(0, 0),
    )

    # Evaluate the shape functions and their derivatives for all elements and gauss points
    R, dR, d2R = shape_funcs_el_vmap(u_full, v_full)

    # Reshape the results to match the expected output shapes
    N = R
    dN_isp = dR.transpose((0, 2, 1, 3))
    d2N_isp = d2R.transpose((0, 2, 3, 1, 4))

    # Calculate the extrapolation matrix
    E_extrap = jnp.linalg.pinv(N)

    # Precompute the scales for each span
    scale_u = 0.5 * (u1[:, None] - u0[:, None])
    scale_v = 0.5 * (v1[:, None] - v0[:, None])

    # Scale the derivatives in the isoparametric space to
    dR_parametric_full = jnp.stack(
        (
            dN_isp[:, 0, :, :] * scale_u[:, :, None],
            dN_isp[:, 1, :, :] * scale_v[:, :, None],
        ),
        axis=1,
    )  # -> (E,2,G,nnz)

    # same for d2R: broadcast the four components
    a00 = d2N_isp[:, 0, 0, :, :] * scale_u[:, :, None] ** 2
    a01 = d2N_isp[:, 0, 1, :, :] * (scale_u[:, :, None] * scale_v[:, :, None])
    a10 = d2N_isp[:, 1, 0, :, :] * (scale_v[:, :, None] * scale_u[:, :, None])
    a11 = d2N_isp[:, 1, 1, :, :] * scale_v[:, :, None] ** 2

    # Build the parametric second derivatives
    row0 = jnp.stack([a00, a01], axis=1)
    row1 = jnp.stack([a10, a11], axis=1)
    d2R_parametric_full = jnp.stack([row0, row1], axis=2)  # -> (E,2,2,G,(q+1)*(p+1))

    # Map the element coordinates
    elem_coords = nodal_coordinates[connectivity]  # shape [E, nnz, 2]

    # Compute the jacobian for each element and gauss point
    jacobian_all = jnp.einsum(
        "eIgn, enJ -> egIJ", dR_parametric_full, elem_coords
    )  # (E,G,2,2)

    # Compute the determinant and inverse of the jacobian
    J_det = jnp.linalg.det(jacobian_all)
    J_inv = jnp.linalg.inv(jacobian_all)
    point_volumes = J_det * gauss_wts[None, :] * thickness

    # Physical mapping to the first derivatives
    dN_phys = jnp.einsum("egij, ejgn->egin", J_inv, dR_parametric_full)

    # Construct the J33 matrix and its inverse
    J33_all = build_J33(jacobian_all)
    J33_inv = jnp.linalg.inv(J33_all)

    # Flatten the second order parametric derivatives
    d2N_xx = d2R_parametric_full[:, 0, 0, :, :]  # [E, G, N]
    d2N_yy = d2R_parametric_full[:, 1, 1, :, :]  # [E, G, N]
    d2N_xy = d2R_parametric_full[:, 0, 1, :, :]  # [E, G, N]
    d2N_flattened = jnp.stack([d2N_xx, d2N_yy, d2N_xy], axis=2)  # [E, G, 3, N]

    # Compute the second order physical derivatives
    jcb_isp_2_phys_2 = jnp.einsum("egkn,end->egkd", d2N_flattened, elem_coords)
    Atemp = jnp.einsum("egkd,egdn->egkn", jcb_isp_2_phys_2, dN_phys)
    A2temp = d2N_flattened - Atemp
    d2N_phys = jnp.einsum("egij,egjn->egin", J33_inv, A2temp)

    # Assemble the B matrix
    @jit
    def build_B(dN_phys_point: Array) -> Array:
        B = jnp.zeros((3, 2 * (p + 1) * (q + 1)))
        B = B.at[0, 0::2].set(dN_phys_point[0])
        B = B.at[1, 1::2].set(dN_phys_point[1])
        B = B.at[2, 0::2].set(dN_phys_point[1])
        B = B.at[2, 1::2].set(dN_phys_point[0])
        return B

    # vmap over the elements and gauss points to build the B matrix
    B = vmap(  # over elements
        vmap(  # over gauss‐points
            build_B,
            in_axes=0,  # index 0 because element axis is dropped by the first vmap
        ),
        in_axes=0,
    )(dN_phys)

    dN_phys_rot = None
    d2N_phys_rot = None
    if rotation_angles is not None:
        s = jnp.sin(rotation_angles)
        c = jnp.cos(rotation_angles)
        rotation_matrices = jnp.array([[c, -s], [s, c]])

        # Transpose to get the correct orientation in material axis
        elem_coords = jnp.einsum("eij, jke -> eik", elem_coords, rotation_matrices)

        # Compute the jacobian for each element and gauss point
        jacobian_all = jnp.einsum(
            "eIgn, enJ -> egIJ", dR_parametric_full, elem_coords
        )  # (E,G,2,2)

        # Compute the determinant and inverse of the jacobian
        J_det = jnp.linalg.det(jacobian_all)
        J_inv = jnp.linalg.inv(jacobian_all)
        point_volumes = J_det * gauss_wts[None, :] * thickness

        # Physical mapping to the first derivatives
        dN_phys_rot = jnp.einsum("egij, ejgn->egin", J_inv, dR_parametric_full)

        # Construct the J33 matrix and its inverse
        J33_all = build_J33(jacobian_all)
        J33_inv = jnp.linalg.inv(J33_all)

        # Flatten the second order parametric derivatives
        d2N_xx = d2R_parametric_full[:, 0, 0, :, :]  # [E, G, N]
        d2N_yy = d2R_parametric_full[:, 1, 1, :, :]  # [E, G, N]
        d2N_xy = d2R_parametric_full[:, 0, 1, :, :]  # [E, G, N]
        d2N_flattened = jnp.stack([d2N_xx, d2N_yy, d2N_xy], axis=2)  # [E, G, 3, N]

        # Compute the second order physical derivatives
        jcb_isp_2_phys_2 = jnp.einsum("egkn,end->egkd", d2N_flattened, elem_coords)
        Atemp = jnp.einsum("egkd,egdn->egkn", jcb_isp_2_phys_2, dN_phys_rot)
        A2temp = d2N_flattened - Atemp
        d2N_phys_rot = jnp.einsum("egij,egjn->egin", J33_inv, A2temp)
    return (
        N,
        dN_isp,
        d2N_isp,
        point_volumes,
        B,
        dN_phys,
        d2N_phys,
        E_extrap,
        dN_phys_rot,
        d2N_phys_rot,
    )


def determine_initialisation_function(
    input_dict_mesh_section: Dict[str, Any],
) -> Callable[..., Any]:
    """Determine the appropriate initialisation function based on mesh type.

    Select the correct Gauss point initialisation function based on the
    element type and number of integration points specified in the mesh
    configuration dictionary.

    Parameters
    ----------
    input_dict_mesh_section : Dict[str, Any]
        Dictionary containing mesh configuration including 'type' and
        'number_of_integration_points' keys.

    Returns
    -------
    Callable[..., Any]
        The appropriate precompute function for the specified element type
        and integration scheme.

    Raises
    ------
    ValueError
        If an unsupported mesh type is provided.
    AssertionError
        If no matching initialisation function is found for the given
        configuration, or if NURBS type does not have -1 integration points.
    """
    # Pre-compute integration point values
    precompute_matrices = None
    match input_dict_mesh_section["type"]:
        case "tri3":
            if input_dict_mesh_section["number_of_integration_points"] == 1:
                precompute_matrices = precompute_tri_1ip
        case "tri6":
            if input_dict_mesh_section["number_of_integration_points"] == 6:
                precompute_matrices = precompute_tri_6ip
        case "quad4":
            if input_dict_mesh_section["number_of_integration_points"] == 4:
                precompute_matrices = precompute_quad_4ip
        case "quad8":
            if input_dict_mesh_section["number_of_integration_points"] == 9:
                precompute_matrices = precompute_quad8_9ip
            if input_dict_mesh_section["number_of_integration_points"] == 4:
                precompute_matrices = precompute_quad8_4ip
        case "quad9":
            if input_dict_mesh_section["number_of_integration_points"] == 9:
                precompute_matrices = precompute_quad9_9ip
            if input_dict_mesh_section["number_of_integration_points"] == 4:
                precompute_matrices = precompute_quad9_4ip
        case "nurbs":
            assert (
                input_dict_mesh_section["number_of_integration_points"] == -1
            ), "NURBS mesh type expects -1 integration points in the input file; provide the integration order in the mesh dictionary."
            precompute_matrices = nurbs_point_initialisation
        case _:
            raise ValueError(
                f'Invalid mesh type. Supported types are: {TRI_TYPES + QUAD_TYPES} or "nurbs"'
            )

    # Check if the precompute_matrices is None
    assert (
        precompute_matrices is not None
    ), f"Invalid mesh type or integration point count. Supported types are: {TRI_TYPES + QUAD_TYPES}"

    return precompute_matrices
