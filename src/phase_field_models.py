"""Phase field fracture models and energy computations."""

from abc import abstractmethod
from typing import Optional, Type

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from utils import timed_jit


class BasePhasefieldModel:
    """Base class for phase field fracture models.

    Parameters
    ----------
    G_c : float
        Critical energy release rate.
    l_0 : float
        Length scale parameter.
    invert : bool, optional
        Whether to invert damage. Default is False.
    """

    # AT1/AT2/PF4/AN
    problem_type: str
    # 2 or 4
    order: int
    G_c: float
    l_0: float

    def __init__(self, G_c: float, l_0: float, invert: bool = False) -> None:
        # Set the model parameters
        self.G_c = G_c
        self.l_0 = l_0
        self.invert_damage = invert

        self._build_kernels()

    def _build_kernels(self) -> None:
        """Build the JIT-compiled kernels."""
        energy_function = (
            lambda c, N, dNdx, H=0, d2=None, gam=None: self._energy_density(
                c, N, dNdx, H, d2, gam
            )
        )
        self._energy_point_jit = timed_jit(energy_function)

        # Map over all of the quantities except for the gamma matrix
        in_axes = (0, 0, 0, 0, 0, None)

        self._energy_vmap_2 = timed_jit(
            jax.vmap(jax.vmap(energy_function, in_axes), in_axes)
        )
        self._residual_vmap_2 = timed_jit(
            jax.vmap(jax.vmap(jax.grad(energy_function), in_axes), in_axes)
        )
        self._hessian_vmap_2 = timed_jit(
            jax.vmap(jax.vmap(jax.hessian(energy_function), in_axes), in_axes)
        )

        def _degradation_in_ip(c_elemental: Array, N: Array) -> Array:
            """Compute degradation at integration points."""
            c_ip = self._interp_c_ip(c_elemental, N)
            return self._degradation(c_ip)

        def _grad_in_ip(c_elemental: Array, dNdx: Array) -> Array:
            """Compute gradient at integration points."""
            return self._grad_c_ip(c_elemental, dNdx)

        def _hess_in_ip(c_elemental: Array, d2Ndx2: Array) -> Array:
            """Compute Hessian-like term at integration points."""
            return self._hess_like_c_ip(c_elemental, d2Ndx2)

        # Map over degradation, gradient, and Hessian
        self._degradation_vmap_2 = timed_jit(jax.vmap(jax.vmap(_degradation_in_ip)))
        self._grad_vmap_2 = timed_jit(jax.vmap(jax.vmap(_grad_in_ip)))
        self._hess_vmap_2 = timed_jit(jax.vmap(jax.vmap(_hess_in_ip)))

    @staticmethod
    def _laplacian_from_d2c(d2c_ip: Optional[Array]) -> Array:
        """Compute the Laplacian of c from second derivatives at a Gauss point.

        Parameters
        ----------
        d2c_ip : Array
            Second derivatives of c in Voigt notation (3 for 2D, 6 for 3D)
            or full Hessian matrix.

        Returns
        -------
        laplacian : Array
            The Laplacian of c at the Gauss point.

        Raises
        ------
        ValueError
            If d2c_ip is None or has unsupported shape.
        """
        if d2c_ip is None:
            raise ValueError("PF-4 isotropic requires second derivatives (d2c_ip).")

        if d2c_ip.ndim == 1:
            s = d2c_ip.shape[0]
            # 2D Voigt
            if s == 3:
                return d2c_ip[0] + d2c_ip[1]
            # 3D Voigt
            elif s == 6:
                return d2c_ip[0] + d2c_ip[1] + d2c_ip[2]
            else:
                raise ValueError(
                    f"Unsupported Voigt length {s}. Expected 3 (2D) or 6 (3D)."
                )
        elif d2c_ip.ndim == 2:
            # Full (dim, dim) Hessian
            return jnp.trace(d2c_ip)
        else:
            raise ValueError(f"Unexpected d2c_ip shape {d2c_ip.shape}.")

    @abstractmethod
    def _energy_surface(
        self,
        c_ip: Array,
        dc_ip: Array,
        d2c_ip: Optional[Array] = None,
        gamma: Optional[Array] = None,
    ) -> Array:
        """Compute the fracture surface energy density.

        Parameters
        ----------
        c_ip : Array
            Phase field value at integration points.
        dc_ip : Array
            Gradient of phase field at integration points.
        d2c_ip : Array, optional
            Second derivatives of phase field. Default is None.
        gamma : Array, optional
            Anisotropy tensor. Default is None.

        Returns
        -------
        psi_surface : Array
            Surface energy density at integration points.
        """

    @staticmethod
    def _interp_c_ip(c_e: Array, N: Array) -> Array:
        """Interpolate phase field from element to Gauss point."""
        return jnp.einsum("n,n->", N, c_e)

    @staticmethod
    def _grad_c_ip(c_e: Array, dNdx: Array) -> Array:
        """Compute the gradient of phase field at Gauss points."""
        return jnp.einsum("dn,n->d", dNdx, c_e)

    @staticmethod
    def _hess_like_c_ip(c_e: Array, d2Ndx2: Optional[Array]) -> Optional[Array]:
        """Compute second derivatives of phase field at Gauss points.

        Parameters
        ----------
        c_e : Array
            Element phase field values.
        d2Ndx2 : Array, optional
            Second derivatives of shape functions. Default is None.

        Returns
        -------
        d2c : Array or None
            Second derivatives of phase field, or None if d2Ndx2 is None.
        """
        if d2Ndx2 is None:
            return None
        # Model-specific contraction
        return jnp.einsum("dn,n->d", d2Ndx2, c_e)

    @staticmethod
    def _degradation(c: Array) -> Array:
        """Compute the degradation factor from phase field value."""
        return c**2

    def _energy_density(
        self,
        c_elemental: Array,
        N: Array,
        dNdx: Array,
        history_field: Array,
        d2Ndx2: Optional[Array] = None,
        gamma: Optional[Array] = None,
    ) -> Array:
        """Compute the phase field energy density.

        Compute the sum of surface energy and irreversibility term for
        the phase field fracture model.

        Parameters
        ----------
        c_elemental : Array
            Element phase field values.
        N : Array
            Shape function values at integration points.
        dNdx : Array
            Shape function gradients at integration points.
        history_field : Array
            History field for irreversibility condition.
        d2Ndx2 : Array, optional
            Second derivatives of shape functions. Default is None.
        gamma : Array, optional
            Anisotropy tensor. Default is None.

        Returns
        -------
        energy : Array
            Total energy density at integration points.
        """
        # Interpolate to obtain the necessary quantities
        c_ip = self._interp_c_ip(c_elemental, N)
        dc_ip = self._grad_c_ip(c_elemental, dNdx)
        d2c_ip = None
        if self.order == 4:
            d2c_ip = self._hess_like_c_ip(c_elemental, d2Ndx2)

        # Obtain the energy surface
        energy_surface = self._energy_surface(
            c_ip=c_ip, dc_ip=dc_ip, d2c_ip=d2c_ip, gamma=gamma
        )

        # Get the degraded tensile driver
        g = self._degradation(c_ip)
        if history_field is None:
            history_field = jnp.zeros_like(g)
        Hf = lax.stop_gradient(history_field)

        return energy_surface + g * Hf

    def energy_density(
        self,
        c_elemental: Array,
        N: Array,
        dNdx: Array,
        history_field: Optional[Array] = None,
        d2Ndx2: Optional[Array] = None,
        gamma: Optional[Array] = None,
    ) -> Array:
        """Compute the fracture energy density.

        Parameters
        ----------
        c_elemental : Array
            Element phase field values.
        N : Array
            Shape function values at integration points.
        dNdx : Array
            Shape function gradients at integration points.
        history_field : Array, optional
            History field for irreversibility. Default is None.
        d2Ndx2 : Array, optional
            Second derivatives of shape functions. Default is None.
        gamma : Array, optional
            Anisotropy tensor. Default is None.

        Returns
        -------
        energy : Array
            Fracture energy density.
        """
        return self._energy_vmap_2(c_elemental, N, dNdx, history_field, d2Ndx2, gamma)

    def residual(
        self,
        c_elemental: Array,
        N: Array,
        dNdx: Array,
        history_field: Array,
        d2Ndx2: Optional[Array] = None,
        gamma: Optional[Array] = None,
    ) -> Array:
        """Compute the residual for the phase field equation.

        Parameters
        ----------
        c_elemental : Array
            Element phase field values.
        N : Array
            Shape function values at integration points.
        dNdx : Array
            Shape function gradients at integration points.
        history_field : Array
            History field for irreversibility condition.
        d2Ndx2 : Array, optional
            Second derivatives of shape functions. Default is None.
        gamma : Array, optional
            Anisotropy tensor. Default is None.

        Returns
        -------
        residual : Array
            First derivative of energy density w.r.t. phase field.
        """
        return self._residual_vmap_2(c_elemental, N, dNdx, history_field, d2Ndx2, gamma)

    def tangent(
        self,
        c_elemental: Array,
        N: Array,
        dNdx: Array,
        history_field: Array,
        d2Ndx2: Optional[Array] = None,
        gamma: Optional[Array] = None,
    ) -> Array:
        """Compute the tangent stiffness for the phase field equation.

        Parameters
        ----------
        c_elemental : Array
            Element phase field values.
        N : Array
            Shape function values at integration points.
        dNdx : Array
            Shape function gradients at integration points.
        history_field : Array
            History field for irreversibility condition.
        d2Ndx2 : Array, optional
            Second derivatives of shape functions. Default is None.
        gamma : Array, optional
            Anisotropy tensor. Default is None.

        Returns
        -------
        tangent : Array
            Second derivative of energy density w.r.t. phase field.
        """
        return (2 * self.l_0 / self.G_c) * self._hessian_vmap_2(
            c_elemental, N, dNdx, history_field, d2Ndx2, gamma
        )

    def degradation_in_ip(self, c_elemental: Array, N: Array) -> Array:
        """Compute the degradation factor at integration points.

        Parameters
        ----------
        c_elemental : Array
            Element phase field values.
        N : Array
            Shape function values at integration points.

        Returns
        -------
        degradation : Array
            Degradation factor at integration points.
        """
        return self._degradation_vmap_2(c_elemental, N)

    def grad_in_ip(self, c_elemental: Array, dNdx: Array) -> Array:
        """Compute the gradient of phase field at integration points.

        Parameters
        ----------
        c_elemental : Array
            Element phase field values.
        dNdx : Array
            Shape function gradients at integration points.

        Returns
        -------
        grad : Array
            Gradient of phase field at integration points.
        """
        return self._grad_vmap_2(c_elemental, dNdx)

    def hess_in_ip(
        self, c_elemental: Array, d2Ndx2: Optional[Array]
    ) -> Optional[Array]:
        """Compute the Hessian-like term at integration points.

        Parameters
        ----------
        c_elemental : Array
            Element phase field values.
        d2Ndx2 : Array, optional
            Second derivatives of shape functions.

        Returns
        -------
        hess : Array or None
            Hessian-like term at integration points, or None if d2Ndx2 is None.
        """
        return self._hess_vmap_2(c_elemental, d2Ndx2)


class AT1PhasefieldModel(BasePhasefieldModel):
    """AT1 phase field fracture model.

    The AT1 model uses a linear regularization for the phase field.
    """

    problem_type = "at1"
    order = 2

    def _energy_surface(self, c_ip: Array, dc_ip: Array, **kwargs) -> Array:
        """Compute the AT1 surface energy density.

        Parameters
        ----------
        c_ip : Array
            Phase field value at integration points.
        dc_ip : Array
            Gradient of phase field at integration points.

        Returns
        -------
        psi_surface : Array
            AT1 surface energy density.
        """
        psi_surface = self.G_c * (
            (c_ip - 1) / self.l_0 + self.l_0 * jnp.dot(dc_ip, dc_ip)
        )
        return psi_surface


class AT2PhasefieldModel(BasePhasefieldModel):
    """AT2 phase field fracture model.

    The AT2 model uses a quadratic regularization for the phase field.
    """

    problem_type = "at2"
    order = 2

    def _energy_surface(self, c_ip: Array, dc_ip: Array, **kwargs) -> Array:
        """Compute the AT2 surface energy density.

        Parameters
        ----------
        c_ip : Array
            Phase field value at integration points.
        dc_ip : Array
            Gradient of phase field at integration points.

        Returns
        -------
        psi_surface : Array
            AT2 surface energy density.
        """
        psi_surface = self.G_c * (
            (c_ip - 1) ** 2 / (4 * self.l_0) + self.l_0 * jnp.dot(dc_ip, dc_ip)
        )
        return psi_surface


class LinearElasticityModel(BasePhasefieldModel):
    """Linear elasticity model for compatibility.

    A placeholder model that inherits from BasePhasefieldModel
    for interface compatibility with linear elasticity problems.
    """

    problem_type = "linear_elasticity"
    order = 2

    # skylos: ignore-start
    def _energy_surface(self, c_ip: Array, dc_ip: Array, **kwargs) -> Array:
        """Placeholder energy surface for linear elasticity.

        Returns
        -------
        None
            Always returns None as this is a placeholder model.
        """
        return None

    # skylos: ignore-end


class Isotropic4thOrderPhasefieldModel(BasePhasefieldModel):
    """4th-order isotropic phase field fracture model.

    A higher-order regularization model that includes both gradient
    and Laplacian terms for improved accuracy.
    """

    problem_type = "isotropic-4"
    order = 4

    def _energy_surface(
        self,
        c_ip: Array,
        dc_ip: Array,
        d2c_ip: Array,
        **kwargs,
    ) -> Array:
        """Compute the 4th-order isotropic surface energy density.

        Parameters
        ----------
        c_ip : Array
            Phase field value at integration points.
        dc_ip : Array
            Gradient of phase field at integration points.
        d2c_ip : Array
            Second derivatives of phase field.

        Returns
        -------
        psi_surface : Array
            4th-order isotropic surface energy density.
        """

        laplacian = self._laplacian_from_d2c(d2c_ip)
        grad_sq = jnp.dot(dc_ip, dc_ip)

        psi_surface = self.G_c * (
            (c_ip - 1) ** 2 / (4 * self.l_0)
            + (self.l_0 / 2) * grad_sq
            + (self.l_0**3 / 4) * laplacian**2
        )

        return psi_surface


class AnisotropicPhasefieldModel(BasePhasefieldModel):
    """Anisotropic phase field fracture model.

    An AT2-based model with anisotropic regularization for
    direction-dependent fracture behavior.
    """

    problem_type = "anisotropic"
    order = 4

    def _energy_surface(
        self,
        c_ip: Array,
        dc_ip: Array,
        d2c_ip: Array,
        gamma: Array,
    ) -> Array:
        """Compute the anisotropic surface energy density.

        Parameters
        ----------
        c_ip : Array
            Phase field value at integration points.
        dc_ip : Array
            Gradient of phase field at integration points.
        d2c_ip : Array
            Second derivatives of phase field.
        gamma : Array
            Anisotropy tensor.

        Returns
        -------
        psi_surface : Array
            Anisotropic surface energy density.
        """
        grad_sq = jnp.dot(dc_ip, dc_ip)
        an_term = jnp.dot(d2c_ip, gamma @ d2c_ip)
        psi_surface = (
            self.G_c * ((c_ip - 1.0) ** 2 / (4.0 * self.l_0) + self.l_0 * grad_sq)
            + self.G_c * (self.l_0**3) * an_term
        )
        return psi_surface


def get_phasefield_model(name: str, G_c: float, l_0: float) -> BasePhasefieldModel:
    """Factory function to create a phase field model instance.

    Parameters
    ----------
    name : str
        Name of the model ('at1', 'at2', 'isotropic-4', 'anisotropic').
    G_c : float
        Critical energy release rate.
    l_0 : float
        Length scale parameter.

    Returns
    -------
    model : BasePhasefieldModel
        An instance of the requested phase field model.

    Raises
    ------
    ValueError
        If the requested model name is not found.
    """
    # Define the registry for the energy models
    reg = {}

    def walk(c: Type) -> None:
        """Recursively walk through subclasses to populate the registry."""
        for s in c.__subclasses__():
            pt = getattr(s, "problem_type", None)
            if pt is not None:
                reg[str(pt).lower()] = s
            walk(s)

    # Fetch the model types available
    walk(BasePhasefieldModel)

    # Get the requested model type
    model_type = reg.get(name.lower(), None)

    # If possible, initialise and return the model. Otherwise, raise error.
    if model_type is not None:
        return model_type(G_c, l_0)
    else:
        raise ValueError(f"Unknown phasefield model: {name}")
