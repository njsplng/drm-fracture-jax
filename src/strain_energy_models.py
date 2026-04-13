"""Strain energy, stress, and tangent computations using JAX."""

from abc import abstractmethod

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array

from utils import timed_jit


def reduce_voigt_for_plane_assumptions(
    C_6x6: Array,
    plane_assumption: str,
    lame_lambda: float,
    lame_mu: float,
) -> Array:
    """Reduce the 6x6 Voigt matrix to 3x3 for plane stress/strain.

    Apply plane stress or plane strain assumptions to reduce the
    constitutive matrix dimensionality.

    Parameters
    ----------
    C_6x6 : Array
        Full 6x6 constitutive matrix in Voigt notation.
    plane_assumption : str
        Plane assumption type: 'stress', 'strain', or 'none' for 3D.
    lame_lambda : float
        Lamé lambda parameter.
    lame_mu : float
        Lamé mu (shear modulus) parameter.

    Returns
    -------
    Array
        Reduced 3x3 matrix for plane modes, or unchanged 6x6 for 3D.

    Raises
    ------
    ValueError
        If plane_assumption is not 'stress', 'strain', or 'none'.
    """
    if plane_assumption == "none":
        # Keep full 6x6
        return C_6x6

    elif plane_assumption == "strain":
        # Plane strain: just extract in-plane components
        # σ_xx, σ_yy, σ_xy from ε_xx, ε_yy, γ_xy
        indices = jnp.array([0, 1, 5])
        # Voigt: [xx, yy, xy]
        C_3x3 = C_6x6[..., indices[:, None], indices]
        return C_3x3

    elif plane_assumption == "stress":
        # Plane stress: static condensation to eliminate σ_zz
        # Extract blocks
        C_aa = C_6x6[..., :2, :2]
        # xx, yy block
        C_ab = C_6x6[..., :2, 2:3]
        # coupling to zz
        C_ba = C_6x6[..., 2:3, :2]
        # coupling from zz
        C_bb = C_6x6[..., 2:3, 2:3]
        # zz-zz
        C_ss = C_6x6[..., 5:6, 5:6]
        # shear xy-xy
        C_as = C_6x6[..., :2, 5:6]
        # coupling normal to shear
        C_sa = C_6x6[..., 5:6, :2]
        # coupling shear to normal

        # Schur complement: C_reduced = C_aa - C_ab @ inv(C_bb) @ C_ba
        K = lame_lambda + 2.0 / 3.0 * lame_mu
        C_bb_safe = jnp.where(jnp.abs(C_bb) < 1e-8 * K, 1e-8 * K, C_bb)
        C_reduced_2x2 = C_aa - C_ab @ (1.0 / C_bb_safe) @ C_ba

        # Assemble 3x3: [σ_xx, σ_yy, σ_xy] from [ε_xx, ε_yy, γ_xy]
        C_3x3 = jnp.block([[C_reduced_2x2, C_as], [C_sa, C_ss]])
        return C_3x3

    else:
        raise ValueError(f"Unknown plane assumption: {plane_assumption}")


# -----------------------------------------------------------
# Base energy model class
# -----------------------------------------------------------


class BaseEnergyModel:
    """Base class for strain energy models.

    Provides the interface for computing strain energy, stress, and
    tangent modulus for various material models.

    Parameters
    ----------
    youngs_modulus : float
        Young's modulus of the material.
    poissons_ratio : float
        Poisson's ratio of the material.
    plane_assumption : str
        Plane assumption type: 'stress', 'strain', or 'none' for 3D.
    **kwargs
        Additional keyword arguments passed to model-specific attributes.
    """

    problem_type: str
    # "stress", "strain", or "none" for 3D
    plane_assumption: str = "stress"
    youngs_modulus: float
    poissons_ratio: float
    lame_mu: float
    lame_lambda: float

    def __init__(
        self,
        youngs_modulus: float,
        poissons_ratio: float,
        plane_assumption: str,
        **kwargs,
    ) -> None:
        """Initialize the base energy model."""
        # Assign input variables
        self.youngs_modulus = float(youngs_modulus)
        self.poissons_ratio = float(poissons_ratio)
        self.lame_mu = self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
        self.lame_lambda = (
            self.youngs_modulus
            * self.poissons_ratio
            / ((1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio))
        )
        self.plane_assumption = plane_assumption

        # Set the necessary kwargs
        for k, v in kwargs.items():
            if k in self.__annotations__.keys():
                self.__dict__.update({k: v})

        # Build and cache kernels
        self._build_kernels()

    def initialise_constitutive_matrices(
        self,
        point_volumes: Array,
        *args,
        **kwargs,
    ) -> Array:
        """Initialize the constitutive matrices for the energy model.

        Parameters
        ----------
        point_volumes : Array
            Array of point volumes for broadcasting.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Array
            Broadcasted constitutive matrix with shape matching point_volumes.
        """
        C = self._initialise_constitutive_matrix()
        C = jnp.broadcast_to(C, (*point_volumes.shape, *C.shape))
        C = reduce_voigt_for_plane_assumptions(
            C, self.plane_assumption, self.lame_lambda, self.lame_mu
        )
        return C

    def _initialise_constitutive_matrix(self) -> Array:
        """Initialize the constitutive matrix for the energy model.

        Returns
        -------
        Array
            6x6 isotropic elastic constitutive matrix in Voigt notation.
        """
        t1 = self.lame_lambda + 2 * self.lame_mu
        t2 = self.lame_lambda
        t3 = self.lame_mu
        C = jnp.array(
            [
                [t1, t2, t2, 0.0, 0.0, 0.0],
                [t2, t1, t2, 0.0, 0.0, 0.0],
                [t2, t2, t1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, t3, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, t3, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, t3],
            ]
        )
        return C

    # -----------------------------------------------------------
    # Strain tensor utilities
    # -----------------------------------------------------------

    def _strain_tensor_to_voigt(self, strain: Array) -> Array:
        """Convert a (3, 3) strain tensor to Voigt vector notation."""
        strain = 0.5 * (strain + strain.T)  # Ensure symmetry
        if self.plane_assumption == "none":
            return jnp.array(
                [
                    strain[0, 0],
                    strain[1, 1],
                    strain[2, 2],
                    2 * strain[1, 2],
                    2 * strain[0, 2],
                    2 * strain[0, 1],
                ]
            )
        else:
            return jnp.array([strain[0, 0], strain[1, 1], 2 * strain[0, 1]])

    def _enforce_plane_constraints_stress(self, tensor: Array) -> Array:
        """Enforce plane stress/strain constraints on a stress tensor.

        Parameters
        ----------
        tensor : Array
            Stress tensor to constrain.

        Returns
        -------
        Array
            Constrained stress tensor.
        """
        if self.plane_assumption == "stress":
            return tensor.at[2, 2].set(0.0)
        elif self.plane_assumption == "strain":
            return tensor.at[2, 2].set(
                self.poissons_ratio * (tensor[0, 0] + tensor[1, 1])
            )
        else:
            return tensor

    # -----------------------------------------------------------
    # Static utility methods
    # -----------------------------------------------------------

    @staticmethod
    def macaulay_softplus(x: Array, k: float = 1e10) -> tuple[Array, Array]:
        """Compute smoothed Macaulay brackets using the softplus function.

        Parameters
        ----------
        x : Array
            Input array to decompose.
        k : float, optional
            Softness parameter. Default is 1e10.

        Returns
        -------
        tuple[Array, Array]
            Positive and negative parts of the input.
        """
        pos = jnn.softplus(k * x) / k
        neg = x - pos
        return pos, neg

    @staticmethod
    def safe_eigvalsh(A: Array, eps: float = 1e-12) -> Array:
        """Compute eigenvalues of a symmetric matrix in a numerically stable way.

        Parameters
        ----------
        A : Array
            Symmetric matrix whose eigenvalues are computed.
        eps : float, optional
            Small perturbation for numerical stability. Default is 1e-12.

        Returns
        -------
        Array
            Eigenvalues of the symmetric matrix.
        """
        # Make exactly symmetric (guards small asymmetries from AD/assembly)
        A = 0.5 * (A + A.T)
        # Tiny anisotropic jitter to break exact degeneracy deterministically
        jitter = eps * jnp.diag(jnp.array([0.0, 1.0, 2.0], dtype=A.dtype))
        # Compute eigenvalues only (cheaper and avoids returning eigenvectors)
        w = jnp.linalg.eigvalsh(A + jitter)
        # Remove the bias introduced by the jitter (to first order)
        w = w - jnp.array([0.0, 1.0, 2.0], dtype=A.dtype) * eps
        return w

    # -----------------------------------------------------------
    # Kernel building and caching
    # -----------------------------------------------------------

    def _build_kernels(self) -> None:
        """Build and cache jitted/vmapped kernels for different input shapes."""

        # Define the function w.r.t. strains and optional constitutive matrix.
        def energy_parts_point(s: Array, C: Array | None = None) -> tuple[Array, Array]:
            """Compute the positive and negative energy parts for a strain point."""
            if C is None:
                return self._energy_parts(s)
            else:
                return self._energy_parts_with_C(s, C)

        def psi_pos(s: Array, C: Array | None = None) -> Array:
            """Compute the positive part of the strain energy."""
            return energy_parts_point(s, C)[0]

        def psi_neg(s: Array, C: Array | None = None) -> Array:
            """Compute the negative part of the strain energy."""
            return energy_parts_point(s, C)[1]

        # Gradients of the energy parts give the stress parts
        grad_pos = jax.grad(psi_pos, argnums=0)
        grad_neg = jax.grad(psi_neg, argnums=0)

        # Value-and-grad versions of the energy parts
        vng_pos = jax.value_and_grad(psi_pos, argnums=0)  # (energy_p, stress_p)
        vng_neg = jax.value_and_grad(psi_neg, argnums=0)  # (energy_m, stress_m)

        def stress_point(s: Array, C: Array | None = None) -> tuple[Array, Array]:
            """Compute the stress parts for a single strain point."""
            # Compute stress parts
            sig_p = grad_pos(s, C)
            sig_m = grad_neg(s, C)
            # Apply plane constraint deterministically (compile-time branch)
            sig_p = self._enforce_plane_constraints_stress(sig_p)
            sig_m = self._enforce_plane_constraints_stress(sig_m)

            # Return the constrained stress parts
            return sig_p, sig_m

        def parts_point(
            s: Array, C: Array | None = None
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            """Compute the stress and energy parts for a single strain point."""
            e_pos, sig_pos = vng_pos(s, C)
            e_neg, sig_neg = vng_neg(s, C)
            # Apply plane constraint deterministically (compile-time branch)
            sig_pos = self._enforce_plane_constraints_stress(sig_pos)
            sig_neg = self._enforce_plane_constraints_stress(sig_neg)
            return (sig_pos, sig_neg), (e_pos, e_neg)

        def hess_point(s: Array, d: float, C: Array | None = None) -> Array:
            """Compute the Hessian of the energy scaled by degradation."""
            return jax.hessian(lambda x: self._energy(x, d, C))(s)

        def _sanitise_stress_postprocessing(s: Array, pe_coeff: float) -> Array:
            """Apply postprocessing to the stress tensor for plane assumptions."""
            if self.plane_assumption != "none":
                s = (
                    s.at[..., 0, 2]
                    .set(0.0)
                    .at[..., 2, 0]
                    .set(0.0)
                    .at[..., 1, 2]
                    .set(0.0)
                    .at[..., 2, 1]
                    .set(0.0)
                )
            if self.plane_assumption == "strain":
                s = s.at[..., 2, 2].set(pe_coeff)
            if self.plane_assumption == "stress":
                s = s.at[..., 2, 2].set(0.0)

            return s

        def _vmises_tensor(s: Array, e: Array) -> Array:
            """Compute von Mises stress from the full stress tensor."""
            # Enforce the plane stress/strain constraints
            tr_eps = jnp.trace(e, axis1=-2, axis2=-1)
            s = _sanitise_stress_postprocessing(s, tr_eps * self.lame_lambda)

            # Extract components
            sxx, syy, szz = s[..., 0, 0], s[..., 1, 1], s[..., 2, 2]
            sxy, sxz, syz = s[..., 0, 1], s[..., 0, 2], s[..., 1, 2]

            # J2 invariant and von Mises
            J2 = (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6.0 * (sxy**2 + sxz**2 + syz**2)
            ) / 6.0
            return jnp.sqrt(jnp.maximum(0.0, 3.0 * J2))

        def _hydro_tensor(s: Array, e: Array) -> float:
            """Compute hydrostatic (mean) stress from the full stress tensor."""
            # Enforce the plane stress/strain constraints
            tr_eps = jnp.trace(e, axis1=-2, axis2=-1)
            s = _sanitise_stress_postprocessing(s, tr_eps * self.lame_lambda)

            return jnp.trace(s, axis1=-2, axis2=-1) / 3.0

        def tangent_to_voigt(C: Array) -> Array:
            """Convert a 4th-order tangent tensor to Voigt notation."""

            def apply(E: Array) -> Array:
                """Apply the 4th-order tensor to a 2nd-order tensor."""
                return jnp.einsum("...ijkl,kl->...ij", C, E)

            # In case of 3D, return full 6x6 Voigt
            if self.plane_assumption == "none":
                Z = jnp.zeros((3, 3))
                E_xx = Z.at[0, 0].set(1.0)
                E_yy = Z.at[1, 1].set(1.0)
                E_zz = Z.at[2, 2].set(1.0)
                E_yz = Z.at[1, 2].set(0.5).at[2, 1].set(0.5)  # γ_yz = 1
                E_xz = Z.at[0, 2].set(0.5).at[2, 0].set(0.5)  # γ_xz = 1
                E_xy = Z.at[0, 1].set(0.5).at[1, 0].set(0.5)  # γ_xy = 1

                S_xx, S_yy, S_zz = apply(E_xx), apply(E_yy), apply(E_zz)
                S_yz, S_xz, S_xy = apply(E_yz), apply(E_xz), apply(E_xy)

                # rows: (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)
                # cols: (ε_xx, ε_yy, ε_zz, γ_yz, γ_xz, γ_xy)
                C_voigt6 = jnp.stack(
                    [
                        jnp.stack(
                            [
                                S_xx[..., 0, 0],
                                S_yy[..., 0, 0],
                                S_zz[..., 0, 0],
                                S_yz[..., 0, 0],
                                S_xz[..., 0, 0],
                                S_xy[..., 0, 0],
                            ],
                            axis=-1,
                        ),
                        jnp.stack(
                            [
                                S_xx[..., 1, 1],
                                S_yy[..., 1, 1],
                                S_zz[..., 1, 1],
                                S_yz[..., 1, 1],
                                S_xz[..., 1, 1],
                                S_xy[..., 1, 1],
                            ],
                            axis=-1,
                        ),
                        jnp.stack(
                            [
                                S_xx[..., 2, 2],
                                S_yy[..., 2, 2],
                                S_zz[..., 2, 2],
                                S_yz[..., 2, 2],
                                S_xz[..., 2, 2],
                                S_xy[..., 2, 2],
                            ],
                            axis=-1,
                        ),
                        jnp.stack(
                            [
                                S_xx[..., 1, 2],
                                S_yy[..., 1, 2],
                                S_zz[..., 1, 2],
                                S_yz[..., 1, 2],
                                S_xz[..., 1, 2],
                                S_xy[..., 1, 2],
                            ],
                            axis=-1,
                        ),
                        jnp.stack(
                            [
                                S_xx[..., 0, 2],
                                S_yy[..., 0, 2],
                                S_zz[..., 0, 2],
                                S_yz[..., 0, 2],
                                S_xz[..., 0, 2],
                                S_xy[..., 0, 2],
                            ],
                            axis=-1,
                        ),
                        jnp.stack(
                            [
                                S_xx[..., 0, 1],
                                S_yy[..., 0, 1],
                                S_zz[..., 0, 1],
                                S_yz[..., 0, 1],
                                S_xz[..., 0, 1],
                                S_xy[..., 0, 1],
                            ],
                            axis=-1,
                        ),
                    ],
                    axis=-2,
                )
                return C_voigt6

            # In case of 2D, return 3x3 Voigt with engineering shear
            E0 = jnp.zeros((3, 3)).at[0, 0].set(1.0)
            E1 = jnp.zeros((3, 3)).at[1, 1].set(1.0)
            E2 = jnp.zeros((3, 3)).at[0, 1].set(0.5).at[1, 0].set(0.5)

            if self.plane_assumption == "stress":
                # Extract components
                Czz = C[..., 2, 2, :, :]
                Cijzz = C[..., :, :, 2, 2]
                Czzzz = C[..., 2, 2, 2, 2]

                def _safe_div(num: Array, den: Array, tiny: float) -> Array:
                    den_safe = jnp.where(jnp.abs(den) < tiny, jnp.sign(den) * tiny, den)
                    return num / den_safe

                def add_plane_stress_correction(E_in: Array) -> Array:
                    # C, Czz, Cijzz, Czzzz are already captured from the outer scope
                    # choose tiny scaled by bulk modulus:
                    K = self.lame_lambda + 2.0 / 3.0 * self.lame_mu
                    schur_tiny = 1e-10 * K
                    num = jnp.einsum("...kl,kl->...", Czz, E_in)
                    ezz = -_safe_div(num, Czzzz, schur_tiny)
                    S = apply(E_in) + ezz[..., None, None] * Cijzz
                    return jnp.nan_to_num(S)

                # Get the plane-stress terms
                S0, S1, S2 = (
                    add_plane_stress_correction(E0),
                    add_plane_stress_correction(E1),
                    add_plane_stress_correction(E2),
                )
            else:
                # Get the plane-strain terms
                S0, S1, S2 = apply(E0), apply(E1), apply(E2)

            # Assemble into 3x3 Voigt
            C_voigt3 = jnp.stack(
                [
                    jnp.stack([S0[..., 0, 0], S1[..., 0, 0], S2[..., 0, 0]], axis=-1),
                    jnp.stack([S0[..., 1, 1], S1[..., 1, 1], S2[..., 1, 1]], axis=-1),
                    jnp.stack([S0[..., 0, 1], S1[..., 0, 1], S2[..., 0, 1]], axis=-1),
                ],
                axis=-2,
            )
            return C_voigt3

        # Cache jitted/vmapped functions to use later
        self._energy_point_jit = timed_jit(energy_parts_point)
        self._energy_vmap_2 = timed_jit(
            jax.vmap(jax.vmap(energy_parts_point, in_axes=(0, 0)), in_axes=(0, 0))
        )
        self._stress_vmap_2 = timed_jit(
            jax.vmap(jax.vmap(stress_point, in_axes=(0, 0)), in_axes=(0, 0))
        )
        self._stress_energy_parts_vmap_2 = timed_jit(
            jax.vmap(jax.vmap(parts_point, in_axes=(0, 0)), in_axes=(0, 0))
        )
        self._hess_vmap_2 = timed_jit(
            jax.vmap(jax.vmap(hess_point, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
        )
        self._tangent_to_voigt_vmap_2 = timed_jit(jax.vmap(jax.vmap(tangent_to_voigt)))
        self._vmises_tensor_vmap_2 = timed_jit(jax.vmap(jax.vmap(_vmises_tensor)))
        self._hydro_tensor_vmap_2 = timed_jit(jax.vmap(jax.vmap(_hydro_tensor)))

    # -----------------------------------------------------------
    # Abstract energy computation methods
    # -----------------------------------------------------------

    @abstractmethod
    def _energy_parts(self, strains: Array) -> Array:
        """Compute the decomposed energy parts for a single strain point."""

    def _energy_parts_with_C(self, strains: Array, C: Array) -> Array:
        """Compute the decomposed energy parts with a constitutive matrix."""
        return self._energy_parts(
            strains
        )  # By default, ignore C. Override in constitutive models.

    def _energy(
        self, strains: Array, degradation: float, C: Array | None = None
    ) -> Array:
        """Compute the overall energy density according to the degradation."""
        e_pos, e_neg = self.energy_parts(strains, C)
        return degradation * e_pos + e_neg

    # -----------------------------------------------------------

    def energy_parts(
        self, strains: Array, C: Array | None = None
    ) -> tuple[Array, Array]:
        """Compute the decomposed energy parts for different input shapes."""
        if strains.ndim == 2:
            return self._energy_point_jit(strains, C)
        if strains.ndim == 4:
            return self._energy_vmap_2(strains, C)
        raise ValueError("strain must have shape (3,3), [N,3,3], or [E,G,3,3].")

    def stress_and_energy(
        self,
        strains: Array,
        C: Array | None = None,
        return_energy: bool = False,
    ) -> tuple[Array, Array] | tuple[Array, Array, Array, Array]:
        """Compute the stress parts and optionally energy density parts.

        Parameters
        ----------
        strains : Array
            Strain tensor array.
        C : Array or None, optional
            Constitutive matrix. Default is None.
        return_energy : bool, optional
            Whether to return energy parts. Default is False.

        Returns
        -------
        tuple
            Stress parts, or stress and energy parts if return_energy is True.
        """
        if not return_energy:
            return self._stress_vmap_2(strains, C)

        return self._stress_energy_parts_vmap_2(strains, C)

    def tangent(
        self, strains: Array, degradation: Array, C: Array | None = None
    ) -> Array:
        """Compute the tangent modulus in Voigt notation.

        Parameters
        ----------
        strains : Array
            Strain tensor array.
        degradation : Array
            Degradation variable array.
        C : Array or None, optional
            Constitutive matrix. Default is None.

        Returns
        -------
        Array
            Tangent modulus in Voigt notation.
        """
        return self._tangent_to_voigt_vmap_2(self._hess_vmap_2(strains, degradation, C))

    def postprocess_stress(
        self, stress: Array, strain: Array, stress_type: str | None = None
    ) -> Array:
        """Postprocess the stress tensor according to the requested type.

        Parameters
        ----------
        stress : Array
            Stress tensor to postprocess.
        strain : Array
            Strain tensor (needed for plane assumption handling).
        stress_type : str or None, optional
            Type of stress: 'von_mises', 'hydrostatic', or ''. Default is None.

        Returns
        -------
        Array
            Postprocessed stress scalar or None.

        Raises
        ------
        ValueError
            If stress_type is not recognized.
        """
        match stress_type:
            case "von_mises":
                return self._vmises_tensor_vmap_2(stress, strain)
            case "hydrostatic":
                return self._hydro_tensor_vmap_2(stress, strain)
            case "":
                return None
            case _:
                raise ValueError(f"Unknown stress type: {stress_type}")


class ConstitutiveEnergyModel(BaseEnergyModel):
    """Base implementation for constitutive material models.

    Extends the base energy model to support material anisotropy through
    constitutive matrices that can be rotated by element orientations.
    """

    def initialise_constitutive_matrices(
        self,
        point_volumes: Array,
        element_rotations: Array,
        **kwargs,
    ) -> Array:
        """Initialize constitutive matrices with element rotations.

        Parameters
        ----------
        point_volumes : Array
            Array of point volumes for broadcasting.
        element_rotations : Array
            Per-element rotation angles in radians.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Array
            Rotated and reduced constitutive matrix.
        """
        C = self._initialise_constitutive_matrix()
        C = jnp.broadcast_to(C, (*point_volumes.shape, *C.shape))
        C = self._rotate_constitutive_6x6(C, element_rotations)
        C = reduce_voigt_for_plane_assumptions(
            C, self.plane_assumption, self.lame_lambda, self.lame_mu
        )
        return C

    # -----------------------------------------------------------
    # Constitutive matrix methods
    # -----------------------------------------------------------

    @abstractmethod
    def _initialise_constitutive_matrix(self) -> Array:
        """Initialize the constitutive matrix for the first time."""

    @abstractmethod
    def _energy_parts_with_C(self, strains: Array, C: Array) -> Array:
        """Compute the decomposed energy parts with a constitutive matrix."""

    @staticmethod
    def _rotate_constitutive_6x6(C_base: Array, rotation_angles: Array) -> Array:
        """Rotate a 6x6 constitutive matrix by per-element angles.

        Apply in-plane rotation about the z-axis to the constitutive
        matrix for 2D problems.

        Parameters
        ----------
        C_base : Array
            Base 6x6 constitutive matrix in Voigt notation.
        rotation_angles : Array
            Per-element rotation angles in radians.

        Returns
        -------
        Array
            Rotated constitutive matrices with shape (E, 6, 6) or (E, G, 6, 6).
        """
        theta = -rotation_angles

        c = jnp.cos(theta)  # (E,)
        s = jnp.sin(theta)  # (E,)
        c2 = c**2
        s2 = s**2
        cs = c * s

        z = jnp.zeros_like(theta)
        o = jnp.ones_like(theta)

        # Stress-like transform A (acts on [σxx,σyy,σzz,σyz,σxz,σxy])
        A0 = jnp.stack([c2, s2, z, z, z, 2.0 * cs], axis=-1)
        A1 = jnp.stack([s2, c2, z, z, z, -2.0 * cs], axis=-1)
        A2 = jnp.stack([z, z, o, z, z, z], axis=-1)
        A3 = jnp.stack([z, z, z, c, -s, z], axis=-1)
        A4 = jnp.stack([z, z, z, s, c, z], axis=-1)
        A5 = jnp.stack([-cs, cs, z, z, z, c2 - s2], axis=-1)
        A = jnp.stack([A0, A1, A2, A3, A4, A5], axis=-2)  # (E,6,6)

        # Strain-like transform B (acts on [εxx,εyy,εzz,γyz,γxz,γxy])
        B0 = jnp.stack([c2, s2, z, z, z, cs], axis=-1)
        B1 = jnp.stack([s2, c2, z, z, z, -cs], axis=-1)
        B2 = jnp.stack([z, z, o, z, z, z], axis=-1)
        B3 = jnp.stack([z, z, z, c, -s, z], axis=-1)
        B4 = jnp.stack([z, z, z, s, c, z], axis=-1)
        B5 = jnp.stack([-2.0 * cs, 2.0 * cs, z, z, z, c2 - s2], axis=-1)
        B = jnp.stack([B0, B1, B2, B3, B4, B5], axis=-2)  # (E,6,6)

        # Inverse of B: for a pure rotation representation, inv(B(theta)) = B(-theta)
        # (this avoids a matrix inverse and is typically more stable)
        theta_m = -theta
        c = jnp.cos(theta_m)
        s = jnp.sin(theta_m)
        c2 = c * c
        s2 = s * s
        cs = c * s

        Bin0 = jnp.stack([c2, s2, z, z, z, cs], axis=-1)
        Bin1 = jnp.stack([s2, c2, z, z, z, -cs], axis=-1)
        Bin2 = jnp.stack([z, z, o, z, z, z], axis=-1)
        Bin3 = jnp.stack([z, z, z, c, -s, z], axis=-1)
        Bin4 = jnp.stack([z, z, z, s, c, z], axis=-1)
        Bin5 = jnp.stack([-2.0 * cs, 2.0 * cs, z, z, z, c2 - s2], axis=-1)
        B_inv = jnp.stack([Bin0, Bin1, Bin2, Bin3, Bin4, Bin5], axis=-2)  # (E,6,6)

        C = C_base
        # if C.ndim == 2:
        #     # (6,6) -> (E,6,6)
        #     C = jnp.broadcast_to(C, (E, 6, 6))
        #     temp = jnp.einsum("eij,ejk->eik", A, C)
        #     return jnp.einsum("eij,ejk->eik", temp, B_inv)
        #
        # if C.ndim == 3:
        #     # (E,6,6)
        #     temp = jnp.einsum("eij,ejk->eik", A, C)
        #     return jnp.einsum("eij,ejk->eik", temp, B_inv)

        # (E,G,6,6)
        temp = jnp.einsum("eij,egjk->egik", A, C)
        return jnp.einsum("egij,ejk->egik", temp, B_inv)

    # -----------------------------------------------------------
    # Energy parts implementation
    # -----------------------------------------------------------

    # skylos: ignore-start
    def _energy_parts(self, strains: Array) -> Array:
        """Compute the decomposed energy parts for a single strain point.

        Raises
        ------
        NotImplementedError
            Constitutive models must override _energy_parts_with_C.
        """
        raise NotImplementedError(
            "Constitutive energy models must implement the _energy_parts_with_C method."
        )

    # skylos: ignore-end


# -----------------------------------------------------------
# No-split energy model
# -----------------------------------------------------------


class NoSplitEnergyModel(BaseEnergyModel):
    """No-split strain energy model.

    Uses the small-strain approximation with Saint-Venant-Kirchhoff
    material model without energy splitting.
    """

    problem_type = "none"

    def _energy_parts(self, strain: Array) -> tuple[Array, Array]:
        """Compute the energy parts.

        Parameters
        ----------
        strain : Array
            Strain tensor with shape (3, 3).

        Returns
        -------
        tuple[Array, Array]
            Positive and negative energy parts. The negative part is zero.

        Raises
        ------
        AssertionError
            If strain does not have shape (3, 3).
        """
        assert strain.shape == (3, 3), "strain must have shape (3,3)."

        # Compute energy
        trace = jnp.trace(strain)
        trace_square = jnp.trace(strain @ strain)
        energy = 0.5 * self.lame_lambda * trace**2 + self.lame_mu * trace_square
        # No split, so second part is zero.
        return energy, jnp.zeros_like(energy)


# -----------------------------------------------------------
# Spectral and volumetric-deviatoric split energy models
# -----------------------------------------------------------


class SpectralSplitEnergyModel(BaseEnergyModel):
    """Spectral split strain energy model.

    Uses the small-strain approximation with Saint-Venant-Kirchhoff
    material model with spectral decomposition for energy splitting.
    """

    problem_type = "spectral"

    def _energy_parts(self, strain: Array) -> tuple[Array, Array]:
        """Compute the spectral energy parts.

        Parameters
        ----------
        strain : Array
            Strain tensor with shape (3, 3).

        Returns
        -------
        tuple[Array, Array]
            Positive and negative energy parts based on spectral decomposition.

        Raises
        ------
        AssertionError
            If strain does not have shape (3, 3).
        """
        assert strain.shape == (3, 3), "strain must have shape (3,3)."

        # Eigen-decomposition of strain
        eigvals = self.safe_eigvalsh(strain)
        lp, ln = self.macaulay_softplus(eigvals)

        # Volumetric trace part: keep as-is but smooth at 0 (optional but consistent)
        tr = jnp.sum(eigvals)
        tr_pos, tr_neg = self.macaulay_softplus(tr)

        # Energies
        energy_positive = 0.5 * self.lame_lambda * (
            tr_pos**2
        ) + self.lame_mu * jnp.sum(lp**2)
        energy_negative = 0.5 * self.lame_lambda * (
            tr_neg**2
        ) + self.lame_mu * jnp.sum(ln**2)

        return energy_positive, energy_negative


class VolumetricDeviatoricSplitEnergyModel(BaseEnergyModel):
    """Volumetric-deviatoric split strain energy model.

    Uses the small-strain approximation with Saint-Venant-Kirchhoff
    material model with volumetric-deviatoric decomposition.
    """

    problem_type = "volumetric"

    def _energy_parts(self, strain: Array) -> tuple[Array, Array]:
        """Compute the volumetric-deviatoric energy parts.

        Parameters
        ----------
        strain : Array
            Strain tensor with shape (3, 3).

        Returns
        -------
        tuple[Array, Array]
            Positive and negative energy parts. Positive part includes
            deviatoric energy, negative part is volumetric only.

        Raises
        ------
        AssertionError
            If strain does not have shape (3, 3).
        """
        assert strain.shape == (3, 3), "strain must have shape (3,3)."

        # Compute volumetric and deviatoric parts
        trace = jnp.trace(strain)
        identity = jnp.eye(3)
        strain_vol = (trace / 3.0) * identity
        strain_dev = strain - strain_vol

        # Bulk modulus
        K = self.lame_lambda + 2.0 / 3.0 * self.lame_mu
        trace_pos, trace_neg = self.macaulay_softplus(trace)

        # Compute energies
        energy_vol_positive = 0.5 * K * trace_pos**2
        energy_vol_negative = 0.5 * K * trace_neg**2
        energy_dev = self.lame_mu * jnp.trace(strain_dev @ strain_dev)

        return energy_vol_positive + energy_dev, energy_vol_negative


# -----------------------------------------------------------
# Constitutive models
# -----------------------------------------------------------


class IsotropicConstitutiveNoSplitModel(ConstitutiveEnergyModel):
    """Isotropic constitutive model without energy splitting."""

    problem_type = "none_constitutive"

    def _initialise_constitutive_matrix(self) -> Array:
        """Initialize the isotropic constitutive matrix."""
        t1 = self.lame_lambda + 2 * self.lame_mu
        t2 = self.lame_lambda
        t3 = self.lame_mu
        C = jnp.array(
            [
                [t1, t2, t2, 0.0, 0.0, 0.0],
                [t2, t1, t2, 0.0, 0.0, 0.0],
                [t2, t2, t1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, t3, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, t3, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, t3],
            ]
        )
        return C

    def _energy_parts_with_C(self, strains: Array, C: Array) -> Array:
        """Compute the energy parts with the constitutive matrix."""
        strains = self._strain_tensor_to_voigt(strains)
        energy = 0.5 * jnp.dot(strains, C @ strains)
        return energy, jnp.zeros_like(energy)


class CubicAnisotropyNoSplitModel(ConstitutiveEnergyModel):
    """Cubic anisotropy model without energy splitting.

    Attributes
    ----------
    shear_modulus : float
        Shear modulus for cubic anisotropy.
    """

    problem_type = "cubic_anisotropy_none"
    shear_modulus: float

    def _initialise_constitutive_matrix(self) -> Array:
        """Initialize the cubic anisotropy constitutive matrix."""
        coeff = self.lame_lambda + 2 * self.lame_mu
        matrix = jnp.zeros((6, 6))
        matrix = matrix.at[0, 0].set(coeff).at[1, 1].set(coeff).at[2, 2].set(coeff)
        matrix = matrix.at[0, 1].set(self.lame_lambda).at[1, 0].set(self.lame_lambda)
        matrix = matrix.at[0, 2].set(self.lame_lambda).at[2, 0].set(self.lame_lambda)
        matrix = matrix.at[2, 1].set(self.lame_lambda).at[1, 2].set(self.lame_lambda)
        matrix = (
            matrix.at[3, 3]
            .set(self.shear_modulus)
            .at[4, 4]
            .set(self.shear_modulus)
            .at[5, 5]
            .set(self.shear_modulus)
        )
        return matrix

    def _energy_parts_with_C(self, strains: Array, C: Array) -> Array:
        """Compute the energy parts with the constitutive matrix."""
        strains = self._strain_tensor_to_voigt(strains)
        energy = 0.5 * jnp.dot(strains, C @ strains)
        return energy, jnp.zeros_like(energy)


class OrthotropicAnisotropyNoSplitModel(ConstitutiveEnergyModel):
    """Orthotropic anisotropy model without energy splitting.

    Attributes
    ----------
    E_11, E_22, E_33 : float
        Young's moduli in principal directions.
    nu_12, nu_23, nu_31 : float
        Poisson's ratios.
    G_12, G_23, G_31 : float
        Shear moduli in principal planes.
    """

    problem_type = "orthotropic_anisotropy_none"
    E_11: float
    E_22: float
    E_33: float
    nu_12: float
    nu_23: float
    nu_31: float
    G_12: float
    G_23: float
    G_31: float

    def _initialise_constitutive_matrix(self) -> Array:
        """Initialize the orthotropic constitutive matrix."""
        delta = (
            1
            - self.nu_12 * self.nu_23
            - self.nu_23 * self.nu_31
            - self.nu_31 * self.nu_12
            - 2 * self.nu_12 * self.nu_23 * self.nu_31
        )
        delta /= self.E_11 * self.E_22 * self.E_33
        assert (
            delta > 0
        ), "Invalid orthotropic constraints, delta needs to be positive definite."
        # Voigt notation matrix set up
        coeff1 = (1 - self.nu_23**2) / (self.E_22 * self.E_33 * delta)
        coeff2 = (1 - self.nu_31**2) / (self.E_33 * self.E_11 * delta)
        coeff3 = (1 - self.nu_12**2) / (self.E_11 * self.E_22 * delta)
        coeff4 = (self.nu_23 + self.nu_31 * self.nu_12) / (
            self.E_11 * self.E_22 * delta
        )
        coeff5 = (self.nu_31 + self.nu_12 * self.nu_23) / (
            self.E_22 * self.E_33 * delta
        )
        coeff6 = (self.nu_12 + self.nu_31 * self.nu_23) / (
            self.E_33 * self.E_11 * delta
        )
        # Matrix population
        matrix = jnp.zeros((6, 6))
        matrix = matrix.at[0, 0].set(coeff1).at[1, 1].set(coeff2).at[2, 2].set(coeff3)
        matrix = matrix.at[1, 2].set(coeff4).at[2, 1].set(coeff4)
        matrix = matrix.at[0, 2].set(coeff5).at[2, 0].set(coeff5)
        matrix = matrix.at[0, 1].set(coeff6).at[1, 0].set(coeff6)
        matrix = matrix.at[3, 3].set(self.G_23)
        matrix = matrix.at[4, 4].set(self.G_31)
        matrix = matrix.at[5, 5].set(self.G_12)
        return matrix

    def _energy_parts_with_C(self, strains: Array, C: Array) -> Array:
        """Compute the energy parts with the constitutive matrix."""
        strains = self._strain_tensor_to_voigt(strains)
        energy = 0.5 * jnp.dot(strains, C @ strains)
        return energy, jnp.zeros_like(energy)


# -----------------------------------------------------------
# Factory function to get energy model by name
# -----------------------------------------------------------


def get_energy_model(
    name: str,
    E: float,
    nu: float,
    plane_mode: str,
    **kwargs,
) -> BaseEnergyModel:
    """Factory function to get the appropriate energy model.

    Parameters
    ----------
    name : str
        Name of the energy model type.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    plane_mode : str
        Plane assumption: 'stress', 'strain', or 'none' for 3D.
    **kwargs
        Additional keyword arguments for model-specific parameters.

    Returns
    -------
    BaseEnergyModel
        Initialized energy model instance.

    Raises
    ------
    ValueError
        If the energy model name is not recognized.
    """
    # Define the registry for the energy models
    reg = {}

    def walk(c: type) -> None:
        """Recursively walk through subclasses to populate the registry."""
        for s in c.__subclasses__():
            pt = getattr(s, "problem_type", None)
            if pt is not None:
                reg[str(pt).lower()] = s
            walk(s)

    # Fetch the model types available
    walk(BaseEnergyModel)

    # Get the requested model type
    model_type = reg.get(name.lower(), None)

    # If possible, initialise and return the model. Otherwise, raise error.
    if model_type is not None:
        return model_type(E, nu, plane_mode, **kwargs)
    else:
        raise ValueError(f"Unknown energy model: {name}")
