"""Define the base model class and specific neural network model classes.

This module provides the BaseModel abstract base class and the Default
model implementation for physics-informed neural networks.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array

from data_handling import (
    DistanceFunctions,
    IPData,
    MaterialParameters,
    MeshData,
    ProblemParameters,
)
from strain_models import nodal_displacement_to_strain, voigt_strain_to_tensor
from utils import timed_filter_jit


class BaseModel(eqx.Module, ABC):
    """Base class for all neural network models used in this framework.

    Attributes
    ----------
    network : eqx.Module
        The neural network module.
    ip_data : IPData
        Integration point data.
    distance_functions : DistanceFunctions
        Distance functions for boundary conditions.
    material_parameters : MaterialParameters
        Material parameters.
    problem_parameters : ProblemParameters
        Problem parameters.
    mesh_data : MeshData
        Mesh data.
    phasefield_constraint : Callable
        Constraint function for phase field.
    displacement_increment : float
        Displacement increment.
    previous_phasefield : tuple
        Previous phase field values.
    loading_angle : float
        Loading angle.
    strain_energy_model : eqx.Module
        Strain energy model.
    phasefield_model : eqx.Module
        Phase field model.
    constitutive_matrices : Array
        Constitutive matrices.
    """

    network: eqx.Module
    ip_data: IPData = eqx.field()
    distance_functions: DistanceFunctions = eqx.field()
    material_parameters: MaterialParameters = eqx.field()
    problem_parameters: ProblemParameters = eqx.field()
    mesh_data: MeshData = eqx.field()
    phasefield_constraint: Callable = eqx.field()
    displacement_increment: float = eqx.field()
    previous_phasefield: tuple = eqx.field()
    loading_angle: float = eqx.field()
    strain_energy_model: eqx.Module = eqx.field()
    phasefield_model: eqx.Module = eqx.field()
    constitutive_matrices: Array = eqx.field()

    def __init__(
        self,
        network: eqx.Module,
        material_parameters: MaterialParameters,
        problem_parameters: ProblemParameters,
        phasefield_constraint: Callable,
        displacement_increment: float = 1e-12,
        ip_data: Optional[IPData] = None,
        mesh_data: Optional[MeshData] = None,
        distance_functions: Optional[DistanceFunctions] = None,
        previous_phasefield: Optional[Array] = None,
        loading_angle: float = 0.0,
        strain_energy_model: Optional[eqx.Module] = None,
        phasefield_model: Optional[eqx.Module] = None,
    ) -> None:
        """Initialize the base model."""
        self.network = network
        self.ip_data = ip_data
        self.distance_functions = distance_functions
        self.material_parameters = material_parameters
        self.problem_parameters = problem_parameters
        self.mesh_data = mesh_data
        self.phasefield_constraint = phasefield_constraint
        self.displacement_increment = displacement_increment
        self.previous_phasefield = tuple(previous_phasefield.tolist())
        self.loading_angle = loading_angle
        self.strain_energy_model = strain_energy_model
        self.phasefield_model = phasefield_model
        self.initialise_constitutive_matrices()

    @abstractmethod
    def __call__(self, x: Array) -> tuple[Array, Array, Array]:
        """Forward pass of the model.

        To be implemented by the specific model classes.

        Parameters
        ----------
        x : Array
            Input coordinates.

        Returns
        -------
        u : Array
            Horizontal displacement.
        v : Array
            Vertical displacement.
        c : Array
            Phase field.
        """
        return

    def initialise_constitutive_matrices(self) -> None:
        """Initialize the constitutive matrices."""
        matrices = self.strain_energy_model.initialise_constitutive_matrices(
            self.ip_data.volumes, self.ip_data.rotation_angles
        )
        object.__setattr__(self, "constitutive_matrices", matrices)

    @timed_filter_jit
    def loss_energy(
        self,
        net_output: Array,
    ) -> tuple[Array, Array, Array]:
        """Compute the energy loss terms.

        Parameters
        ----------
        net_output : Array
            Network output containing displacement and phase field.

        Returns
        -------
        E_elastic : Array
            Elastic strain energy.
        E_fracture : Array
            Fracture energy.
        E_irreversibility : Array
            Irreversibility energy.
        """
        u, v, c = net_output

        # Form the elemental phasefield values
        c_elemental = c[self.mesh_data.connectivities]
        c_elemental = jnp.repeat(
            c_elemental[:, None, :], self.ip_data.N.shape[1], axis=1
        )
        c_elemental = 1 - c_elemental

        # Get the degradation at the integration points
        ip_g = self.phasefield_model.degradation_in_ip(c_elemental, self.ip_data.N)

        # Get the fracture energy
        ip_fracture_energy_density = self.phasefield_model.energy_density(
            c_elemental=c_elemental,
            N=self.ip_data.N,
            dNdx=(
                self.ip_data.physical_derivatives_rot
                if self.ip_data.physical_derivatives_rot is not None
                else self.ip_data.physical_derivatives
            ),
            d2Ndx2=(
                self.ip_data.physical_derivatives_2_rot
                if self.ip_data.physical_derivatives_2_rot is not None
                else self.ip_data.physical_derivatives_2
            ),
            gamma=self.ip_data.gamma_matrix,
        )
        ip_fracture_energy = ip_fracture_energy_density * self.ip_data.volumes

        # Compute the irreversibility energy
        irreversibility_penalty = (
            (self.material_parameters.G_c / self.material_parameters.l_0)
            * (1.0 / self.material_parameters.pf_irreversibility_tolerance**2 - 1)
            * self.material_parameters.energy_scaling
        )
        irreversibility_energy = (
            0.5
            * irreversibility_penalty
            * jnn.relu(jnp.array(self.previous_phasefield) - c) ** 2
        )

        if self.problem_parameters.problem_type == "linear_elasticity":
            ip_g = jnp.ones_like(self.ip_data.volumes)
            ip_fracture_energy = jnp.array([0.0])
            irreversibility_energy = jnp.array([0.0])

        # Shape the NN outputs into FEM-like nodal displacements
        nodal_displacements = jnp.stack((u, v), axis=-1).flatten()

        # Get the strains
        ip_strain = nodal_displacement_to_strain(
            nodal_displacements=(
                nodal_displacements[::2],
                nodal_displacements[1::2],
            ),
            connectivity=self.mesh_data.connectivities,
            B=self.ip_data.B,
        )
        ip_strain_full = voigt_strain_to_tensor(
            strains=ip_strain,
            nu=self.material_parameters.nu,
            plane_mode=self.problem_parameters.plane_mode,
        )
        ip_strain_density_plus, ip_strain_density_minus = (
            self.strain_energy_model.energy_parts(
                ip_strain_full, self.constitutive_matrices
            )
        )
        ip_strain_energy = ip_strain_density_plus * ip_g + ip_strain_density_minus
        ip_strain_energy = ip_strain_energy * self.ip_data.volumes

        return (
            jnp.sum(ip_strain_energy),
            jnp.sum(ip_fracture_energy),
            jnp.sum(irreversibility_energy),
        )

    def loss_weight_decay(self, coefficient: float) -> float:
        """Compute the weight decay loss for the model.

        Parameters
        ----------
        coefficient : float
            Weight decay coefficient.

        Returns
        -------
        loss : float
            Weight decay loss value.
        """
        # Partition the model into trainable parameters vs. static objects.
        params, _ = eqx.partition(self.network, eqx.is_inexact_array)

        # Build the weight decay term
        decay_tree = jax.tree_util.tree_map(lambda p: jnp.sum(jnp.square(p)), params)

        # Zero out any activation parameters
        if hasattr(self.network, "activation_list"):
            for i, act in enumerate(self.network.activation_list):
                # Zero out the activation parameters
                zero_subtree = jax.tree_util.tree_map(lambda _: 0.0, act)
                decay_tree = eqx.tree_at(
                    lambda n, i=i: n.activation_list[i],
                    decay_tree,
                    replace=zero_subtree,
                )

        # Trainable resnets: zero out the skip alpha parameters
        if hasattr(self.network, "skip_alphas"):
            for i, alpha in enumerate(self.network.skip_alphas):
                # Zero out the skip alpha parameters
                zero_subtree = jax.tree_util.tree_map(lambda _: 0.0, alpha)
                decay_tree = eqx.tree_at(
                    lambda n, i=i: n.skip_alphas[i],
                    decay_tree,
                    replace=zero_subtree,
                )

        # Projected resnets: zero out the projection matrices
        if hasattr(self.network, "projection_input"):
            decay_tree = eqx.tree_at(
                lambda n: n.projection_input, decay_tree, replace=0.0
            )
        if hasattr(self.network, "projection_output"):
            decay_tree = eqx.tree_at(
                lambda n: n.projection_output, decay_tree, replace=0.0
            )

        loss = sum(jax.tree_util.tree_leaves(decay_tree))
        return coefficient * loss

    def get_loss_values(
        self,
        loss_terms: dict,
        weight_decay: float = 1e-5,
        energy_scale: str = "log",
    ) -> float:
        """Get the loss values for the model for plotting the loss landscapes.

        Parameters
        ----------
        loss_terms : dict
            Dictionary specifying which loss terms to include.
        weight_decay : float, optional
            Weight decay coefficient. Default is 1e-5.
        energy_scale : str, optional
            Scaling for energy terms ('log' or 'linear'). Default is 'log'.

        Returns
        -------
        loss : float
            Total loss value.
        """
        loss = 0.0
        E_elastic, E_damage, E_irreversibility = self.loss_energy(
            self(self.mesh_data.nodal_coordinates)
        )

        if loss_terms["energy_elastic"]:
            loss += E_elastic
        if loss_terms["energy_phasefield"]:
            loss += E_damage
        if loss_terms["energy_irreversibility"]:
            loss += E_irreversibility

        if energy_scale == "log":
            loss = jnp.log10(loss)

        if loss_terms["weight_decay"] and weight_decay > 0:
            loss += self.loss_weight_decay(weight_decay)

        return loss


class Default(BaseModel):
    """Default model that sets up boundary conditions and neural network.

    This model applies Dirichlet boundary conditions and loading window
    constraints to the neural network outputs.
    """

    @timed_filter_jit
    def __call__(
        self,
        x: Array,
        **kwargs,
    ) -> tuple[Array, Array, Array]:
        """Forward pass of the Default model.

        Parameters
        ----------
        x : Array
            Nodal coordinates.

        Returns
        -------
        u : Array
            Horizontal displacement.
        v : Array
            Vertical displacement.
        c : Array
            Phase field.
        """
        out = jax.vmap(self.network)(x)
        u_pred = out[:, 0]
        v_pred = out[:, 1]
        c = out[:, 2]

        load_window_x = self.distance_functions.load_window * jnp.cos(
            jnp.deg2rad(self.loading_angle)
        )
        load_window_y = self.distance_functions.load_window * jnp.sin(
            jnp.deg2rad(self.loading_angle)
        )

        # Dirichlet boundary conditions and loading window for displacement
        u = (
            u_pred * (1 - self.distance_functions.fixed_window_x) * (1 - load_window_x)
            + load_window_x
        ) * self.displacement_increment
        v = (
            v_pred * (1 - self.distance_functions.fixed_window_y) * (1 - load_window_y)
            + load_window_y
        ) * self.displacement_increment

        c = self.phasefield_constraint(c)

        return u, v, c


def get_model_nn() -> BaseModel:
    """Factory function to get the model based on the type.

    Returns
    -------
    model : type
        The Default model class.
    """
    return Default
