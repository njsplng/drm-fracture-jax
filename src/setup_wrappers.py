"""Host major setup functions used in the main scripts."""

import pathlib
import sys
from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array

# from compute_energy import compute_fracture_energy
from data_handling import DistanceFunctionsClass, IPDataClass, MeshDataClass
from distance_functions import CompositeDistanceFunction
from initialise_points import determine_initialisation_function
from mesh import load_and_rescale_mesh

# Set up the necessary paths
current_path = pathlib.Path(__file__).parent.parent.resolve()

# Link the fem source
fem_source_path = current_path / "fem" / "src"
if fem_source_path not in sys.path:
    sys.path.append(str(fem_source_path))
from nodal_computations import solve_phasefield


def setup_geometry(
    config: dict,
    pretraining_mesh: Optional[dict] = None,
    dataclasses_frozen: bool = True,
) -> Tuple[
    "MeshData",
    "IPData",
    "DistanceFunctions",
    Tuple[
        CompositeDistanceFunction,
        CompositeDistanceFunction,
        CompositeDistanceFunction,
    ],
]:
    """Set up the geometry for the simulation.

    Assemble the mesh data, integration point data, and distance
    functions. Allows overriding the mesh parameters with a
    pretraining mesh if provided.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing mesh and problem
        parameters.
    pretraining_mesh : dict, optional
        Pretraining mesh parameters to override config mesh.
        Default is None.
    dataclasses_frozen : bool, optional
        Whether to create frozen dataclasses. Default is True.

    Returns
    -------
    tuple
        A tuple containing (mesh_data, ip_data, distance_functions)
        where mesh_data is the mesh data object, ip_data is the
        integration point data object, and distance_functions is
        the distance functions object.
    """
    # Assemble the dataclasses
    MeshData = MeshDataClass(frozen=dataclasses_frozen)
    IPData = IPDataClass(frozen=dataclasses_frozen)
    DistanceFunctions = DistanceFunctionsClass(frozen=dataclasses_frozen)

    mesh_params = config.mesh
    if pretraining_mesh is not None:
        mesh_params = pretraining_mesh

    # Parse in the parent mesh
    nodal_coordinates, connectivities, dofs, aux_nurbs = load_and_rescale_mesh(
        mesh_params, config.problem_domain
    )
    ip_initialiser = determine_initialisation_function(mesh_params)
    mesh_data = MeshData(
        nodal_coordinates=nodal_coordinates,
        connectivities=connectivities,
        dofs=dofs,
        aux_nurbs=aux_nurbs,
    )
    ip_data = IPData.from_initializer(
        ip_initialiser,
        connectivities,
        nodal_coordinates,
        config.material_parameters.thickness,
        aux_nurbs,
        config.problem_type,
        config.gamma_parameters,
        config.material_rotation,
        config.problem_type == "anisotropic"
        or "anisotropy" in config.strain_split
        or config.strain_split == "none_constitutive",
    )
    distance_functions = DistanceFunctions.from_parameters(
        fixed_window_params=config.boundary_conditions.fixed_window_parameters,
        load_window_params=config.boundary_conditions.load_window_parameters,
        nodal_coordinates=nodal_coordinates,
        fixed_window_directions=config.boundary_conditions.fixed_window_directions,
    )
    return (mesh_data, ip_data, distance_functions)


def setup_initial_phasefield(
    pf_params: dict,
    mesh_data: "MeshData",
    ip_data: "IPData",
    material_parameters: "MaterialParameters",
    phasefield_model: "BasePhasefieldModel",
) -> Tuple[Array, Array, Array]:
    """Set up the initial phasefield values based on parameters.

    Compute the initial phasefield distribution from crack
    parameters and return the history field, degradation,
    and phasefield values.

    Parameters
    ----------
    pf_params : dict
        Phasefield parameters including initial crack parameters.
    mesh_data : MeshData
        Mesh data object containing nodal coordinates and
        connectivities.
    ip_data : IPData
        Integration point data object.
    material_parameters : MaterialParameters
        Material parameters object.
    phasefield_model : BasePhasefieldModel
        Phasefield model object.

    Returns
    -------
    tuple
        A tuple of three arrays: ip_history_field, ip_g,
        and phasefield values.
    """
    # Set/extract the parameters
    phasefield_value_target = 1e-3
    crack_parameters = pf_params.initial_crack_parameters

    # Condition the distance function
    for entry in crack_parameters:
        entry["d0"] = material_parameters.l_0
        entry["order"] = 1

    # Get the coefficients
    B = 1 / phasefield_value_target - 1
    coefficient = B * material_parameters.G_c / (4 * material_parameters.l_0)

    # Get the distance function to the nodes
    phasefield_distance_fn = CompositeDistanceFunction(crack_parameters)
    crack_decay = phasefield_distance_fn(mesh_data.nodal_coordinates)

    # Get the nodal history field and extrapolate to the integration points
    nodal_history_field = coefficient * crack_decay
    elemental_history_field = nodal_history_field[mesh_data.connectivities]
    ip_history_field = jnp.einsum("en, egn -> eg", elemental_history_field, ip_data.N)

    # Compute initial phasefield values
    phasefield = jnp.ones(mesh_data.nodal_coordinates.shape[0])
    phasefield = solve_phasefield(
        phasefield, phasefield_model, ip_data, mesh_data, ip_history_field
    )
    c_elemental = phasefield[mesh_data.connectivities]
    c_elemental = jnp.repeat(c_elemental[:, None, :], ip_data.N.shape[1], axis=1)
    ip_g = phasefield_model.degradation_in_ip(c_elemental, ip_data.N)

    return ip_history_field, ip_g, phasefield
