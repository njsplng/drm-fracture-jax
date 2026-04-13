"""Dataclasses for finite element computations.

Provide dataclass wrappers for integration point data, distance functions,
material parameters, problem parameters, and mesh data for use in
phase-field fracture simulations.
"""

import inspect
import sys
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node
from jaxtyping import Array

from distance_functions import CompositeDistanceFunction
from utils import set_up_rotation_array


def IPDataClass(frozen: bool) -> Callable[[], "IPData"]:
    """Create a frozen or unfrozen IPData dataclass.

    Factory function that returns an IPData class for storing integration
    point data in finite element computations.

    Parameters
    ----------
    frozen : bool
        Whether to create a frozen (immutable) dataclass.

    Returns
    -------
    IPData : type
        The IPData dataclass type.
    """

    @dataclass(frozen=frozen)
    class IPData:
        """Integration point data for finite element computations.

        Store shape functions, their derivatives, volumes, and related
        quantities computed at integration points for finite element
        analysis.

        Parameters
        ----------
        N : Array
            Shape functions at integration points.
        dN : Array
            Derivatives of shape functions.
        d2N : Array
            Second derivatives of shape functions.
        volumes : Array
            Integration point volumes.
        B : Array
            Strain-displacement matrix.
        physical_derivatives : Array
            Physical derivatives of shape functions.
        physical_derivatives_2 : Array
            Second physical derivatives of shape functions.
        extrapolations : Array
            Extrapolation matrix.
        physical_derivatives_rot : Array or None
            Rotated physical derivatives.
        physical_derivatives_2_rot : Array or None
            Rotated second physical derivatives.
        gamma_matrix : Array or None
            Anisotropy gamma matrix.
        rotation_angles : Array or None
            Material rotation angles.
        """

        N: Array
        dN: Array
        d2N: Array
        volumes: Array
        B: Array
        physical_derivatives: Array
        physical_derivatives_2: Array
        extrapolations: Array
        physical_derivatives_rot: Optional[Array] = None
        physical_derivatives_2_rot: Optional[Array] = None

        # Anisotropy-only fields
        gamma_matrix: Optional[Array] = None
        rotation_angles: Optional[Array] = None

        @classmethod
        def from_initializer(
            cls,
            initializer: Callable,
            connectivity: Array,
            nodal_coordinates: Array,
            thickness: float,
            info: Optional[Dict],
            problem_type: str,
            gamma_params: Optional[Dict] = None,
            rotation_dict: Optional[Dict] = None,
            build_material_rotations: bool = False,
        ) -> "IPData":
            """Create IPData from an initializer function.

            Parameters
            ----------
            initializer : Callable
                Function to compute base integration point data.
            connectivity : Array
                Element connectivity array.
            nodal_coordinates : Array
                Nodal coordinates array.
            thickness : float
                Element thickness.
            info : Dict or None
                Additional information dictionary.
            problem_type : str
                Problem type ('isotropic' or 'anisotropic').
            gamma_params : Dict, optional
                Anisotropy gamma parameters.
            rotation_dict : Dict, optional
                Material rotation parameters.
            build_material_rotations : bool, optional
                Whether to build material rotations. Default is False.

            Returns
            -------
            IPData
                Initialized IPData instance.
            """
            # Always compute the base IP data
            outs = initializer(
                connectivity=connectivity,
                nodal_coordinates=nodal_coordinates,
                thickness=thickness,
                info=info,
            )
            base = cls(*outs)

            base = replace(base, gamma_matrix=jnp.zeros((3, 3)))

            if build_material_rotations:
                # Initialize the rotation angles if the material rotation is enabled
                rotation_angles = set_up_rotation_array(
                    nodal_coordinates,
                    connectivity,
                    rotation_angle=rotation_dict["angle"],
                    direction=rotation_dict["slicing_direction"],
                    slices=rotation_dict["number_of_slices"],
                )
                base = replace(base, rotation_angles=rotation_angles)

            # If anisotropic, compute gamma and rotated derivatives
            if problem_type == "anisotropic":
                # Build gamma matrix
                g11, g22, g44, g12 = (
                    gamma_params["gamma11"],
                    gamma_params["gamma22"],
                    gamma_params["gamma44"],
                    gamma_params["gamma12"],
                )
                gamma = jnp.array([[g11, g12, 0], [g12, g22, 0], [0, 0, 4 * g44]])

                # Call initializer again for rotated derivatives only
                *_, der_rot, der2_rot = initializer(
                    connectivity=connectivity,
                    nodal_coordinates=nodal_coordinates,
                    thickness=thickness,
                    info=info,
                    rotation_angles=rotation_angles,
                )

                return replace(
                    base,
                    gamma_matrix=gamma,
                    physical_derivatives_rot=der_rot,
                    physical_derivatives_2_rot=der2_rot,
                )

            return base

    register_dataclass_pytree(IPData)
    return IPData


def DistanceFunctionsClass(
    frozen: bool,
) -> Callable[[], "DistanceFunctions"]:
    """Create a frozen or unfrozen DistanceFunctions dataclass.

    Factory function that returns a DistanceFunctions class for storing
    distance functions for fixed and load windows.

    Parameters
    ----------
    frozen : bool
        Whether to create a frozen (immutable) dataclass.

    Returns
    -------
    DistanceFunctions : type
        The DistanceFunctions dataclass type.
    """

    @dataclass(frozen=frozen)
    class DistanceFunctions:
        """Distance functions for fixed and load windows.

        Parameters
        ----------
        fixed_window : Array
            Distance function values for fixed window.
        fixed_window_x : Array
            Distance function values for fixed window in x direction.
        fixed_window_y : Array
            Distance function values for fixed window in y direction.
        load_window : Array
            Distance function values for load window.
        """

        fixed_window: Array
        fixed_window_x: Array
        fixed_window_y: Array
        load_window: Array

        @classmethod
        def from_parameters(
            cls,
            fixed_window_params: Union[List, Dict],
            load_window_params: Union[List, Dict],
            nodal_coordinates: Array,
            fixed_window_directions: Optional[List] = None,
        ) -> "DistanceFunctions":
            """Create DistanceFunctions from parameter dictionaries.

            Parameters
            ----------
            fixed_window_params : list or dict
                Parameters for fixed window distance functions.
            load_window_params : list or dict
                Parameters for load window distance functions.
            nodal_coordinates : Array
                Nodal coordinates array.
            fixed_window_directions : list, optional
                Directions for fixed window functions.

            Returns
            -------
            DistanceFunctions
                Initialized DistanceFunctions instance.
            """
            if fixed_window_directions is not None:
                assert len(fixed_window_directions) == len(
                    fixed_window_params
                ), "When provided, fixed window directions must have the same length as the parameter dictionary list."
            fixed_x, fixed_y = [], []
            for direction, function in zip(
                fixed_window_directions, fixed_window_params
            ):
                if "x" in direction:
                    fixed_x.append(function)
                if "y" in direction:
                    fixed_y.append(function)
            fixed_window_fn = CompositeDistanceFunction(fixed_window_params)
            fixed_window = fixed_window_fn(nodal_coordinates)
            fixed_window_fn_x = CompositeDistanceFunction(fixed_x)
            fixed_window_x = fixed_window_fn_x(nodal_coordinates)
            fixed_window_fn_y = CompositeDistanceFunction(fixed_y)
            fixed_window_y = fixed_window_fn_y(nodal_coordinates)
            load_window_fn = CompositeDistanceFunction(load_window_params)
            load_window = load_window_fn(nodal_coordinates)

            return cls(
                fixed_window=fixed_window,
                fixed_window_x=fixed_window_x,
                fixed_window_y=fixed_window_y,
                load_window=load_window,
            )

    register_dataclass_pytree(DistanceFunctions)
    return DistanceFunctions


# skylos: ignore-start
def MaterialParametersClass(
    frozen: bool,
) -> Callable[[], "MaterialParameters"]:
    """Create a frozen or unfrozen MaterialParameters dataclass.

    Factory function that returns a MaterialParameters class for storing
    material parameters for the problem.

    Parameters
    ----------
    frozen : bool
        Whether to create a frozen (immutable) dataclass.

    Returns
    -------
    MaterialParameters : type
        The MaterialParameters dataclass type.
    """

    @dataclass(frozen=frozen)
    class MaterialParameters:
        """Material parameters for the problem.

        Parameters
        ----------
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        l_0 : float
            Length scale parameter.
        G_c : float
            Critical energy release rate.
        mu : float
            Shear modulus.
        lambda_ : float
            First Lamé parameter.
        pf_irreversibility_tolerance : float
            Tolerance for phase field irreversibility.
        energy_scaling : float
            Energy scaling factor.
        displacement_scaling : float
            Displacement scaling factor.
        force_scaling : float
            Force scaling factor.
        cubic_anisotropy_G : float
            Shear modulus for cubic anisotropy.
        orthotropic_anisotropy_params : dict
            Parameters for orthotropic anisotropy.
        penalty_scaling : float
            Penalty scaling factor.
        """

        E: float
        nu: float
        l_0: float
        G_c: float
        mu: float
        lambda_: float
        pf_irreversibility_tolerance: float
        energy_scaling: float
        displacement_scaling: float
        force_scaling: float
        cubic_anisotropy_G: float
        orthotropic_anisotropy_params: dict
        penalty_scaling: float

        @classmethod
        def from_dict(cls, params: Dict) -> "MaterialParameters":
            """Create MaterialParameters from a dictionary.

            Parameters
            ----------
            params : dict
                Dictionary containing material and phasefield parameters.

            Returns
            -------
            MaterialParameters
                Initialized MaterialParameters instance.
            """
            E = params["material_parameters"]["youngs_modulus"]
            penalty_scaling = 1
            nu = params["material_parameters"]["poissons_ratio"]
            l_0 = params["phasefield_parameters"]["l_0"]
            G_c = params["phasefield_parameters"]["G_c"]
            cubic_anisotropy_G = params["material_parameters"]["cubic_anisotropy"][
                "shear_modulus"
            ]
            orthotropic_anisotropy_params = params["material_parameters"][
                "orthotropic_anisotropy"
            ]

            energy_scaling = 1.0
            displacement_scaling = 1.0
            if params["nondimensionalise"]:
                cubic_anisotropy_G /= E
                displacement_scaling = jnp.sqrt(E * l_0 / G_c)
                energy_scaling = l_0 / G_c
                penalty_scaling = 1 / E
                G_c = l_0
                E = 1.0
                ortho_max_modulus = max(
                    [
                        orthotropic_anisotropy_params["E_11"],
                        orthotropic_anisotropy_params["E_22"],
                        orthotropic_anisotropy_params["E_33"],
                    ]
                )
                orthotropic_anisotropy_params = {
                    "E_11": orthotropic_anisotropy_params["E_11"] / ortho_max_modulus,
                    "E_22": orthotropic_anisotropy_params["E_22"] / ortho_max_modulus,
                    "E_33": orthotropic_anisotropy_params["E_33"] / ortho_max_modulus,
                    "nu_12": orthotropic_anisotropy_params["nu_12"],
                    "nu_23": orthotropic_anisotropy_params["nu_23"],
                    "nu_31": orthotropic_anisotropy_params["nu_31"],
                    "G_12": orthotropic_anisotropy_params["G_12"] / ortho_max_modulus,
                    "G_23": orthotropic_anisotropy_params["G_23"] / ortho_max_modulus,
                    "G_31": orthotropic_anisotropy_params["G_31"] / ortho_max_modulus,
                }

            return cls(
                E=E,
                nu=nu,
                l_0=l_0,
                G_c=G_c,
                mu=E / (2 * (1 + nu)),
                lambda_=(E * nu) / ((1 + nu) * (1 - 2 * nu)),
                pf_irreversibility_tolerance=params.get("phasefield", {}).get(
                    "irreversibility_tolerance", 0
                ),
                energy_scaling=energy_scaling,
                displacement_scaling=displacement_scaling,
                force_scaling=energy_scaling / displacement_scaling,
                cubic_anisotropy_G=cubic_anisotropy_G,
                orthotropic_anisotropy_params=orthotropic_anisotropy_params,
                penalty_scaling=penalty_scaling,
            )

    register_dataclass_pytree(MaterialParameters)
    return MaterialParameters


# skylos: ignore-end


# skylos: ignore-start
def ProblemParametersClass(
    frozen: bool,
) -> Callable[[], "ProblemParameters"]:  # skylos: ignore
    """Create a frozen or unfrozen ProblemParameters dataclass.

    Factory function that returns a ProblemParameters class for storing
    problem parameters for the problem.

    Parameters
    ----------
    frozen : bool
        Whether to create a frozen (immutable) dataclass.

    Returns
    -------
    ProblemParameters : type
        The ProblemParameters dataclass type.
    """

    @dataclass(frozen=frozen)
    class ProblemParameters:
        """Problem parameters for the problem.

        Parameters
        ----------
        plane_mode : str
            Plane stress or plane strain mode.
        strain_split : str
            Strain split method.
        problem_type : str
            Problem type (isotropic or anisotropic).
        """

        plane_mode: str
        strain_split: str
        problem_type: str

        @classmethod
        def from_dict(cls, params: Dict) -> "ProblemParameters":
            """Create ProblemParameters from a dictionary.

            Parameters
            ----------
            params : dict
                Dictionary containing problem parameters.

            Returns
            -------
            ProblemParameters
                Initialized ProblemParameters instance.
            """
            return cls(
                plane_mode=params["plane"],
                strain_split=params["strain_split"],
                problem_type=params["problem_type"],
            )

    register_dataclass_pytree(ProblemParameters)
    return ProblemParameters


# skylos: ignore-end


def MeshDataClass(frozen: bool) -> Callable[[], "MeshData"]:
    """Create a frozen or unfrozen MeshData dataclass.

    Factory function that returns a MeshData class for storing mesh data.

    Parameters
    ----------
    frozen : bool
        Whether to create a frozen (immutable) dataclass.

    Returns
    -------
    MeshData : type
        The MeshData dataclass type.
    """

    @dataclass(frozen=frozen)
    class MeshData:
        """Lightweight wrapper for mesh data.

        Parameters
        ----------
        nodal_coordinates : Array
            Nodal coordinates array.
        connectivities : Array
            Element connectivity array.
        dofs : Array
            Degrees of freedom array.
        aux_nurbs : dict, optional
            Auxiliary NURBS data.
        """

        nodal_coordinates: Array
        connectivities: Array
        dofs: Array
        aux_nurbs: Optional[dict] = None

    register_dataclass_pytree(MeshData)
    return MeshData


_ARRAY_TYPES = (jnp.ndarray, np.ndarray)


def _is_array_annotation(ann: object) -> bool:
    """Determine if a type annotation represents an array type.

    Check if the annotation is an Array type, numpy/jax ndarray, or an
    Optional/Union containing array types.

    Parameters
    ----------
    ann : object
        Type annotation to check.

    Returns
    -------
    bool
        True if the annotation represents an array type.
    """
    if ann is Array or ann in _ARRAY_TYPES:
        return True
    origin = get_origin(ann)
    if origin in (Optional, Union):
        return any(_is_array_annotation(a) for a in get_args(ann) if a is not None)
    return False


def register_dataclass_pytree(cls: type) -> None:
    """Register a dataclass as a pytree with JAX.

    Parameters
    ----------
    cls : type
        Dataclass type to register as a pytree.

    Raises
    ------
    AssertionError
        If cls is not a dataclass.
    """
    assert is_dataclass(cls), f"{cls} must be a dataclass"

    # Decide CHILD FIELDS ONCE from annotations (stable across values)
    child_names = tuple(f.name for f in fields(cls) if _is_array_annotation(f.type))

    def flatten(obj: type) -> Tuple[List, Dict[str, Any]]:
        """Flatten a pytree node.

        Parameters
        ----------
        obj : type
            Pytree node to flatten.

        Returns
        -------
        tuple
            Tuple of (children list, aux dict).
        """
        children = []
        aux = {}
        none_mask = {}
        for f in fields(obj):
            v = getattr(obj, f.name)
            if f.name in child_names:
                if v is None:
                    pass
                    # raise ValueError(f"Field {f.name} cannot be None")
                children.append(v)
            else:
                aux[f.name] = v
        aux["_child_names"] = child_names
        aux["_none_mask"] = none_mask if none_mask else None
        return children, aux

    def unflatten(aux: Dict[str, Any], children: List[Any]) -> type:
        """Unflatten a pytree node.

        Parameters
        ----------
        aux : dict
            Auxiliary data dictionary.
        children : list
            List of child values.

        Returns
        -------
        type
            Reconstructed pytree node instance.
        """
        kwargs = dict(aux)
        names = kwargs.pop("_child_names")
        none_mask = kwargs.pop("_none_mask", None) or {}
        it = iter(children)
        for name in names:
            val = next(it)
            if none_mask.get(name, False):
                val = None
            kwargs[name] = val
        return cls(**kwargs)

    register_pytree_node(cls, flatten, unflatten)


_mod = sys.modules[__name__]
for func_name, factory in inspect.getmembers(_mod, inspect.isfunction):
    if func_name.endswith("Class"):
        cls = factory(frozen=True)
        public_name = func_name[:-5]
        setattr(_mod, public_name, cls)
