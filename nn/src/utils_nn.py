"""Additional utility functions for neural network simulations."""

import pathlib
from dataclasses import fields, is_dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from networks import FNN, ResNet

from data_handling import MeshData
from io_handlers import (
    dump_blosc,
    output_paraview,
    read_blosc,
)
from utils import (
    split_displacement,
)

current_path = pathlib.Path(__file__).parent.resolve()
project_root = current_path.parent.parent


def update_displacement_increment(
    model: eqx.Module, displacement_increment: float
) -> eqx.Module:
    """Update the displacement increment in the model.

    Parameters
    ----------
    model : eqx.Module
        The model to update.
    displacement_increment : float
        The displacement increment value to set.

    Returns
    -------
    eqx.Module
        The updated model with new displacement increment.
    """
    object.__setattr__(model, "displacement_increment", displacement_increment)
    return model


def update_previous_phasefield(
    model: eqx.Module, previous_phasefield: Array
) -> eqx.Module:
    """Update the previous phasefield in the model.

    Parameters
    ----------
    model : eqx.Module
        The model to update.
    previous_phasefield : Array
        The previous phasefield array to set.

    Returns
    -------
    eqx.Module
        The updated model with new previous phasefield.
    """
    object.__setattr__(
        model, "previous_phasefield", tuple(previous_phasefield.tolist())
    )
    return model


def calculate_weight_sparsity(nn: eqx.Module, eps: tuple = (1e-6, 1e-8, 1e-10)) -> dict:
    """Compute the fraction of weights above given epsilon thresholds.

    Parameters
    ----------
    nn : eqx.Module
        The neural network model to analyze.
    eps : tuple, optional
        Epsilon thresholds for sparsity calculation. Default is
        (1e-6, 1e-8, 1e-10).

    Returns
    -------
    dict
        Dictionary mapping epsilon thresholds to lists of nonzero
        weight fractions (as percentages).
    """
    sparsity_dict = {f"{entry:.0e}": [] for entry in eps}

    # -----------------------------------------------------------
    # Initialise list of linear weights
    # -----------------------------------------------------------
    linear_weights = []

    # -----------------------------------------------------------
    # In case of blocked networks, need to extract the linears first
    # -----------------------------------------------------------
    if hasattr(nn, "blocks"):
        for block in nn.blocks:
            linear_weights.extend(block.linear_list)
    elif hasattr(nn, "linears"):
        linear_weights.extend(nn.linears)

    for entry in linear_weights:
        weights_flat = entry.weight.flatten()
        for epsilon in sparsity_dict:
            nonzero_entries = jnp.sum(jnp.abs(weights_flat) > float(epsilon))
            nonzero_fraction = (
                round(float(nonzero_entries / len(weights_flat)) * 10000) / 100
            )
            sparsity_dict[epsilon].append(nonzero_fraction)

    return sparsity_dict


def predict_model_output(
    model: eqx.Module,
    inp: Array,
) -> tuple[Array, Array]:
    """Predict the output of the model for a given input.

    Parameters
    ----------
    model : eqx.Module
        The model to make predictions with.
    inp : Array
        Input array for the model.

    Returns
    -------
    tuple[Array, Array]
        A tuple containing the displacement array and phasefield
        array.
    """
    # Predict the nodal displacements and phasefield with jit turned off
    with jax.disable_jit():
        u, v, c = model(inp)

    # -----------------------------------------------------------
    # Reshape the outputs
    # -----------------------------------------------------------
    displacement = jnp.stack([u, v], axis=-1).ravel()
    return displacement, c


def output_training_snapshot(
    output_dict: dict, filename: str, increment_index: int
) -> None:
    """Output the training prediction snapshot to a file.

    Parameters
    ----------
    output_dict : dict
        Dictionary containing the output data to save.
    filename : str
        Name of the run folder.
    increment_index : int
        Index of the current increment.
    """
    output_path = project_root / "output" / "training_snapshots"

    # -----------------------------------------------------------
    # Locate the run path and form the file path
    # -----------------------------------------------------------
    run_path = output_path / filename
    file_path = run_path / f"increment_{increment_index}.dat"
    run_path.mkdir(parents=True, exist_ok=True)
    dump_blosc(output_dict, file_path)


def postprocess_training_snapshots(
    filename: str,
    mesh_data: MeshData,
    mesh_type: str,
    burn_negative_increment: bool = True,
) -> None:
    """Post-process training snapshots and export to ParaView format.

    Parameters
    ----------
    filename : str
        Name of the run folder containing training snapshots.
    mesh_data : MeshData
        Mesh data containing nodal coordinates and connectivity.
    mesh_type : str
        Type of mesh (e.g., '2D', '3D').
    burn_negative_increment : bool, optional
        Whether to skip the -1 increment file. Default is True.
    """
    output_path = project_root / "output" / "training_snapshots"
    paraview_path = project_root / "output" / "paraview"

    # -----------------------------------------------------------
    # Resolve the run folder path
    # -----------------------------------------------------------
    run_folder = output_path / filename

    # -----------------------------------------------------------
    # Extract each file's name and sort them
    # -----------------------------------------------------------
    dat_files = [file for file in run_folder.glob(f"*.dat")]
    sorted_dat_files = sorted(dat_files)

    # -----------------------------------------------------------
    # Form the paraview directory
    # -----------------------------------------------------------
    pvd_filedir = f"training_{filename}"
    training_pvd_dir = paraview_path / pvd_filedir
    training_pvd_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Form the PVD filenames
    # -----------------------------------------------------------
    filenames = [x.stem for x in sorted_dat_files]
    if burn_negative_increment and "-1" in filenames[0]:
        filenames = filenames[1:]  # Remove the first entry if it is -1 increment

    for input_file, pvd_file in zip(sorted_dat_files, filenames):
        # -----------------------------------------------------------
        # Extract the data from the file
        # -----------------------------------------------------------
        data = read_blosc(input_file)

        # -----------------------------------------------------------
        # Form a dummy increment list
        # -----------------------------------------------------------
        increments = jnp.arange(len(data["displacement"]) + 1)

        # -----------------------------------------------------------
        # Split the displacements into paraview-friendly format
        # -----------------------------------------------------------
        pp_displacements = []
        for entry in data["displacement"]:
            pp_displacements.append(split_displacement(entry))
        data["displacement"] = jnp.array(pp_displacements)

        # -----------------------------------------------------------
        # Invert the phasefield values
        # -----------------------------------------------------------
        phasefields = []
        for entry in data["phasefield"]:
            phasefields.append(1 - entry)
        data["phasefield"] = jnp.array(phasefields)

        # -----------------------------------------------------------
        # Write the paraview file for the current training step
        # -----------------------------------------------------------
        _ = output_paraview(
            file_name=pvd_file,
            increment_list=increments,
            qoi_dict=data,
            coordinates=mesh_data.nodal_coordinates,
            connectivity=mesh_data.connectivities,
            mesh_type=mesh_type,
            aux_nurbs=mesh_data.aux_nurbs,
            subdir="training_" + filename,
        )

    for file in sorted_dat_files:
        # -----------------------------------------------------------
        # Remove the original files after processing
        # -----------------------------------------------------------
        file.unlink()
    run_folder.rmdir()  # Remove the run folder if empty


def reset_activation_coefficients(
    model: eqx.Module, trainable_global: bool, initial_coefficient: float
) -> eqx.Module:
    """Reset the activation coefficients in the model.

    Parameters
    ----------
    model : eqx.Module
        The model containing activation layers to reset.
    trainable_global : bool
        Whether to use global trainable coefficients.
    initial_coefficient : float
        The initial coefficient value to set.

    Returns
    -------
    eqx.Module
        The model with reset activation coefficients.
    """
    if not hasattr(model.network, "activation_list"):
        return model
    if trainable_global:
        model = eqx.tree_at(
            lambda m: m.network.activation_list[0].coeff,
            model,
            jnp.array(initial_coefficient),
        )
    else:
        activation_list = model.network.activation_list
        for entry in activation_list:
            if hasattr(entry, "coeff"):
                # -----------------------------------------------------------
                # Reset the coefficient to the initial value
                # -----------------------------------------------------------
                eqx.tree_at(
                    lambda m: m.coeff,
                    entry,
                    jnp.array(initial_coefficient),
                )
        model = eqx.tree_at(lambda m: m.network.activation_list, model, activation_list)

    return model


def subtract_fnn_pytree(model: object) -> object:
    """Return a Python pytree excluding FNN-declared attributes.

    Parameters
    ----------
    model : object
        The model object to process.

    Returns
    -------
    object
        A pytree containing only non-FNN attributes.
    """
    fnn_field_names = {f.name for f in fields(FNN)}

    def _nonempty(x: object) -> bool:  # type: ignore
        if x is None:
            return False
        if isinstance(x, (list, tuple, dict)):
            return len(x) > 0
        return True

    def keep(x: object) -> object:  # type: ignore
        if isinstance(x, eqx.Module) or is_dataclass(x):
            out = {}
            for f in fields(type(x)):
                if f.name in fnn_field_names:
                    continue
                try:
                    v = getattr(x, f.name)
                except Exception:
                    continue
                k = keep(v)
                if _nonempty(k):
                    out[f.name] = k
            return out
        if isinstance(x, list):
            xs = [keep(v) for v in x]
            xs = [v for v in xs if _nonempty(v)]
            return xs
        if isinstance(x, tuple):
            xs = tuple(v for v in (keep(v) for v in x) if _nonempty(v))
            return xs
        if isinstance(x, dict):
            xs = {k: keep(v) for k, v in x.items()}
            return {k: v for k, v in xs.items() if _nonempty(v)}
        return x

    return keep(model)


def block_stats(net: object) -> dict[str, object]:
    """Extract monitoring statistics from a blocked network instance.

    Parameters
    ----------
    net : object
        The network object to extract statistics from.

    Returns
    -------
    dict[str, object]
        A JSON-serializable dictionary containing statistics
        such as alpha coefficient values, mean, std, min, and max.
    """
    out = {}

    # -----------------------------------------------------------
    # residual coefficients alpha per block
    # -----------------------------------------------------------
    if hasattr(net, "blocks") and len(net.blocks) > 0:
        alphas = jnp.array([blk.alpha for blk in net.blocks])
        out["alpha"] = {
            "values": [float(a) for a in alphas],
            "mean": float(jnp.mean(alphas)),
            "std": float(jnp.std(alphas)),
            "min": float(jnp.min(alphas)),
            "max": float(jnp.max(alphas)),
        }

    return out


def collect_auxiliary_data(network: eqx.Module) -> dict[str, object]:
    """Collect auxiliary data from the model for logging or analysis.

    Parameters
    ----------
    network : eqx.Module
        The network model to collect data from.

    Returns
    -------
    dict[str, object]
        A dictionary containing auxiliary data from the network.

    Raises
    ------
    NotImplementedError
        If the network type is not supported (not ResNet or FNN).
    """
    if isinstance(network, ResNet):
        return block_stats(network)
    elif isinstance(network, FNN):
        return subtract_fnn_pytree(network)
    raise NotImplementedError(
        f"Auxiliary data collection not implemented for network type {type(network)}"
    )
