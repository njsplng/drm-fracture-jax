"""Input and output handlers for simulations.

This module provides functions for reading and writing simulation data,
including JSON input parsing, checkpoint management, Paraview output
generation, and various serialization formats.
"""

import json
import logging
import os
import pathlib
import pickle
import shlex
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import blosc
import jsonschema
import numpy as np
import pyvista as pv
from jaxtyping import Array
from PIL import Image
from tqdm import tqdm
from vtk import (
    VTK_BIQUADRATIC_QUAD,
    VTK_QUAD,
    VTK_QUADRATIC_QUAD,
    VTK_QUADRATIC_TRIANGLE,
    VTK_TRIANGLE,
)

from json_encoder import format_json_string
from utils import (
    build_quad4_connectivity,
    check_input_placeholders,
    deep_update,
    timed_func,
)

BACKEND_SUFFIXES = {"blosc": ".dat", "pickle": ".pkl"}


def _to_numpy(o: object) -> object:
    """Recursively convert JAX arrays to NumPy arrays for pickling.

    Parameters
    ----------
    o : object
        Object potentially containing JAX arrays.

    Returns
    -------
    object
        Object with all JAX arrays converted to NumPy arrays.
    """
    if isinstance(o, dict):
        return {k: _to_numpy(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_numpy(v) for v in o]
    if isinstance(o, tuple):
        return tuple(_to_numpy(v) for v in o)
    if hasattr(o, "__jax_array__") or type(o).__module__.startswith("jax"):
        return np.asarray(o)
    return o


@timed_func
def dump_blosc(obj: object, path: str) -> None:
    """Compress and write an object to disk using blosc and pickle.

    Parameters
    ----------
    obj : object
        The object to serialize and compress.
    path : str
        File path where the compressed data will be written.
    """
    blosc.set_nthreads(4)
    pickled_data = pickle.dumps(_to_numpy(obj), protocol=5)
    compressed = blosc.compress(
        pickled_data,
        cname="lz4",
        clevel=1,
        shuffle=blosc.SHUFFLE,
    )
    with open(path, "wb") as f:
        f.write(compressed)


@timed_func
def read_blosc(path: str) -> object:
    """Read and decompress an object from disk.

    Parameters
    ----------
    path : str
        File path of the compressed data to read.

    Returns
    -------
    object
        The deserialized object from the compressed file.
    """
    compressed = pathlib.Path(path).read_bytes()
    data = blosc.decompress(compressed, as_bytearray=True)
    return pickle.loads(data)


def input_pickle(
    filename: str,
    backend: Literal["blosc", "pickle"] = "blosc",
    appended_file: bool = False,
) -> object:
    """Read an object from a pickle file.

    Parameters
    ----------
    filename : str
        Name of the pickle file (without extension).
    backend : {"blosc", "pickle"}, optional
        Serialization backend. Default is "blosc".
    appended_file : bool, optional
        Whether the file was appended. Default is False.

    Returns
    -------
    object
        The deserialized object from the pickle file.
    """
    # Get the current path
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent

    # Extract the suffix based on the backend. Default to .pkl
    suffix = BACKEND_SUFFIXES.get(backend, ".pkl")

    def in_pickle(pickle_file_path: pathlib.Path) -> Optional[Dict[str, object]]:
        """Shorthand for calling the default pickling method."""
        data = []
        with open(str(pickle_file_path), "rb") as file:
            while True:
                try:
                    data.append(pickle.load(file))
                except EOFError:
                    break
        # Exit if no returns
        if len(data) == 0:
            return None

        # If len 1, return
        if len(data) == 1:
            return data[0]

        merged_dict = defaultdict(list)
        for entry in data:
            for key, value in entry.items():
                merged_dict[key].extend(value)
        data = dict(merged_dict)
        return data

    # Find the pickle file path
    pickle_file_path = project_root / "output" / "pickle" / f"{filename}{suffix}"

    # Read in the pickled file
    if backend == "blosc" and not appended_file:
        try:
            return read_blosc(pickle_file_path)
        except:
            return in_pickle(pickle_file_path)
    else:
        return in_pickle(pickle_file_path)


def output_pickle(
    target: object,
    filename: str,
    backend: Literal["blosc", "pickle"] = "blosc",
    append_mode: bool = False,
) -> None:
    """Write an object to a pickle file.

    Parameters
    ----------
    target : object
        The object to serialize.
    filename : str
        Name of the output file (without extension).
    backend : {"blosc", "pickle"}, optional
        Serialization backend. Default is "blosc".
    append_mode : bool, optional
        Whether to append to existing file. Default is False.
    """
    # Get the current path
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    # Construct the output file path
    output_path = project_root / "output" / "pickle"
    # If non-existent, create the directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialise the empty file name for now
    full_file_name = f"{filename}"

    # Extract the suffix based on the backend. Default to .pkl
    suffix = BACKEND_SUFFIXES.get(backend, ".pkl")

    # Finally, add the name
    full_file_name += suffix

    # Find the pickle file path
    pickle_file_path = output_path / full_file_name

    def out_pickle(pickle_file_path: pathlib.Path) -> None:
        """Shorthand for calling the default pickling method."""
        with open(str(pickle_file_path), "wb" if not append_mode else "ab") as file:
            pickle.dump(target, file, protocol=5)
            # Extra safety
            file.flush()

    # Write the file
    if backend == "blosc" and not append_mode:
        try:
            dump_blosc(target, str(pickle_file_path))
        except:
            out_pickle(pickle_file_path)
    else:
        out_pickle(pickle_file_path)


def get_input_file_extension(
    input_dict: Dict[str, object], target_directory: str
) -> Dict[str, object]:
    """Resolve extends chain and apply overrides from input dictionary.

    Parameters
    ----------
    input_dict : Dict[str, object]
        Input dictionary potentially containing 'extends' and 'overrides' keys.
    target_directory : str
        Directory containing the input JSON files.

    Returns
    -------
    Dict[str, object]
        Merged dictionary with extends resolved and overrides applied.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    project_root = current_path.parent

    # helper to load a JSON file by stem
    def load_file(stem: str) -> Dict[str, object]:
        p = project_root / "input" / target_directory / f"{stem}.json"
        with open(p) as f:
            return json.load(f)

    # If nothing to extend, return a copy
    stem = input_dict.get("extends")
    if not stem:
        return input_dict.copy()

    # Load the parent and recurse
    parent_dict = load_file(stem)
    parent_flat = get_input_file_extension(parent_dict, target_directory)

    # Now merge in *this* file’s overrides
    overrides = input_dict.get("overrides", {})
    deep_update(parent_flat, overrides)

    # Return the merged result
    return parent_flat


def format_errors(errors: List[object]) -> List[str]:
    """Format jsonschema validation errors into human-readable strings.

    Parameters
    ----------
    errors : List[object]
        List of jsonschema validation error objects.

    Returns
    -------
    List[str]
        List of formatted error messages.
    """
    out = []
    for e in errors:
        # build a “dotted” path (or <root> if it’s the document itself)
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"

        # choose a friendly message based on which keyword failed
        if e.validator in ("minItems", "maxItems"):
            count = len(e.instance)
            bound = e.validator_value
            kind = "at least" if e.validator == "minItems" else "at most"
            msg = f"array has {count} element(s) but requires {kind} {bound}"
        elif e.validator in (
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
        ):
            val = e.instance
            bound = e.validator_value
            op = {
                "minimum": "≥",
                "exclusiveMinimum": ">",
                "maximum": "≤",
                "exclusiveMaximum": "<",
            }[e.validator]
            msg = f"value {val!r} is not {op} {bound}"
        elif e.validator == "enum":
            val = e.instance
            opts = e.validator_value
            msg = f"value {val!r} is not one of {opts}"
        else:
            # fallback on the default message
            msg = e.message

        out.append(f"{path}: {msg}")
    return out


def validate_input(
    input_dict: Dict[str, object], schema: Dict[str, object]
) -> List[object]:
    """Validate input dictionary against a JSON schema.

    Parameters
    ----------
    input_dict : Dict[str, object]
        Dictionary to validate.
    schema : Dict[str, object]
        JSON schema to validate against.

    Returns
    -------
    List[object]
        List of validation error objects (empty if valid).
    """
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(input_dict), key=lambda e: e.path)
    return errors


def parse_schema(schema_type: str) -> Dict[str, object]:
    """Parse a schema JSON file and return the dictionary.

    Parameters
    ----------
    schema_type : str
        Type of schema to load (filename without .json extension).

    Returns
    -------
    Dict[str, object]
        Parsed schema dictionary.
    """
    # Get the current path
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    # Find the pickle file path
    schema_file_path = project_root / "input" / "schemas" / f"{schema_type}.json"

    # Open the JSON
    with open(str(schema_file_path), "r") as f:
        schema_dict = json.load(f)

    return schema_dict


def check_schema_conformance(
    input_dict: Dict[str, object], input_file: str, schema_type: str
) -> None:
    """Check if input dictionary conforms to a pre-defined schema.

    Parameters
    ----------
    input_dict : Dict[str, object]
        Dictionary to validate.
    input_file : str
        Name of the input file (for error messages).
    schema_type : str
        Type of schema to validate against.

    Raises
    ------
    jsonschema.ValidationError
        If input_dict does not conform to the schema.
    """
    # Validate the input file against the pre-defined schema
    schema = parse_schema(schema_type)
    errors = validate_input(input_dict, schema)

    # If there are errors, format them and raise a ValidationError
    if errors:
        formatted_errors = format_errors(errors)
        error_message = "\n".join(formatted_errors)
        raise jsonschema.ValidationError(
            f"Input file '{input_file}' does not conform to the schema (type {schema_type}):\n{error_message}"
        )


def parse_parent_input_json(input_file: str) -> Dict[str, object]:
    """Parse a parent input JSON file and return the dictionary.

    Parameters
    ----------
    input_file : str
        Name of the parent input file (without .json extension).

    Returns
    -------
    Dict[str, object]
        Parsed and validated input dictionary.
    """
    # Get the current path
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    # Find the pickle file path
    input_file_path = project_root / "input" / "parent" / f"{input_file}.json"

    # Open the JSON
    with open(str(input_file_path), "r") as f:
        input_dict = json.load(f)

    # Override the input file if it has an "extends" key
    input_dict = get_input_file_extension(
        input_dict=input_dict, target_directory="parent"
    )

    # Check if the input file conforms to the schema
    check_schema_conformance(
        input_dict=input_dict, input_file=input_file, schema_type="parent"
    )

    return input_dict


def parse_input_json(
    filename: str,
    target_directory: str,
    problem_specific_function: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    """Read input parameters from a JSON file and validate them.

    Parameters
    ----------
    filename : str
        Name of the input file (without .json extension).
    target_directory : str
        Directory containing the input file.
    problem_specific_function : Optional[Callable[[Dict[str, object]], None]]
        Optional function to perform problem-specific processing.

    Returns
    -------
    Dict[str, object]
        Validated and merged input dictionary.
    """
    # Get the current path
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    # Find the pickle file path
    input_file_path = project_root / "input" / target_directory / f"{filename}.json"

    # Open the JSON
    with open(str(input_file_path), "r") as f:
        input_dict = json.load(f)

    # Update the input dict if it extends on other files
    input_dict = get_input_file_extension(
        input_dict=input_dict, target_directory=target_directory
    )

    # Get the parent file
    parent_input_dict = parse_parent_input_json(input_dict["parent_input_file"])

    # Check if the input file conforms to the schema
    check_schema_conformance(
        input_dict=input_dict, input_file=filename, schema_type=target_directory
    )

    # Update the parent input dict with the child input dict
    parent_input_dict.update(input_dict)

    # Rename the parent input dict to input_dict
    input_dict = parent_input_dict.copy()

    # Perform the problem-specific procressing if provided
    if problem_specific_function is not None:
        problem_specific_function(input_dict)

    # Check if the input dictionary contains placeholders
    assert check_input_placeholders(
        input_dict
    ), "Input dictionary contains placeholders. Please replace them with actual values."

    return input_dict


@dataclass
class ParaviewContext:
    """Static data computed by setup_paraview_output for Paraview output.

    Attributes
    ----------
    file_name : str
        Base name for output files.
    pvd_path : pathlib.Path
        Path to the .pvd index file.
    vtu_path : pathlib.Path
        Directory path for .vtu files.
    cells : np.ndarray
        Cell connectivity array.
    cell_types : np.ndarray
        Cell type array.
    points_3d : np.ndarray
        3D point coordinates.
    ip_N : Optional[np.ndarray]
        Shape functions at integration points.
    gp_coordinates_flat : Optional[np.ndarray]
        Flattened Gauss point coordinates.
    connectivity : np.ndarray
        Element connectivity array.
    n_elems : int
        Number of elements.
    n_gauss : int
        Number of Gauss points per element.
    excluded_keys : List[str]
        Keys to exclude from output.
    aux_nurbs : Optional[Dict[str, object]]
        Auxiliary NURBS data.
    """

    file_name: str
    pvd_path: pathlib.Path
    vtu_path: pathlib.Path
    cells: np.ndarray
    cell_types: np.ndarray
    points_3d: np.ndarray
    ip_N: Optional[np.ndarray]
    gp_coordinates_flat: Optional[np.ndarray]
    connectivity: np.ndarray
    n_elems: int
    n_gauss: int
    excluded_keys: List[str]
    aux_nurbs: Optional[Dict[str, object]]
    written_increments: List[float] = field(default_factory=list)
    _grid: object = field(default=None, repr=False)
    _gp_cloud: object = field(default=None, repr=False)


def setup_paraview_output(
    file_name: str,
    coordinates: Array,
    connectivity: Array,
    mesh_type: str = "quad4",
    ip_N: Optional[Array] = None,
    aux_nurbs: Optional[Dict[str, object]] = None,
    problem_dict: Optional[Dict[str, object]] = None,
    excluded_keys: Optional[List[str]] = None,
    subdir: Optional[str] = None,
    resume: bool = False,
) -> ParaviewContext:
    """Create output directories and precompute static geometry data.

    Set up Paraview output by creating directories, writing problem settings,
    and precomputing geometry data. Returns a ParaviewContext used by
    write_paraview_step and write_paraview_pvd.

    Parameters
    ----------
    file_name : str
        Base name for output files.
    coordinates : Array
        Node coordinates array.
    connectivity : Array
        Element connectivity array.
    mesh_type : str, optional
        Mesh type ("quad4", "quad8", etc.). Default is "quad4".
    ip_N : Optional[Array]
        Shape functions at integration points.
    aux_nurbs : Optional[Dict[str, object]]
        Auxiliary NURBS data.
    problem_dict : Optional[Dict[str, object]]
        Problem settings dictionary.
    excluded_keys : Optional[List[str]]
        Keys to exclude from output.
    subdir : Optional[str]
        Subdirectory for output files.
    resume : bool
        Whether to attempt resuming an interrupted simulation.

    Returns
    -------
    ParaviewContext
        Context object containing precomputed geometry data.
    """
    if excluded_keys is None:
        excluded_keys = ["incremental_displacements"]

    current_path = pathlib.Path(__file__).parent.resolve()
    project_root = current_path.parent
    paraview_root = project_root / "output" / "paraview"
    paraview_root.mkdir(parents=True, exist_ok=True)
    if subdir is not None:
        paraview_root = paraview_root / subdir
        paraview_root.mkdir(parents=True, exist_ok=True)

    pvd_path = paraview_root / f"{file_name}.pvd"
    vtu_path = paraview_root / file_name
    vtu_path.mkdir(parents=True, exist_ok=True)

    # Clean up any existing files from a previous run if resuming is not explicitly flagged
    if not resume:
        if pvd_path.exists():
            pvd_path.unlink()
        for item in vtu_path.iterdir():
            if item.is_file():
                item.unlink()

    # Write problem settings for reproducibility
    if problem_dict is not None:
        with open(str(vtu_path / "_problem_settings.json"), "w") as f:
            f.write(format_json_string(problem_dict))

    # Build cell arrays
    n_nodes = coordinates.shape[0]
    n_elems = connectivity.shape[0]

    match mesh_type:
        case "quad4":
            element_type = VTK_QUAD
        case "quad8":
            element_type = VTK_QUADRATIC_QUAD
        case "quad9":
            element_type = VTK_BIQUADRATIC_QUAD
        case "tri3":
            element_type = VTK_TRIANGLE
        case "tri6":
            element_type = VTK_QUADRATIC_TRIANGLE
        case "nurbs":
            element_type = VTK_QUAD

    cell_connectivity = (
        build_quad4_connectivity(aux_nurbs) if aux_nurbs is not None else connectivity
    )
    cells = np.array(
        [[len(elem)] + elem.tolist() for elem in cell_connectivity], dtype=int
    ).ravel()
    cell_types = np.full(cell_connectivity.shape[0], element_type, dtype=np.uint8)
    points_3d = np.column_stack(
        [coordinates, np.zeros(n_nodes, dtype=coordinates.dtype)]
    )

    # Precompute Gauss point coordinates if shape functions are provided
    gp_coordinates_flat = None
    n_gauss = 0
    if ip_N is not None:
        elemental_coordinates = coordinates[connectivity]
        gauss_point_xy = np.einsum("end, egn -> egd", elemental_coordinates, ip_N)
        gp_coords = np.stack(
            [
                gauss_point_xy[..., 0],
                gauss_point_xy[..., 1],
                np.zeros_like(gauss_point_xy[..., 0]),
            ],
            axis=-1,
        )
        n_gauss = gp_coords.shape[1]
        gp_coordinates_flat = gp_coords.reshape((n_elems * n_gauss, 3))

    # Construct and cache the base grid — geometry never changes between steps
    grid = pv.UnstructuredGrid(cells, cell_types, points_3d)

    # Construct and cache the GP cloud if shape functions are provided
    gp_cloud = None
    if gp_coordinates_flat is not None:
        gp_cloud = pv.PolyData(gp_coordinates_flat)

    ctx = ParaviewContext(
        file_name=file_name,
        pvd_path=pvd_path,
        vtu_path=vtu_path,
        cells=cells,
        cell_types=cell_types,
        points_3d=points_3d,
        ip_N=ip_N,
        gp_coordinates_flat=gp_coordinates_flat,
        connectivity=connectivity,
        n_elems=n_elems,
        n_gauss=n_gauss,
        excluded_keys=excluded_keys,
        aux_nurbs=aux_nurbs,
        _grid=grid,
        _gp_cloud=gp_cloud,
    )
    rehydrate_written_increments(ctx)
    return ctx


def rehydrate_written_increments(ctx: ParaviewContext) -> None:
    """Rebuild written_increments from existing .vtu files on disk.

    Scans ``ctx.vtu_path`` for .vtu files whose names encode increment
    values (produced by write_paraview_step) and repopulates
    ``ctx.written_increments`` in ascending order. Files are matched by
    the ``"{value:.2e}".replace(".", "_")`` convention used when writing.

    Parameters
    ----------
    ctx : ParaviewContext
        Context whose written_increments list should be rebuilt.
    """
    recovered: List[float] = []
    for item in sorted(ctx.vtu_path.iterdir()):
        if not item.is_file() or item.suffix != ".vtu":
            continue
        stem = item.stem
        if stem.endswith("_gp"):
            continue
        # Invert "{value:.2e}".replace(".", "_") — the last underscore is
        # the exponent separator (e.g. "1_23e-02"), the first is the decimal.
        try:
            # Replace only the first underscore back to a dot
            restored = stem.replace("_", ".", 1)
            recovered.append(float(restored))
        except ValueError:
            logging.warning(
                f"ParaviewContext: could not parse increment from '{item.name}', skipping."
            )
            continue
    recovered.sort()
    ctx.written_increments = recovered
    logging.info(
        f"ParaviewContext: rehydrated {len(recovered)} written increment(s) "
        f"from {ctx.vtu_path.name}."
    )


@timed_func
def write_paraview_step(
    ctx: ParaviewContext,
    step_data: Dict[str, object],
    increment_value: float,
) -> None:
    """Write one .vtu file (and optionally _gp.vtp) for a single timestep.

    Write Paraview output for a single timestep using cached geometry from
    the context. The step_data dictionary contains single arrays (not lists).
    Only point_data is updated each step; geometry is reused from cache.

    Parameters
    ----------
    ctx : ParaviewContext
        Context object with precomputed geometry data.
    step_data : Dict[str, object]
        Dictionary of field data for this timestep.
    increment_value : float
        Scalar displacement/load factor for this step.
    """
    safe_string = f"{increment_value:.2e}".replace(".", "_")

    # ── Nodal .vtu ────────────────────────────────────────────────────────────
    # Clear previous step's data and assign new arrays in-place.
    # Avoids reconstructing the UnstructuredGrid geometry every step.
    ctx._grid.clear_point_data()

    for key, arr in step_data.items():
        if key in ctx.excluded_keys:
            continue
        if key.startswith("ip_") and not key.endswith("_nodes"):
            continue
        pv_key = key[:-6] if key.endswith("_nodes") else key
        ctx._grid.point_data[pv_key] = np.asarray(arr)

    ctx._grid.save(str(ctx.vtu_path / f"{safe_string}.vtu"), binary=True)

    # ── Gauss point .vtp ──────────────────────────────────────────────────────
    if ctx._gp_cloud is not None:
        non_ip_keys = [
            k
            for k in step_data
            if not k.startswith("ip_") and k not in ctx.excluded_keys
        ]
        ip_keys = [
            k
            for k in step_data
            if k.startswith("ip_")
            and not k.endswith("_nodes")
            and k not in ctx.excluded_keys
        ]

        if ip_keys or non_ip_keys:
            ctx._gp_cloud.clear_point_data()

            for key in ip_keys:
                data = np.asarray(step_data[key])
                flat = (
                    data.reshape((ctx.n_elems * ctx.n_gauss,))
                    if data.ndim == 2
                    else data.reshape((ctx.n_elems * ctx.n_gauss,) + data.shape[2:])
                )
                ctx._gp_cloud.point_data[key] = flat

            for key in non_ip_keys:
                elemental = np.asarray(step_data[key])[ctx.connectivity]
                if elemental.ndim == 2:
                    elemental = elemental[:, :, None]
                gp_data = np.einsum("end..., egn -> egd...", elemental, ctx.ip_N)
                flat = (
                    gp_data.reshape((ctx.n_elems * ctx.n_gauss,))
                    if gp_data.ndim == 2
                    else gp_data.reshape(
                        (ctx.n_elems * ctx.n_gauss,) + gp_data.shape[2:]
                    )
                )
                ctx._gp_cloud.point_data[key] = flat

            ctx._gp_cloud.save(str(ctx.vtu_path / f"{safe_string}_gp.vtp"), binary=True)

    ctx.written_increments.append(increment_value)


def write_paraview_pvd(ctx: ParaviewContext) -> None:
    """Write the .pvd index file referencing all .vtu files.

    Generate the Paraview .pvd index file that references all .vtu files
    written during the simulation. Call once after all timesteps complete.

    Parameters
    ----------
    ctx : ParaviewContext
        Context object containing written increment values and paths.
    """
    has_gp = ctx.ip_N is not None

    with open(str(ctx.pvd_path), "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for i, entry in enumerate(ctx.written_increments):
            safe_string = f"{entry:.2e}".replace(".", "_")
            rel_path = f"{ctx.file_name}/{safe_string}.vtu"
            f.write(
                f'    <DataSet timestep="{i+1}" group="" part="0" file="{rel_path}"/>\n'
            )
            if has_gp:
                rel_gp = f"{ctx.file_name}/{safe_string}_gp.vtp"
                f.write(
                    f'    <DataSet timestep="{i+1}" group="" part="1" file="{rel_gp}"/>\n'
                )
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")


def output_paraview(
    file_name: str,
    increment_list: List[float],
    coordinates: Array,
    connectivity: Array,
    qoi_dict: Optional[Dict[str, object]] = None,
    ip_N: Optional[Array] = None,
    mesh_type: str = "quad4",
    excluded_keys: Optional[List[str]] = None,
    problem_dict: Optional[Dict[str, object]] = None,
    aux_nurbs: Optional[Dict[str, object]] = None,
    subdir: Optional[str] = None,
) -> str:
    """Batch entry point for Paraview output generation.

    Generate Paraview output files for all timesteps. Delegates to
    setup_paraview_output, write_paraview_step, and write_paraview_pvd
    internally.

    Parameters
    ----------
    file_name : str
        Base name for output files.
    increment_list : List[float]
        List of increment values for each timestep.
    coordinates : Array
        Node coordinates array.
    connectivity : Array
        Element connectivity array.
    qoi_dict : Optional[Dict[str, object]]
        Dictionary of quantities of interest.
    ip_N : Optional[Array]
        Shape functions at integration points.
    mesh_type : str, optional
        Mesh type. Default is "quad4".
    excluded_keys : Optional[List[str]]
        Keys to exclude from output.
    problem_dict : Optional[Dict[str, object]]
        Problem settings dictionary.
    aux_nurbs : Optional[Dict[str, object]]
        Auxiliary NURBS data.
    subdir : Optional[str]
        Subdirectory for output files.

    Returns
    -------
    str
        The file name used for output.
    """
    if excluded_keys is None:
        excluded_keys = ["incremental_displacements"]

    ctx = setup_paraview_output(
        file_name=file_name,
        coordinates=coordinates,
        connectivity=connectivity,
        mesh_type=mesh_type,
        ip_N=ip_N,
        aux_nurbs=aux_nurbs,
        problem_dict=problem_dict,
        excluded_keys=excluded_keys,
        subdir=subdir,
    )

    # Determine whether to read from checkpoints or qoi_dict
    checkpoint_filenames = (
        None if qoi_dict else list_checkpoints(filename=file_name, file_prefix="pp")
    )

    for i, entry in enumerate(increment_list[1:]):
        if checkpoint_filenames:
            raw = read_blosc(checkpoint_filenames[i])
            # checkpoint dicts store single-element lists — unwrap
            step_data = {k: v[0] for k, v in raw.items()}
        else:
            step_data = {k: v[i] for k, v in qoi_dict.items()}

        write_paraview_step(ctx, step_data, entry)

    write_paraview_pvd(ctx)
    return file_name


def generate_paraview_fig_local(
    paraview_file_path: Path,
    field_arg: str,
    time_index: str = "-1",
    mesh_outline: int = 0,
    axis_limits: Optional[List[float]] = None,
    distortion_scaling: float = 0.0,
    paper_mode: bool = False,
) -> None:
    """Generate a Paraview figure from a given Paraview file path.

    Requires the PARAVIEW_PYTHON environment variable to be set to the
    path of the pvpython executable. The time_index can be "all", a single
    index, or a range in the form "start-end".

    When paper_mode is True:
        - Renders a square image (500x500) with tight margins (camera zoom).
        - The colorbar is saved as a separate file (no field title, with
          min/max labels and ticks).
        - No text or colorbar on the main render.

    Parameters
    ----------
    paraview_file_path : Path
        Path to the Paraview output directory.
    field_arg : str
        Field name to visualize.
    time_index : str, optional
        Time index ("all", single index, or "start-end"). Default is "-1".
    mesh_outline : int, optional
        Whether to show mesh outline. Default is 0.
    axis_limits : Optional[List[float]], optional
        Axis limits for the plot.
    distortion_scaling : float, optional
        Scaling factor for distortion. Default is 0.0.
    paper_mode : bool, optional
        Whether to use paper mode rendering. Default is False.
    """
    # Check if the environment variable is set
    env_variable = "PARAVIEW_PYTHON"
    pvpython_path = os.environ.get(env_variable)
    if pvpython_path:
        # Check if the file exists
        if os.path.isfile(pvpython_path):
            # Check if it is an actual Python interpreter
            try:
                result = subprocess.run(
                    [pvpython_path, "--version"], capture_output=True, text=True
                )
                if not "paraview" in result.stdout:
                    raise Exception(
                        "The file is not a valid paraview Python interpreter."
                    )
            except Exception as e:
                raise Exception(f"Error running the file: {e}")
        else:
            raise Exception(f"The file {env_variable} does not exist.")
    else:
        raise Exception(
            f"Environment variable {env_variable} not set. Please set it to the path of the pvpython executable."
        )
    # Call the paraview script
    current_path = pathlib.Path(__file__).parent.resolve()
    paraview_script_file = current_path / "paraview_fig.py"

    cmd = [
        str(pvpython_path),
        str(paraview_script_file),
        str(paraview_file_path),
        str(field_arg),
        str(time_index),
        str(mesh_outline),
        str(distortion_scaling),
    ]

    # Append optional scaling argument (positional, must come before paper_mode)
    # Use "none" sentinel instead of empty string — pvpython cannot decode empty args
    if axis_limits:
        cmd.append(str(axis_limits))
    else:
        cmd.append("none")

    # Append paper_mode flag
    cmd.append("1" if paper_mode else "0")

    result = subprocess.run(cmd)


def generate_paraview_fig(
    paraview_file_path: pathlib.Path,
    field_arg: str,
    time_index: str = "-1",
    mesh_outline: int = 0,
    axis_limits: Optional[List[float]] = None,
    distortion_scaling: float = 0.0,
    use_pvbatch: bool = False,
    paper_mode: bool = False,
) -> None:
    """Generate a Paraview figure headlessly via Xvfb + Mesa llvmpipe.

    Render Paraview figures without a display using Xvfb virtual framebuffer
    and Mesa software rendering. The time_index can be "all", a single index,
    or a range in the form "start-end".

    When paper_mode is True:
        - Renders a square image (500x500) with tight margins (camera zoom).
        - The colorbar is saved as a separate file (no field title, with
          min/max labels).
        - No text or colorbar on the main render.

    Parameters
    ----------
    paraview_file_path : pathlib.Path
        Path to the Paraview output directory.
    field_arg : str
        Field name to visualize.
    time_index : str, optional
        Time index ("all", single index, or "start-end"). Default is "-1".
    mesh_outline : int, optional
        Whether to show mesh outline. Default is 0.
    axis_limits : Optional[List[float]], optional
        Axis limits for the plot.
    distortion_scaling : float, optional
        Scaling factor for distortion. Default is 0.0.
    use_pvbatch : bool, optional
        Whether to use pvbatch instead of pvpython. Default is False.
    paper_mode : bool, optional
        Whether to use paper mode rendering. Default is False.
    """
    script_dir = pathlib.Path(__file__).parent.resolve()
    paraview_script = script_dir / "paraview_fig.py"
    exe = "pvbatch" if use_pvbatch else "pvpython"
    module_load_script = script_dir.parent / "bash" / "module_loads_pvrender.sh"
    module_loads = module_load_script.read_text().strip()

    # build args
    args = [
        exe,
        shlex.quote(str(paraview_script)),
        shlex.quote(str(paraview_file_path)),
        shlex.quote(field_arg),
        shlex.quote(str(time_index)),
        shlex.quote(str(mesh_outline)),
        shlex.quote(str(distortion_scaling)),
    ]
    if axis_limits is not None:
        args.append(shlex.quote(str(axis_limits)))
    else:
        args.append("none")

    args.append("1" if paper_mode else "0")

    inner = " ".join(args)
    cmd = f"""
        module purge
        module load {module_loads}
        xvfb-run -a -s "-screen 0 1920x1080x24 -nolisten tcp -ac" {inner}
        """
    subprocess.run(["bash", "-lc", cmd], check=True)


def generate_gif_from_paraview(
    paraview_dir: str,
    field_arg: str,
    mesh_outline: bool = False,
    axis_limits: Tuple[float, float] = (0, 1),
    distortion_scaling: float = 0.0,
    fps: int = 20,
    include_every_n: int = 1,
    parent_dir: Optional[str] = None,
) -> None:
    """Generate a GIF animation from Paraview output files.

    Parameters
    ----------
    paraview_dir : str
        Paraview output directory name.
    field_arg : str
        Field name to visualize.
    mesh_outline : bool, optional
        Whether to show mesh outline. Default is False.
    axis_limits : Tuple[float, float], optional
        Axis limits for the plot. Default is (0, 1).
    distortion_scaling : float, optional
        Scaling factor for distortion. Default is 0.0.
    fps : int, optional
        Frames per second for the GIF. Default is 20.
    include_every_n : int, optional
        Include every nth frame. Default is 1.
    parent_dir : Optional[str], optional
        Parent directory for output files.
    """
    # If path is not explicitly provided, assume the default output path
    if parent_dir is None:
        current_path = pathlib.Path(__file__).parent.resolve()
        output_path = current_path.parent / "output"
    # If provided, point to new folder
    else:
        output_path = pathlib.Path(parent_dir)

    # Find the target directory
    target_directory = output_path / "paraview" / paraview_dir
    n_vtu_files = len(list(target_directory.glob("*.vtu")))

    # Parse to pathlib to extract the stem
    paraview_dir_filename = str(pathlib.Path(paraview_dir).stem)

    generated_file_names = []
    for time_index in range(n_vtu_files):
        filename = f"pv_render_{paraview_dir_filename}_{field_arg}_{time_index}.png"
        generated_file_names.append(filename)

    # Select the appropriate figure generation function
    fig_generation_function = generate_paraview_fig
    pvpython_path = os.environ.get("PARAVIEW_PYTHON")
    if pvpython_path is not None:
        fig_generation_function = generate_paraview_fig_local

    # Generate the paraview figure for each time index
    fig_generation_function(
        target_directory,
        field_arg,
        "all",
        distortion_scaling=distortion_scaling,
        mesh_outline="1" if mesh_outline else "0",
        axis_limits=None if axis_limits == "auto" else axis_limits,
    )

    # Assemble the full path for the generated images
    generated_image_directory = output_path / "plots"
    image_paths = [
        generated_image_directory / filename.replace("/", "_")
        for filename in generated_file_names
    ]

    def quantize_png(path: pathlib.Path, colors: int = 128) -> None:
        """Quantize a PNG image to a specified number of colors."""
        # load with alpha & composite on white
        im = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        flat = Image.alpha_composite(bg, im).convert("RGB")

        # convert to P-mode palette
        p = flat.quantize(colors=colors, method=Image.MEDIANCUT)
        # Make sure no palette-transparency info rides along
        if "transparency" in p.info:
            p.info.pop("transparency", None)
        # save with PNG optimizer on
        p.save(path, optimize=True)

    for img_path in image_paths:
        quantize_png(img_path, colors=128)

    # Get the gifs directory
    gifs_directory = output_path / "gifs"
    gifs_directory.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, img_path in enumerate(image_paths):
        if i % include_every_n != 0:
            continue
        im = Image.open(img_path)
        # Convert to RGB to throw away any transparency
        frames.append(im.convert("RGB"))

    # Form the gif path relative to the output/paraview directory
    gif_path = (
        gifs_directory / f"{paraview_dir_filename.replace('/', '_')}_{field_arg}.gif"
    )
    # duration per frame = 1/fps
    duration_ms = int(1000 / fps)
    frames[0].save(
        gif_path,
        save_all=True,
        # Rest of the frames
        append_images=frames[1:],
        # Ms per frame
        duration=duration_ms,
        # Infinite loop
        loop=0,
        # Completely replace each frame
        disposal=2,
    )

    # clean up intermediate PNGs
    for img_path in image_paths:
        try:
            img_path.unlink()
        except Exception as e:
            print(f"Could not delete {img_path}: {e}")


def generate_paraview_figs_recursively(
    field_arg: str,
    target_dir: Optional[str] = None,
    silent: bool = False,
    **kwargs,
) -> None:
    """Generate Paraview figures for all Paraview directories.

    Parameters
    ----------
    field_arg : str
        Field name to visualize.
    target_dir : Optional[str], optional
        Target directory containing Paraview output.
    silent : bool, optional
        Whether to suppress progress output. Default is False.
    **kwargs
        Additional arguments passed to generate_gif_from_paraview.
    """
    if target_dir is None:
        current_path = pathlib.Path(__file__).parent.resolve()
        output_path = current_path.parent / "output"
        paraview_path = output_path / "paraview"
    else:
        # Assume that the provided dir is the paraview directory
        paraview_path = pathlib.Path(target_dir)
        # In case the provided path was. the parent output, find the paraview folder inside it
        for obj in paraview_path.glob("paraview"):
            paraview_path = obj

    output_paths = []
    # Iterate through all directories in the paraview folder
    for dir_entry in paraview_path.iterdir():
        # For each main file, generate the paraview figure
        if dir_entry.suffix == ".pvd":
            output_paths.append(dir_entry.stem)

    # Create the iterator based on the silent flag
    iterator = (
        tqdm(output_paths, desc="Generating paraview figures")
        if not silent
        else output_paths
    )

    # Generate the paraview figures for each directory
    for i, entry in enumerate(iterator):
        if silent:
            print(f"Generating paraview figures for {i}/{len(output_paths)}...")
        generate_gif_from_paraview(entry, field_arg, **kwargs, parent_dir=target_dir)


def output_paraview_gif(
    filename: str,
    axis_limits: List[Union[float, str]],
    target_fields: List[str],
    distortion_scaling: float = 0.0,
    fps: int = 20,
    include_every_n: int = 1,
    mesh_outline: bool = False,
) -> None:
    """Generate GIFs for requested fields from Paraview output.

    Parameters
    ----------
    filename : str
        Name of the Paraview output file.
    axis_limits : List[Union[float, str]]
        Axis limits for each field.
    target_fields : List[str]
        List of field names to visualize.
    distortion_scaling : float, optional
        Scaling factor for distortion. Default is 0.0.
    fps : int, optional
        Frames per second for the GIF. Default is 20.
    include_every_n : int, optional
        Include every nth frame. Default is 1.
    mesh_outline : bool, optional
        Whether to show mesh outline. Default is False.
    """
    # Ensure that the lengths of target_fields and axis_limits are the same
    if len(target_fields) > len(axis_limits):
        axis_limits.extend(["auto"] * (len(target_fields) - len(axis_limits)))

    # For each of the fields to generate, create a gif
    for field, limits in zip(target_fields, axis_limits):
        generate_gif_from_paraview(
            paraview_dir=filename,
            field_arg=field,
            axis_limits=limits,
            distortion_scaling=distortion_scaling,
            fps=fps,
            include_every_n=include_every_n,
            mesh_outline=mesh_outline,
        )


@timed_func
def write_checkpoint(
    qoi_dict: Dict[str, object],
    filename: str,
    timestep: int,
    reset_qoi_dict: bool = True,
    file_prefix: str = "",
) -> None:
    """Write the current qoi_dict iteration to a pickle checkpoint file.

    Parameters
    ----------
    qoi_dict : Dict[str, object]
        Dictionary of quantities of interest to save.
    filename : str
        Base name for the checkpoint file.
    timestep : int
        Current timestep number.
    reset_qoi_dict : bool, optional
        Whether to clear qoi_dict after writing. Default is True.
    file_prefix : str, optional
        Prefix for the checkpoint filename. Default is "".
    """
    # Nothing to do if test files are being run
    if filename.startswith("test_"):
        return
    # Get the current path
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    # Find the pickle file path
    checkpoint_parent_path = project_root / "output" / "checkpoint" / filename
    checkpoint_parent_path.mkdir(parents=True, exist_ok=True)

    # Add an underscore to the end of the prefix if prefix is provided
    file_prefix += "_" if file_prefix != "" else ""

    # Form the checkpoint file name
    checkpoint_file_name = f"{file_prefix}checkpoint_{timestep}.dat"

    # Find the checkpoint file path
    checkpoint_file_path = checkpoint_parent_path / checkpoint_file_name

    # Write the file
    dump_blosc(qoi_dict, checkpoint_file_path)

    # Reset the qoi_dict if required
    if reset_qoi_dict:
        qoi_dict.clear()


def list_checkpoints(filename: str, file_prefix: Optional[str] = None) -> List[Path]:
    """Return all matching checkpoint file paths for a given filename.

    Parameters
    ----------
    filename : str
        Base name of the checkpoint files.
    file_prefix : Optional[str], optional
        Optional prefix to filter checkpoint files.

    Returns
    -------
    List[Path]
        Sorted list of checkpoint file paths.
    """
    # Get the current path
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    checkpoint_parent_dir = project_root / "output" / "checkpoint"
    # Find the pickle file path
    checkpoint_parent_path = checkpoint_parent_dir / filename

    # Form the file prefix such that it would match the checkpoint files
    file_prefix = file_prefix + "_" if file_prefix is not None else ""

    # Get the list of all checkpoint files
    checkpoint_files = list(checkpoint_parent_path.glob(f"{file_prefix}*.dat"))

    # Return early if nothing to process
    if checkpoint_files == []:
        return []

    # Sort the files by name
    checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))

    return checkpoint_files


def attach_config_to_checkpoints(
    filename: str, config: Dict[str, object]
) -> List[Path]:
    """Attach configuration to checkpoint directory.

    Parameters
    ----------
    filename : str
        Base name of the checkpoint files.
    config : Dict[str, object]
        Configuration dictionary to save.

    Returns
    -------
    List[Path]
        List of checkpoint file paths.
    """
    # Get the current path
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    checkpoint_parent_dir = project_root / "output" / "checkpoint"
    # Find the pickle file path
    checkpoint_parent_path = checkpoint_parent_dir / filename

    json_file_path = checkpoint_parent_path / "_problem_settings.json"
    with open(str(json_file_path), "w") as json_file:
        json_file.write(format_json_string(dict(config)))


def read_checkpoints(
    filename: str,
    var_names: Optional[List[str]] = None,
    file_prefix: Optional[str] = None,
) -> Tuple[object, List[Path]]:
    """Reconstruct the qoi_dict from checkpoint files.

    Parameters
    ----------
    filename : str
        Base name of the checkpoint files.
    var_names : Optional[List[str]], optional
        List of variable names to extract.
    file_prefix : Optional[str], optional
        Optional prefix to filter checkpoint files.

    Returns
    -------
    Tuple[object, List[Path]]
        Reconstructed data and list of checkpoint file paths.
    """
    # Get the current path
    # Nothing to do if test files are being run
    if filename.startswith("test_"):
        return None, []

    checkpoint_files = list_checkpoints(filename=filename, file_prefix=file_prefix)

    output_dict = {}
    output_list = []
    output_list_object = False

    # Read the files and reconstruct the output dict
    for checkpoint_file in checkpoint_files:
        checkpoint_in = read_blosc(checkpoint_file)
        if type(checkpoint_in) is list:
            output_list.extend(checkpoint_in)
            output_list_object = True
        elif type(checkpoint_in) is dict:
            for key, value in checkpoint_in.items():
                if key not in output_dict:
                    output_dict[key] = []
                output_dict[key].extend(value)

            # Remove the untracked keys
            if var_names is not None:
                for key in list(output_dict.keys()):
                    if key not in var_names:
                        output_dict.pop(key)

    out = output_list if output_list_object else output_dict

    return out, checkpoint_files


def wipe_all_checkpoints(filename: str) -> None:
    """Completely destroy all checkpoints associated with a filename.

    Parameters
    ----------
    filename : str
        Base name of the checkpoint files to delete.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    checkpoint_parent_dir = project_root / "output" / "checkpoint"
    # Find the pickle file path
    checkpoint_parent_path = checkpoint_parent_dir / filename
    if checkpoint_parent_path.exists():
        shutil.rmtree(checkpoint_parent_path)


def wipe_all_models(filename: str) -> None:
    """Completely destroy all saved models associated with a filename.

    Parameters
    ----------
    filename : str
        Base name of the model files to delete.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    # Find the parent directory
    project_root = current_path.parent
    models_parent_dir = project_root / "output" / "models"
    # Find the model file path
    models_parent_path = models_parent_dir / filename
    if models_parent_path.exists():
        shutil.rmtree(models_parent_path)


def cleanup_checkpoints(
    checkpoint_files: List[Path], unlink_parent: bool = False
) -> None:
    """Clean up checkpoint files.

    Parameters
    ----------
    checkpoint_files : List[Path]
        List of checkpoint file paths to remove.
    unlink_parent : bool, optional
        Whether to remove the parent directory. Default is False.
    """
    # Remove the checkpoint files
    for checkpoint_file in checkpoint_files:
        checkpoint_file.unlink()
    if unlink_parent:
        checkpoint_parent_path = checkpoint_files[0].parent
        checkpoint_parent_path.rmdir()


def read_reference_file(filename: str) -> Dict[str, object]:
    """Read a reference data file and return the stored dictionary.

    Parameters
    ----------
    filename : str
        Name of the reference file (without .dat extension).

    Returns
    -------
    Dict[str, object]
        Dictionary containing the reference data.
    """
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    reference_data_directory = project_root / "reference_data"
    reference_file_path = reference_data_directory / f"{filename}.dat"

    data = read_blosc(reference_file_path)

    return data
