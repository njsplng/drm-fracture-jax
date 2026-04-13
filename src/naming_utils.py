"""Utility functions and classes for handling naming conventions in the project."""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jsonpath_ng import parse as jsonpath_parse

from io_handlers import get_input_file_extension, parse_parent_input_json
from json_encoder import format_json_string


def list_changed_files_since_last_commit(repo_root: Path) -> set[Path]:
    """Return files changed since the last commit.

    Return all modified, added, deleted, or renamed files tracked by Git.
    Untracked files are excluded from the results.

    Parameters
    ----------
    repo_root : Path
        Root directory of the repository.

    Returns
    -------
    set[Path]
        Set of file paths that have been changed.
    """
    files: set[Path] = set()

    def run_git(args: List[str]) -> List[str]:
        try:
            out = subprocess.check_output(
                args, cwd=repo_root, stderr=subprocess.DEVNULL
            )
            return [line.strip() for line in out.decode().splitlines() if line.strip()]
        except Exception:
            return []

    # Modified/staged (exclude deletions)
    diff_unstaged = run_git(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"]
    )
    diff_staged = run_git(
        ["git", "diff", "--name-only", "--cached", "--diff-filter=ACMR", "HEAD"]
    )

    # Untracked (not yet added to Git)
    untracked = run_git(["git", "ls-files", "--others", "--exclude-standard"])

    files.update(repo_root / p for p in diff_unstaged + diff_staged + untracked)
    # Keep only files that exist
    return {p for p in files if p.exists()}


def list_all_json_files(repo_root: Path) -> set[Path]:
    """Return all JSON files in the repository.

    Recursively search the repository root for all files with .json extension.

    Parameters
    ----------
    repo_root : Path
        Root directory of the repository.

    Returns
    -------
    set[Path]
        Set of paths to all JSON files found.
    """
    return set(repo_root.rglob("*.json"))


def elem_abbr(value: Optional[str]) -> str:
    """Return an abbreviated string for the element type.

    Map element type names to their abbreviations (e.g., 'tri3' -> 'T3').

    Parameters
    ----------
    value : Optional[str]
        Element type name.

    Returns
    -------
    str
        Abbreviated element type string.
    """
    if value is None:
        return ""
    v = value.lower()
    return {
        "tri3": "T3",
        "quad4": "Q4",
        "quad8": "Q8",
        "tri6": "T6",
        "quad9": "Q9",
        "nurbs": "IGA",
    }.get(v, value[0].upper() if value else "")


def problem_abbr(value: Optional[str]) -> str:
    """Return an abbreviated string for the problem type.

    Map problem type names to their abbreviations (e.g., 'linear_elasticity' -> 'LE').

    Parameters
    ----------
    value : Optional[str]
        Problem type name.

    Returns
    -------
    str
        Abbreviated problem type string.
    """
    if value is None:
        return ""
    v = value.lower()
    if v == "linear_elasticity":
        return "LE"
    if v == "at1":
        return "AT1"
    if v == "at2":
        return "AT2"
    if v == "isotropic-4":
        return "PF4"
    if v == "anisotropic":
        return "AN"
    return "".join(w[0].upper() for w in v.split("_"))


def plane_abbr(value: Optional[str]) -> str:
    """Return an abbreviated string for the plane type.

    Map plane type names to their abbreviations ('stress' -> 'PS', 'strain' -> 'PE').

    Parameters
    ----------
    value : Optional[str]
        Plane type name.

    Returns
    -------
    str
        Abbreviated plane type string.
    """
    if value is None:
        return ""
    v = value.lower()
    return {"stress": "PS", "strain": "PE"}.get(v, value[0].upper())


def strain_abbr(value: Optional[str]) -> str:
    """Return an abbreviated string for the strain type.

    Map strain type names to their abbreviations (e.g., 'spectral' -> 'SP').

    Parameters
    ----------
    value : Optional[str]
        Strain type name.

    Returns
    -------
    str
        Abbreviated strain type string.
    """
    if value is None:
        return ""
    v = value.lower()
    name_dict = {
        "spectral": "SP",
        "volumetric": "VD",
        "none": "NO",
        "star-convex": "SX",
        "cubic_anisotropy_none": "CA",
        "orthotropic_anisotropy_none": "OA",
        "none_constitutive": "NC",
    }
    return name_dict.get(v, value[0].upper())


def rotation_str(
    angle: Optional[str],
    number_of_slices: Optional[int],
    slicing_direction: Optional[str],
    problem_type: Optional[str],
    strain_type: Optional[str],
) -> str:
    """Return a formatted string for the rotation angle.

    Generate a rotation string with signed angle and optional slice information.
    Only returns non-empty string for anisotropic problems or specific strain types.

    Parameters
    ----------
    angle : Optional[str]
        Rotation angle as string.
    number_of_slices : Optional[int]
        Number of slices for the material.
    slicing_direction : Optional[str]
        Direction of slicing.
    problem_type : Optional[str]
        Problem type.
    strain_type : Optional[str]
        Strain type.

    Returns
    -------
    str
        Formatted rotation string or empty string.
    """
    try:
        a = int(angle)
    except (TypeError, ValueError):
        return ""
    if problem_abbr(problem_type) != "AN" and strain_abbr(strain_type) not in (
        "CA",
        "OA",
    ):
        return ""
    out_angle = f"{a:+d}"
    out_rotation = (
        f"_{number_of_slices}{slicing_direction[0].upper()}"
        if number_of_slices > 1
        else ""
    )
    return out_angle + out_rotation


def dimensionalisation_str(nondimensionalisation: bool) -> str:
    """Return a string indicating whether the problem is non-dimensionalised.

    Parameters
    ----------
    nondimensionalisation : bool
        Whether the problem is non-dimensionalised.

    Returns
    -------
    str
        'ND' if non-dimensionalised, 'SD' otherwise.
    """
    if nondimensionalisation:
        return "ND"
    return "SD"


def transfer_learning_abbr(transfer_learning: Dict) -> str:
    """Return a string indicating whether transfer learning is enabled.

    Parameters
    ----------
    transfer_learning : Dict
        Configuration dictionary for transfer learning.

    Returns
    -------
    str
        '_TL' if enabled, '_TLO' if optimiser transfer enabled, empty string
        otherwise.
    """
    if transfer_learning.get("enabled", False):
        optimiser_suffix = ""
        if transfer_learning.get("transfer_optimiser", False):
            optimiser_suffix = "O"
        return "_TL" + optimiser_suffix
    return ""


def network_abbr(value: Optional[str]) -> str:
    """Return an abbreviated string for the network architecture.

    Parameters
    ----------
    value : Optional[str]
        Network architecture name.

    Returns
    -------
    str
        Abbreviated network architecture string.
    """
    if value is None:
        return ""
    v = value.lower()
    if v == "fnn":
        return "FNN"
    if v.startswith("resnet"):
        blocks = int(value[len("resnet") :])
        return f"RN{blocks}"
    return "".join(w[0].upper() for w in v.split("_"))


def activation_abbr(value: Optional[Dict]) -> str:
    """Return an abbreviated string for the activation function.

    Parameters
    ----------
    value : Optional[Dict]
        Activation function configuration dictionary.

    Returns
    -------
    str
        Abbreviated activation function string.
    """
    if value is None:
        return ""
    name = value.get("function", "")
    coeff = value.get("initial_coefficient", "")
    trainable = value.get("trainable", False)
    trainable_global = value.get("trainable_global", False)
    trainable_str = "T" if trainable else ""
    trainable_str += "G" if (trainable_str == "T" and trainable_global) else ""
    return name + f"{coeff:.1f}".replace(".", "-") + trainable_str


def optimiser_abbr(value: Optional[List]) -> str:
    """Return an abbreviated string for the optimiser.

    Parameters
    ----------
    value : Optional[List]
        List of optimiser names.

    Returns
    -------
    str
        Abbreviated optimiser string, joined with '+' if multiple.
    """
    if value is None or not isinstance(value, list) or len(value) == 0:
        return ""
    return "+".join(entry.split("_")[0] for entry in value)


def displacement_steps_abbr(value: Optional[Dict]) -> str:
    """Return the displacement configuration string.

    Generate an abbreviated string from the neural network displacement
    configuration dictionary.

    Parameters
    ----------
    value : Optional[Dict]
        Displacement configuration dictionary.

    Returns
    -------
    str
        Configuration string starting with 'D' followed by parameters,
        or 'D?' if value is None.
    """
    if value == None:
        return "D?"
    displacement_start = value.get("start", -1)
    increments_coarse = value.get("increments_coarse", -1)
    coarse_end = format_json_string(value.get("coarse_end", -1))
    increments_fine = value.get("increments_fine", -1)
    displacement_end = format_json_string(value.get("end", -1))
    displacement_str = "D"
    displacement_str += (
        "S" + str(displacement_start).replace(".", "_")
        if displacement_start != 0
        else ""
    )
    displacement_str += "C" + str(increments_coarse) if increments_coarse != 0 else ""
    displacement_str += (
        "CE" + str(coarse_end).replace(".", "_") if coarse_end != "0.0" else ""
    )
    displacement_str += "F" + str(increments_fine) if increments_fine != 0 else ""
    displacement_str += "E" + str(displacement_end).replace(".", "_")
    return displacement_str


def rff_abbr(value: Optional[Dict]) -> str:
    """Generate the RFF abbreviation for the filename.

    Parameters
    ----------
    value : Optional[Dict]
        RFF configuration dictionary.

    Returns
    -------
    str
        RFF abbreviation string or empty string if disabled.
    """
    if not value.get("enabled"):
        return ""
    return f"_RFF{value.get('features')}x{str(value.get('scale')).replace('.', '_')}"


class JsonJinjaRenamer:
    """Handle renaming JSON files using Jinja2 templates.

    Parameters
    ----------
    base_dir : str
        Base directory containing JSON files to process.
    config_path : str
        Path to the configuration file.
    auto_yes : bool, optional
        Automatically confirm renames without prompting. Default is False.
    """

    def __init__(
        self,
        base_dir: str,
        config_path: str,
        auto_yes: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir)
        cfg = Path(config_path)
        self.config = json.loads(cfg.read_text())
        self.env = Environment(
            loader=FileSystemLoader(cfg.parent),
            undefined=StrictUndefined,
            autoescape=False,
        )
        # register filters
        self.env.filters.update(
            {
                "elem_abbr": elem_abbr,
                "problem_abbr": problem_abbr,
                "plane_abbr": plane_abbr,
                "strain_abbr": strain_abbr,
                "rotation_str": rotation_str,
                "dimensionalisation_str": dimensionalisation_str,
                "transfer_learning_abbr": transfer_learning_abbr,
                "network_abbr": network_abbr,
                "optimiser_abbr": optimiser_abbr,
                "activation_abbr": activation_abbr,
                "displacement_steps_abbr": displacement_steps_abbr,
                "rff_abbr": rff_abbr,
            }
        )
        self.template = self.env.get_template(self.config["template_file"])
        self.mappings = self.config["mappings"]
        self.auto_yes = auto_yes
        self.rename_mapping: dict[str, str] = {}
        self.references_updated = 0
        self.test_references_updated = 0

    def extract_vars(self, data: Dict) -> Dict:
        """Extract variables from JSON data using JSONPath expressions.

        Parameters
        ----------
        data : Dict
            JSON data dictionary.

        Returns
        -------
        Dict
            Dictionary of extracted variables.
        """
        ctx = {}
        for var_name, jp_expr in self.mappings.items():
            matches = jsonpath_parse(jp_expr).find(data)
            ctx[var_name] = matches[0].value if matches else None
        return ctx

    def update_single_reference(self, old_name: str, new_name: str) -> None:
        """Update references in JSON files based on the rename_mapping.

        When running 'parent':
            • In parent/: update data['extends']
            • In fem/ and nn/: update data['parent_input_file'] and
              data['overrides']['parent_input_file']
        When running 'fem' or 'nn':
            • Only update data['parent_input_file'] and
              data['overrides']['parent_input_file'] in that same subdir.

        Parameters
        ----------
        old_name : str
            Old file name to replace.
        new_name : str
            New file name to use.
        """
        project_root = self.base_dir.parent
        cmd = self.base_dir.name  # 'parent', 'fem', or 'nn'

        # Create a single mapping for the old and new names
        single_mapping = {old_name: new_name}

        # Add to the global rename mapping
        self.rename_mapping.update(single_mapping)

        # Scan through all JSON files in the project root
        for fp in project_root.rglob("*.json"):
            if str(fp.name) == new_name.split("/")[-1] + ".json":
                continue

            rel = fp.relative_to(project_root)
            top = rel.parts[0]

            # Decide if we should even open this file
            if cmd == "parent":
                in_parent = top == "parent"
                in_children = top in ("fem", "nn")
                if not (in_parent or in_children):
                    continue
            else:
                # cmd is 'fem' or 'nn'
                if top != cmd:
                    continue
                in_parent = False
                in_children = True

            # Load the JSON once
            try:
                data = json.loads(fp.read_text())
            except json.JSONDecodeError:
                continue

            updated = False

            # If cmd==parent and this is under parent/, update `extends`
            if (in_parent and cmd == "parent") or (
                in_children and cmd in ("fem", "nn")
            ):
                if "extends" in data:
                    old = data["extends"]
                    if old in single_mapping:
                        data["extends"] = single_mapping[old]
                        updated = True

            # In any child-scan, update `parent_input_file` and overrides
            if in_children and "parent_input_file" in data:
                old = data["parent_input_file"]
                if old in single_mapping:
                    data["parent_input_file"] = single_mapping[old]
                    updated = True

            # Also update nested overrides.parent_input_file if present
            if in_children and isinstance(data.get("overrides"), dict):
                old = data["overrides"].get("parent_input_file")
                if old in single_mapping:
                    data["overrides"]["parent_input_file"] = single_mapping[old]
                    updated = True

            # If any updates were made, write the file back
            if updated:
                logging.info(f"Updating references in {rel}")
                with open(fp, "w") as f:
                    f.write(format_json_string(data))
                self.references_updated += 1

    def process_commit_references(self) -> None:
        """Update references in commit files based on the rename_mapping.

        Update the reference_files.sh file in the commit directory to reflect
        renamed files.
        """
        commit_dir = self.base_dir.parent.parent / "commit"
        reference_file = commit_dir / "reference_files.sh"

        if not reference_file.is_file():
            logging.info(f"Reference file {reference_file} does not exist.")
            return

        update = False
        with open(reference_file, "r") as f:
            data = f.readlines()

        target_line = None
        line_index = None
        for i, line in enumerate(data):
            if line.lower().startswith(self.base_dir.name):
                target_line = line
                line_index = i

        for key, value in self.rename_mapping.items():
            if key in target_line:
                logging.info(f"Updating reference {key} -> {value}")
                target_line = target_line.replace(key, value)
                self.references_updated += 1
                update = True
                data[line_index] = target_line
                break

        if update:
            with open(reference_file, "w") as f:
                f.writelines(data)

    def prepare_data(self, data: Dict, dir_target: str) -> Dict:
        """Always load the parent JSON, then merge overrides if present.

        For a simple file (no 'extends'), orig has 'parent_input_file' → load
        that JSON. For an extended file, orig has 'extends' and overrides → load
        the parent from overrides.

        Parameters
        ----------
        data : Dict
            Input data dictionary.
        dir_target : str
            Target directory name.

        Returns
        -------
        Dict
            Merged data dictionary.
        """
        for key in ("extends", "parent_input_file"):
            if key in data and data[key]:
                stem = data[key]
                if stem in self.rename_mapping:
                    # map old_stem → new_filename.json → new_stem
                    new_fname = self.rename_mapping[stem]
                    data[key] = Path(new_fname).stem

        # Extend the data with the input file extension if possible
        data = get_input_file_extension(data, dir_target)

        # Determine which parent to load
        parent_ref = data.get("parent_input_file")
        if parent_ref is not None:
            parent_data = parse_parent_input_json(parent_ref)

            # Merge parent data into the original data
            data.update(parent_data)

        # If the mesh is NURBS, add it to the data
        if data.get("mesh", {}).get("type", None) == "nurbs":
            current_path = Path(__file__).parent.resolve()
            mesh_path = (
                current_path.parent
                / "mesh"
                / "nurbs"
                / f"{data['mesh']['filename']}.json"
            )

            if mesh_path.is_file():
                mesh_data = json.loads(mesh_path.read_text())
                data.update(mesh_data)

        return data

    def generate_name(self, data: Dict) -> str:
        """Generate a new name for the JSON file based on the data and template.

        Custom logic for forming fields goes here.

        Parameters
        ----------
        data : Dict
            Data dictionary containing configuration.

        Returns
        -------
        str
            Generated file name.
        """
        ctx = self.extract_vars(data)

        # Normalize mesh_filename
        if ctx.get("mesh_filename"):
            ctx["mesh_filename"] = Path(ctx["mesh_filename"]).name
        # Compute formatted fields
        ctx["material_rotation_str"] = rotation_str(
            ctx.get("material_rotation_angle"),
            ctx.get("number_of_slices"),
            ctx.get("slicing_direction"),
            ctx.get("problem_type"),
            ctx.get("strain_split"),
        )

        et = ctx.get("elem_type") or ""
        if et:
            et = et.lower()
        if et == "nurbs":
            ox = ctx.get("order_x") or ""
            oy = ctx.get("order_y") or ""
            gp = ctx.get("gauss_integration_order") ** 2 or ""
            # e.g. IGA3x3_3IP
            ctx["elem_abbr_str"] = f"IGA{ox}x{oy}_{gp}IP"
        else:
            ctx["elem_abbr_str"] = elem_abbr(ctx.get("elem_type") or "")

        ctx["dimensionalisation_str"] = dimensionalisation_str(
            ctx.get("nondimensionalisation", False)
        )

        ctx["transfer_learning_abbr"] = transfer_learning_abbr(
            data.get("transfer_learning", {})
        )

        return self.template.render(**ctx)

    def process_file(self, file_path: Path) -> None:
        """Process a single JSON file.

        Read the file, generate a new name, and rename if necessary.

        Parameters
        ----------
        file_path : Path
            Path to the JSON file to process.
        """
        try:
            original = json.loads(file_path.read_text())
        except json.JSONDecodeError:
            logging.warning(f"Skipping invalid JSON: {file_path}")
            return

        data = self.prepare_data(original, self.base_dir.stem)
        new_name = self.generate_name(data)
        new_path = file_path.parent / new_name

        rel = file_path.relative_to(self.base_dir)
        if file_path.name != new_name:
            old_stem = file_path.stem
            # Extract the folder name for the new file if exists
            folder_name = str(rel.parent) + "/" if str(rel.parent) != "." else ""
            logging.info(f"Renaming {rel} -> {folder_name}{new_name}")
            if self.auto_yes or input("Proceed? (y/n) ").strip().lower() == "y":
                os.rename(file_path, new_path)
                self.update_single_reference(
                    folder_name + old_stem, folder_name + new_name.split(".")[0]
                )
                self.references_updated += 1
        else:
            logging.info(f"OK")

    def process_test_files(self, files_path: Path) -> None:
        """Process test files to update references.

        Rename test files and their associated data files according to the
        rename mapping.

        Parameters
        ----------
        files_path : Path
            Path to the directory containing test files.
        """
        # For each test file in a given path, check if it matches the rename mapping.
        for py_file in files_path.glob("test_*.py"):
            for key, value in self.rename_mapping.items():
                # Only check the key if it starts with 'tests/' and matches the test file name.
                if "tests/" in key and key.split("/")[1] in py_file.name:
                    # Sanitise the key and value for renaming.
                    safe_key = key.split("/")[1]
                    safe_value = value.split("/")[1]
                    # Rename the test file and update the data file reference.
                    logging.info(
                        f"\nUpdating test file {py_file.name} -> {py_file.name.replace(safe_key, safe_value)}"
                    )
                    os.rename(
                        py_file,
                        py_file.with_name(py_file.name.replace(safe_key, safe_value)),
                    )
                    logging.info(f"Renaming data file {key}.dat -> {value}.dat")
                    os.rename(
                        f"{py_file.parent}/data/{safe_key}.dat",
                        f"{py_file.parent}/data/{safe_value}.dat",
                    )
                    self.test_references_updated += 1

    def run(self) -> None:
        """Process all JSON files in the base directory.

        Scan for changed JSON files, generate new names, rename files,
        and update all references throughout the project.
        """
        sep = "-" * 50
        logging.info(
            f"{sep}\nProcessing input files directory for {self.base_dir.name}\n{sep}"
        )

        # Get the changed files since last commit
        repo_root = self.base_dir.parent.parent  # adjust if needed (repo root)

        scan_all = os.environ.get("SCAN_ALL_FILES", "").strip().lower() == "true"
        if scan_all:
            logging.info("SCAN_ALL_FILES=true — scanning all JSON files in repository.")
            candidate_files = list_all_json_files(repo_root)
        else:
            logging.info("Scanning only git-affected JSON files.")
            candidate_files = list_changed_files_since_last_commit(repo_root)

        # Only include JSON files under this base_dir
        targets = [
            p
            for p in candidate_files
            if p.suffix == ".json" and self.base_dir in p.parents
        ]

        if not targets:
            logging.info("No changed JSON files since last commit.")
        else:
            for fp in sorted(targets):
                rel = fp.relative_to(self.base_dir)
                logging.info(f"\nProcessing changed file: {rel}")
                try:
                    self.process_file(fp)
                except Exception as e:
                    logging.error(f"Error processing {rel}: {e}")

        logging.info(f"\n{sep}\nProcessing commit references\n{sep}")
        self.process_commit_references()

        # Process tests directory if it exists
        target_dir = self.base_dir.name
        tests_dir = self.base_dir.parent.parent / target_dir / "tests"
        if tests_dir.exists() and tests_dir.is_dir():
            logging.info(f"\n{sep}\nProcessing tests directory\n{sep}")
            self.process_test_files(tests_dir)

        # Summary output
        logging.info(f"\n{sep}")
        logging.info(f"Updated references in JSON files: {self.references_updated}")
        if tests_dir.exists() and tests_dir.is_dir():
            logging.info(
                f"Updated references in test files: {self.test_references_updated}"
            )
        logging.info(sep)
