#!/usr/bin/env python3

"""Crawl the input files in a directory and validate them against a reference JSON file."""

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

current_path = Path(__file__).parent.resolve()
src_path = current_path.parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
from io_handlers import get_input_file_extension, parse_parent_input_json
from json_encoder import format_json_string


class OrderedJsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that preserves order.
    """

    def encode(self, obj: object) -> str:
        if isinstance(obj, OrderedDict):
            return (
                "{"
                + ", ".join(
                    f"{self.encode(k)}: {self.encode(v)}" for k, v in obj.items()
                )
                + "}"
            )
        return super().encode(obj)


class JsonValidator:
    """
    Given a reference JSON file, validate and fix JSON files in a directory.
    """

    def __init__(
        self,
        target_dir: str,
        reference_file: str,
        auto_yes: bool = False,
        auto_no: bool = False,
    ) -> None:
        """Initialize the JsonValidator with target directory, reference file, and options."""
        current_path = Path(__file__).parent.resolve()
        input_path = current_path.parent / "input"
        self.target_dir = input_path / target_dir
        self.reference_file = self.target_dir / reference_file
        self.auto_yes = auto_yes
        self.auto_no = auto_no
        self.reference_data = self._load_reference()
        self.files_processed = 0
        self.files_modified = 0

    def _load_reference(self) -> OrderedDict:
        """Load the reference JSON file.

        Returns:
            OrderedDict: The loaded reference JSON data.

        Raises:
            SystemExit: If reference file is not found or invalid JSON.
        """
        try:
            with open(self.reference_file, "r") as f:
                return json.load(f, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            logging.error(f"Reference file '{self.reference_file}' not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            logging.error(f"Reference file '{self.reference_file}' is not valid JSON.")
            sys.exit(1)

    def _find_json_files(self) -> List[Path]:
        """Find all JSON files in the target directory.

        Returns:
            List[Path]: List of JSON file paths.
        """
        json_files = []
        for file in self.target_dir.glob("**/*.json"):
            if file.is_file() and file != self.reference_file:
                json_files.append(file)
        return json_files

    def _check_missing_keys(self, target_data: OrderedDict) -> OrderedDict:
        """Find keys that are in the reference but missing in target.

        Args:
            target_data: The target JSON data to check.

        Returns:
            OrderedDict: Missing keys with their reference values.
        """
        missing_keys = OrderedDict()

        def check_recursive(
            ref: OrderedDict, target: OrderedDict, path: str = ""
        ) -> None:
            if isinstance(ref, (dict, OrderedDict)) and isinstance(
                target, (dict, OrderedDict)
            ):
                for key, value in ref.items():
                    current_path = f"{path}.{key}" if path else key
                    if key not in target:
                        missing_keys[current_path] = value
                    elif isinstance(value, (dict, OrderedDict, list)):
                        check_recursive(
                            value,
                            target.get(
                                key,
                                (
                                    OrderedDict()
                                    if isinstance(value, (dict, OrderedDict))
                                    else []
                                ),
                            ),
                            current_path,
                        )
            elif isinstance(ref, list) and isinstance(target, list):
                # Do not check lists, all of the hierarchy is already checked in the dicts
                return

        check_recursive(self.reference_data, target_data)
        return missing_keys

    def _check_extra_keys(
        self,
        target_data: OrderedDict,
        reference_data: Optional[OrderedDict] = None,
    ) -> OrderedDict:
        """Find keys that are in target but not in reference.

        Args:
            target_data: The target JSON data to check.
            reference_data: Optional reference data to check against.

        Returns:
            OrderedDict: Extra keys with their values.
        """
        extra_keys = OrderedDict()

        def check_recursive(
            ref: OrderedDict, target: OrderedDict, path: str = ""
        ) -> None:
            if isinstance(ref, (dict, OrderedDict)) and isinstance(
                target, (dict, OrderedDict)
            ):
                for key, value in target.items():
                    current_path = f"{path}.{key}" if path else key
                    if key not in ref:
                        extra_keys[current_path] = value
                    elif isinstance(value, (dict, OrderedDict, list)) and isinstance(
                        ref.get(key), (dict, OrderedDict, list)
                    ):
                        check_recursive(ref.get(key), value, current_path)
            elif isinstance(ref, list) and isinstance(target, list):
                # Do not check lists, all of the hierarchy is already checked in the dicts
                return

        if reference_data is None:
            reference_data = self.reference_data
        check_recursive(reference_data, target_data)
        return extra_keys

    def _add_missing_keys(
        self, target_data: OrderedDict, missing_keys: OrderedDict
    ) -> OrderedDict:
        """Add missing keys to the target data with default value -1.

        Args:
            target_data: The target JSON data to modify.
            missing_keys: OrderedDict of missing keys to add.

        Returns:
            OrderedDict: Modified target data with missing keys added.
        """
        # First, get a deep copy of the target data
        data = self._deep_copy_ordered(target_data)

        # Process each missing key
        for path, value in missing_keys.items():
            keys = path.split(".")
            last_key = keys[-1]

            # Navigate to find the parent dictionary that will contain the new key
            current = data
            ref_current = self.reference_data
            parent_path = []

            for i, key in enumerate(keys[:-1]):
                parent_path.append(key)
                if key not in current:
                    current[key] = OrderedDict()
                current = current[key]

                # Keep track of where we are in the reference
                if key in ref_current:
                    ref_current = ref_current[key]
                else:
                    ref_current = {}

            # If we're at a leaf node, just add the default value
            if not isinstance(value, (dict, OrderedDict)):
                # Create a new OrderedDict with keys in reference order
                if isinstance(ref_current, (dict, OrderedDict)) and isinstance(
                    current, (dict, OrderedDict)
                ):
                    new_dict = OrderedDict()

                    # First add all keys from reference that exist in current
                    for ref_key in ref_current:
                        if ref_key in current and ref_key != last_key:
                            new_dict[ref_key] = current[ref_key]
                        elif ref_key == last_key:
                            new_dict[last_key] = self._define_placeholder_values(value)

                    # Then add the new key in correct position
                    if last_key not in new_dict:
                        new_dict[last_key] = -1

                    # Finally add any remaining keys from current not in reference
                    for curr_key in current:
                        if curr_key not in new_dict:
                            new_dict[curr_key] = current[curr_key]

                    # Replace the current dict with our reordered one
                    parent = data
                    for p_key in parent_path[:-1]:
                        parent = parent[p_key]

                    if parent_path:
                        parent[parent_path[-1]] = new_dict
                    else:
                        data = new_dict
                else:
                    # If reference isn't a dict or we can't reorder, just add the key
                    current[last_key] = -1
            else:
                # For nested structures, create with default values
                current[last_key] = self._create_structure_with_defaults(value, path)

        return data

    def _reorder_dict_like_reference(self, target_dict: Dict, ref_dict: Dict) -> Dict:
        """Recursively reorder target_dict to match the key order in ref_dict.

        Args:
            target_dict: The dictionary to reorder.
            ref_dict: The reference dictionary to match ordering against.

        Returns:
            Dict: Reordered dictionary matching ref_dict's key order.
        """
        if not isinstance(target_dict, (dict, OrderedDict)) or not isinstance(
            ref_dict, (dict, OrderedDict)
        ):
            return target_dict

        result = OrderedDict()

        # First add keys that exist in both, in reference order
        for key in ref_dict:
            if key in target_dict:
                if isinstance(ref_dict[key], (dict, OrderedDict)) and isinstance(
                    target_dict[key], (dict, OrderedDict)
                ):
                    result[key] = self._reorder_dict_like_reference(
                        target_dict[key], ref_dict[key]
                    )
                else:
                    result[key] = target_dict[key]

        # Then add any keys in target that aren't in reference (preserve their order)
        for key in target_dict:
            if key not in result:
                result[key] = target_dict[key]

        return result

    def _deep_copy_ordered(self, obj: object) -> OrderedDict:
        """Create a deep copy of an object, preserving OrderedDict structures.

        Args:
            obj: The object to deep copy.

        Returns:
            OrderedDict: Deep copy of the object.
        """
        if isinstance(obj, (dict, OrderedDict)):
            return OrderedDict((k, self._deep_copy_ordered(v)) for k, v in obj.items())
        elif isinstance(obj, list):
            return [self._deep_copy_ordered(item) for item in obj]
        else:
            return obj

    def _create_structure_with_defaults(
        self, structure: object, path: str = ""
    ) -> OrderedDict:
        """Recursively create a structure with default values.

        Args:
            structure: The structure to create defaults for.
            path: Current path in the structure.

        Returns:
            OrderedDict: Structure with default values.
        """
        if isinstance(structure, (dict, OrderedDict)):
            result = OrderedDict()
            for k, v in structure.items():
                current_path = f"{path}.{k}" if path else k
                result[k] = self._create_structure_with_defaults(v, current_path)
            return result
        else:
            return self._define_placeholder_values(structure)

    def _define_placeholder_values(self, structure: object) -> object:
        """Define placeholder values for different types.

        Args:
            structure: The structure to define placeholder for.

        Returns:
            object: Placeholder value for the type.
        """
        if isinstance(structure, list):
            return []
        elif isinstance(structure, str):
            return "placeholder_string"
        elif isinstance(structure, bool):
            return "paceholder_boolean (true/false)"
        elif isinstance(structure, int):
            return "placeholder_integer"
        elif isinstance(structure, float):
            return "placeholder_float"
        else:
            # For any other type, try to use the same type with a sensible default
            logging.warning(f"Encountered unknown type {type(structure)}")
            return -999999  # Fallback

    def _remove_extra_keys(
        self, target_data: OrderedDict, extra_keys: OrderedDict
    ) -> OrderedDict:
        """Remove extra keys from the target data.

        Args:
            target_data: The target JSON data to modify.
            extra_keys: OrderedDict of extra keys to remove.

        Returns:
            OrderedDict: Modified target data with extra keys removed.
        """
        data = self._deep_copy_ordered(target_data)

        for path in extra_keys:
            keys = path.split(".")
            current = data
            parent_chain = []

            # Navigate to the parent of the key to remove
            for i, key in enumerate(keys[:-1]):
                if key not in current:
                    break
                parent_chain.append(current)
                current = current[key]

            # Remove the extra key
            last_key = keys[-1]
            if last_key in current:
                del current[last_key]

        return data

    def process_file(self, file_path: str) -> None:
        """Process a single JSON file.

        Args:
            file_path: Path to the JSON file to process.
        """
        try:
            with open(file_path, "r") as f:
                target_data = json.load(f, object_pairs_hook=OrderedDict)
        except json.JSONDecodeError:
            logging.warning(f"Skipping invalid JSON file: {file_path}")
            return

        # Initialise flag
        data_extended = False
        # Check if the target data extends another file. Need to branch to not check the missing keys
        if target_data.get("extends", None):
            # Set the flag
            data_extended = True
            # Check against the override keys
            override_keys = target_data.get("overrides", {})
            # If keys are found
            if override_keys:
                # Link to the reference data if it exists
                extended_target_data = get_input_file_extension(
                    target_data, self.target_dir
                )
                # Set an empty flag (defaults to self.reference_data by default)
                reference_data = None

                # Check if there is a parent input file and link to it
                if extended_target_data.get("parent_input_file", None) is not None:
                    # Parse the parent input file to check for overrides
                    parent_input_file = parse_parent_input_json(
                        extended_target_data["parent_input_file"]
                    )
                    # Pass it as the reference data
                    reference_data = extended_target_data.update(parent_input_file)

                # Find the extra keys in the overrides
                extra_keys = self._check_extra_keys(override_keys, reference_data)

        # Initialise as empty for safety
        missing_keys = None

        # If the data is not extended, we can check for missing keys and the extra keys in the input file
        if not data_extended:
            missing_keys = self._check_missing_keys(target_data)
            extra_keys = self._check_extra_keys(target_data)

        # Flag to check if the file was modified
        modified = False

        # Want to do the missing keys check only if the data is not extended
        if not data_extended and missing_keys:
            logging.info(f"\nFile: {file_path}")
            logging.info("Missing keys:")
            for key in missing_keys:
                logging.info(f"  - {key}")

            if not self.auto_no and (
                self.auto_yes or input("Add missing keys? (y/n) ").lower() == "y"
            ):
                target_data = self._add_missing_keys(target_data, missing_keys)
                modified = True
                logging.info("Missing keys added.")

        if extra_keys:
            # Print the file name again only if there are no missing keys and the input was not an extension to avoid redundancy
            if not missing_keys or data_extended:
                logging.info(f"\nFile: {file_path}")

            logging.info("Extra keys not in reference:")
            for key in extra_keys:
                logging.info(f"  - {key}")

            if not self.auto_no and (
                self.auto_yes or input("Remove extra keys? (y/n) ").lower() == "y"
            ):
                if not data_extended:
                    target_data = self._remove_extra_keys(target_data, extra_keys)
                else:
                    target_data["overrides"] = self._remove_extra_keys(
                        target_data.get("overrides", OrderedDict()), extra_keys
                    )
                modified = True
                logging.info("Extra keys removed.")

        # Inside process_file, right before writing to file
        if modified:
            # Reorder the entire structure to match reference
            target_data = self._reorder_dict_like_reference(
                target_data, self.reference_data
            )

            with open(file_path, "w") as f:
                f.write(format_json_string(target_data))

            self.files_modified += 1

        self.files_processed += 1

    def run(self) -> None:
        """Run the validation process on all JSON files.

        Returns:
            None
        """
        json_files = self._find_json_files()

        if not json_files:
            logging.info(f"No JSON files found in '{self.target_dir}'.")
            return

        logging.info(f"Found {len(json_files)} JSON files to check.")

        for file_path in json_files:
            try:
                self.process_file(file_path)
            except Exception as e:
                logging.error(f"File {file_path} failed: {e}")

        logging.info(
            f"\nSummary: Processed {self.files_processed} files, modified {self.files_modified} files."
        )


def main() -> None:
    """Main function to parse arguments and run the validator.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Validate and fix JSON files against a reference template."
    )
    parser.add_argument("directory", help="Directory containing JSON files to check")  # type: ignore[arg-type]
    parser.add_argument("reference", help="Path to the reference JSON file")  # type: ignore[arg-type]
    parser.add_argument("-y", "--yes", action="store_true", help="Confirm all changes")  # type: ignore[arg-type]
    parser.add_argument("-n", "--no", action="store_true", help="Decline all changes")  # type: ignore[arg-type]
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")  # type: ignore[arg-type]

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    validator = JsonValidator(args.directory, args.reference, args.yes, args.no)
    validator.run()


if __name__ == "__main__":
    main()
