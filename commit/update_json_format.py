#!/usr/bin/env python3

"""
Update the JSON format for input files.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Union

from fast_json_repair import repair_json

# Ensure src is in path for io_handlers
current_path = Path(__file__).parent.resolve()
src_path = current_path.parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
from json_encoder import format_json_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_json_with_repair(file: Path) -> Union[dict, list, None]:
    """
    Load JSON from a file, attempting to repair it if it's malformed.
    """
    with open(file, "r") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logging.warning(f"Malformed JSON at {file}: {e} — attempting repair")
        repaired = repair_json(
            raw, return_objects=True
        )  # returns parsed object directly
        if repaired is None:
            logging.error(f"Could not repair {file}, skipping")
            return None
        logging.info(f"Repaired {file}")
        return repaired


def main() -> None:
    """
    Main function to update JSON files.
    """
    for root, dirs, files in os.walk(current_path.parent):
        # Prevent os.walk from recursing into "output" and files starting with "."
        dirs[:] = [d for d in dirs if d != "output" and not d.startswith(".")]

        for name in files:
            if name.endswith(".json"):
                file = Path(root) / name
                content = load_json_with_repair(file)
                # skip files that couldn't be repaired
                if content is None:
                    continue
                with open(file, "w") as f:
                    f.write(format_json_string(content))


if __name__ == "__main__":
    main()
