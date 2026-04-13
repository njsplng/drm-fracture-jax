#!/usr/bin/env python3

"""Normalise separator lines in code files to a standard format."""

import re
import sys

SEPARATOR = "# -----------------------------------------------------------"

for filepath in sys.argv[1:]:
    with open(filepath, "r") as f:
        content = f.read()
    new_content = re.sub(r"# -{9,}", SEPARATOR, content)
    if new_content != content:
        with open(filepath, "w") as f:
            f.write(new_content)
