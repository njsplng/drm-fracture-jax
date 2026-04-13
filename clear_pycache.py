#!/usr/bin/env python3

"""Remove all __pycache__ directories in the current directory tree."""

import pathlib
import shutil


def clean_pycache() -> int:
    """Remove all __pycache__ directories in the current directory tree.

    Recursively search for and delete all __pycache__ directories starting
    from the current working directory.

    Returns
    -------
    int
        Number of __pycache__ directories removed.
    """
    count = 0
    for p in pathlib.Path(".").rglob("__pycache__"):
        shutil.rmtree(p)
        count += 1
    return count


if __name__ == "__main__":
    count = clean_pycache()
    print(f"Removed {count} __pycache__ directories")
