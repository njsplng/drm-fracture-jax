"""Utilities for testing FEM code against reference data."""

import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Set up the necessary paths
current_path = pathlib.Path(__file__).parent.resolve()
# Link the fem source
fem_source_path = current_path.parent / "src"
if fem_source_path not in sys.path:
    sys.path.append(str(fem_source_path))

# Link the generic source
generic_source_path = current_path.parent.parent / "src"
if generic_source_path not in sys.path:
    sys.path.append(str(generic_source_path))

from io_handlers import input_pickle, read_blosc


def read_in_reference(filename: str) -> dict:
    """
    Reads in the reference data from a file.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    return read_blosc(current_path / f"data/{filename}.dat")


def execute_fem(filename: str) -> dict:
    """
    Executes the FEM code given the filename.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    fem_executable = str(current_path.parent / "main.py")
    subprocess.run(
        ["python3", fem_executable, f"tests/{filename}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return input_pickle(f"fem_tests/{filename}")


def setup_testing_pickle_path() -> pathlib.Path:
    """
    Sets up the pickle path for the test.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    test_pickle_path = current_path.parent.parent / "output" / "pickle" / "fem_tests"
    test_pickle_path.mkdir(parents=True, exist_ok=True)

    return test_pickle_path


def cleanup_testing_pickle_path() -> None:
    """
    Cleans up the pickle path for the testing.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    test_pickle_path = current_path.parent.parent / "output" / "pickle" / "fem_tests"
    if test_pickle_path.exists():
        for file in test_pickle_path.iterdir():
            if file.is_file():
                file.unlink()
        test_pickle_path.rmdir()


def generic_test(filename: str) -> None:
    """
    Generic test function for comparing FEM runs.
    """
    reference_data = read_in_reference(filename)
    test_data = execute_fem(filename)
    for key, value in reference_data.items():
        if key in test_data:
            assert jnp.allclose(
                jnp.array(value), jnp.array(test_data[key])
            ), f"Mismatch in {key}, largest difference: {jnp.max(jnp.abs(jnp.array(value) - jnp.array(test_data[key])))}"
        else:
            print(f"{key} not found in test data")
