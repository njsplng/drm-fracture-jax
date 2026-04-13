"""Script for generating reference data for FEM tests."""

import pathlib
from datetime import datetime

from utils_tests import (
    cleanup_testing_pickle_path,
    execute_fem,
    setup_testing_pickle_path,
)

if __name__ == "__main__":
    current_path = pathlib.Path(__file__).parent.resolve()
    fem_tests_input_path = current_path.parent.parent / "input" / "fem" / "tests"
    reference_data_folder_path = current_path / "data"

    # Set up the pickle path for FEM tests generation
    pickle_output_path = setup_testing_pickle_path()

    # Keep track of what was executed for bookkeeping
    files_executed = []

    # If the input path exists and is a directory, execute the FEM code for each file
    if fem_tests_input_path.exists() and fem_tests_input_path.is_dir():
        for file_path in fem_tests_input_path.glob("*"):
            print("Running FEM code for file:", file_path.name)
            execute_fem(file_path.name.split(".")[0])
            files_executed.append(file_path.name.split(".")[0])  # Bookkeeping
    # Otherwise, raise an error that something has gone wrong
    else:
        raise FileNotFoundError(
            f"Directory {fem_tests_input_path} does not exist or is not a directory"
        )

    # If "data" folder does not exist, create it
    if not reference_data_folder_path.exists():
        reference_data_folder_path.mkdir(parents=True, exist_ok=True)
    # If it does, wipe all the files in it
    else:
        for file in reference_data_folder_path.iterdir():
            if file.is_file():
                file.unlink()

    # Move the generated pickle files to the data folder
    for file_path in pickle_output_path.glob("*"):
        file_path.rename(reference_data_folder_path / file_path.name)

    # Clean up the pickle path for FEM tests generation
    cleanup_testing_pickle_path()

    # Leave a fingerprint for the last date of execution and what was executed
    fingerprint = (
        f"Last run: {datetime.now().isoformat()}; Files executed: {files_executed}\n"
    )
    # Write the fingerprint to a file
    with open(str(current_path / "last_run.txt"), "a") as f:
        f.write(fingerprint)
