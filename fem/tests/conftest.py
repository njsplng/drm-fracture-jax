"""Testing configuration for FEM simulations."""

import pytest
from utils_tests import cleanup_testing_pickle_path, setup_testing_pickle_path


@pytest.fixture(scope="session", autouse=True)
def fem_test_environment() -> None:
    """
    Set up the fem_tests pickle folder for writing testfiles.
    """
    # Set up the testing pickle path
    _ = setup_testing_pickle_path()

    # Yield to allow test execution
    yield

    # Clean up the testing pickle path
    cleanup_testing_pickle_path()
