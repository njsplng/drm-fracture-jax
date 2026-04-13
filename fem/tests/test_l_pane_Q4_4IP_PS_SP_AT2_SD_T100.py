from utils_tests import generic_test


def test() -> None:
    generic_test(__file__.split("/")[-1].split(".")[0][5:])
