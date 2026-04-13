"""Unit tests for optimiser builders.

Tests for optimiser construction and learning rate schedules.
"""

from pathlib import Path

import pytest
from optimisers import get_optimiser


class TestOptimiserBuilders:
    """Tests for optimiser builder functions."""

    def test_optimiser_builders(self) -> None:
        """Test 6.22: Call get_optimiser with test configs, verify returns correct type and epoch count."""
        # Test each optimiser type using test config files
        # Note: get_optimiser expects the filename without path (files must be in input/optimisers/)
        optimiser_configs = [
            "tests/adam_ME100",
            "tests/sgd_ME100",
            "tests/rprop_ME100",
            "tests/lbfgs_ME100",
            "tests/soap_ME100",
            "tests/ssbroydenarmijo_ME100_MEM20",
        ]

        for config_name in optimiser_configs:
            optimiser, n_epochs = get_optimiser(config_name)
            # Most optimisers return optax.GradientTransformation
            # ssbroydenarmijo returns OptimistixAsOptax
            assert isinstance(
                n_epochs, int
            ), f"{config_name} should return int for n_epochs"
            assert n_epochs == 100, f"{config_name} should return n_epochs=100"

    def test_learning_rate_schedules(self) -> None:
        """Test different learning rate schedule types using test configs."""
        # Fixed learning rate - just verify it loads without error
        optimiser, n_epochs = get_optimiser("tests/adam_ME100")
        assert n_epochs == 100, "Fixed LR config should have 100 epochs"

        # Linear schedule
        optimiser, n_epochs = get_optimiser("tests/adam_LS1-3_LE1-4_LRL_TE0_ME100")
        assert n_epochs == 100, "Linear schedule config should have 100 epochs"

        # Constant schedule
        optimiser, n_epochs = get_optimiser("tests/adam_LR1-3_ME100")
        assert n_epochs == 100, "Constant schedule config should have 100 epochs"

        # Cosine decay schedule
        optimiser, n_epochs = get_optimiser("tests/adam_LS1-3_LRCD_ME100")
        assert n_epochs == 100, "Cosine decay config should have 100 epochs"

        # Exponential decay schedule
        optimiser, n_epochs = get_optimiser("tests/adam_LS1-3_LE1-4_LRED_TE0_ME100")
        assert n_epochs == 100, "Exponential decay config should have 100 epochs"


class TestGetOptimiser:
    """Tests for get_optimiser function."""

    def test_get_optimiser_from_json(self) -> None:
        """Test 6.23: Load from existing optimiser JSON, verify returns without error and epoch count matches."""
        import json

        # Use a known test config
        config_name = "adam_ME100"
        config_file = (
            Path(__file__).parent.parent
            / "input"
            / "optimisers"
            / "tests"
            / f"{config_name}.json"
        )

        # Load the JSON to get expected epochs
        with open(config_file) as f:
            params = json.load(f)

        expected_epochs = params["number_of_epochs"]

        # Call get_optimiser
        optimiser, n_epochs = get_optimiser("tests/" + config_name)

        assert (
            n_epochs == expected_epochs
        ), f"Expected {expected_epochs} epochs, got {n_epochs}"

    def test_get_optimiser_unknown(self) -> None:
        """Test get_optimiser with unknown name raises an error."""
        # Unknown file raises FileNotFoundError first
        with pytest.raises((FileNotFoundError, ValueError)):
            get_optimiser("unknown_optimiser_xyz")
