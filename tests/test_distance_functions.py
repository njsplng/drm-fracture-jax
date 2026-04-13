"""Tests for distance functions.

Tests DistanceFunction1D, DistanceFunction2D, and CompositeDistanceFunction
from src/distance_functions.py.
"""

import jax.numpy as jnp

from distance_functions import (
    CompositeDistanceFunction,
    DistanceFunction1D,
    DistanceFunction2D,
)


class TestDistanceFunction1D:
    """Tests for DistanceFunction1D."""

    def test_distance_function_1d_basic(self):
        """Test 5.1: DistanceFunction1D produces correct values at known locations.

        Create with x_init=0, L=1, d0=0.5, order=2.
        Verify: value is 1 inside interval, 0 far outside, decays correctly in transition zones.
        """
        df = DistanceFunction1D(x_init=0.0, L=1.0, d0=0.5, order=2)

        # Inside the interval [0, 1] - should be 1.0
        inside_points = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        values_inside = df(inside_points)
        assert jnp.allclose(
            values_inside, 1.0, atol=1e-6
        ), "Values inside interval should be 1.0"

        # Far outside - should be 0.0
        far_points = jnp.array([-2.0, -1.0, 2.0, 3.0])
        values_far = df(far_points)
        assert jnp.allclose(
            values_far, 0.0, atol=1e-6
        ), "Values far outside should be 0.0"

        # In transition zone (left decay: -0.5 < x < 0)
        left_transition = jnp.array([-0.4, -0.25, -0.1])
        values_left = df(left_transition)
        assert jnp.all(values_left > 0), "Left transition values should be positive"
        assert jnp.all(values_left < 1), "Left transition values should be less than 1"

        # In transition zone (right decay: 1 < x < 1.5)
        right_transition = jnp.array([1.1, 1.25, 1.4])
        values_right = df(right_transition)
        assert jnp.all(values_right > 0), "Right transition values should be positive"
        assert jnp.all(
            values_right < 1
        ), "Right transition values should be less than 1"

    def test_distance_function_1d_order_1(self):
        """Test 5.1 variant: First order distance function."""
        df = DistanceFunction1D(x_init=0.0, L=1.0, d0=0.5, order=1)

        # Inside the interval - should be 1.0
        inside_points = jnp.array([0.5])
        values_inside = df(inside_points)
        assert jnp.allclose(values_inside, 1.0, atol=1e-6)

    def test_distance_function_1d_shape_handling(self):
        """Test 5.1 variant: Input shape handling."""
        df = DistanceFunction1D(x_init=0.0, L=1.0, d0=0.5, order=2)

        # Test 1D input (N,)
        inp_1d = jnp.array([0.5, 0.75])
        result_1d = df(inp_1d)
        assert result_1d.shape == (2,), "Output shape should match input count"

        # Test 2D input (N, 1)
        inp_2d = jnp.array([[0.5], [0.75]])
        result_2d = df(inp_2d)
        assert result_2d.shape == (2,), "Output shape should match input count"

        # Results should be the same
        assert jnp.allclose(result_1d, result_2d), "Results should be identical"


class TestDistanceFunction2D:
    """Tests for DistanceFunction2D."""

    def test_distance_function_2d_basic(self):
        """Test 5.2: DistanceFunction2D produces correct values for a horizontal line segment.

        Verify: ≈1 on the line, decays for y≠0, 0 far away.
        """
        # Horizontal line segment from (0, 0) to (1, 0)
        df = DistanceFunction2D(
            x_init=0.0, y_init=0.0, theta=0.0, L=1.0, d0=0.5, order=2
        )

        # On the line segment - should be ≈1.0
        on_line = jnp.array([[0.25, 0.0], [0.5, 0.0], [0.75, 0.0]])
        values_on_line = df(on_line)
        assert jnp.all(
            values_on_line > 0.9
        ), f"Values on line should be ≈1, got {values_on_line}"

        # Off the line (y ≠ 0) but within decay distance - should decay
        off_line = jnp.array([[0.5, 0.2], [0.5, 0.3], [0.5, 0.4]])
        values_off_line = df(off_line)
        assert jnp.all(
            values_off_line > 0
        ), "Off-line values within decay should be positive"
        assert jnp.all(values_off_line < 1), "Off-line values should be less than 1"

        # Far away - should be ≈0
        far_away = jnp.array([[0.5, 2.0], [5.0, 5.0], [-2.0, -2.0]])
        values_far = df(far_away)
        assert jnp.allclose(values_far, 0.0, atol=1e-6), "Far away values should be 0"

    def test_distance_function_2d_rotation(self):
        """Test 5.3: DistanceFunction2D with theta=90 rotates the segment correctly.

        A vertical line segment from (0, 0) to (0, 1).
        """
        # Vertical line segment (theta = 90 degrees)
        df = DistanceFunction2D(
            x_init=0.0, y_init=0.0, theta=90.0, L=1.0, d0=0.5, order=2
        )

        # On the vertical line segment - should be ≈1.0
        on_vertical = jnp.array([[0.0, 0.25], [0.0, 0.5], [0.0, 0.75]])
        values_on_vertical = df(on_vertical)
        assert jnp.all(
            values_on_vertical > 0.9
        ), f"Values on vertical line should be ≈1, got {values_on_vertical}"

        # Off the vertical line - should decay
        off_vertical = jnp.array([[0.2, 0.5], [0.3, 0.5], [0.4, 0.5]])
        values_off_vertical = df(off_vertical)
        assert jnp.all(
            values_off_vertical > 0
        ), "Off-vertical values within decay should be positive"
        assert jnp.all(
            values_off_vertical < 1
        ), "Off-vertical values should be less than 1"

        # Far away - should be ≈0
        far_away = jnp.array([[2.0, 0.5], [5.0, 5.0], [-2.0, -2.0]])
        values_far = df(far_away)
        assert jnp.allclose(values_far, 0.0, atol=1e-6), "Far away values should be 0"

    def test_distance_function_2d_45_degree_rotation(self):
        """Test 5.3 variant: 45 degree rotation."""
        # Line at 45 degrees
        df = DistanceFunction2D(
            x_init=0.0, y_init=0.0, theta=45.0, L=1.0, d0=0.5, order=2
        )

        # Points along the 45-degree line (x = y)
        on_diagonal = jnp.array([[0.25, 0.25], [0.5, 0.5], [0.7, 0.7]])
        values_on_diagonal = df(on_diagonal)
        assert jnp.all(
            values_on_diagonal > 0.9
        ), f"Values on diagonal should be ≈1, got {values_on_diagonal}"


class TestCompositeDistanceFunction:
    """Tests for CompositeDistanceFunction."""

    def test_composite_distance_function_1d(self):
        """Test 5.4: CompositeDistanceFunction sums multiple 1D distance functions."""
        # Two separate intervals
        input_list = [
            {"x_init": 0.0, "L": 0.3, "d0": 0.1, "order": 2},
            {"x_init": 0.7, "L": 0.3, "d0": 0.1, "order": 2},
        ]
        cdf = CompositeDistanceFunction(input_list, dimension=1)

        # In first interval only
        in_first = jnp.array([0.15])
        result_first = cdf(in_first)
        assert jnp.allclose(
            result_first, 1.0, atol=1e-6
        ), "Should be 1 in first interval"

        # In second interval only
        in_second = jnp.array([0.85])
        result_second = cdf(in_second)
        assert jnp.allclose(
            result_second, 1.0, atol=1e-6
        ), "Should be 1 in second interval"

        # Between intervals
        between = jnp.array([0.5])
        result_between = cdf(between)
        assert jnp.allclose(
            result_between, 0.0, atol=1e-6
        ), "Should be 0 between intervals"

    def test_composite_distance_function_2d(self):
        """Test 5.4: CompositeDistanceFunction sums multiple 2D distance functions."""
        # Two horizontal segments at different y positions
        input_list = [
            {
                "x_init": 0.0,
                "y_init": 0.0,
                "theta": 0.0,
                "L": 1.0,
                "d0": 0.2,
                "order": 2,
            },
            {
                "x_init": 0.0,
                "y_init": 1.0,
                "theta": 0.0,
                "L": 1.0,
                "d0": 0.2,
                "order": 2,
            },
        ]
        cdf = CompositeDistanceFunction(input_list, dimension=2)

        # On first segment
        on_first = jnp.array([[0.5, 0.0]])
        result_first = cdf(on_first)
        assert result_first[0] > 0.9, "Should be ≈1 on first segment"

        # On second segment
        on_second = jnp.array([[0.5, 1.0]])
        result_second = cdf(on_second)
        assert result_second[0] > 0.9, "Should be ≈1 on second segment"

        # Between segments
        between = jnp.array([[0.5, 0.5]])
        result_between = cdf(between)
        assert jnp.allclose(
            result_between, 0.0, atol=1e-6
        ), "Should be 0 between segments"

    def test_composite_distance_function_aggregate_true(self):
        """Test 5.4: aggregate=True returns a list of individual outputs."""
        input_list = [
            {"x_init": 0.0, "L": 0.3, "d0": 0.1, "order": 2},
            {"x_init": 0.7, "L": 0.3, "d0": 0.1, "order": 2},
        ]
        cdf = CompositeDistanceFunction(input_list, dimension=1)

        # Test point in first interval
        test_point = jnp.array([0.15, 0.85, 0.5])

        # aggregate=False (default) - returns summed array
        result_summed = cdf(test_point, aggregate=False)
        assert isinstance(
            result_summed, jnp.ndarray
        ), "Should return array when aggregate=False"
        assert result_summed.shape == (
            3,
        ), "Should return array of same length as input"

        # aggregate=True - returns list of individual outputs
        result_list = cdf(test_point, aggregate=True)
        assert isinstance(result_list, list), "Should return list when aggregate=True"
        assert len(result_list) == 2, "Should return one array per distance function"
        assert all(
            r.shape == (3,) for r in result_list
        ), "Each array should have same length as input"

        # Verify the sum matches
        summed_manual = sum(result_list)
        assert jnp.allclose(
            result_summed, summed_manual
        ), "Summed result should equal sum of individual results"

    def test_composite_distance_function_single_entry(self):
        """Test 5.4 variant: Single entry wrapped in list."""
        # Single entry as dict (should be wrapped)
        cdf = CompositeDistanceFunction(
            {"x_init": 0.0, "L": 1.0, "d0": 0.5, "order": 2}, dimension=1
        )

        result = cdf(jnp.array([0.5]))
        assert jnp.allclose(
            result, 1.0, atol=1e-6
        ), "Should work with single dict entry"


class TestDistanceFunctionEdgeCases:
    """Edge case tests for distance functions."""

    def test_distance_function_1d_zero_length(self):
        """Test behavior with very small L."""
        df = DistanceFunction1D(x_init=0.5, L=1e-6, d0=0.1, order=2)
        result = df(jnp.array([0.5]))
        # Should still produce a valid result
        assert jnp.isfinite(result).all(), "Should produce finite result"

    def test_distance_function_2d_zero_length(self):
        """Test behavior with very small L."""
        df = DistanceFunction2D(
            x_init=0.5, y_init=0.5, theta=0.0, L=1e-6, d0=0.1, order=2
        )
        result = df(jnp.array([[0.5, 0.5]]))
        # Should still produce a valid result
        assert jnp.isfinite(result).all(), "Should produce finite result"

    def test_distance_function_2d_smooth_end_false(self):
        """Test with smooth_end=False."""
        df = DistanceFunction2D(
            x_init=0.0, y_init=0.0, theta=0.0, L=1.0, d0=0.5, order=2, smooth_end=False
        )

        # On the line - should still be ≈1
        on_line = jnp.array([[0.5, 0.0]])
        result = df(on_line)
        assert result[0] > 0.9, "Should be ≈1 on line even with smooth_end=False"

    def test_composite_empty_list(self):
        """Test with empty input list."""
        cdf = CompositeDistanceFunction([], dimension=1)
        result = cdf(jnp.array([0.5, 1.0]))
        assert jnp.allclose(result, 0.0), "Empty composite should return zeros"
