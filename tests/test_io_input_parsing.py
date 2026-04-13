"""Unit tests for input JSON parsing and schema validation.

Tests the parse_input_json, get_input_file_extension, check_schema_conformance,
validate_input, format_errors, and parse_schema functions.
"""

import jsonschema
import pytest

from io_handlers import (
    check_schema_conformance,
    format_errors,
    get_input_file_extension,
    parse_input_json,
    parse_schema,
    validate_input,
)


class TestParseInputJson:
    """Test input JSON parsing."""

    def test_parse_input_json_valid(self) -> None:
        """Test 1.10: parse_input_json successfully loads, extends, validates and merges a FEM test input.

        Use one of the existing test input files (cantilever_Q4_4IP_PE_SP_AT2_SD_T100).
        Verify the returned dict has both parent and child keys merged.
        Verify schema validation passed (no exception raised).
        """
        # Parse the FEM test input - file is in input/fem/tests/
        input_dict = parse_input_json(
            "tests/cantilever_Q4_4IP_PE_SP_AT2_SD_T100", "fem"
        )

        # Verify parent keys are present (from parent config)
        assert "title_prefix" in input_dict
        assert "boundary_conditions" in input_dict

        # Verify child keys are present (from child config)
        assert "loop_parameters" in input_dict
        assert "output_parameters" in input_dict

        # Verify merged values
        assert input_dict["loop_parameters"]["timesteps"] == 100

    def test_get_input_file_extension_chain(self) -> None:
        """Test 1.12: The extends → overrides resolution chain in get_input_file_extension.

        Use an input file that extends a parent.
        Verify that the returned dict has parent values overridden only where specified.
        Verify a file with no extends key returns a plain copy.
        """
        # Test with an extending file - use parent/tests directory where the file exists
        extending_dict = {
            "extends": "cantilever_Q4_4IP_PE_SP_AT2_SD",
            "overrides": {
                "loop_parameters": {"timesteps": 200},
            },
        }

        result = get_input_file_extension(extending_dict, "parent/tests")

        # Verify parent values are present (title_prefix is in parent config)
        assert "title_prefix" in result

        # Verify override is applied
        assert result["loop_parameters"]["timesteps"] == 200

        # Test with no extends
        simple_dict = {"key": "value", "nested": {"a": 1}}
        result_simple = get_input_file_extension(simple_dict, "parent/tests")
        assert result_simple == simple_dict
        assert result_simple is not simple_dict  # Should be a copy

    def test_get_input_file_extension_no_extends(self) -> None:
        """Test that get_input_file_extension returns a copy when no extends key."""
        original = {"key": "value", "nested": {"a": 1}}
        result = get_input_file_extension(original, "fem/tests")
        assert result == original
        assert result is not original  # Should be a copy
        # Modify result and verify original is unchanged
        result["key"] = "modified"
        assert original["key"] == "value"


class TestFormatErrors:
    """Test error formatting."""

    def test_format_errors_readable(self) -> None:
        """Test 1.13: format_errors produces human-readable strings for each validator type.

        Create synthetic jsonschema errors for minItems, maximum, enum, and generic cases.
        Call format_errors and verify each produces the expected format string.
        """
        # Create synthetic errors
        errors = []

        # minItems error
        class MinItemsError:
            validator = "minItems"
            validator_value = 2
            instance = [1]
            absolute_path = ["array_key"]
            message = "minItems error"

        errors.append(MinItemsError())

        # maximum error
        class MaximumError:
            validator = "maximum"
            validator_value = 100
            instance = 150
            absolute_path = ["value"]
            message = "maximum error"

        errors.append(MaximumError())

        # enum error
        class EnumError:
            validator = "enum"
            validator_value = ["a", "b", "c"]
            instance = "d"
            absolute_path = []
            message = "enum error"

        errors.append(EnumError())

        # generic error
        class GenericError:
            validator = "type"
            validator_value = "string"
            instance = 123
            absolute_path = ["nested", "key"]
            message = "123 is not of type 'string'"

        errors.append(GenericError())

        # Format errors
        formatted = format_errors(errors)

        # Verify formatting
        assert len(formatted) == 4
        assert "array_key" in formatted[0]
        assert "at least 2" in formatted[0]
        assert "value" in formatted[1]
        assert "≤ 100" in formatted[1]
        assert "<root>" in formatted[2]
        assert "'d'" in formatted[2]
        assert "nested.key" in formatted[3]


class TestSchemaFunctions:
    """Test schema parsing and validation functions."""

    def test_parse_schema(self) -> None:
        """Test parse_schema loads schema files correctly."""
        # Parse fem schema
        fem_schema = parse_schema("fem")
        assert "type" in fem_schema
        assert fem_schema["type"] == "object"

        # Parse parent schema
        parent_schema = parse_schema("parent")
        assert "type" in parent_schema

    def test_validate_input(self) -> None:
        """Test validate_input returns errors for invalid input."""
        # Valid input
        valid_input = {"name": "test", "value": 42}
        valid_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
        }
        errors = validate_input(valid_input, valid_schema)
        assert errors == []

        # Invalid input
        invalid_input = {"name": 123, "value": "not a number"}
        errors = validate_input(invalid_input, valid_schema)
        assert len(errors) > 0

    def test_check_schema_conformance(self) -> None:
        """Test check_schema_conformance raises on invalid input."""
        # Valid input should not raise - use a minimal valid parent config
        # Note: extends/overrides are processed BEFORE schema validation,
        # so we provide a fully valid input that conforms to the parent schema
        valid_input = {
            "title_prefix": "test",
            "aggregate_timings": False,
            "boundary_conditions": {
                "fixed_window_parameters": [],
                "load_window_parameters": [],
            },
            "debug": False,
            "gamma_parameters": {
                "gamma11": 0.0,
                "gamma12": 0.0,
                "gamma22": 0.0,
                "gamma44": 0.0,
            },
            "generate_plots": [],  # Must be array, not boolean
            "material_parameters": {
                "youngs_modulus": 1.0,
                "poissons_ratio": 0.3,
                "thickness": 1.0,  # Required
                "cubic_anisotropy": {"shear_modulus": 0.0},
                "orthotropic_anisotropy": {
                    "E_11": 1.0,
                    "E_22": 1.0,
                    "E_33": 1.0,
                    "G_12": 0.5,
                    "G_23": 0.5,
                    "G_31": 0.5,
                    "nu_12": 0.3,
                    "nu_23": 0.3,
                    "nu_31": 0.3,
                },
            },
            "material_rotation": {
                "angle": 0.0,
                "slicing_direction": "x",
                "number_of_slices": 1,
            },
            "mesh": {
                "type": "quad4",
                "number_of_integration_points": 4,
                "filename": "test_mesh",
            },
            "nondimensionalise": False,
            "phasefield_parameters": {
                "l_0": 0.1,
                "G_c": 1.0,
                "initial_crack_parameters": [],
            },
            "plane": "strain",
            "postprocessing_stress_type": "",  # Must be empty string or hydrostatic/von_mises
            "problem_domain": {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
            },
            "problem_type": "at2",  # Must be one of the allowed values
            "recorded_values": [],
            "split_output_keys": [],
            "strain_split": "spectral",
            "time_functions": False,  # Must be boolean, not array
            "target_dof": {
                "coordinates": [1.0, 1.0],
                "direction": "y",
            },
            "profile_memory": False,
        }
        check_schema_conformance(valid_input, "test_input", "parent")

        # Invalid input should raise - use type mismatch
        invalid_input = {"title_prefix": 12345}  # title_prefix should be string
        with pytest.raises(jsonschema.ValidationError):
            check_schema_conformance(invalid_input, "test_input", "parent")
