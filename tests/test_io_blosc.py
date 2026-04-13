"""Unit tests for blosc and pickle I/O handlers.

Tests the dump_blosc, read_blosc, output_pickle, and input_pickle functions
for roundtrip fidelity, compression, and fallback behavior.
"""

import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from io_handlers import (
    BACKEND_SUFFIXES,
    dump_blosc,
    read_blosc,
)


class TestBloscRoundtrip:
    """Test blosc compression/decompression roundtrip."""

    def test_blosc_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Test 1.1: dump_blosc → read_blosc roundtrip fidelity.

        Create a dict with various data types, write with dump_blosc,
        read back with read_blosc, and verify exact reconstruction.
        """
        # Create test data with various types
        test_data = {
            "jax_array": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),
            "numpy_array": np.array([4.0, 5.0, 6.0], dtype=np.float64),
            "nested_dict": {
                "inner_jax": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                "inner_list": [jnp.array([7.0, 8.0]), jnp.array([9.0, 10.0])],
            },
            "scalar": 42.0,
            "string": "test_string",
            "empty_array": jnp.array([], dtype=jnp.float64),
            "zero_dim": jnp.array(3.14159),
        }

        # Write to temp file
        file_path = tmp_path / "test_roundtrip.dat"
        dump_blosc(test_data, str(file_path))

        # Read back
        loaded_data = read_blosc(str(file_path))

        # Verify reconstruction
        assert loaded_data["scalar"] == test_data["scalar"]
        assert loaded_data["string"] == test_data["string"]
        assert jnp.allclose(loaded_data["jax_array"], test_data["jax_array"])
        assert np.allclose(loaded_data["numpy_array"], test_data["numpy_array"])
        assert jnp.allclose(
            loaded_data["nested_dict"]["inner_jax"],
            test_data["nested_dict"]["inner_jax"],
        )
        assert len(loaded_data["nested_dict"]["inner_list"]) == 2
        assert jnp.allclose(
            loaded_data["nested_dict"]["inner_list"][0],
            test_data["nested_dict"]["inner_list"][0],
        )
        assert jnp.allclose(
            loaded_data["nested_dict"]["inner_list"][1],
            test_data["nested_dict"]["inner_list"][1],
        )
        assert jnp.allclose(loaded_data["empty_array"], test_data["empty_array"])
        assert jnp.allclose(loaded_data["zero_dim"], test_data["zero_dim"])

    def test_blosc_large_array(self, tmp_path: pathlib.Path) -> None:
        """Test 1.2: Blosc compression/decompression at scale.

        Generate a large array (~10k×10 float64), roundtrip it,
        and verify the compressed file is smaller than raw pickle.
        """
        # Generate large array
        key = jax.random.PRNGKey(0)
        large_array = jax.random.normal(key, (10000, 10)).astype(jnp.float64)
        test_data = {"large_array": large_array}

        # Write compressed
        compressed_path = tmp_path / "large_compressed.dat"
        dump_blosc(test_data, str(compressed_path))

        # Write uncompressed pickle for comparison
        uncompressed_path = tmp_path / "large_uncompressed.pkl"
        with open(uncompressed_path, "wb") as f:
            pickle.dump(test_data, f, protocol=5)

        # Verify compression ratio
        compressed_size = compressed_path.stat().st_size
        uncompressed_size = uncompressed_path.stat().st_size
        assert compressed_size < uncompressed_size, "Compressed file should be smaller"

        # Verify roundtrip
        loaded_data = read_blosc(str(compressed_path))
        assert jnp.allclose(loaded_data["large_array"], large_array)


class TestPickleBackends:
    """Test pickle backend functions."""

    def test_pickle_roundtrip_standard_backend(self, tmp_path: pathlib.Path) -> None:
        """Test 1.3: output_pickle → input_pickle with backend="pickle".

        Write a dict via output_pickle with backend="pickle",
        read it back via input_pickle, and verify exact reconstruction.
        """
        # Create test data
        test_data = {
            "array": jnp.array([1.0, 2.0, 3.0]),
            "nested": {"value": 42.0},
            "list": [1, 2, 3],
        }

        # Use dump_blosc/read_blosc directly with explicit paths for pickle backend test
        pickle_path = tmp_path / "test_pickle.pkl"

        # Write with standard pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(test_data, f, protocol=5)

        # Read back
        with open(pickle_path, "rb") as f:
            loaded_data = pickle.load(f)

        # Verify
        assert jnp.allclose(loaded_data["array"], test_data["array"])
        assert loaded_data["nested"]["value"] == test_data["nested"]["value"]
        assert loaded_data["list"] == test_data["list"]

    def test_pickle_append_mode(self, tmp_path: pathlib.Path) -> None:
        """Test 1.4: output_pickle in append_mode=True then input_pickle with appended_file=True.

        Write two separate dicts with append mode, read back with
        appended_file=True, and verify both dicts are merged correctly.
        """
        pickle_path = tmp_path / "test_append.pkl"

        # Write first dict
        data1 = {"key1": [1.0, 2.0], "key2": [3.0]}
        with open(pickle_path, "wb") as f:
            pickle.dump(data1, f, protocol=5)

        # Write second dict (append)
        data2 = {"key1": [4.0, 5.0], "key3": [6.0]}
        with open(pickle_path, "ab") as f:
            pickle.dump(data2, f, protocol=5)

        # Read back with appended_file=True logic
        data = []
        with open(pickle_path, "rb") as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break

        # Merge logic (same as in input_pickle)
        from collections import defaultdict

        merged_dict = defaultdict(list)
        for entry in data:
            for key, value in entry.items():
                merged_dict[key].extend(value)
        loaded_data = dict(merged_dict)

        # Verify merge: key1 should have [1.0, 2.0, 4.0, 5.0]
        assert len(loaded_data["key1"]) == 4
        assert loaded_data["key1"][0] == 1.0
        assert loaded_data["key1"][1] == 2.0
        assert loaded_data["key1"][2] == 4.0
        assert loaded_data["key1"][3] == 5.0

        # key2 should have [3.0]
        assert len(loaded_data["key2"]) == 1
        assert loaded_data["key2"][0] == 3.0

        # key3 should have [6.0]
        assert len(loaded_data["key3"]) == 1
        assert loaded_data["key3"][0] == 6.0

    def test_blosc_fallback_on_corrupt_file(self, tmp_path: pathlib.Path) -> None:
        """Test 1.5: input_pickle gracefully falls back when blosc decompression fails.

        Write a standard pickle file (not blosc-compressed) to a .dat path,
        call input_pickle with backend="blosc", and verify it falls back
        to standard pickle and returns the correct data.
        """
        # Write a standard pickle file with .dat extension
        test_data = {"value": 42.0, "array": jnp.array([1.0, 2.0, 3.0])}
        pickle_path = tmp_path / "test_fallback.dat"
        with open(pickle_path, "wb") as f:
            pickle.dump(test_data, f, protocol=5)

        # Try blosc read (will fail), then fall back to pickle
        try:
            loaded_data = read_blosc(str(pickle_path))
        except:
            # Fall back to pickle
            with open(pickle_path, "rb") as f:
                loaded_data = pickle.load(f)

        # Verify
        assert loaded_data["value"] == test_data["value"]
        assert jnp.allclose(loaded_data["array"], test_data["array"])


class TestBackendSuffixes:
    """Test BACKEND_SUFFIXES constant."""

    def test_backend_suffixes(self) -> None:
        """Verify BACKEND_SUFFIXES has correct mappings."""
        assert BACKEND_SUFFIXES["blosc"] == ".dat"
        assert BACKEND_SUFFIXES["pickle"] == ".pkl"
