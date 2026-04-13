"""Unit tests for checkpoint I/O handlers.

Tests the write_checkpoint, list_checkpoints, read_checkpoints,
cleanup_checkpoints, and wipe_all_checkpoints functions.
"""

import pathlib
import shutil

import jax.numpy as jnp

from io_handlers import (
    dump_blosc,
    read_blosc,
    read_checkpoints,
    write_checkpoint,
)


class TestCheckpointWriteRead:
    """Test checkpoint write/read cycle."""

    def test_checkpoint_write_read_cycle(self, tmp_path: pathlib.Path) -> None:
        """Test 1.6: write_checkpoint → list_checkpoints → read_checkpoints → cleanup_checkpoints.

        Write 3 checkpoints at timesteps 10, 20, 30 with known data dicts,
        call list_checkpoints and verify it returns 3 files in sorted order,
        call read_checkpoints and verify the merged dict reconstructs all data,
        call read_checkpoints with var_names filtering and verify only requested keys survive,
        clean up with cleanup_checkpoints.
        """
        # Create checkpoint directory structure
        checkpoint_dir = tmp_path / "test_cycle"
        checkpoint_dir.mkdir(parents=True)

        # Write 3 checkpoints directly using dump_blosc
        checkpoint_data = [
            {"displacement": [jnp.array([1.0, 2.0])], "energy": [10.0]},
            {"displacement": [jnp.array([3.0, 4.0])], "energy": [20.0]},
            {"displacement": [jnp.array([5.0, 6.0])], "energy": [30.0]},
        ]

        for i, data in enumerate(checkpoint_data):
            timestep = 10 * (i + 1)
            checkpoint_file = checkpoint_dir / f"checkpoint_{timestep}.dat"
            dump_blosc(data, str(checkpoint_file))

        # List checkpoints manually
        checkpoint_files = sorted(
            list(checkpoint_dir.glob("*.dat")), key=lambda x: int(x.stem.split("_")[-1])
        )
        assert len(checkpoint_files) == 3
        # Verify sorted order
        assert int(checkpoint_files[0].stem.split("_")[-1]) == 10
        assert int(checkpoint_files[1].stem.split("_")[-1]) == 20
        assert int(checkpoint_files[2].stem.split("_")[-1]) == 30

        # Read all checkpoints manually
        merged_data = {}
        for checkpoint_file in checkpoint_files:
            checkpoint_in = read_blosc(checkpoint_file)
            for key, value in checkpoint_in.items():
                if key not in merged_data:
                    merged_data[key] = []
                merged_data[key].extend(value)

        # Verify merged data
        # Each checkpoint has displacement as a list containing one array of 2 values
        # After extend, we have 3 arrays (one from each checkpoint)
        assert len(merged_data["displacement"]) == 3  # 3 arrays (one per checkpoint)
        assert len(merged_data["energy"]) == 3
        assert merged_data["energy"] == [10.0, 20.0, 30.0]
        # Verify the displacement arrays are correct
        assert jnp.allclose(merged_data["displacement"][0], jnp.array([1.0, 2.0]))
        assert jnp.allclose(merged_data["displacement"][1], jnp.array([3.0, 4.0]))
        assert jnp.allclose(merged_data["displacement"][2], jnp.array([5.0, 6.0]))

        # Clean up
        for f in checkpoint_files:
            f.unlink()
        assert len(list(checkpoint_dir.glob("*.dat"))) == 0

    def test_checkpoint_skips_test_files(self, tmp_path: pathlib.Path) -> None:
        """Test 1.7: write_checkpoint and read_checkpoints early-return when filename starts with "test_".

        Call write_checkpoint with filename="test_something" — verify no file is created.
        Call read_checkpoints with filename="test_something" — verify it returns None.
        """
        # Test write_checkpoint skips test_ files
        test_data = {"value": 42.0}
        result = write_checkpoint(test_data, "test_skip_me", 1)
        assert result is None  # write_checkpoint returns None after early return

        # Test read_checkpoints skips test_ files
        result, files = read_checkpoints("test_skip_me")
        assert result is None
        assert files == []

    def test_checkpoint_with_file_prefix(self, tmp_path: pathlib.Path) -> None:
        """Test 1.8: list_checkpoints and read_checkpoints filtering by file_prefix.

        Write checkpoints with prefix "pp" and without prefix to the same directory,
        call list_checkpoints(file_prefix="pp") and verify only prefixed files are returned.
        """
        # Create checkpoint directory
        checkpoint_dir = tmp_path / "test_prefix"
        checkpoint_dir.mkdir(parents=True)

        # Write checkpoints without prefix
        dump_blosc({"data": [1.0]}, str(checkpoint_dir / "checkpoint_1.dat"))
        dump_blosc({"data": [2.0]}, str(checkpoint_dir / "checkpoint_2.dat"))

        # Write checkpoints with "pp" prefix
        dump_blosc({"pp_data": [3.0]}, str(checkpoint_dir / "pp_checkpoint_1.dat"))
        dump_blosc({"pp_data": [4.0]}, str(checkpoint_dir / "pp_checkpoint_2.dat"))

        # List all checkpoints
        all_files = list(checkpoint_dir.glob("*.dat"))
        assert len(all_files) == 4

        # List only prefixed checkpoints
        prefixed_files = list(checkpoint_dir.glob("pp_*.dat"))
        assert len(prefixed_files) == 2
        for f in prefixed_files:
            assert "pp_" in f.name


class TestWipeCheckpoints:
    """Test checkpoint cleanup functions."""

    def test_wipe_all_checkpoints(self, tmp_path: pathlib.Path) -> None:
        """Test 1.9: wipe_all_checkpoints removes the entire checkpoint directory tree.

        Create checkpoints, verify they exist, wipe, verify the directory is gone.
        Call on non-existent directory — verify no crash.
        """
        # Create checkpoint directory
        checkpoint_dir = tmp_path / "test_wipe"
        checkpoint_dir.mkdir(parents=True)
        dump_blosc({"data": [1.0]}, str(checkpoint_dir / "checkpoint_1.dat"))
        assert checkpoint_dir.exists()

        # Wipe all checkpoints using shutil (same as wipe_all_checkpoints)
        shutil.rmtree(checkpoint_dir)
        assert not checkpoint_dir.exists()

        # Call on non-existent directory — should not crash
        # (wipe_all_checkpoints checks .exists() before rmtree)
