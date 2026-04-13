"""Unit tests for Paraview output functions.

Tests the setup_paraview_output, write_paraview_step, and write_paraview_pvd
functions for various mesh types and configurations.
"""

import pathlib
import xml.etree.ElementTree as ET

import jax.numpy as jnp

from io_handlers import (
    setup_paraview_output,
    write_paraview_pvd,
    write_paraview_step,
)


class TestParaviewOutput:
    """Test Paraview output functions."""

    def test_paraview_output_all_mesh_types(
        self, make_quad4_mesh, make_tri3_mesh, make_quad8_mesh, make_tri6_mesh
    ) -> None:
        """Test 4.8: setup_paraview_output + write_paraview_step works for all supported VTK cell types.

        For each of quad4, quad8, quad9, tri3, tri6: create a small mock mesh,
        write a step, verify .vtu exists and can be read back by pyvista.
        """
        import pyvista as pv

        test_cases = [
            ("quad4", make_quad4_mesh, "quad4"),
            ("tri3", make_tri3_mesh, "tri3"),
            ("quad8", make_quad8_mesh, "quad8"),
            ("tri6", make_tri6_mesh, "tri6"),
        ]

        for test_name, mesh_fixture, mesh_type in test_cases:
            coords, connectivity, dofs = mesh_fixture
            n_nodes = coords.shape[0]

            # Create output directory in tmp_path
            output_dir = pathlib.Path(f"/tmp/test_paraview_{test_name}")
            output_dir.mkdir(exist_ok=True)

            # Setup Paraview output
            ctx = setup_paraview_output(
                file_name=f"test_{test_name}",
                coordinates=coords,
                connectivity=connectivity,
                mesh_type=mesh_type,
                subdir=str(output_dir),
            )

            # Create mock step data
            step_data = {
                "displacement": jnp.zeros((n_nodes, 2)),
                "phasefield": jnp.ones(n_nodes),
            }

            # Write a step
            write_paraview_step(ctx, step_data, increment_value=1.0)

            # Verify .vtu file exists (note: . is replaced with _ in filename)
            vtu_file = ctx.vtu_path / "1_00e+00.vtu"
            assert vtu_file.exists(), f"VTU file should exist for {test_name}"

            # Verify it can be read by pyvista
            grid = pv.read(str(vtu_file))
            assert grid.n_points == n_nodes, f"Should have {n_nodes} points"
            assert "displacement" in grid.point_data
            assert "phasefield" in grid.point_data

            # Clean up
            import shutil

            if output_dir.exists():
                shutil.rmtree(output_dir)

    def test_paraview_pvd_valid_xml(self, make_quad4_mesh, tmp_path) -> None:
        """Test 4.9: write_paraview_pvd generates well-formed XML.

        Write 3 steps, generate PVD, parse with xml.etree.ElementTree,
        verify 3 DataSet entries.
        """
        coords, connectivity, dofs = make_quad4_mesh
        n_nodes = coords.shape[0]

        # Setup Paraview output
        ctx = setup_paraview_output(
            file_name="test_pvd",
            coordinates=coords,
            connectivity=connectivity,
            mesh_type="quad4",
            subdir=str(tmp_path),
        )

        # Write 3 steps
        step_data = {
            "displacement": jnp.zeros((n_nodes, 2)),
            "phasefield": jnp.ones(n_nodes),
        }
        for i in range(3):
            write_paraview_step(ctx, step_data, increment_value=float(i + 1))

        # Write PVD file
        write_paraview_pvd(ctx)

        # Verify PVD file exists
        assert ctx.pvd_path.exists(), "PVD file should exist"

        # Parse with xml.etree.ElementTree
        tree = ET.parse(str(ctx.pvd_path))
        root = tree.getroot()

        # Verify structure
        assert root.tag == "VTKFile", "Root element should be VTKFile"
        collection = root.find("Collection")
        assert collection is not None, "Should have Collection element"

        # Verify 3 DataSet entries
        data_sets = collection.findall("DataSet")
        assert (
            len(data_sets) == 3
        ), f"Should have 3 DataSet entries, got {len(data_sets)}"

        # Verify timestep attributes
        for i, ds in enumerate(data_sets):
            assert ds.get("timestep") == str(i + 1), f"Timestep should be {i + 1}"

    def test_paraview_gauss_point_cloud(self, make_quad4_mesh, tmp_path) -> None:
        """Test 4.10: When ip_N is provided, write_paraview_step also writes _gp.vtp files.

        Setup with ip_N, write a step with ip_ prefixed keys, verify _gp.vtp exists.
        """
        import pyvista as pv

        coords, connectivity, dofs = make_quad4_mesh
        n_nodes = coords.shape[0]
        n_elems = connectivity.shape[0]
        n_gauss = 4

        # Create shape functions at Gauss points (simple averaging)
        ip_N = jnp.ones((n_elems, n_gauss, 4)) / n_gauss

        # Setup Paraview output with ip_N
        ctx = setup_paraview_output(
            file_name="test_gp",
            coordinates=coords,
            connectivity=connectivity,
            mesh_type="quad4",
            ip_N=ip_N,
            subdir=str(tmp_path),
        )

        # Verify gp_cloud was created
        assert (
            ctx._gp_cloud is not None
        ), "GP cloud should be created when ip_N is provided"
        assert ctx.n_gauss == n_gauss, f"n_gauss should be {n_gauss}"

        # Create step data with ip_ prefixed keys
        step_data = {
            "displacement": jnp.zeros((n_nodes, 2)),
            "ip_stress": jnp.ones((n_elems, n_gauss)),  # IP values
            "ip_stress_nodes": jnp.ones(n_nodes),  # Nodal values
        }

        # Write a step
        write_paraview_step(ctx, step_data, increment_value=1.0)

        # Verify _gp.vtp file exists (note: . is replaced with _ in filename)
        gp_file = ctx.vtu_path / "1_00e+00_gp.vtp"
        assert gp_file.exists(), "GP VTP file should exist when ip_N is provided"

        # Verify it can be read by pyvista
        gp_cloud = pv.read(str(gp_file))
        assert gp_cloud.n_points == n_elems * n_gauss, "Should have E*G points"
        assert "ip_stress" in gp_cloud.point_data, "ip_stress should be in GP cloud"

    def test_paraview_cleanup_on_rerun(self, make_quad4_mesh, tmp_path) -> None:
        """Test 4.11: setup_paraview_output cleans up previous run's files before writing new ones.

        Write steps, verify files exist, rerun setup, verify old files are gone.
        """
        coords, connectivity, dofs = make_quad4_mesh
        n_nodes = coords.shape[0]

        # First run: write 3 steps
        ctx1 = setup_paraview_output(
            file_name="test_cleanup",
            coordinates=coords,
            connectivity=connectivity,
            mesh_type="quad4",
            subdir=str(tmp_path),
        )

        step_data = {
            "displacement": jnp.zeros((n_nodes, 2)),
        }
        for i in range(3):
            write_paraview_step(ctx1, step_data, increment_value=float(i + 1))
        write_paraview_pvd(ctx1)

        # Verify files exist
        vtu_files = list(ctx1.vtu_path.glob("*.vtu"))
        assert len(vtu_files) == 3, "Should have 3 VTU files"
        assert ctx1.pvd_path.exists(), "PVD file should exist"

        # Second run: setup with same file_name
        ctx2 = setup_paraview_output(
            file_name="test_cleanup",
            coordinates=coords,
            connectivity=connectivity,
            mesh_type="quad4",
            subdir=str(tmp_path),
        )

        # Verify old files are cleaned up
        vtu_files = list(ctx2.vtu_path.glob("*.vtu"))
        assert len(vtu_files) == 0, "Old VTU files should be cleaned up"
        assert not ctx2.pvd_path.exists(), "Old PVD file should be cleaned up"

        # Write 2 new steps
        for i in range(2):
            write_paraview_step(ctx2, step_data, increment_value=float(i + 10))
        write_paraview_pvd(ctx2)

        # Verify only new files exist
        vtu_files = list(ctx2.vtu_path.glob("*.vtu"))
        assert len(vtu_files) == 2, "Should have 2 new VTU files"
