"""Host the abstract simulation class and specific subclasses.

This module provides the base Simulation class and its concrete
implementations for finite element method and Deep Ritz Method
simulations.
"""

import logging

logging.getLogger("jax").setLevel(logging.ERROR)

import pathlib
import sys
from typing import Dict, Optional, Tuple

current_path = pathlib.Path(__file__).parent.resolve()
nn_source_path = current_path.parent / "nn" / "src"
fem_source_path = current_path.parent / "fem" / "src"
sys.path.append(str(nn_source_path))
sys.path.append(str(fem_source_path))
import copy
import os
import tempfile
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from time import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from mlflow_tracker import DiagnosticsPayload, MLflowTracker
from models_nn import get_model_nn
from nodal_computations import (
    hyplas_residual,
    pointwise_D_to_global_stiffness_coeffs,
    pointwise_D_to_global_stiffness_sparse_matrix,
    pointwise_internal_forces_to_global,
    pre_newton_raphson_load,
    solve_displacement_disp_control_initial,
    solve_displacement_disp_control_subsequent,
    solve_displacement_load_control,
    solve_phasefield,
)
from optimisers import get_optimiser
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from setup_fem import (
    set_up_body_F,
    set_up_concentrated_F,
    set_up_disp_factor,
    set_up_lagrange_parameters,
    set_up_load_factor,
    set_up_load_window_lagrange,
    set_up_load_window_penalty,
    set_up_window_boundary,
    set_up_window_dofs,
    set_up_window_F,
)
from setup_nn import construct_displacement, construct_network, construct_pf_constraint
from tqdm import tqdm
from train_networks import EarlyStopping, training_step
from utils_fem import (
    cleanup_solver,
    reorder_qoi_dict,
    set_up_sparse_solver,
)
from utils_nn import (
    calculate_weight_sparsity,
    collect_auxiliary_data,
    postprocess_training_snapshots,
    predict_model_output,
    reset_activation_coefficients,
)

from data_handling import MaterialParameters, ProblemParameters
from io_handlers import (
    cleanup_checkpoints,
    dump_blosc,
    list_checkpoints,
    output_paraview,
    output_paraview_gif,
    output_pickle,
    parse_input_json,
    read_blosc,
    read_checkpoints,
    setup_paraview_output,
    wipe_all_checkpoints,
    write_checkpoint,
    write_paraview_pvd,
    write_paraview_step,
)
from log import log_separator, setup_logging
from memory import log_mem
from phase_field_models import get_phasefield_model
from plots_streaming import _SCALE_RULES, StreamingPlotManager
from plots_utils import run_plot_jobs
from plots_wrapper import dump_plot_data
from setup_wrappers import setup_geometry, setup_initial_phasefield
from strain_energy_models import get_energy_model
from strain_models import nodal_displacement_to_strain, voigt_strain_to_tensor
from utils import (
    StrictDict,
    handle_errors_class,
    matrix_to_voigt,
    output_timing_table,
    postprocess_qoi_dict,
    prune_qoi_dict,
    raise_osx_notification,
    rescale_qoi_dict,
    set_aggregate_timings,
    set_enable_timing,
    set_up_target_dof,
    timing_aggregator,
)

warnings.filterwarnings("ignore", message="A JAX array is being set as static")
jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=6, linewidth=800, suppress=False)


@handle_errors_class
class Simulation(ABC):
    """Abstract base class for simulations.

    Provides common initialization and interface for both FEM and
    DRM simulation implementations.

    Parameters
    ----------
    title : str
        Simulation title used for output file naming.
    config : Dict[str, object] or None, optional
        Configuration dictionary. If None, loaded from JSON file.
    """

    run_type: str

    def __init__(self, title: str, config: Optional[Dict[str, object]] = None) -> None:
        """Initialize the simulation with title and configuration."""
        # Parse the input JSON file
        self.start_time = time()
        self.title = f"{self.run_type}_{title}"
        if config is None:
            self.config = parse_input_json(title, self.run_type)
            # manual seed override
            seed_value = os.environ.get("SEED")
            if seed_value is None:
                seed_value = os.environ.get("SLURM_ARRAY_TASK_ID")
            if seed_value is not None:
                self.config["seed"] = int(seed_value)
            self.config = StrictDict(self.config)
        else:
            self.config = StrictDict(config)
        self._setup_generic()
        self._setup_specific()
        self._setup_output()
        # Get the online_output flag
        if hasattr(self, "online_output"):
            logging.info(f"Output streaming into checkpoints: {self.online_output}")
        logging.info(f"Set up time: {(time() - self.start_time):.6f}")
        logging.info(log_separator)

    @abstractmethod
    def run(self) -> None:
        """Execute the simulation."""

    def _setup_generic(
        self,
    ) -> None:
        """Set up generic simulation parameters common to all simulations."""
        # Timestamp the title if required
        self.title += (
            datetime.now().strftime("_%Y_%m_%dT%H_%M")
            if self.config.output_parameters.timestamp
            else ""
        )

        # Set up timing options
        set_enable_timing(self.config.time_functions)
        set_aggregate_timings(self.config.aggregate_timings)

        # Set up logging
        skip_log = getattr(self, "_no_log", None)
        if not skip_log:
            setup_logging(title=self.title, config=self.config, debug=self.config.debug)

        # Extract silent run flag from command line arguments
        self.silent = True if "nobar" in sys.argv else False

        # Setup the geometry files
        self.mesh_data, self.ip_data, self.distance_functions = setup_geometry(
            self.config, dataclasses_frozen=False
        )
        logging.info(
            f"Found {self.mesh_data.nodal_coordinates.shape[0]} nodes "
            f"and {self.mesh_data.connectivities.shape[0]} elements."
        )

        # Locate the target degree of freedom
        self.target_dof = set_up_target_dof(self.mesh_data, self.config.target_dof)
        logging.info(f"Target DOF located at index {self.target_dof}.")

        # Set up the material and problem parameters
        self.material_parameters = MaterialParameters.from_dict(self.config)
        self.problem_parameters = ProblemParameters.from_dict(self.config)
        logging.info("Finished creating dataclasses.")

        # Set up the strain energy and phase field models
        self.strain_energy_model = get_energy_model(
            name=self.problem_parameters.strain_split,
            E=self.material_parameters.E,
            nu=self.material_parameters.nu,
            plane_mode=self.problem_parameters.plane_mode,
            shear_modulus=self.material_parameters.cubic_anisotropy_G,
            **self.material_parameters.orthotropic_anisotropy_params,
        )

        # Set up the initial constitutive matrices
        self.ip_D = self.strain_energy_model.initialise_constitutive_matrices(
            self.ip_data.volumes, element_rotations=self.ip_data.rotation_angles
        )
        self.ip_D_undegraded = self.ip_D

        # Set up the phase field model
        self.phasefield_model = get_phasefield_model(
            name=self.problem_parameters.problem_type,
            G_c=self.material_parameters.G_c,
            l_0=self.material_parameters.l_0,
        )
        logging.info(
            f"Finished setting up strain energy ({self.problem_parameters.strain_split}) "
            f"and phase field ({self.problem_parameters.problem_type}) models."
        )

        if not self.problem_parameters.problem_type == "linear_elasticity":
            # Set up initial phasefield variables
            self._initial_phasefield_tuple = setup_initial_phasefield(
                pf_params=self.config.phasefield_parameters,
                mesh_data=self.mesh_data,
                ip_data=self.ip_data,
                material_parameters=self.material_parameters,
                phasefield_model=self.phasefield_model,
            )
            logging.info("Initial phasefield values set up completed.")

        # Extract the tracked variables
        self.tracked_variables = self.config.recorded_values

        # Get the rendering flag
        self.render_output = bool(os.environ.get("RENDER_OUTPUT", False))
        logging.info(f"Render output flag set to {self.render_output}.")

    def _setup_output(
        self,
    ) -> None:
        """Set up paraview and plotting context managers for streaming output."""
        self._setup_streaming_plots()
        self._setup_paraview_context()

    def _setup_streaming_plots(
        self,
    ) -> None:
        """Initialize the streaming plot manager for active stream mode."""
        self.streaming_plot_manager = None
        if not self.online_output:
            return
        self.streaming_plot_manager = StreamingPlotManager(
            jobs=self.config.generate_plots,
            target_dof=self.target_dof,
            title=self.title,
        )

    def _setup_paraview_context(
        self,
    ) -> None:
        """Initialize ParaviewContext for per-step .vtu writing.

        Only active when save_pvd and online_output are both enabled.
        """
        self.paraview_ctx = None
        if not self.config.output_parameters.save_pvd:
            return
        if not self.online_output:
            return
        if hasattr(self, "perm_1d"):
            coordinates = self.mesh_data.nodal_coordinates[self.inv_perm_1d]
            connectivity = self.perm_1d[self.mesh_data.connectivities]
        else:
            coordinates = self.mesh_data.nodal_coordinates
            connectivity = self.mesh_data.connectivities

        resume = getattr(self, "_resumed_from_checkpoint", False) or hasattr(
            self, "_resume_step"
        )
        self.paraview_ctx = setup_paraview_output(
            file_name=self.title,
            coordinates=coordinates,
            connectivity=connectivity,
            mesh_type=self.config.mesh.type,
            ip_N=self.ip_data.N,
            aux_nurbs=self.mesh_data.aux_nurbs,
            problem_dict=self.config,
            excluded_keys=["incremental_displacements"],
            resume=resume,
        )
        logging.info("ParaviewContext initialised for per-step .vtu writing.")

    @abstractmethod
    def _setup_specific(
        self,
    ) -> None:
        """Set up simulation-specific parameters."""

    def postprocess(
        self,
    ) -> None:
        """Postprocess the simulation results."""
        self._postprocess_specific()
        assert (
            hasattr(self, "qoi_dict") or self.online_output
        ), "No qoi_dict found for postprocessing."
        assert hasattr(
            self, "paraview_increments"
        ), "No paraview increments found for postprocessing."
        self._postprocess_generic()

    def _postprocess_generic(
        self,
    ) -> None:
        """Perform generic postprocessing of the simulation results."""
        assert hasattr(self, "online_output"), "online_output is not detected!"
        self._rescale_qois()
        self._output_pickle()
        self._generate_plots()
        self._output_paraview()
        self._output_paraview_gif()

    def _rescale_qois(
        self,
    ) -> None:
        """Rescale quantities of interest back to dimensional units."""
        if not self.online_output:
            rescale_qoi_dict(
                qoi_dict=self.qoi_dict,
                displacement_scaling=self.material_parameters.displacement_scaling,
                energy_scaling=self.material_parameters.energy_scaling,
                force_scaling=self.material_parameters.force_scaling,
            )
        else:
            files = list_checkpoints(filename=self.title, file_prefix="pp")
            for entry in files:
                checkpoint_data = read_blosc(entry)
                rescale_qoi_dict(
                    qoi_dict=checkpoint_data,
                    displacement_scaling=self.material_parameters.displacement_scaling,
                    energy_scaling=self.material_parameters.energy_scaling,
                    force_scaling=self.material_parameters.force_scaling,
                )
                dump_blosc(checkpoint_data, entry)

        if hasattr(self, "incremental_displacements"):
            self.incremental_displacements /= (
                self.material_parameters.displacement_scaling
            )

    def _rescale_step(self, step_dict: Dict[str, Array]) -> Dict[str, Array]:
        """Rescale a flat dict from nondimensional to dimensional units.

        Parameters
        ----------
        step_dict : Dict[str, Array]
            Dictionary mapping variable names to arrays.

        Returns
        -------
        Dict[str, Array]
            New dictionary with rescaled values; input is not mutated.
        """
        inv_energy = 1.0 / self.material_parameters.energy_scaling
        inv_force = 1.0 / self.material_parameters.force_scaling
        inv_displacement = 1.0 / self.material_parameters.displacement_scaling

        out = {}
        for key, arr in step_dict.items():
            scale = None
            for substring, kind in _SCALE_RULES:
                if substring in key:
                    scale = (
                        inv_energy
                        if kind == "energy"
                        else inv_force if kind == "force" else inv_displacement
                    )
                    break
            if scale is None:
                out[key] = arr
            elif isinstance(arr, np.ndarray) and arr.flags.writeable:
                np.multiply(arr, scale, out=arr)
                out[key] = arr
            else:
                out[key] = arr * scale
        return out

    def _postprocess_and_output_step(
        self,
        step_dict: Dict[str, Array],
        increment_value: float,
    ) -> None:
        """Execute the shared per-step output pipeline for stream mode.

        Performs postprocessing, appends to pickle, and writes .vtu file.

        Parameters
        ----------
        step_dict : Dict[str, Array]
            Flat dictionary mapping variable names to arrays (rescaled).
        increment_value : float
            Increment value for the current step.
        """
        wrapped = {k: [v] for k, v in step_dict.items()}

        if self.config.output_parameters.save_pvd:
            # Use unpermuted connectivity/extrapolations since _reorder_step has
            # already put data arrays in original space
            if hasattr(self, "perm_1d"):
                connectivities = self.perm_1d[self.mesh_data.connectivities]
                extrapolations = self.ip_data.extrapolations
            else:
                connectivities = self.mesh_data.connectivities
                extrapolations = self.ip_data.extrapolations

            processed = postprocess_qoi_dict(
                qoi_dict=wrapped,
                connectivities=connectivities,
                conns_size=self.mesh_data.nodal_coordinates.shape[0],
                extrapolation_matrices=extrapolations,
                keys_to_split=self.config.split_output_keys,
                sanitise_keys=True,
            )
        else:
            processed = wrapped

        # Append to pickle
        # FEM check
        has_attr_and_true = (
            hasattr(self.config.output_parameters, "output_pickle")
            and self.config.output_parameters.output_pickle
        )
        # NN check - always output
        doesnt_have_attr = not hasattr(self.config.output_parameters, "output_pickle")
        if has_attr_and_true or doesnt_have_attr:
            output_pickle(target=wrapped, filename=self.title, append_mode=True)

        # Write .vtu
        if self.paraview_ctx is not None:
            flat_processed = {k: v[0] for k, v in processed.items()}
            write_paraview_step(self.paraview_ctx, flat_processed, increment_value)

    def _get_increment_value(self, step: int) -> float:
        """Return the dimensional increment value for the given step.

        Parameters
        ----------
        step : int
            Step index.

        Returns
        -------
        float
            Dimensional increment value for Paraview filename labeling.
        """
        match self.config.solution_parameters.mode:
            case "displacement control":
                return float(
                    self.incremental_displacements[
                        len(self.paraview_ctx.written_increments) + 1
                    ]
                    / self.material_parameters.displacement_scaling
                )
            case "load control":
                return float(
                    self.load_factor[len(self.paraview_ctx.written_increments) + 1]
                )
            case _:
                return float(step)

    def _generate_plots(
        self,
    ) -> None:
        """Generate plots for the simulation results."""
        if self.online_output:
            logging.info("Plot generation performed incrementally already.")
            return
        logging.info("Generating plots...")
        run_plot_jobs(self.title, self.target_dof, self.config.generate_plots)
        logging.info("Dumping plotting data...")
        dump_plot_data(title=self.title)
        logging.info("Data plotted and plotting data dumped successfully.")

    def _output_pickle(
        self,
    ) -> None:
        """Output the quantities of interest dictionary as a pickle file."""
        has_attr_and_true = (
            hasattr(self.config.output_parameters, "output_pickle")
            and self.config.output_parameters.output_pickle
        )
        doesnt_have_attr = not hasattr(self.config.output_parameters, "output_pickle")
        # Explicitly check if attribute is set or output if attribute doesn't exist
        if has_attr_and_true or doesnt_have_attr:
            logging.info("Outputting pickle file...")
            if self.online_output:
                files = list_checkpoints(filename=self.title, file_prefix="pp")
                for entry in files:
                    checkpoint_data = read_blosc(entry)
                    output_pickle(
                        target=checkpoint_data, filename=self.title, append_mode=True
                    )
            else:
                output_pickle(target=self.qoi_dict, filename=self.title)
            logging.info("Pickle file output complete.")

    def _output_paraview(
        self,
    ) -> None:
        """Output the simulation results in Paraview PVD format."""
        if not self.config.output_parameters.save_pvd:
            return
        logging.info("Saving the results to PVD...")
        if self.online_output:
            # .vtu files already written per-step — just write the .pvd index
            if self.paraview_ctx is not None:
                write_paraview_pvd(self.paraview_ctx)
            logging.info("PVD index written.")
            return
        # Non-stream: full offline batch path
        self.qoi_dict = postprocess_qoi_dict(
            qoi_dict=self.qoi_dict,
            connectivities=self.mesh_data.connectivities,
            conns_size=self.mesh_data.nodal_coordinates.shape[0],
            extrapolation_matrices=self.ip_data.extrapolations,
            keys_to_split=self.config.split_output_keys,
            sanitise_keys=True,
        )
        _ = output_paraview(
            file_name=self.title,
            increment_list=self.paraview_increments,
            qoi_dict=self.qoi_dict,
            coordinates=self.mesh_data.nodal_coordinates,
            connectivity=self.mesh_data.connectivities,
            ip_N=self.ip_data.N,
            mesh_type=self.config.mesh.type,
            problem_dict=self.config,
            aux_nurbs=self.mesh_data.aux_nurbs,
        )
        logging.info("PVD saved.")

    def _output_paraview_gif(
        self,
    ) -> None:
        """Render and save a GIF animation of the Paraview results."""
        # Construct the rendering flag
        render_gif = (
            self.config.output_parameters.save_pvd
            and self.config.output_parameters.pvd_gif_parameters.enabled
            and self.render_output
        )
        # If not rendering, return
        if not render_gif:
            return

        logging.info("Rendering Paraview GIF...")
        output_paraview_gif(
            filename=self.title,
            axis_limits=self.config.output_parameters.pvd_gif_parameters.axis_limits,
            target_fields=self.config.output_parameters.pvd_gif_parameters.target_fields,
            distortion_scaling=self.config.output_parameters.pvd_gif_parameters.distortion_scaling,
            mesh_outline=self.config.output_parameters.pvd_gif_parameters.mesh_outline,
            fps=self.config.output_parameters.pvd_gif_parameters.fps,
            include_every_n=self.config.output_parameters.pvd_gif_parameters.include_every_n,
        )
        logging.info("Paraview GIF rendering complete.")

    @abstractmethod
    def _postprocess_specific(
        self,
    ) -> None:
        """Perform simulation-specific postprocessing."""

    def _track_qois(
        self,
    ) -> None:
        """Track quantities of interest at the current timestep."""
        # Ensure the qoi_dict exists
        if not hasattr(self, "qoi_dict"):
            self.qoi_dict = {name: [] for name in self.tracked_variables}

        # Ensure checkpoint keys are tracked
        if hasattr(self, "checkpoint_keys"):
            for key in self.checkpoint_keys:
                if key not in self.tracked_variables:
                    self.tracked_variables.append(key)

        # Append the variables to the qoi_dict
        for name in self.tracked_variables:
            if hasattr(self, name):
                self.qoi_dict.setdefault(name, []).append(getattr(self, name))

    def _write_checkpoint(self, step: int, file_prefix: Optional[str] = None) -> None:
        """Write a checkpoint of the current simulation state.

        Parameters
        ----------
        step : int
            Current step/timestep identifier.
        file_prefix : str or None, optional
            Optional prefix for the checkpoint filename.
        """
        write_checkpoint(
            qoi_dict=self.qoi_dict,
            filename=self.title,
            timestep=step,
            reset_qoi_dict=True,
            file_prefix=file_prefix or "",
        )

    def finalise(
        self,
    ) -> None:
        """Finalize the simulation and clean up resources."""
        if (
            hasattr(self, "streaming_plot_manager")
            and self.streaming_plot_manager is not None
        ):
            self.streaming_plot_manager.close()
        # Free up any remainder checkpoint files
        wipe_all_checkpoints(self.title)
        # Timing diagnostics
        if self.config.aggregate_timings and self.config.time_functions:
            output_timing_table(timing_aggregator)
        raise_osx_notification(self.title)
        logging.info("Simulation complete.")
        sys.exit(0)


@handle_errors_class
class FEMSimulation(Simulation):
    """Finite Element Method simulation class.

    Implements the standard FEM solution procedure for phase-field
    fracture mechanics problems using Newton-Raphson iteration.
    """

    run_type = "fem"

    def _setup_specific(
        self,
    ) -> None:
        """Set up FEM-specific simulation parameters."""
        self._initialise_variables()
        self._setup_solution_iterator()
        self._setup_forcing()
        self._setup_boundary_conditions()
        self._setup_lagrange_constraints()
        self._setup_penalty_constraints()
        self._enforce_no_constraint_overlap()
        self._setup_sparse_stiffness_indices()
        self._load_checkpoints()
        set_up_sparse_solver(self.config.sparse_solver, self.config.n_threads_petsc)

    def _initialise_variables(
        self,
    ) -> None:
        """Initialize variables for the FEM simulation."""
        # Output variable storage
        self.qoi_dict = {name: [] for name in self.tracked_variables}

        # Newton Raphson variables
        self.total_nr_iterations = 0
        self.max_nr_iterations = (
            self.config.loop_parameters.newton_raphson_maximum_steps
        )
        self.nr_tolerance = self.config.loop_parameters.newton_raphson_tolerance
        self.converged = False

        # Set up fields
        self._initialise_field_variables()

        # Set up constraints
        self.penalty_method_enabled = self.config.penalty_method_parameters.enabled
        self.lagrange_multiplier_enabled = (
            self.config.lagrange_multiplier_parameters.enabled
        )
        assert not (
            self.lagrange_multiplier_enabled and self.penalty_method_enabled
        ), "Cannot enable both Lagrange multipliers and penalty method simultaneously."
        self.bound_dofs = None
        self.penalty_value = (
            self.config.penalty_method_parameters.penalty_value
            * self.material_parameters.penalty_scaling
        )
        self.penalty_master_dof = self.target_dof
        self.penalty_tied_dofs = None
        self.lagrange_slice = 2 * self.mesh_data.nodal_coordinates.shape[0]
        self.lagrange_multipliers_incremental = None
        self.lagrange_mask = None
        self.B_full, self.V = None, None

        # Construct the timestep iterator
        self.step_iterator = range(self.config.loop_parameters.timesteps + 1)

        if self.problem_parameters.problem_type != "linear_elasticity":
            # Unzip the initial phasefield tuple
            self.ip_history_field, self.ip_g, self.phasefield = (
                self._initial_phasefield_tuple
            )

        self.online_output = self.config.output_parameters.stream_outputs
        logging.debug("Finished FEM-specific variable initialisation.")

    def _initialise_field_variables(
        self,
    ) -> None:
        """Initialize empty field arrays for displacement, strain, stress."""
        self.ip_history_field = jnp.zeros_like(self.ip_data.volumes)
        self.ip_strain_tp1 = jnp.repeat(
            jnp.expand_dims(self.ip_history_field, axis=-1),
            repeats=3,
            axis=-1,
        )
        self.ip_strain_t = jnp.zeros_like(self.ip_strain_tp1)
        self.ip_stress_tp1 = jnp.zeros_like(self.ip_strain_tp1)
        self.ip_stress_t = jnp.zeros_like(self.ip_strain_tp1)
        self.ip_strain_incremental = jnp.zeros_like(self.ip_strain_tp1)
        self.ip_g = jnp.ones_like(self.ip_data.volumes)
        self.phasefield = jnp.zeros(self.mesh_data.nodal_coordinates.shape[0])
        self.displacement = jnp.zeros(2 * self.mesh_data.nodal_coordinates.shape[0])

    def _setup_solution_iterator(
        self,
    ) -> None:
        """Set up the solution iterator for displacement or load control."""
        match self.config.solution_parameters.mode:
            case "displacement control":
                self.incremental_displacements = set_up_disp_factor(self.config)
                self.incremental_displacements *= (
                    self.material_parameters.displacement_scaling
                )
            case "load control":
                self.load_factor = set_up_load_factor(self.config)

        logging.debug("Finished setting up solution iterator.")

    def _setup_forcing(
        self,
    ) -> None:
        """Set up the external forcing vector for the FEM simulation."""
        match self.config.forcing_parameters.mode:
            case "concentrated":
                self.F = set_up_concentrated_F(
                    config=self.config,
                    dofs_size=2 * self.mesh_data.nodal_coordinates.shape[0],
                    target_dof=self.target_dof,
                )
            case "window":
                self.F = set_up_window_F(
                    config=self.config,
                    nodal_coordinates=self.mesh_data.nodal_coordinates,
                )
            case "body":
                self.F = set_up_body_F(
                    config=self.config,
                    dofs_size=2 * self.mesh_data.nodal_coordinates.shape[0],
                    dofs=self.mesh_data.dofs,
                    point_volumes=self.ip_data.volumes,
                    N=self.ip_data.N,
                )

        logging.debug("Finished setting up forcing vector.")

    def _setup_boundary_conditions(
        self,
    ) -> None:
        """Set up the boundary conditions for the FEM simulation."""
        self.boundary_conditions = set_up_window_dofs(
            dof_selection_mode=self.config.boundary_conditions_parameters.dof_selection_mode,
            bound_dofs=self.config.boundary_conditions_parameters.constrained_degrees_of_freedom,
            nodal_coordinates=self.mesh_data.nodal_coordinates,
            config=self.config,
            window_function=set_up_window_boundary,
        )
        self.free_dofs = jnp.setdiff1d(
            jnp.arange(2 * len(self.mesh_data.nodal_coordinates)),
            self.boundary_conditions,
        )
        self.F = self.F.at[self.boundary_conditions].set(0)

        logging.debug("Finished setting up boundary conditions.")

    def _setup_lagrange_constraints(
        self,
    ) -> None:
        """Set up Lagrange multiplier constraints for the FEM simulation."""
        if not self.lagrange_multiplier_enabled:
            return

        # Extract Lagrange multiplier parameters and set up
        lagrange_params = self.config.lagrange_multiplier_parameters
        self.bound_dofs = set_up_window_dofs(
            dof_selection_mode=lagrange_params.dof_selection_mode,
            bound_dofs=lagrange_params.bound_degrees_of_freedom,
            nodal_coordinates=self.mesh_data.nodal_coordinates,
            config=self.config,
            window_function=set_up_load_window_lagrange,
        )
        self.B_full, self.B, self.V, self.lagrange_multipliers = (
            set_up_lagrange_parameters(
                bound_dofs=self.bound_dofs.astype(int),
                free_dofs=self.free_dofs,
                dofs_size=2 * self.mesh_data.nodal_coordinates.shape[0],
            )
        )
        i_B, j_B = jnp.where(self.B_full != 0)
        self.lagrange_mask = tuple(i_B.tolist()), tuple(j_B.tolist())

        logging.debug("Finished setting up Lagrange multiplier constraints.")

    def _setup_penalty_constraints(
        self,
    ) -> None:
        """Set up penalty method constraints for the FEM simulation."""
        if not self.penalty_method_enabled:
            return
        penalty_params = self.config.penalty_method_parameters
        self.bound_dofs = set_up_window_dofs(
            dof_selection_mode=penalty_params.dof_selection_mode,
            bound_dofs=penalty_params.bound_degrees_of_freedom,
            nodal_coordinates=self.mesh_data.nodal_coordinates,
            config=self.config,
            window_function=set_up_load_window_penalty,
        )
        self.penalty_master_dof = self.target_dof
        self.penalty_tied_dofs = self.bound_dofs[
            self.bound_dofs != self.penalty_master_dof
        ]

        logging.debug("Finished setting up penalty method constraints.")

    def _enforce_no_constraint_overlap(
        self,
    ) -> None:
        """Ensure no overlap between bound DOFs and boundary conditions."""
        if not self.penalty_method_enabled and not self.lagrange_multiplier_enabled:
            return

        # Cast to lists and ensure there is no overlap
        bound_set = set(self.bound_dofs.tolist())
        bc_set = set(self.boundary_conditions.tolist())
        if bound_set.intersection(bc_set) != set():
            logging.error(
                "Bound dofs and boundary conditions intersect! This will lead to instability."
            )
            logging.error("Intersection: " + str(bound_set.intersection(bc_set)))
            raise ValueError("Bound dofs and boundary conditions intersect.")

        logging.debug("No overlap between bound dofs and boundary conditions.")

    def _setup_sparse_stiffness_indices(
        self,
    ) -> None:
        """Set up the sparse stiffness matrix indices for the FEM simulation."""
        row_idx, col_idx, values = pointwise_D_to_global_stiffness_coeffs(
            ip_D=self.ip_D,
            ip_volumes=self.ip_data.volumes,
            ip_B=self.ip_data.B,
            dofs=self.mesh_data.dofs,
        )

        if self.config.optimise_sparsity:
            row_idx, col_idx = self._optimise_sparsity_pattern(row_idx, col_idx, values)

        # Need numpy here as jnp arrays are unhashable
        import numpy as np

        # Create a hashable object for JAX to trace
        boundary_set = set(np.array(self.boundary_conditions))
        indices = []
        row_idx_jnp = np.array(row_idx)
        col_idx_jnp = np.array(col_idx)

        # Collect non-boundary indices
        for i, (r, c) in enumerate(zip(row_idx_jnp, col_idx_jnp)):
            if r not in boundary_set and c not in boundary_set:
                indices.append(i)

        # Cast to constant data type
        self.indices_sparse = tuple(indices)

        logging.debug("Finished setting up sparse stiffness matrix indices.")

    def _optimise_sparsity_pattern(
        self,
        row_idx: Array,
        col_idx: Array,
        values: Array,
    ) -> Tuple[Array, Array]:
        """Optimize the sparsity pattern of the stiffness matrix.

        Applies the reverse Cuthill-McKee algorithm to reduce matrix
        bandwidth and improve solver performance.

        Parameters
        ----------
        row_idx : Array
            Row indices of non-zero entries.
        col_idx : Array
            Column indices of non-zero entries.
        values : Array
            Values of non-zero entries.

        Returns
        -------
        Tuple[Array, Array]
            Permuted row and column indices.
        """
        K_optim = csc_matrix((values, (row_idx, col_idx)))
        perm_2d = reverse_cuthill_mckee(K_optim, symmetric_mode=True)

        # Returns y-x ordered permutation, we are working with x-y. Need to reorder
        perm_2d_blocks = perm_2d.reshape(self.mesh_data.nodal_coordinates.shape[0], 2)
        perm_2d_blocks = jnp.sort(perm_2d_blocks, axis=1)
        perm_2d = perm_2d_blocks.reshape(-1)

        # Get the inverse permutation for dof mapping
        inv_perm_2d = jnp.argsort(perm_2d).astype(int)

        # Permute the indices
        self.F = self.F[perm_2d]

        # Permute the connectivities
        self.free_dofs = inv_perm_2d[self.free_dofs]
        self.boundary_conditions = inv_perm_2d[self.boundary_conditions]
        self.mesh_data.dofs = inv_perm_2d[self.mesh_data.dofs]

        # Create 1D permutation
        inv_perm_1d = inv_perm_2d[jnp.where(inv_perm_2d % 2 == 0)] // 2
        perm_1d = perm_2d[jnp.where(perm_2d % 2 == 0)] // 2

        # Permute 1D arrays
        self.mesh_data.connectivities = inv_perm_1d[
            self.mesh_data.connectivities
        ].astype(int)
        self.mesh_data.nodal_coordinates = self.mesh_data.nodal_coordinates[perm_1d]

        # Get the inverse target dof
        self.target_dof = int(inv_perm_2d[self.target_dof])

        # Reorder the bound dofs
        if self.lagrange_multiplier_enabled or self.penalty_method_enabled:
            self.bound_dofs = inv_perm_2d[self.bound_dofs]
            self.penalty_master_dof = int(inv_perm_2d[self.penalty_master_dof])
            self.penalty_tied_dofs = inv_perm_2d[self.penalty_tied_dofs]
        # Reorder the Lagrange multiplier matrices
        if self.lagrange_multiplier_enabled:
            self.B_full = self.B_full[:, perm_2d]
            i_B, j_B = jnp.where(self.B_full != 0)
            self.lagrange_mask = tuple(i_B.tolist()), tuple(j_B.tolist())

        # Save the permutations
        self.perm_1d = perm_1d
        self.inv_perm_1d = inv_perm_1d
        self.perm_2d = perm_2d
        self.inv_perm_2d = inv_perm_2d

        # Permute the indices
        row_idx = inv_perm_2d[row_idx]
        col_idx = inv_perm_2d[col_idx]

        logging.debug("Finished optimising sparsity pattern.")

        return row_idx, col_idx

    def _load_checkpoints(
        self,
    ) -> None:
        """Load existing checkpoints to resume the simulation if present."""
        self.checkpoint_keys = [
            "displacement",
            "ip_D",
            "ip_strain_tp1",
            "ip_stress_tp1",
            "ip_history_field",
            "ip_g",
        ]

        checkpoint_obj = read_checkpoints(
            filename=self.title, var_names=self.checkpoint_keys
        )
        n_checkpoints = 0
        # If checkpoints are present, load them and continue from the last timestep
        if checkpoint_obj[0]:
            checkpoint_dict, _ = checkpoint_obj
            n_checkpoints = len(checkpoint_dict["displacement"])
            t = len(checkpoint_dict["displacement"])
            self.ip_stress_tp1 = checkpoint_dict["ip_stress_tp1"][-1]
            self.ip_strain_tp1 = checkpoint_dict["ip_strain_tp1"][-1]
            self.ip_D = checkpoint_dict["ip_D"][-1]
            self.ip_history_field = checkpoint_dict["ip_history_field"][-1]
            self.displacement = checkpoint_dict["displacement"][-1]
            self.ip_g = checkpoint_dict["ip_g"][-1]
            if self.lagrange_multiplier_enabled:
                self.lagrange_multipliers = checkpoint_dict["lagrange_multipliers"][-1]
            self.step_iterator = range(t, self.config.loop_parameters.timesteps + 1)

        self._resumed_from_checkpoint = n_checkpoints > 0
        logging.info(f"Found {n_checkpoints} checkpoints")

    def run(
        self,
    ) -> None:
        """Execute the FEM simulation main loop."""
        # If not silent, wrap the iterator with tqdm
        if not self.silent:
            self.step_iterator = tqdm(self.step_iterator, unit="increment")

        # Execute the stepping loop
        for increment in self.step_iterator:
            # Displacement field update
            self.converged = False
            increment_start_time = time()
            logging.info(log_separator)
            logging.info(f"Current iteration step: {increment} to {increment + 1}")
            self._update_variables()
            self._set_increment(increment)
            for nr_iteration in range(self.max_nr_iterations):
                self.total_nr_iterations += 1
                self._form_stiffness_matrix()
                self._obtain_displacement(first_loop_step=(nr_iteration == 0))
                self._displacement_field_postprocess()
                self._check_convergence()
                if self.converged:
                    break
            logging.info(
                f"Newton-Raphson {'converged' if self.converged else 'diverged'} in "
                f"{nr_iteration + 1} iterations. Time taken: {(time() - increment_start_time):.6f}"
            )
            # If not converged, means NR failed. Truncate the incremental arrays and break
            if not self.converged:
                self._truncate_increments(increment)
                break

            # Append the incremental quantities to the total ones
            self._append_iteration_data()

            # If solving linear elasticity, finalise the step here
            if self.problem_parameters.problem_type == "linear_elasticity":
                self._finalise_step(increment)
                continue

            # Phase field update
            phasefield_start_time = time()
            self.ip_history_field = jnp.maximum(
                self.ip_history_field, self.ip_strain_energy_density_plus
            )
            self._solve_phasefield()
            self._phase_field_postprocess()
            logging.info(
                f"Phase-field solution time: {(time() - phasefield_start_time):.6f}"
            )

            # Finalise the step
            self._finalise_step(increment)

        # Print final diagnostics
        logging.info(log_separator)
        logging.info(
            f"Total NR iterations: {self.total_nr_iterations}. Total time: {(time() - self.start_time):.6f}"
        )

    def _update_variables(
        self,
    ) -> None:
        """Update variables at the start of each timestep."""
        self.ip_strain_t = self.ip_strain_tp1
        self.ip_stress_t = self.ip_stress_tp1
        self.displacement_iterative = jnp.zeros_like(self.displacement)
        self.displacement_incremental = jnp.zeros_like(self.displacement)
        self.internal_forces_incremental = jnp.zeros_like(self.displacement)
        if self.lagrange_multiplier_enabled:
            self.lagrange_multipliers_incremental = jnp.zeros_like(
                self.lagrange_multipliers
            )

    def _set_increment(self, increment: int) -> None:
        """Set the current target increment.

        Parameters
        ----------
        increment : int
            Increment index.
        """
        match self.config.solution_parameters.mode:
            case "load control":
                self.F_incremental = self.F * (
                    self.load_factor[increment + 1] - self.load_factor[increment]
                )
                self.residual_iterative = pre_newton_raphson_load(
                    F_incremental=self.F_incremental,
                    displacement_incremental=self.displacement_incremental,
                    B_full=self.B_full,
                    V=self.V,
                    lagrange_multiplier_enabled=self.lagrange_multiplier_enabled,
                )
            case "displacement control":
                self.displacement_target_incremental = (
                    self.incremental_displacements[increment + 1]
                    - self.incremental_displacements[increment]
                )
                self.load_factor_incremental = jnp.array([0])
                self.residual_iterative = -self.F

        logging.debug(f"Set increment {increment}.")

    def _form_stiffness_matrix(
        self,
    ) -> None:
        """Assemble the global stiffness matrix."""
        self.K = pointwise_D_to_global_stiffness_sparse_matrix(
            ip_D=self.ip_D,
            ip_volumes=self.ip_data.volumes,
            ip_B=self.ip_data.B,
            dofs=self.mesh_data.dofs,
            dofs_size=2 * self.mesh_data.nodal_coordinates.shape[0],
            fixed_dofs=tuple(self.boundary_conditions.tolist()),
            lagrange_multiplier_enabled=self.lagrange_multiplier_enabled,
            B_full=self.B_full,
            mask=self.indices_sparse,
            lagrange_mask=self.lagrange_mask,
            penalty_method_enabled=self.penalty_method_enabled,
            penalty_value=self.penalty_value,
            penalty_master_dof=self.penalty_master_dof,
            penalty_tied_dofs=self.penalty_tied_dofs,
        )

    def _obtain_displacement(self, first_loop_step: bool) -> None:
        """Solve for the incremental displacement field.

        Parameters
        ----------
        first_loop_step : bool
            Whether this is the first iteration of the NR loop.
        """
        # Solve for the displacements
        match self.config.solution_parameters.mode:
            case "load control":
                self.displacement_iterative, self.lagrange_multipliers_iterative = (
                    solve_displacement_load_control(
                        lagrange_multiplier_enabled=self.lagrange_multiplier_enabled,
                        K=self.K,
                        displacement_iterative=self.displacement_iterative,
                        residual_iterative=self.residual_iterative,
                        lagrange_slice=self.lagrange_slice,
                    )
                )
            case "displacement control":
                if first_loop_step:
                    (
                        self.displacement_iterative,
                        self.F_incremental,
                        self.lagrange_multipliers_iterative,
                        self.load_factor_incremental,
                    ) = solve_displacement_disp_control_initial(
                        lagrange_multiplier_enabled=self.lagrange_multiplier_enabled,
                        F=self.F,
                        V=self.V,
                        lagrange_slice=self.lagrange_slice,
                        K=self.K,
                        displacement_target_incremental=self.displacement_target_incremental,
                        target_dof=self.target_dof,
                        load_factor_incremental=self.load_factor_incremental,
                    )
                else:
                    (
                        self.displacement_iterative,
                        self.F_incremental,
                        self.lagrange_multipliers_iterative,
                        self.load_factor_incremental,
                    ) = solve_displacement_disp_control_subsequent(
                        lagrange_multiplier_enabled=self.lagrange_multiplier_enabled,
                        F=self.F,
                        V=self.V,
                        lagrange_slice=self.lagrange_slice,
                        K=self.K,
                        residual_iterative=self.residual_iterative,
                        target_dof=self.target_dof,
                        load_factor_incremental=self.load_factor_incremental,
                    )

        # Increment the displacement value
        self.displacement_incremental += self.displacement_iterative

        # Increment the lagrange multipliers
        if self.lagrange_multiplier_enabled:
            self.lagrange_multipliers_incremental += self.lagrange_multipliers_iterative

        logging.debug("Displacement field solved")

    def _displacement_field_postprocess(
        self,
    ) -> None:
        """Postprocess the displacement field after solving."""
        self.ip_strain_incremental = nodal_displacement_to_strain(
            nodal_displacements=(
                self.displacement_incremental[::2],
                self.displacement_incremental[1::2],
            ),
            connectivity=self.mesh_data.connectivities,
            B=self.ip_data.B,
        )

        # Get the strain at new timestep (strain_incremental + strain_t)
        self.ip_strain_tp1 = self.ip_strain_incremental + self.ip_strain_t

        # Get the full strain tensor
        self.ip_strain_full = voigt_strain_to_tensor(
            strains=self.ip_strain_tp1,
            nu=self.material_parameters.nu,
            plane_mode=self.problem_parameters.plane_mode,
        )

        # Compute the stress and strain energy density
        (
            (self.ip_stress_plus, self.ip_stress_minus),
            (self.ip_strain_energy_density_plus, self.ip_strain_energy_density_minus),
        ) = self.strain_energy_model.stress_and_energy(
            self.ip_strain_full, self.ip_D_undegraded, return_energy=True
        )

        logging.debug("Stress and strain energy density parts calculated")

        # Compose the full strain energy density
        self.ip_strain_energy_density = (
            self.ip_g * self.ip_strain_energy_density_plus
            + self.ip_strain_energy_density_minus
        )

        # Weight by the integration point volumes
        self.ip_strain_energy = self.ip_strain_energy_density * self.ip_data.volumes

        # Combine the stress to final value
        self.ip_stress_tp1 = (
            jnp.einsum("eg, egij -> egij", self.ip_g, self.ip_stress_plus)
            + self.ip_stress_minus
        ).block_until_ready()
        # Need the blockuntilready, otherwise the simulation sometimes hangs

        # Generate postprocessed stress if needed
        self.ip_stress_pp = self.strain_energy_model.postprocess_stress(
            self.ip_stress_tp1,
            self.ip_strain_full,
            self.config.postprocessing_stress_type,
        )

        logging.debug("Full stress and strain energy obtained")

        # Convert to Voigt notation
        self.ip_stress_tp1 = matrix_to_voigt(
            self.ip_stress_tp1, double_off_diagonal=False
        )[:, :, [0, 1, 5]]
        # Index only the relevant in-plane stress components

        # Get the pointwise internal forces
        self.ip_internal_forces_incremental = jnp.einsum(
            "egij, egj, eg -> egi",
            jnp.matrix_transpose(self.ip_data.B),
            self.ip_stress_tp1 - self.ip_stress_t,
            self.ip_data.volumes,
        )

        # Convert the pointwise internal forces to global
        self.internal_forces_incremental = pointwise_internal_forces_to_global(
            ip_internal_forces=self.ip_internal_forces_incremental,
            dofs=self.mesh_data.dofs,
            dofs_size=2 * self.mesh_data.nodal_coordinates.shape[0],
        )

        logging.debug("Internal forces calculated")

        # Compute the tangent modulus
        self.ip_D = self.strain_energy_model.tangent(
            self.ip_strain_full, self.ip_g, self.ip_D_undegraded
        )

        logging.debug("Tangential matrices recalculated")

    def _check_convergence(
        self,
    ) -> bool:
        """Check for convergence of the Newton-Raphson iteration."""
        self.ratio, self.residual_iterative, self.converged = hyplas_residual(
            displacement_incremental=self.displacement_incremental,
            F_incremental=self.F_incremental,
            internal_forces_incremental=self.internal_forces_incremental,
            free_dofs=self.free_dofs,
            boundary_conditions=self.boundary_conditions,
            B_full=self.B_full,
            V=self.V,
            lagrange_multipliers_incremental=self.lagrange_multipliers_incremental,
            newton_raphson_tolerance=self.config.loop_parameters.newton_raphson_tolerance,
            lagrange_multiplier_enabled=self.lagrange_multiplier_enabled,
            penalty_method_enabled=self.penalty_method_enabled,
            penalty_value=self.penalty_value,
            penalty_master_dof=self.penalty_master_dof,
            penalty_tied_dofs=self.penalty_tied_dofs,
        )
        # Log the residual ratio
        logging.info(f"Residual ratio: {self.ratio:.2e}")

    def _truncate_increments(self, step: int) -> None:
        """Truncate incremental arrays to the current step.

        Parameters
        ----------
        step : int
            Current step index.
        """
        if hasattr(self, "incremental_displacements"):
            self.incremental_displacements = self.incremental_displacements[: step + 1]
        if hasattr(self, "load_factor"):
            self.load_factor = self.load_factor[: step + 1]

    def _append_iteration_data(
        self,
    ) -> None:
        """Append incremental quantities to total displacement and multipliers."""
        self.displacement += self.displacement_incremental
        if self.lagrange_multiplier_enabled:
            self.lagrange_multipliers += self.lagrange_multipliers_incremental

    def _finalise_step(self, step: int) -> None:
        """Finalize the current timestep.

        Parameters
        ----------
        step : int
            Current step index.
        """
        self._track_qois()
        if self.online_output:
            self._reorder_step()
            # Build flat single-step dict from last entry — raw (unscaled) copy
            raw_dict = {k: np.asarray(v[-1]) for k, v in self.qoi_dict.items() if v}
            # Rescaled copy for output only — does not mutate checkpoint data
            step_dict = self._rescale_step(dict(raw_dict))
            inc = (
                self._get_increment_value(step)
                if self.paraview_ctx is not None
                else float(step)
            )
            self._postprocess_and_output_step(step_dict, inc)
            if self.streaming_plot_manager is not None:
                self.streaming_plot_manager.update(self, step)
        self._write_checkpoint(step)
        if self.config.profile_memory:
            log_mem(f"after step {step}")

    def _solve_phasefield(
        self,
    ) -> None:
        """Solve the phase field evolution equation."""
        self.phasefield = solve_phasefield(
            self.phasefield,
            self.phasefield_model,
            self.ip_data,
            self.mesh_data,
            self.ip_history_field,
        )

        logging.debug("Phase-field solution completed")

    def _phase_field_postprocess(
        self,
    ) -> None:
        """Postprocess the phase field after solving."""
        # Extract the elemental quantities
        phasefield_elemental = self.phasefield[self.mesh_data.connectivities]
        phasefield_elemental = jnp.repeat(
            phasefield_elemental[:, None, :], self.ip_data.N.shape[1], axis=1
        )

        # Get the phasefield derivatives at the integration points
        self.ip_phasefield_derivatives = self.phasefield_model.grad_in_ip(
            c_elemental=phasefield_elemental,
            dNdx=(
                self.ip_data.physical_derivatives
                if self.ip_data.physical_derivatives_rot is None
                else self.ip_data.physical_derivatives_rot
            ),
        )
        self.ip_phasefield_derivatives_2 = self.phasefield_model.hess_in_ip(
            c_elemental=phasefield_elemental,
            d2Ndx2=(
                self.ip_data.physical_derivatives_2
                if self.ip_data.physical_derivatives_2_rot is None
                else self.ip_data.physical_derivatives_2_rot
            ),
        )

        # Get the fracture energy at the integration points
        self.ip_fracture_energy_density = self.phasefield_model.energy_density(
            c_elemental=phasefield_elemental,
            N=self.ip_data.N,
            dNdx=(
                self.ip_data.physical_derivatives
                if self.ip_data.physical_derivatives_rot is None
                else self.ip_data.physical_derivatives_rot
            ),
            d2Ndx2=(
                self.ip_data.physical_derivatives_2
                if self.ip_data.physical_derivatives_2_rot is None
                else self.ip_data.physical_derivatives_2_rot
            ),
            gamma=self.ip_data.gamma_matrix,
        )
        self.ip_fracture_energy = jnp.einsum(
            "eg,eg->eg", self.ip_fracture_energy_density, self.ip_data.volumes
        )
        logging.debug("Fracture energy computed")

        # Get the degradation function at the integration points
        self.ip_g = self.phasefield_model.degradation_in_ip(
            c_elemental=phasefield_elemental, N=self.ip_data.N
        ).block_until_ready()

    def _postprocess_specific(
        self,
    ) -> None:
        """Perform FEM-specific postprocessing."""
        # Make sure to set the paraview increments correctly
        match self.config.solution_parameters.mode:
            case "displacement control":
                self.paraview_increments = (
                    self.incremental_displacements
                    / self.material_parameters.displacement_scaling
                )
            case "load control":
                self.paraview_increments = self.load_factor

        if self.online_output:
            # All heavy work done per-step — just ensure qoi_dict exists
            if not hasattr(self, "qoi_dict"):
                self.qoi_dict = {}
        else:
            # Non-stream: load checkpoints, reorder, rescale for offline output
            self._load_checkpoints_postprocess()
            self._reorder_qois()

        cleanup_solver()

    def _load_checkpoints_postprocess(
        self,
    ) -> None:
        """Load checkpoint data for postprocessing."""
        if not self.online_output:
            logging.info("Loading checkpoints for postprocessing...")
            self.qoi_dict, checkpoint_files = read_checkpoints(
                filename=self.title, var_names=self.tracked_variables
            )
            logging.info(
                "Results loaded from checkpoints. Cleaning up any extra keys..."
            )
            self.qoi_dict = prune_qoi_dict(
                qoi_dict=self.qoi_dict, keys_to_keep=self.tracked_variables
            )
            logging.info("Checkpoint loading complete.")
            cleanup_checkpoints(checkpoint_files, unlink_parent=True)
            logging.info("Checkpoint files cleaned up.")
        else:
            logging.info("Loading checkpoints one-by-one for postprocessing...")
            checkpoint_files = list_checkpoints(self.title)
            for i, entry in enumerate(checkpoint_files):
                checkpoint_data = read_blosc(entry)
                checkpoint_data = prune_qoi_dict(
                    qoi_dict=checkpoint_data, keys_to_keep=self.tracked_variables
                )
                write_checkpoint(
                    qoi_dict=checkpoint_data,
                    filename=self.title,
                    timestep=i,
                    file_prefix="pp",
                    reset_qoi_dict=True,
                )
                logging.info(f"Checkpoint {i} pruned from non-relevant quantities.")
            cleanup_checkpoints(checkpoint_files)

    def _reorder_qois(
        self,
    ) -> None:
        """Reorder quantities of interest to original mesh ordering."""
        if self.config.optimise_sparsity:
            logging.info("Reverting the global variables to the original order")
            if not self.online_output:
                reorder_qoi_dict(
                    qoi_dict=self.qoi_dict,
                    inv_perm_2d=self.inv_perm_2d,
                    inv_perm_1d=self.inv_perm_1d,
                    dofs_size=2 * self.mesh_data.nodal_coordinates.shape[0],
                )
            else:
                files = list_checkpoints(self.title, file_prefix="pp")
                for i, entry in enumerate(files):
                    checkpoint_data = read_blosc(entry)
                    reorder_qoi_dict(
                        qoi_dict=checkpoint_data,
                        inv_perm_2d=self.inv_perm_2d,
                        inv_perm_1d=self.inv_perm_1d,
                        dofs_size=2 * self.mesh_data.nodal_coordinates.shape[0],
                    )
                    dump_blosc(checkpoint_data, entry)
                    logging.info(f"Checkpoint {i} reordered.")

            self.mesh_data.nodal_coordinates = self.mesh_data.nodal_coordinates[
                self.inv_perm_1d
            ]
            self.mesh_data.connectivities = self.perm_1d[self.mesh_data.connectivities]
            self.target_dof = self.perm_2d[self.target_dof]
            logging.info("Reordering complete")

    def _reorder_step(
        self,
    ) -> None:
        """Reorder the current step data to original mesh ordering."""
        if not self.config.optimise_sparsity:
            return
        dofs_size = 2 * self.mesh_data.nodal_coordinates.shape[0]
        for key, entries in self.qoi_dict.items():
            if not entries or key.startswith("ip"):
                continue
            arr = entries[-1]
            if arr.shape[0] == dofs_size:
                entries[-1] = arr[self.inv_perm_2d.astype(int)]
            elif arr.shape[0] == dofs_size // 2:
                entries[-1] = arr[self.inv_perm_1d.astype(int)]


class PostprocessFields(FEMSimulation):
    """Extract postprocessing functionality from the FEM class.

    Provides field postprocessing capabilities independent of the
    full FEM simulation, useful for DRMSimulation.

    Parameters
    ----------
    parent_config : Dict[str, object]
        Configuration dictionary with parent information.
    title : str
        Simulation title for output identification.
    """

    run_type = "parent"

    def __init__(self, parent_config: Dict[str, object], title: str) -> None:
        """Initialize the postprocessing object with parent config."""
        self._no_log = True
        self.config = StrictDict(parent_config)
        self.config.output_parameters.timestamp = False
        self.title = title
        self._setup_generic()
        self._initialise_field_variables()

    def assign_field_variables(self, displacement: Array, phasefield: Array) -> None:
        """Attach the provided fields to the object.

        Parameters
        ----------
        displacement : Array
            Displacement field values.
        phasefield : Array
            Phase field values.
        """
        # Generate the incremental displacement
        self.displacement_incremental = displacement - self.displacement
        self.displacement = displacement

        # Assign the phasefield
        self.phasefield = phasefield

    def postprocess_fields(
        self,
    ) -> None:
        """Generate postprocessed quantities from assigned step-wise fields."""
        # For displacement fields, need: self.displacements_incremental
        self._displacement_field_postprocess()

        # For phase fields, need self.phasefield
        self._phase_field_postprocess()

        # Track
        self._track_qois()

        # Increment time-based qois
        self.ip_strain_t = self.ip_strain_tp1
        self.ip_stress_t = self.ip_stress_tp1

    def get_step_dict(self) -> Dict[str, Array]:
        """Return a flat dict of the most recently tracked step.

        Returns
        -------
        Dict[str, Array]
            Dictionary mapping variable names to their latest values.
        """
        return {k: np.asarray(v[-1]) for k, v in self.qoi_dict.items() if v}


@handle_errors_class
class DRMSimulation(Simulation):
    """Deep Ritz Method simulation class.

    Implements neural network-based solution of phase-field fracture
    mechanics problems using variational principles.
    """

    run_type = "nn"

    def _setup_generic(
        self,
    ) -> None:
        """Override generic setup to add seed suffix to DRM simulation title."""
        self.title += "_S" + str(self.config.seed)
        super()._setup_generic()

    def _setup_specific(
        self,
    ) -> None:
        """Set up DRM-specific simulation parameters."""
        self._initialise_variables()
        self._setup_displacement_iterator()
        self._setup_training_variables()
        self._setup_pretraining_variables()
        self._setup_model()
        self._setup_readability_flags()
        self._setup_postprocess_fields()
        self._setup_mlflow()
        logging.info("Main training parameters set up.")

    def _initialise_variables(
        self,
    ) -> None:
        """Initialize variables for the DRM simulation."""
        self.activation_coefficients = {}
        self.extra_attributes = {}
        self.sparsity_information = {}
        self.optimiser_states = (
            []
            if self.config.transfer_learning.enabled
            & self.config.transfer_learning.transfer_optimiser
            else None
        )
        self.diverged = False
        self.online_output = self.config.output_parameters.stream_outputs
        self._sparsity_records = []
        self._activation_records = []
        self._extra_records = []

    def _setup_readability_flags(
        self,
    ) -> None:
        """Set up boolean flags for DRM simulation configuration."""
        # Booleans
        self.calculate_sparsity = (
            self.config.output_parameters.log_parameters.calculate_sparsity
        )
        self.output_activation_coefficients = (
            self.config.network.activation.trainable
            and self.config.output_parameters.log_parameters.output_activation_coefficients
        )
        self.reset_activations = (
            self.config.network.activation.trainable
            and self.config.pretraining_parameters.reset_activation_coefficient
        )
        self.output_extra_attributes = collect_auxiliary_data(self.network) != {}
        self.pretrain = self.config.pretraining_parameters.enabled
        self.transfer_learning = self.config.transfer_learning.enabled

        # Dictionaries
        self.training_snapshots = self.config.output_parameters.training_snapshots

    def _setup_displacement_iterator(
        self,
    ) -> None:
        """Set up the displacement iterator for the DRM simulation."""
        self.incremental_displacements = construct_displacement(
            self.config.displacement_parameters
        )
        self.incremental_displacements *= self.material_parameters.displacement_scaling

        logging.info(
            f"Displacement iterator set up with length {len(self.incremental_displacements)}."
        )

    def _setup_pretraining_variables(
        self,
    ) -> None:
        """Set up pretraining-specific variables for the DRM simulation."""
        # Form the pretraining optimiser list
        wrapped_list = [
            get_optimiser(name) for name in self.config.pretraining_parameters.optimiser
        ]
        self.optimiser_list_pretrain = [item[0] for item in wrapped_list]
        self.optimiser_name_list_pretrain = [
            name for name in self.config.pretraining_parameters.optimiser
        ]
        self.number_of_epochs_list_pretrain = [item[1] for item in wrapped_list]

        (
            self.mesh_data_pretrain,
            self.ip_data_pretrain,
            self.distance_functions_pretrain,
        ) = setup_geometry(self.config, self.config.pretraining_parameters.mesh)

        *_, initial_phasefield_pretrain = setup_initial_phasefield(
            pf_params=self.config.phasefield_parameters,
            mesh_data=self.mesh_data_pretrain,
            ip_data=self.ip_data_pretrain,
            material_parameters=self.material_parameters,
            phasefield_model=self.phasefield_model,
        )
        self.initial_phasefield_pretrain = 1 - initial_phasefield_pretrain

        logging.info(
            f"Found pretraining mesh with {self.mesh_data_pretrain.nodal_coordinates.shape[0]}"
            f" nodes and {self.mesh_data_pretrain.connectivities.shape[0]} elements."
        )

    def _setup_training_variables(
        self,
    ) -> None:
        """Set up training-specific variables for the DRM simulation."""
        # Form the training optimiser list
        wrapped_list = [
            get_optimiser(name) for name in self.config.training_parameters.optimiser
        ]
        self.optimiser_list = [item[0] for item in wrapped_list]
        self.optimiser_name_list = [
            name for name in self.config.training_parameters.optimiser
        ]
        self.number_of_epochs_list = [item[1] for item in wrapped_list]

        # Set up early stopping
        self.early_stop = EarlyStopping(
            patience=self.config.training_parameters.early_stopping.patience,
            relative_threshold=self.config.training_parameters.early_stopping.relative_threshold,
            absolute_threshold=self.config.training_parameters.early_stopping.absolute_threshold,
            mode=self.config.training_parameters.early_stopping.mode,
        )

    def _setup_model(
        self,
    ) -> None:
        """Set up the main neural network model for the DRM simulation."""
        # Construct the neural network
        self.network = construct_network(self.config.network, seed=self.config.seed)

        # Set up the phasefield model
        self.phasefield_constraint = construct_pf_constraint(
            self.config.phasefield.constraint
        )

        # Extract and invert the initial phasefield tuple
        *_, previous_phasefield = self._initial_phasefield_tuple
        self.previous_phasefield = 1 - previous_phasefield

        # Set up the model
        model_type = get_model_nn()
        self.model = model_type(
            network=self.network,
            material_parameters=self.material_parameters,
            problem_parameters=self.problem_parameters,
            phasefield_constraint=self.phasefield_constraint,
            ip_data=self.ip_data,
            mesh_data=self.mesh_data,
            distance_functions=self.distance_functions,
            previous_phasefield=self.previous_phasefield,
            loading_angle=self.config.displacement_parameters.loading_angle,
            strain_energy_model=self.strain_energy_model,
            phasefield_model=self.phasefield_model,
        )
        self.model_initial = copy.deepcopy(self.model)
        logging.info(
            f"Trainable network initialised with {self.config.network.hidden_count} "
            f"layers ({self.config.network.hidden_size} neurons per hidden)."
        )

    def _setup_postprocess_fields(
        self,
    ) -> None:
        """Initialize PostprocessFields helper for converting NN predictions."""
        self._pp_fields = PostprocessFields(
            parent_config=dict(self.config),
            title=self.title,
        )
        logging.info("PostprocessFields helper initialised for DRM output.")

    def _build_step_dict_drm(self) -> Dict[str, Array]:
        """Convert NN predictions into a postprocessed flat step dict."""
        phasefield_fem_convention = 1.0 - self.phasefield_prediction
        self._pp_fields.assign_field_variables(
            displacement=self.displacement_prediction,
            phasefield=phasefield_fem_convention,
        )
        self._pp_fields.postprocess_fields()
        return self._pp_fields.get_step_dict()

    def _setup_mlflow(
        self,
    ) -> None:
        """Initialize MLflow tracking for this simulation run."""
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
        self._mlflow_tracker = MLflowTracker(
            config=self.config,
            title=self.title,
            n_increments=len(self.incremental_displacements),
        )
        resume_info = self._mlflow_tracker.setup()
        if resume_info is not None:
            self._apply_resume(resume_info)

    def _apply_resume(self, info: "ResumeInfo") -> None:
        """Restore simulation state from ResumeInfo data."""
        self.title = info.title

        # Load network via the tracker's loader (returns a callable)
        loader = self._mlflow_tracker.get_network_loader(info.run_id, info.latest_step)
        skeleton = construct_network(self.config.network, seed=self.config.seed)
        restored_network = loader(skeleton)

        self.model = eqx.tree_at(lambda m: m.network, self.model, restored_network)
        self.model_initial = copy.deepcopy(self.model)

        # Optionally restore optimiser states
        if self.config.transfer_learning.transfer_optimiser:
            try:
                skeleton_states = [
                    opt.init(eqx.filter(self.model.network, eqx.is_array))
                    for opt in self.optimiser_list
                ]
                opt_loader = self._mlflow_tracker.get_optimiser_loader(
                    info.run_id, info.latest_step
                )
                self.optimiser_states = opt_loader(skeleton_states)
            except Exception as e:
                logging.warning(f"[Resume] Could not restore optimiser states: {e}")

        # Restore phasefield
        pf = self._load_phasefield_for_step(info.latest_step)
        if pf is not None:
            self.previous_phasefield = pf
        else:
            logging.warning("Could not restore phasefield — exiting.")
            exit(1)

        self._resume_step = info.latest_step

    def _load_phasefield_for_step(self, step: int) -> Optional[Array]:
        """Recover the phasefield prediction at a specific step from checkpoint.

        Parameters
        ----------
        step : int
            Step index to load phasefield for.

        Returns
        -------
        Array or None
            Phasefield array if found, None if file is missing or unreadable.
        """
        try:
            predictions, _ = read_checkpoints(self.title, file_prefix="predictions")
            pf = predictions[step + 1].get("phasefield")
            if pf is not None:
                logging.debug(
                    f"[Resume] Loaded phasefield from checkpoint for step {step}."
                )
            return pf
        except Exception as e:
            logging.error(
                f"[Resume] Could not read phasefield checkpoint at step {step}: {e}"
            )
            return None

    def _collect_diagnostics_payload(self, step: int) -> "DiagnosticsPayload":
        """Collect diagnostic data payload for MLflow logging.

        Parameters
        ----------
        step : int
            Current step index.

        Returns
        -------
        DiagnosticsPayload
            Payload containing sparsity, activations, and extra attributes.
        """
        sparsity_rows = None
        activation_coefficients = None
        extra_rows = None

        if self.calculate_sparsity:
            info = calculate_weight_sparsity(self.model.network)
            sparsity_rows = []
            for threshold, layer_values in info.items():
                row = {"step": step, "threshold": threshold}
                row.update({f"layer_{i}": float(v) for i, v in enumerate(layer_values)})
                row["mean"] = float(sum(layer_values) / len(layer_values))
                sparsity_rows.append(row)

        if self.output_activation_coefficients:
            activation_coefficients = {
                i: float(act.coeff)
                for i, act in enumerate(self.model.network.activation_list)
            }

        if self.output_extra_attributes:
            extra = collect_auxiliary_data(self.model.network)
            row = {"step": step}
            for attr_name, stats in extra.items():
                for stat_name, val in stats.items():
                    if stat_name == "values":
                        row.update(
                            {
                                f"{attr_name}/value_{i}": float(v)
                                for i, v in enumerate(val)
                            }
                        )
                    else:
                        row[f"{attr_name}/{stat_name}"] = float(val)
            extra_rows = [row]

        return DiagnosticsPayload(
            sparsity_rows=sparsity_rows,
            activation_coefficients=activation_coefficients,
            extra_attribute_rows=extra_rows,
        )

    def run(
        self,
    ) -> None:
        """Execute the Deep Ritz Method simulation."""
        training_time = time()
        resume_step = getattr(self, "_resume_step", None)
        if self.pretrain:
            if resume_step is None:
                self._run_step(-1)
            else:
                # Trim the displacement iterator to exclude the pretraining step, so the main training loop starts at step 0 as normal
                self.incremental_displacements = self.incremental_displacements[1:]
                logging.info(
                    f"[Resume] Skipping pretraining step -1 as it was already done."
                )
        for i in range(len(self.incremental_displacements)):
            if resume_step is not None and i <= resume_step:
                logging.info(
                    f"[Resume] Skipping training step {i} as it was already done."
                )
                continue
            self._run_step(i)
        logging.info(log_separator)
        logging.info(f"Total training time: {(time() - training_time):.6f} seconds.")
        logging.info(log_separator)

    def _run_step(self, step: int) -> None:
        """Execute a single training step for the DRM simulation.

        Parameters
        ----------
        step : int
            Step index (-1 for pretraining).
        """
        if step < 0:
            self._attach_pretraining_parameters()
        if not self.transfer_learning:
            self.model = self.model_initial
        self._train_model_step(step=step)
        if step < 0:
            self._reattach_training_parameters()
            self._finalise_pretraining()
        self._check_divergence(step=step)
        if self.diverged:
            return

        self._predict_model_output()
        self._finalise_step(step=step)

    def _set_model_attr(self, target_attribute: str, value: object) -> None:
        """Set a model attribute using equinox tree_at.

        Parameters
        ----------
        target_attribute : str
            Name of the attribute to set.
        value : object
            Value to set the attribute to.
        """
        self.model = eqx.tree_at(
            lambda m: getattr(m, target_attribute), self.model, value
        )

    def _attach_pretraining_parameters(
        self,
    ) -> None:
        """Attach pretraining parameters to the model before pretraining."""
        # Save current data to buffers
        logging.info("Saving current training parameters to buffers...")
        self.optimiser_list_buffer = self.optimiser_list
        self.optimiser_name_list_buffer = self.optimiser_name_list
        self.number_of_epochs_list_buffer = self.number_of_epochs_list
        self.transferred_optimiser_list_buffer = self.optimiser_states

        # Overwrite training parameters with pretraining ones
        logging.info("Overwriting training parameters with pretraining ones...")
        self.optimiser_list = self.optimiser_list_pretrain
        self.optimiser_name_list = self.optimiser_name_list_pretrain
        self.number_of_epochs_list = self.number_of_epochs_list_pretrain

        # Model-internal data to buffers
        logging.info("Saving current model parameters to buffers...")
        self.mesh_data_buffer = self.mesh_data
        self.ip_data_buffer = self.ip_data
        self.distance_functions_buffer = self.distance_functions
        self.previous_phasefield_buffer = self.previous_phasefield

        # Update the model tree
        logging.info("Attaching pretraining parameters to the model...")
        self._set_model_attr("mesh_data", self.mesh_data_pretrain)
        self._set_model_attr("ip_data", self.ip_data_pretrain)
        self._set_model_attr("distance_functions", self.distance_functions_pretrain)
        self._set_model_attr("previous_phasefield", self.initial_phasefield_pretrain)

        # Re-set the mesh for predictions of NN
        self.mesh_data = self.mesh_data_pretrain
        self.previous_phasefield = self.initial_phasefield_pretrain

        logging.info("Pretraining parameters attached.")
        self.model.initialise_constitutive_matrices()

    def _reattach_training_parameters(
        self,
    ) -> None:
        """Reattach training parameters to the model after pretraining."""
        # Restore data from buffers
        logging.info("Restoring training parameters from buffers...")
        self.optimiser_list = self.optimiser_list_buffer
        self.optimiser_name_list = self.optimiser_name_list_buffer
        self.number_of_epochs_list = self.number_of_epochs_list_buffer
        self.optimiser_states = self.transferred_optimiser_list_buffer

        # Free memory
        logging.info("Freeing buffers...")
        del (
            self.optimiser_list_buffer,
            self.optimiser_name_list_buffer,
            self.number_of_epochs_list_buffer,
            self.transferred_optimiser_list_buffer,
        )

        # Reset internal model data
        logging.info("Reattaching training parameters to the model...")
        self._set_model_attr("mesh_data", self.mesh_data_buffer)
        self._set_model_attr("ip_data", self.ip_data_buffer)
        self._set_model_attr("distance_functions", self.distance_functions_buffer)
        self._set_model_attr("previous_phasefield", self.previous_phasefield_buffer)

        # Re-set the mesh for predictions of NN
        self.mesh_data = self.mesh_data_buffer
        self.previous_phasefield = self.previous_phasefield_buffer

        # Free memory
        logging.info("Freeing buffers...")
        del (
            self.mesh_data_buffer,
            self.ip_data_buffer,
            self.distance_functions_buffer,
            self.previous_phasefield_buffer,
        )

        self.model.initialise_constitutive_matrices()
        logging.info("Training parameters restored.")

    def _finalise_pretraining(
        self,
    ) -> None:
        """Finalize the pretraining process."""
        self.incremental_displacements = self.incremental_displacements[1:]
        self.model_initial = copy.deepcopy(self.model)

        if self.reset_activations:
            logging.info("Resetting activation coefficients after pretraining...")
            self.model = reset_activation_coefficients(
                model=self.model,
                trainable_global=self.config.network.activation.trainable_global,
                initial_coefficient=self.config.network.activation.initial_coefficient,
            )
            logging.info("Activation coefficients reset.")

    def _train_model_step(self, step: int) -> None:
        """Train the model for a single step.

        Parameters
        ----------
        step : int
            Step index (-1 for pretraining).
        """
        training_start = time()
        logging.info(log_separator)
        logging.info(
            f"Starting training step {step+1}/{len(self.incremental_displacements)}..."
            if step >= 0
            else "Starting pretraining..."
        )

        # If not applicable to pretraining step, set early stopping to None
        early_stop = (
            None
            if step < 0
            and not self.config.training_parameters.early_stopping.pretraining_stop
            else self.early_stop
        )

        self.model, self.optimiser_states, (self.loss_history, self.aux_history) = (
            training_step(
                model=self.model,
                optimiser_list=self.optimiser_list,
                optimiser_name_list=self.optimiser_name_list,
                input_dict=self.config,
                title=self.title,
                silent=self.silent,
                displacement_increment=self.incremental_displacements[step + 1],
                increment_index=step,
                mesh_data=self.mesh_data,
                number_of_epochs_list=self.number_of_epochs_list,
                loss_terms=self.config.training_parameters.loss_terms,
                early_stop=early_stop,
                previous_phasefield=self.previous_phasefield,
                tqdm_training_string=(
                    f"Training {step+1}/{len(self.incremental_displacements)}"
                    if step >= 0
                    else "Pretraining"
                ),
                track_training_output=self.training_snapshots.enabled,
                training_output_frequency=self.training_snapshots.frequency,
                transfer_optimiser_states=self.config.transfer_learning.transfer_optimiser,
                transferred_optimiser_list=self.optimiser_states,
            )
        )
        prefix = f"Step {step+1} training" if step >= 0 else "Pretraining"
        logging.info(f"{prefix} completed in {(time() - training_start):.6f} seconds.")

    def _check_divergence(self, step: int) -> None:
        """Check if the simulation has diverged.

        Parameters
        ----------
        step : int
            Current step index.
        """
        if self.model is None or self.previous_phasefield is None:
            self.diverged = True
            logging.warning(f"Simulation diverged at step {step}. Ending simulation.")
            self.incremental_displacements = (
                self.incremental_displacements[: step + 1]
                if step >= 0
                else self.incremental_displacements[0]
            )
            self._mlflow_tracker.set_tag("diverged", "true")
            self._mlflow_tracker.set_tag("divergence_step", str(step))

    def _predict_model_output(
        self,
    ) -> None:
        """Predict model output for the current displacement."""
        self.displacement_prediction, self.phasefield_prediction = predict_model_output(
            model=self.model,
            inp=self.mesh_data.nodal_coordinates,
        )
        self.previous_phasefield = self.phasefield_prediction

    def _finalise_step(self, step: int) -> None:
        """Finalize the current training step.

        Parameters
        ----------
        step : int
            Current step index.
        """
        if self.optimiser_states is not None:
            self.optimiser_states = self.optimiser_states[
                : self.config.transfer_learning.n_transferable_states
            ]

        diagnostics = self._collect_diagnostics_payload(step)

        # Always write prediction checkpoint for potential resume, even in online mode.
        write_checkpoint(
            [
                {
                    "displacement": self.displacement_prediction,
                    "phasefield": self.phasefield_prediction,
                }
            ],
            filename=self.title,
            timestep=step,
            file_prefix="predictions",
        )

        if self.online_output:
            step_dict = self._build_step_dict_drm()
            step_dict = self._rescale_step(step_dict)

            # Compute increment value directly from step index before
            # incremental_displacements gets modified by _finalise_pretraining
            if self.paraview_ctx is not None:
                if step < 0:
                    # Pretraining — use first displacement value
                    inc = 0.0
                else:
                    inc = float(
                        self.incremental_displacements[step]
                        / self.material_parameters.displacement_scaling
                    )
            else:
                inc = float(step)
            self._postprocess_and_output_step(step_dict, inc)
            if self.streaming_plot_manager is not None:
                self.streaming_plot_manager.update(self._pp_fields, step)

        self._mlflow_tracker.log_step(
            step=step,
            loss_history=self.loss_history,
            aux_history=self.aux_history,
            network=self.model.network,
            optimiser_states=(
                self.optimiser_states
                if self.config.transfer_learning.transfer_optimiser
                else None
            ),
            diagnostics=diagnostics,
        )

        # Loss and aux always written regardless of mode
        if self.config.profile_memory:
            log_mem(f"after increment {step}", cleanup=True)

    def _postprocess_specific(
        self,
    ) -> None:
        """Perform DRM-specific postprocessing."""
        self._build_paraview_increments()
        self._clean_training_snapshots()
        if not self.online_output:
            self._read_and_postprocess_predictions()
        self._mlflow_tracker.teardown()

    def _build_paraview_increments(
        self,
    ) -> None:
        """Realign displacement increments for Paraview postprocessing."""
        prepend = [0.0]
        if self.pretrain:
            prepend.append(0.0)
        # Form the paraview increments
        self.paraview_increments = jnp.array(
            prepend + self.incremental_displacements.tolist()
        )
        # Scale back the increments
        self.paraview_increments /= self.material_parameters.displacement_scaling

    def _clean_training_snapshots(
        self,
    ) -> None:
        """Postprocess and clean up training snapshot files."""
        if not self.training_snapshots.enabled:
            return
        logging.info("Cleaning up training snapshots...")
        postprocess_training_snapshots(
            filename=self.title,
            mesh_data=self.mesh_data,
            mesh_type=self.config.mesh.type,
            burn_negative_increment=False,
        )
        logging.info("Training snapshots cleaned up.")

    def _read_and_postprocess_predictions(
        self,
    ) -> None:
        """Load checkpoint data and postprocess predictions."""
        # Read the raw predictions
        predictions, prediction_checkpoints = read_checkpoints(
            self.title, file_prefix="predictions"
        )
        cleanup_checkpoints(prediction_checkpoints, unlink_parent=True)
        # Generate the postprocessed quantities of interest (QoIs)
        logging.info("Generating postprocessed quantities of interest (QoIs)...")
        for prediction in predictions:
            displacement = prediction["displacement"]
            phasefield = prediction["phasefield"]

            # DRM phasefield convention is inverted (1-c) relative to FEM
            phasefield_fem_convention = 1.0 - phasefield

            self._pp_fields.assign_field_variables(
                displacement=displacement,
                phasefield=phasefield_fem_convention,
            )
            self._pp_fields.postprocess_fields()

        # Pull the accumulated qoi_dict onto self for downstream consumers
        self.qoi_dict = self._pp_fields.qoi_dict
        logging.info("Postprocessed quantities of interest (QoIs) generated.")

