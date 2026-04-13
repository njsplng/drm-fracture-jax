"""Streaming plot manager for incremental PDF updates during simulation.

This module provides a manager that maintains live matplotlib figures and
incrementally updates them as simulation data arrives. Each registered plot
function gets its own streaming entry with persistent figure/axes objects
that are mutated in-place rather than reconstructed each step.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

# Headless backend — safe for HPC, no DISPLAY required
matplotlib.use("Agg")

from io_handlers import dump_blosc, read_blosc
from plots_functions import _get_ys_dict, get_plot_function

linestyle_list = ["-", "--", "-.", ":"]


# -----------------------------------------------------------
# Internal per-function entry
# -----------------------------------------------------------
@dataclass
class _StreamingEntry:
    """Hold mutable state for a single streaming plot function.

    Parameters
    ----------
    func_name : str
        Name of the registered plot function.
    x_fn : callable
        Callable that returns the x-axis value.
    ys : dict[str, callable]
        Dictionary of series callables, each returning a y value.
    requires : list[str]
        Attribute names required on the simulation object.
    needs_target_dof : bool
        Whether the callables need target_dof to be set.
    x_buf : list[float], optional
        Buffer for x-axis values. Default is an empty list.
    y_bufs : dict[str, list[float]], optional
        Per-series y-axis buffers. Default is an empty dict.
    fig : matplotlib.figure.Figure, optional
        Figure object kept alive between steps.
    ax : matplotlib.axes.Axes, optional
        Axes object kept alive between steps.
    lines : dict[str, matplotlib.lines.Line2D], optional
        Line objects per series, mutated in-place.
    pdf_path : pathlib.Path, optional
        Path for saving the PDF plot.
    dat_path : pathlib.Path, optional
        Path for saving the data file.
    call_kwargs : dict, optional
        Additional keyword arguments for the plot function.
    """

    func_name: str
    x_fn: Callable[..., float]
    ys: Dict[str, Callable[..., float]]
    requires: List[str]
    needs_target_dof: bool
    x_buf: List[float] = field(default_factory=list)
    y_bufs: Dict[str, List[float]] = field(default_factory=dict)
    fig: Optional[object] = field(default=None, repr=False)
    ax: Optional[object] = field(default=None, repr=False)
    lines: Dict[str, object] = field(default_factory=dict)
    pdf_path: pathlib.Path = field(default=None)
    dat_path: pathlib.Path = field(default=None)
    call_kwargs: Dict[str, object] = field(default_factory=dict)

    def initialise_figure(
        self, series_names: List[str], plot_fn: Callable[..., object]
    ) -> None:
        """Create and configure the figure and axes for streaming updates.

        Parameters
        ----------
        series_names : list[str]
            Names of the y-series to plot.
        plot_fn : callable
            The registered plot function, used to extract default labels.
        """
        import inspect

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        title = self.call_kwargs.get("title", self.func_name)
        self.ax.set_title(title, fontsize=18)

        sig = inspect.signature(plot_fn)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        self.ax.set_xlabel(
            self.call_kwargs.get("xlabel", defaults.get("xlabel", "Step")), fontsize=12
        )
        self.ax.set_ylabel(
            self.call_kwargs.get("ylabel", defaults.get("ylabel", "Value")), fontsize=12
        )
        self.ax.grid(True)

        discriminate = self.call_kwargs.get("discriminate_linestyles", False)
        legend_list = self.call_kwargs.get("legend_list", None)

        for i, name in enumerate(series_names):
            label = legend_list[i] if legend_list and i < len(legend_list) else name
            ls = linestyle_list[i % len(linestyle_list)] if discriminate else "-"
            (line,) = self.ax.plot([], [], label=label, linestyle=ls)
            self.lines[name] = line
            self.y_bufs[name] = []

        self.ax.legend()

    def rehydrate_from_disk(self) -> None:
        """Reload x/y buffers from the data file and redraw the figure.

        If a data file exists at ``self.dat_path`` from a previous run, its
        contents are loaded back into the x and y buffers and pushed into
        the Line2D objects so the figure reflects the full history before
        streaming resumes. Missing or unreadable files are silently ignored,
        as are series whose names no longer match the current spec.

        Notes
        -----
        Series whose y-functions carry internal state (e.g. RunningSum)
        cannot be fully recovered from disk — only the plotted values are
        restored, not the accumulator state.
        """
        if not self.dat_path.exists():
            return
        try:
            dat = read_blosc(self.dat_path)
        except Exception as exc:
            logging.warning(
                f"StreamingPlotManager: could not rehydrate '{self.func_name}' "
                f"from {self.dat_path}: {exc}"
            )
            return

        x_arr = dat.get("x")
        if x_arr is None or len(x_arr) == 0:
            return

        self.x_buf = [float(v) for v in x_arr]
        for name in self.y_bufs:
            if name in dat:
                self.y_bufs[name] = [float(v) for v in dat[name]]

        # Mutate Line2D data in place to reflect restored history
        for name, line in self.lines.items():
            line.set_xdata(self.x_buf)
            line.set_ydata(self.y_bufs[name])

        # Rescale axes to fit restored data
        self.ax.relim()
        self.ax.autoscale_view()

        logging.info(
            f"StreamingPlotManager: rehydrated '{self.func_name}' with "
            f"{len(self.x_buf)} step(s) from {self.dat_path.name}."
        )

    def append_and_save(self, x_val: float, y_vals: Dict[str, float]) -> None:
        """Append new values, update line data in place, and save the PDF."""
        self.x_buf.append(x_val)
        for name, y_val in y_vals.items():
            self.y_bufs[name].append(y_val)

        # Mutate Line2D data in place — no axis reconstruction
        for name, line in self.lines.items():
            line.set_xdata(self.x_buf)
            line.set_ydata(self.y_bufs[name])

        # Rescale axes to fit new data
        self.ax.relim()
        self.ax.autoscale_view()

        # Save PDF to disk
        self.fig.savefig(str(self.pdf_path), bbox_inches="tight")

        # Dump plot data buffer to disk (overwrites each step — crash-safe)
        dat: Dict[str, object] = {"x": jnp.asarray(self.x_buf)}
        for name in self.y_bufs:
            dat[name] = jnp.asarray(self.y_bufs[name])
        dump_blosc(dat, self.dat_path)

    def close(self) -> None:
        """Close the matplotlib figure and release resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.lines = {}


# -----------------------------------------------------------
# Public manager
# -----------------------------------------------------------
class StreamingPlotManager:
    """Manage incremental PDF plot updates during a streaming FEM simulation.

    This manager maintains live matplotlib figures for each registered plot
    function, updating them incrementally as simulation data arrives.

    Parameters
    ----------
    jobs : list[dict]
        List of plot job configurations from config.generate_plots.
    target_dof : int
        Target degree of freedom index for the simulation.
    title : str
        Simulation title used in output filenames.
    output_dir : path-like, optional
        Output directory for plots. Defaults to project output folder.

    Examples
    --------
    Create a manager during simulation setup:

    >>> manager = StreamingPlotManager(
    ...     jobs=config.generate_plots,
    ...     target_dof=target_dof,
    ...     title="my_simulation",
    ... )

    Update at each step:

    >>> manager.update(simulation, step)

    Close when done:

    >>> manager.close()
    """

    def __init__(
        self,
        jobs: List[Dict[str, object]],
        target_dof: int,
        title: str,
        output_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """Initialise streaming plot entries based on config jobs.

        Parameters
        ----------
        jobs : list[dict]
            List of plot job configurations.
        target_dof : int
            Target degree of freedom index.
        title : str
            Simulation title used in output filenames.
        output_dir : path-like, optional
            Output directory for plots. Defaults to project output folder.
        """
        self._entries: List[_StreamingEntry] = []
        self._step_counter: int = 0

        if output_dir is None:
            current_path = pathlib.Path(__file__).parent.resolve()
            output_dir = current_path.parent / "output"
        output_dir = pathlib.Path(output_dir)
        plots_dir = output_dir / "plots"
        dat_dir = output_dir / "plot_data"
        plots_dir.mkdir(parents=True, exist_ok=True)
        dat_dir.mkdir(parents=True, exist_ok=True)

        for job_dict in jobs:
            func_name = list(job_dict.keys())[0]
            # Shallow copy
            call_kwargs = dict(list(job_dict.values())[0])

            # Resolve the plot function
            try:
                plot_fn = get_plot_function(func_name)
            except ValueError:
                logging.warning(
                    f"StreamingPlotManager: plot function '{func_name}' not found, skipping."
                )
                continue

            # Only stream functions that carry a __stream_spec__
            if not hasattr(plot_fn, "__stream_spec__"):
                logging.warning(
                    f"StreamingPlotManager: '{func_name}' has no __stream_spec__, "
                    "cannot stream. It will be skipped and plotted offline instead."
                )
                continue

            spec = plot_fn.__stream_spec__
            # Fresh instance — owns RunningSum state
            ys = _get_ys_dict(spec)

            entry = _StreamingEntry(
                func_name=func_name,
                x_fn=spec["x"],
                ys=ys,
                requires=list(spec.get("requires", [])),
                needs_target_dof=bool(spec.get("needs_target_dof", False)),
                call_kwargs=call_kwargs,
                pdf_path=plots_dir / f"stream_{func_name}_{title}.pdf",
                dat_path=dat_dir / f"stream_{func_name}_{title}.dat",
            )
            entry.initialise_figure(series_names=list(ys.keys()), plot_fn=plot_fn)
            entry.rehydrate_from_disk()
            self._entries.append(entry)

        self._target_dof = target_dof
        logging.info(
            f"StreamingPlotManager initialised with {len(self._entries)} plot(s): "
            + ", ".join(e.func_name for e in self._entries)
        )

    # -----------------------------------------------------------

    def update(self, sim: "FEMSimulation", step: int) -> None:
        """Update all registered plots with current simulation data.

        Parameters
        ----------
        sim : FEMSimulation
            Live simulation object with current state.
        step : int
            Current step index (0-based).
        """
        for entry in self._entries:
            # Guard: skip if a required attribute is missing this step
            missing: List[str] = [r for r in entry.requires if not hasattr(sim, r)]
            if missing:
                logging.debug(
                    f"StreamingPlotManager: skipping '{entry.func_name}' at step {step} "
                    f"— missing attributes: {missing}"
                )
                continue

            # Build a thin adapter that exposes sim attributes + target_dof
            adapter = _SimAdapter(sim, step, self._target_dof)

            # Evaluate x
            try:
                x_val: float = float(entry.x_fn(adapter, step))
            except Exception as exc:
                logging.warning(
                    f"StreamingPlotManager: x_fn failed for '{entry.func_name}' "
                    f"at step {step}: {exc}"
                )
                continue

            # Evaluate each y series
            y_vals: Dict[str, float] = {}
            failed = False
            for name, fn in entry.ys.items():
                try:
                    y_vals[name] = float(fn(adapter, step))
                except Exception as exc:
                    logging.warning(
                        f"StreamingPlotManager: ys['{name}'] failed for "
                        f"'{entry.func_name}' at step {step}: {exc}"
                    )
                    failed = True
                    break
            if failed:
                continue

            entry.append_and_save(x_val, y_vals)

        self._step_counter += 1

    # -----------------------------------------------------------

    def close(self) -> None:
        """Close all matplotlib figures and release resources."""
        for entry in self._entries:
            entry.close()
        logging.info("StreamingPlotManager: all figures closed.")

    @property
    def n_registered(self) -> int:
        """Return the number of registered streaming plot functions."""
        return len(self._entries)


# -----------------------------------------------------------
# Thin simulation adapter (mirrors _StepView but for live sim objects)
# -----------------------------------------------------------
# Scaling rules mirror rescale_qoi_dict — keyed on substrings in attribute name.
# Order matters: more specific checks first.
_SCALE_RULES: Tuple[Tuple[str, str], ...] = (
    ("energy", "energy"),
    ("stress", "force"),
    ("force", "force"),
    ("F_", "force"),
    ("strain", "displacement"),
    ("displacement", "displacement"),
)


def _scale_factor_for(name: str, material_parameters: "MaterialParameters") -> float:
    """Return the dimensional scaling factor for the given attribute name.

    The scaling follows the same substring-matching logic as rescale_qoi_dict,
    returning 1.0 if no rule matches.

    Parameters
    ----------
    name : str
        Attribute name to check against scaling rules.
    material_parameters : MaterialParameters
        Object containing energy, force, and displacement scaling factors.

    Returns
    -------
    float
        Scaling factor to divide the attribute value by.
    """
    for substring, kind in _SCALE_RULES:
        if substring in name:
            if kind == "energy":
                return float(material_parameters.energy_scaling)
            if kind == "force":
                return float(material_parameters.force_scaling)
            if kind == "displacement":
                return float(material_parameters.displacement_scaling)
    return 1.0


class _SimAdapter:
    """Wrap a live FEMSimulation for use with stream_spec callables.

    This adapter exposes simulation attributes directly with dimensional
    rescaling applied transparently on read. It also provides a hist() stub
    that raises an error since random-access history is not available in
    live streaming mode.

    Parameters
    ----------
    sim : FEMSimulation
        The live simulation object to wrap.
    step : int
        Current step index.
    target_dof : int
        Target degree of freedom index.

    Notes
    -----
    Scaling follows these rules:
    - "energy" -> divide by energy_scaling
    - "stress"/"force"/"F_" -> divide by force_scaling
    - "strain"/"displacement" -> divide by displacement_scaling
    - everything else -> no scaling
    """

    __slots__ = ("_sim", "_step", "target_dof")

    def __init__(self, sim: "FEMSimulation", step: int, target_dof: int) -> None:
        """Initialise the adapter with the simulation, step, and target DOF."""
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_step", step)
        object.__setattr__(self, "target_dof", target_dof)

    def __getattr__(self, name: str) -> object:
        """Get an attribute from the simulation with dimensional rescaling."""
        sim = object.__getattribute__(self, "_sim")
        value = getattr(sim, name)
        scale = _scale_factor_for(name, sim.material_parameters)
        if scale == 1.0:
            return value
        return value / scale

    def hist(self, name: str, step: int) -> None:
        """Raise an error since random-access history is unavailable.

        This method exists to be called by series functions that need
        previous step values, but live streaming mode does not support
        random-access history.

        Parameters
        ----------
        name : str
            Name of the attribute to access.
        step : int
            Step index to access.

        Raises
        ------
        NotImplementedError
            Always raised since history access is not available.
        """
        raise NotImplementedError(
            f"hist('{name}', {step}) called in live streaming mode. "
            "Random-access to previous steps is not available. "
            "This series will be skipped."
        )
