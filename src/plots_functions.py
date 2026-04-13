"""Plotting utilities for visualizing FEM/DRM simulations.

This module provides functions for plotting force-displacement curves,
energy curves, and other simulation outputs.
"""

import inspect
import logging
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array

from io_handlers import (
    input_pickle,
)
from plots_wrapper import collect_plot_data, stream_spec

plt.style.use("seaborn-v0_8-dark-palette")
linestyle_list = ["-", "--", "-.", ":"]


# -----------------------------------------------------------
# NoPlotDataError exception
# -----------------------------------------------------------
class NoPlotDataError(Exception):
    """Raised when no plot data is available for a given plot function."""


def plot_output_data(
    output_pickle_names: Union[str, List[str]],
    extractors: List[Callable],
    required_variables: List[str],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Displacement Increment",
    ylabel: str = "Values",
    ax: Optional[plt.Axes] = None,
    grid: bool = True,
    show_legend: bool = True,
    legend_list: Optional[List] = None,
    discriminate_linestyles: bool = False,
) -> plt.Axes:
    """Plot data from a FEM/DRM simulation.

    Load simulation data from pickle files and plot multiple data
    series using the provided extractor functions.

    Parameters
    ----------
    output_pickle_names : str or list of str
        Name(s) of the pickle file(s) containing FEM simulation data.
    extractors : list of callable
        Each function takes the input_pickle dictionary and returns
        a tuple of (x_data, y_data).
    required_variables : list of str
        Variable names that must be present in input_pickle for all
        extractors.
    labels : list of str, optional
        Labels for each data series.
    title : str, optional
        Title of the plot. If None, a default title is used.
    xlabel : str, optional
        Label for the x-axis. Default is "Displacement Increment".
    ylabel : str, optional
        Label for the y-axis. Default is "Values".
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    grid : bool, optional
        Whether to add a grid to the plot. Default is True.
    show_legend : bool, optional
        Whether to show a legend on the plot. Default is True.
    legend_list : list of str, optional
        Custom legend labels for each data series.
    discriminate_linestyles : bool, optional
        Whether to use different linestyles for each data series.
        Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If legend_list length does not match number of output pickle
        files.
    NoPlotDataError
        If no data was plotted for any of the input files.
    """
    # Load in the FEM file
    if isinstance(output_pickle_names, str):
        output_pickle_names = [output_pickle_names]

    # Check if the axes are provided, if not create a new figure
    plt_show = ax is None
    if ax is None:
        _, ax = plt.subplots()

    # Ensure the legends passed are the same length as the extractors
    if legend_list is not None and len(legend_list) != len(output_pickle_names):
        raise ValueError(
            "The number of legends must match the number of output pickle files. "
            f"Got {len(legend_list)} legends for {len(output_pickle_names)} output pickle files."
        )

    # Iterate over each output pickle file
    for idx, output_pickle_name in enumerate(output_pickle_names):
        input_pickle_obj = input_pickle(output_pickle_name)

        # Check for required variables
        for var in required_variables:
            assert var in input_pickle_obj, f"{var} not found in the file"

        # Keep track of whether any data was plotted for this file
        plotted_count = 0
        # Plot each data series
        for i, extractor in enumerate(extractors):
            x_data, y_data = extractor(input_pickle_obj)
            if x_data is None or y_data is None:
                continue
            # Skip if either x_data or y_data is empty
            plotted_count += 1

            label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
            # In case of multiple pickle files, append the name to the label
            if len(output_pickle_names) > 1 or legend_list is not None:
                unique_label = output_pickle_name
                # It will always be the same length as the output_pickle_names, just index the legends
                if legend_list is not None:
                    unique_label = legend_list[idx]
                label += f" ({unique_label})"

            if discriminate_linestyles:
                ls_index = idx if len(output_pickle_names) > 1 else i
                linestyle = linestyle_list[ls_index % len(linestyle_list)]
            else:
                linestyle = "-"

            # Plot the data
            ax.plot(x_data, y_data, label=label, linestyle=linestyle)

        if plotted_count == 0:
            raise NoPlotDataError(
                f"No data was plotted for file '{output_pickle_name}'. "
            )

    # Set labels and styling
    if grid:
        ax.grid()

    if show_legend and labels:
        ax.legend()

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or f"Plot for {output_pickle_names}", fontsize=18)

    # Set title and show plot if needed
    if plt_show:
        plt.show()


# -----------------------------------------------------------
# Helper classes and functions for extractors
# -----------------------------------------------------------
class RunningSum:
    """Callable that maintains a running sum of values returned by inc_fn.

    Parameters
    ----------
    inc_fn : callable
        Function that takes (sim, step) and returns a value to add to
        the running sum.
    """

    def __init__(self, inc_fn: Callable) -> None:
        """Initialize the running sum with the increment function."""
        self.inc_fn = inc_fn
        self.total = 0.0

    def __call__(self, sim: "Simulation", step: int) -> float:
        """Update and return the running sum for the given step."""
        self.total += _scalar_float(self.inc_fn(sim, step))
        return self.total


def _scalar_float(x: Array) -> float:
    """Convert a scalar-like JAX array to a Python float.

    Parameters
    ----------
    x : Array
        A JAX array containing a single value.

    Returns
    -------
    float
        The Python float value of the array.

    Raises
    ------
    TypeError
        If the input array does not contain exactly one element.
    """
    x = jnp.asarray(x)
    if x.size != 1:
        raise TypeError(
            f"Expected scalar-like value, got shape={x.shape}, size={x.size}"
        )
    return float(jnp.reshape(x, ()))  # reshape to 0-D scalar


def _ys_force_fd_factory() -> Dict[str, Callable]:
    """Create a dict of series callables for force-displacement plotting.

    Returns
    -------
    dict of str to callable
        Dictionary mapping "Force" to a RunningSum callable.
    """
    return {"Force": RunningSum(lambda sim, _: sim.F_incremental[sim.target_dof])}


def _ys_internal_force_fd_factory() -> Dict[str, Callable]:
    """Create a dict of series callables for internal force-displacement plotting.

    Returns
    -------
    dict of str to callable
        Dictionary mapping "Internal force" to a RunningSum callable.
    """
    return {
        "Internal force": RunningSum(
            lambda sim, _: sim.internal_forces_incremental[sim.target_dof]
        )
    }


def _ys_energy_plus_work_factory() -> Dict[str, Callable]:
    """Create a dict of series callables for energy and work plotting.

    Returns
    -------
    dict of str to callable
        Dictionary mapping "Elastic energy", "Fracture energy", and
        "External work" to their respective callables.

    Notes
    -----
    The work_increment function adapts to the calling context:
    - Live streaming (_SimAdapter): displacement_incremental is available
      directly on the sim object.
    - Offline replay (_StepView): displacement_incremental may not be in
      recorded_values, so it falls back to differencing displacement
      values via hist().
    """

    def work_increment(sim: "Simulation", step: int) -> float:
        """Calculate incremental work for the current step.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        step : int
            The current step index.

        Returns
        -------
        float
            The incremental work at this step.
        """
        f = jnp.ravel(sim.F_incremental)

        if hasattr(sim, "displacement_incremental"):
            # Live streaming path: delta is already computed on the sim object.
            du = jnp.ravel(sim.displacement_incremental)
        else:
            # Offline replay path: reconstruct delta from history.
            if step == 0:
                du = jnp.zeros_like(jnp.ravel(sim.hist("displacement", 0)))
            else:
                du = jnp.ravel(sim.hist("displacement", step)) - jnp.ravel(
                    sim.hist("displacement", step - 1)
                )

        return jnp.sum(f * du)

    return {
        "Elastic energy": lambda sim, _: float(jnp.sum(sim.ip_strain_energy)),
        "Fracture energy": lambda sim, _: float(jnp.sum(sim.ip_fracture_energy)),
        "External work": RunningSum(work_increment),
    }


# -----------------------------------------------------------
# Step view class and related functions
# -----------------------------------------------------------
class _StepView:
    """Adapter exposing ip[key][step] as attributes with target_dof.

    Parameters
    ----------
    ip : dict
        Dictionary containing simulation data keyed by variable name.
    step : int
        The current step index.
    target_dof : int
        The target degree of freedom.
    """

    def __init__(self, ip: Dict, step: int, target_dof: int) -> None:
        """Initialize the step view with simulation data."""
        self._ip = ip
        self._step = int(step)
        self.target_dof = int(target_dof)

    def __getattr__(self, name: str) -> Array:
        """Get ip[name] at the current step."""
        try:
            return self._ip[name][self._step]
        except KeyError:
            # Must raise AttributeError so that hasattr() returns False cleanly
            raise AttributeError(
                f"_StepView has no attribute '{name}' " f"(key not found in ip dict)"
            ) from None

    def hist(self, name: str, step: int) -> Array:
        """Access ip[name] at an arbitrary step.

        Parameters
        ----------
        name : str
            The variable name to look up.
        step : int
            The step index to access.

        Returns
        -------
        Array
            The value of ip[name] at the specified step.
        """
        try:
            return self._ip[name][int(step)]
        except KeyError:
            raise AttributeError(
                f"_StepView.hist: key '{name}' not found in ip dict"
            ) from None


def _infer_target_dof(ip: Dict) -> int:
    """Infer the target degree of freedom from simulation data.

    Parameters
    ----------
    ip : dict
        Dictionary containing simulation data including F_incremental.

    Returns
    -------
    int
        The index of the first non-zero entry in F_incremental[0].
    """
    # Same heuristic you already used in your old extractors.
    return int(jnp.where(ip["F_incremental"][0] != 0)[0][0])


def _get_ys_dict(spec: Dict) -> Dict[str, Callable]:
    """Return a fresh dict of series callables.

    Parameters
    ----------
    spec : dict
        Specification dict containing either "ys_factory" or "ys" key.

    Returns
    -------
    dict of str to callable
        Dictionary mapping series names to their callables.

    Raises
    ------
    TypeError
        If ys_factory() does not return a dict.
    ValueError
        If spec defines neither "ys" nor "ys_factory".
    """
    if spec.get("ys_factory") is not None:
        ys = spec["ys_factory"]()
        if not isinstance(ys, dict):
            raise TypeError("ys_factory() must return dict[str, callable]")
        return ys
    ys = spec.get("ys")
    if ys is None:
        raise ValueError("Spec must define either `ys` or `ys_factory`.")
    return ys


def replay_spec_series(
    ip: Dict,
    spec: Dict,
    *,
    target_dof: Optional[int] = None,
    nsteps: Optional[int] = None,
) -> Dict[str, Array]:
    """Run per-step spec over pickle history to build full series.

    Parameters
    ----------
    ip : dict
        Input pickle dictionary containing simulation history.
    spec : dict
        Specification dict with "x" and "ys_factory" or "ys" keys.
    target_dof : int, optional
        Target degree of freedom. Inferred if not provided and
        needs_target_dof is True.
    nsteps : int, optional
        Number of steps to replay. Defaults to length of displacement
        array in ip.

    Returns
    -------
    dict of str to Array
        Dictionary mapping series names to arrays, plus "x" for x-values.

    Raises
    ------
    KeyError
        If a required key is not found in the pickle.
    ValueError
        If a required key is empty or has fewer steps than expected.
    """
    nsteps = nsteps or len(ip.get("displacement", []))
    for k in spec.get("requires", []):
        if k not in ip:
            raise KeyError(f"Required key '{k}' not found in pickle.")
        if len(ip[k]) == 0:
            raise ValueError(f"Required key '{k}' exists but is empty.")
        if len(ip[k]) < nsteps:
            raise ValueError(
                f"Required key '{k}' has {len(ip[k])} steps, "
                f"but {nsteps} are expected."
            )

    if spec.get("needs_target_dof", False):
        if target_dof is None:
            target_dof = _infer_target_dof(ip)  # or raise with a good message
    else:
        target_dof = 0 if target_dof is None else target_dof

    if nsteps is None:
        nsteps = len(ip["displacement"])

    ys = _get_ys_dict(spec)  # fresh state per replay

    xs = []
    out = {name: [] for name in ys.keys()}

    for step in range(nsteps):
        sim = _StepView(ip, step, target_dof)
        xs.append(_scalar_float(spec["x"](sim, step)))
        for name, fn in ys.items():
            out[name].append(_scalar_float(fn(sim, step)))

    out = {k: jnp.asarray(v) for k, v in out.items()}
    out["x"] = jnp.asarray(xs)
    return out


def make_offline_extractors_from_metadata(
    plot_fn: Callable, *, target_dof: Optional[int] = None
) -> Tuple[List[Callable], List[str], List[str]]:
    """Build plot_output_data-compatible extractors from plot_fn metadata.

    Parameters
    ----------
    plot_fn : callable
        A plotting function with a __stream_spec__ attribute.
    target_dof : int, optional
        Target degree of freedom for the simulation.

    Returns
    -------
    extractors : list of callable
        List of functions that take an input_pickle and return
        (x_data, y_data).
    required_variables : list of str
        List of variable names required for the plot.
    labels : list of str
        List of labels for each data series.
    """
    spec = plot_fn.__stream_spec__
    required_variables = spec.get("requires", [])

    # Determine series names (fresh ys dict, but only for keys)
    series_names = list(_get_ys_dict(spec).keys())

    # Replay once per ip dict and cache results (avoid re-replaying per series)
    cache: dict[int, dict[str, Array]] = {}

    def _get(ip_obj: Dict) -> Optional[Dict[str, Array]]:
        """Get replayed series data for the given input pickle, with caching."""
        key = id(ip_obj)
        if key not in cache:
            try:
                cache[key] = replay_spec_series(ip_obj, spec, target_dof=target_dof)
            except (AttributeError, IndexError, KeyError, ValueError) as e:
                logging.warning(f"Skipping plot '{plot_fn.__name__}': {e}")
                cache[key] = None
        return cache[key]

    extractors = []
    for name in series_names:
        # Bind loop var now; otherwise all lambdas would use the last name. [web:149]
        def ex(
            ip_obj: Dict, name: str = name
        ) -> Tuple[Optional[Array], Optional[Array]]:
            """Extractor that returns (x, y) for the given series name."""
            s = _get(ip_obj)
            # Silently skip if replay failed (e.g. missing keys, empty data, etc.)
            if s is None:
                return None, None
            return s["x"], s[name]

        extractors.append(ex)

    return extractors, required_variables, series_names


# -----------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------
@collect_plot_data
@stream_spec(
    x=lambda sim, _: float(sim.displacement[sim.target_dof]),
    ys_factory=_ys_force_fd_factory,
    requires=["displacement", "F_incremental"],
    needs_target_dof=True,
)
def plot_force_displacement(
    output_pickle_name: str,
    target_dof: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    legend_list: Optional[List[str]] = None,
    discriminate_linestyles: bool = False,
) -> plt.Axes:
    """Plot the force-displacement curve for a FEM simulation.

    Parameters
    ----------
    output_pickle_name : str
        Name of the pickle file containing FEM simulation data.
    target_dof : int, optional
        Target degree of freedom. Inferred from data if not provided.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    title : str, optional
        Title of the plot. If None, a default title is used.
    legend_list : list of str, optional
        Custom legend labels for each data series.
    discriminate_linestyles : bool, optional
        Whether to use different linestyles for each data series.
        Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    extractors, required_variables, labels = make_offline_extractors_from_metadata(
        plot_force_displacement,
        target_dof=target_dof,
    )

    plot_output_data(
        output_pickle_name,
        extractors,
        required_variables,
        labels=labels,
        xlabel="Displacement",
        ylabel="Force",
        ax=ax,
        title=title,
        legend_list=legend_list,
        discriminate_linestyles=discriminate_linestyles,
    )
    return ax


@collect_plot_data
@stream_spec(
    x=lambda sim, _: float(sim.displacement[sim.target_dof]),
    ys_factory=_ys_internal_force_fd_factory,
    requires=["displacement", "internal_forces_incremental"],
    needs_target_dof=True,
)
def plot_internal_force_displacement(
    output_pickle_name: str,
    target_dof: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    legend_list: Optional[List[str]] = None,
    discriminate_linestyles: bool = False,
) -> plt.Axes:
    """Plot the internal force-displacement curve for a FEM simulation.

    Parameters
    ----------
    output_pickle_name : str
        Name of the pickle file containing FEM simulation data.
    target_dof : int, optional
        Target degree of freedom. Inferred from data if not provided.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    title : str, optional
        Title of the plot. If None, a default title is used.
    legend_list : list of str, optional
        Custom legend labels for each data series.
    discriminate_linestyles : bool, optional
        Whether to use different linestyles for each data series.
        Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    extractors, required_variables, labels = make_offline_extractors_from_metadata(
        plot_internal_force_displacement,
        target_dof=target_dof,
    )

    plot_output_data(
        output_pickle_name,
        extractors,
        required_variables,
        labels=labels,  # ["Internal force"]
        xlabel="Displacement",
        ylabel="Force",
        ax=ax,
        title=title,
        legend_list=legend_list,
        discriminate_linestyles=discriminate_linestyles,
    )
    return ax


@collect_plot_data
@stream_spec(
    x=lambda sim, step: float(step),
    ys={
        "Elastic energy": lambda sim, _: float(jnp.sum(sim.ip_strain_energy)),
        "Fracture energy": lambda sim, _: float(jnp.sum(sim.ip_fracture_energy)),
    },
    requires=["ip_strain_energy", "ip_fracture_energy"],
)
def plot_energy(
    output_pickle_name: Union[str, List[str]],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    legend_list: Optional[List[str]] = None,
    discriminate_linestyles: bool = False,
) -> plt.Axes:
    """Plot elastic and fracture energy curves for a FEM simulation.

    Parameters
    ----------
    output_pickle_name : str or list of str
        Name(s) of the pickle file(s) containing FEM simulation data.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    title : str, optional
        Title of the plot. If None, a default title is used.
    legend_list : list of str, optional
        Custom legend labels for each data series.
    discriminate_linestyles : bool, optional
        Whether to use different linestyles for each data series.
        Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    extractors, required_variables, labels = make_offline_extractors_from_metadata(
        plot_energy
    )
    plot_output_data(
        output_pickle_name,
        extractors,
        required_variables,
        labels=labels,
        xlabel="Displacement",
        ylabel="Energy Values",
        ax=ax,
        title=title,
        legend_list=legend_list,
        discriminate_linestyles=discriminate_linestyles,
    )
    return ax


@collect_plot_data
@stream_spec(
    x=lambda sim, step: float(step),
    ys_factory=_ys_energy_plus_work_factory,
    requires=[
        "ip_strain_energy",
        "ip_fracture_energy",
        "F_incremental",
        "displacement",
    ],
    needs_target_dof=False,
)
def plot_energy_plus_work(
    output_pickle_name: Union[str, List[str]],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    legend_list: Optional[List[str]] = None,
    discriminate_linestyles: bool = False,
) -> plt.Axes:
    """Plot elastic energy, fracture energy, and external work curves.

    Parameters
    ----------
    output_pickle_name : str or list of str
        Name(s) of the pickle file(s) containing FEM simulation data.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    title : str, optional
        Title of the plot. If None, a default title is used.
    legend_list : list of str, optional
        Custom legend labels for each data series.
    discriminate_linestyles : bool, optional
        Whether to use different linestyles for each data series.
        Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    extractors, required_variables, labels = make_offline_extractors_from_metadata(
        plot_energy_plus_work
    )

    plot_output_data(
        output_pickle_name,
        extractors,
        required_variables,
        labels=labels,  # ["Elastic energy","Fracture energy","External work"]
        xlabel="Step",
        ylabel="Energy / Work",
        ax=ax,
        title=title,
        legend_list=legend_list,
        discriminate_linestyles=discriminate_linestyles,
    )
    return ax


# skylos: ignore-start
@collect_plot_data
def plot_comparison_force_displacement_and_energy(
    filename1: str,
    filename2: str,
    title1: str,
    title2: str,
    figsize: Tuple[int, int] = (12, 12),
    synchronize_y: bool = True,
) -> None:
    """Plot force-displacement and energy curves for two FEM simulations.

    Create a 2x2 subplot comparing force-displacement and energy curves
    for two different simulations.

    Parameters
    ----------
    filename1 : str
        Path to the first pickle file.
    filename2 : str
        Path to the second pickle file.
    title1 : str
        Title for the first simulation plots.
    title2 : str
        Title for the second simulation plots.
    figsize : tuple of int, optional
        Figure size as (width, height). Default is (12, 12).
    synchronize_y : bool, optional
        Whether to synchronize y-axis limits across corresponding
        plots. Default is True.

    Notes
    -----
    This is a debugging utility. The left column shows force-displacement
    curves and the right column shows energy curves.
    """
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    plot_force_displacement(
        filename1,
        ax=ax[0, 0],
        title=title1,
    )
    plot_energy(filename1, ax=ax[0, 1], title=title1)
    plot_force_displacement(
        filename2,
        ax=ax[1, 0],
        title=title2,
    )
    plot_energy(filename2, ax=ax[1, 1], title=title2)

    if synchronize_y:
        # Synchronize y-axis for force-displacement plots (left column: [0,0] and [1,0])
        force_ylims = [ax[0, 0].get_ylim(), ax[1, 0].get_ylim()]
        force_ymin = min([ylim[0] for ylim in force_ylims])
        force_ymax = max([ylim[1] for ylim in force_ylims])
        ax[0, 0].set_ylim(force_ymin, force_ymax)
        ax[1, 0].set_ylim(force_ymin, force_ymax)

        # Synchronize y-axis for energy plots (right column: [0,1] and [1,1])
        energy_ylims = [ax[0, 1].get_ylim(), ax[1, 1].get_ylim()]
        energy_ymin = min([ylim[0] for ylim in energy_ylims])
        energy_ymax = max([ylim[1] for ylim in energy_ylims])
        ax[0, 1].set_ylim(energy_ymin, energy_ymax)
        ax[1, 1].set_ylim(energy_ymin, energy_ymax)


# skylos: ignore-end


_mod = sys.modules[__name__]
plot_fns = {
    name: func
    for name, func in inspect.getmembers(_mod, inspect.isfunction)
    if name.startswith("plot_")
}


def get_plot_function(name: str) -> Callable:
    """Return the plotting function corresponding to the given name.

    Parameters
    ----------
    name : str
        Name of the plot function (must start with "plot_").

    Returns
    -------
    callable
        The requested plotting function.

    Raises
    ------
    ValueError
        If no plot function with the given name exists.
    """
    # Try to match the key and return the parameters
    try:
        return plot_fns[name]
    except KeyError:
        raise ValueError(
            f"Plot function {name} not found. Available plot functions: {list(plot_fns.keys())}."
        )
