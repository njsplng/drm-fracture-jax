"""Wrapper for plotting functions to save captured plot data."""

import pathlib
from contextlib import ContextDecorator
from functools import wraps
from typing import Callable, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array

from io_handlers import dump_blosc

# Global registry: function_name -> captured dict
_PLOT_DATA_BY_FUNC: dict[str, dict[str, object]] = {}


def _series_key(idx: int, label: Optional[str]) -> str:
    """Generate a key for a series based on its label or index.

    Parameters
    ----------
    idx : int
        Index of the series (used when label is empty or None).
    label : str or None
        Label of the series. If empty or None, falls back to index.

    Returns
    -------
    str
        The series key (label if available, otherwise "yN").
    """
    if label and str(label).strip():
        return str(label).strip()
    return f"y{idx+1}"


def _is_arraylike(a: object) -> bool:
    """Check if `a` is array-like (has __len__), excluding strings/bytes."""
    # Treat strings/bytes as format specifiers, not data
    if isinstance(a, (str, bytes)):
        return False
    return hasattr(a, "__len__")


def _parse_plot_args(args: tuple[object, ...]) -> list[tuple[Array, Array]]:
    """Parse matplotlib plot args into a list of (x, y) pairs.

    Supports:
        plot(y)
        plot(x, y)
        plot(x, y, fmt)
        plot(x1, y1, fmt1, x2, y2, fmt2, ...)
        plot(y1, y2, ...)

    Format strings (e.g., "k--") are ignored.

    Parameters
    ----------
    args : tuple of objects
        Positional arguments passed to matplotlib plot.

    Returns
    -------
    list of tuples
        List of (x, y) pairs parsed from the arguments.
    """
    pairs = []
    i = 0
    n = len(args)

    while i < n:
        a = args[i]

        if not _is_arraylike(a):
            # Skip non-arraylike (e.g., fmt before any data)
            i += 1
            continue

        # Case A: plot(y)
        if i + 1 >= n or not _is_arraylike(args[i + 1]):
            y = jnp.asarray(a)
            x = jnp.arange(y.shape[0])
            pairs.append((x, y))
            i += 1
            # If next is fmt string, skip it
            if i < n and isinstance(args[i], (str, bytes)):
                i += 1
            continue

        # Case B: plot(x, y) [optionally followed by fmt]
        x = jnp.asarray(a)
        y_candidate = args[i + 1]
        if _is_arraylike(y_candidate):
            y = jnp.asarray(y_candidate)
            pairs.append((x, y))
            i += 2
            # Skip optional fmt
            if i < n and isinstance(args[i], (str, bytes)):
                i += 1
            continue

        # Fallback: advance to avoid infinite loops
        i += 1

    return pairs


class _PlotTracker(ContextDecorator):
    """Context manager that temporarily wraps plt functions to capture series.

    Wraps plt.plot and plt.Axes.plot to intercept and store plotted data
    during the context block, then restores the original functions on exit.
    """

    def __init__(self) -> None:
        """Initialize the plot tracker with empty state."""
        self._orig_plt_plot: Optional[Callable] = None
        self._orig_axes_plot: Optional[Callable] = None
        self._series: list[dict[str, object]] = []

    def __enter__(self) -> "self":
        """Enter the context and wrap plt.plot functions."""
        self._orig_plt_plot = plt.plot
        self._orig_axes_plot = plt.Axes.plot

        def _wrap_function(original_func: Callable) -> Callable:
            """Wrap plt.plot function."""

            @wraps(original_func)
            def wrapped(*args, **kwargs) -> object:
                """Wrapped plt.plot function to capture data."""
                label = kwargs.get("label", None)
                for x, y in _parse_plot_args(args):
                    self._series.append({"x": x, "y": y, "label": label})
                return original_func(*args, **kwargs)

            return wrapped

        def _wrap_method(original_method: Callable) -> Callable:
            """Wrap Axes.plot method."""

            @wraps(original_method)
            def wrapped(self_ax: object, *args, **kwargs) -> object:
                """Wrapped Axes.plot method to capture data."""
                label = kwargs.get("label", None)
                for x, y in _parse_plot_args(args):
                    self._series.append({"x": x, "y": y, "label": label})
                return original_method(self_ax, *args, **kwargs)

            return wrapped

        plt.plot = _wrap_function(plt.plot)
        plt.Axes.plot = _wrap_method(plt.Axes.plot)
        return self

    def __exit__(self, *exc: object) -> bool:
        """Exit the context and restore original plt functions."""
        if self._orig_plt_plot is not None:
            plt.plot = self._orig_plt_plot
        if self._orig_axes_plot is not None:
            plt.Axes.plot = self._orig_axes_plot
        return False

    @property
    def series(self) -> list[dict[str, object]]:
        """Return the list of captured series.

        Returns
        -------
        list of dicts
            Each dict contains "x", "y", and "label" keys.
        """
        return self._series


def _series_list_to_function_dict(
    series_list: list[dict[str, object]],
) -> dict[str, object]:
    """Convert captured series to a tidy dictionary.

    If all x arrays are equal (allclose), output
    {"x": x, "<label or y1>": y, ...}; otherwise emit
    per-series x/y pairs.

    Parameters
    ----------
    series_list : list of dicts
        List of captured series, each with "x", "y", and "label" keys.

    Returns
    -------
    dict
        Tidy dictionary with shared x if all x arrays are equal,
        otherwise per-series x/y pairs.
    """
    if not series_list:
        return {}

    # Normalize arrays
    xs = [jnp.asarray(s["x"]) for s in series_list]
    ys = [jnp.asarray(s["y"]) for s in series_list]
    labels = [s.get("label") for s in series_list]

    # Check if all x arrays are equal
    all_equal_x = True
    base_x = xs[0]
    for x in xs[1:]:
        if base_x.shape != x.shape or not jnp.allclose(base_x, x):
            all_equal_x = False
            break

    out: dict[str, object] = {}
    if all_equal_x:
        out["x"] = base_x
        for i, (y, lab) in enumerate(zip(ys, labels)):
            key = _series_key(i, lab)
            out[key] = y
    else:
        for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
            key = _series_key(i, lab)
            out[f"{key}_x"] = x
            out[f"{key}_y"] = y
    return out


def collect_plot_data(func: Callable) -> Callable:
    """Decorator that captures series plotted within func and stores them.

    Parameters
    ----------
    func : callable
        The plotting function to decorate.

    Returns
    -------
    callable
        Wrapped function that captures plot data.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> object:
        """Wrapper function to collect plot data."""
        func_name = func.__name__
        with _PlotTracker() as tracker:
            result = func(*args, **kwargs)
        _PLOT_DATA_BY_FUNC[func_name] = _series_list_to_function_dict(tracker.series)
        return result

    return wrapper


def dump_plot_data(title: str) -> None:
    """Dump all captured plot data to files.

    Parameters
    ----------
    title : str
        Title prefix for the output files.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    root = current_path.parent
    output_dir = root / "output"
    data_dir = output_dir / "plot_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for func_name, data in _PLOT_DATA_BY_FUNC.items():
        filename = data_dir / f"{title}__{func_name}.dat"
        dump_blosc(data, filename)


def stream_spec(
    *,
    x: Array,
    ys: Optional[Array] = None,
    ys_factory: Optional[Callable] = None,
    requires: tuple = (),
    needs_target_dof: bool = False,
) -> Callable:
    """Decorator that attaches streaming metadata to a function.

    Exactly one of `ys` or `ys_factory` must be provided.

    Parameters
    ----------
    x : Array
        X values for the stream.
    ys : Array, optional
        Y values or callable. Must provide either ys or ys_factory.
    ys_factory : callable, optional
        Factory function returning Y values. Must provide either
        ys or ys_factory.
    requires : tuple, optional
        Tuple of required dependencies. Default is empty.
    needs_target_dof : bool, optional
        Whether target degrees of freedom are needed. Default is False.

    Returns
    -------
    callable
        Decorator function that attaches stream metadata.

    Raises
    ------
    ValueError
        If neither or both of ys and ys_factory are provided.
    """
    if (ys is None) == (ys_factory is None):
        raise ValueError("Provide exactly one of `ys` or `ys_factory`.")

    def deco(fn: Callable) -> Callable:
        fn.__stream_spec__ = {
            "x": x,
            "ys": ys,
            "ys_factory": ys_factory,
            "requires": list(requires),
            "needs_target_dof": needs_target_dof,
        }
        return fn

    return deco
