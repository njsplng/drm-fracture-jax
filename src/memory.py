"""Memory profiling utilities.

This module provides utilities for monitoring and logging memory usage
including host process memory and JAX device memory statistics.
"""

import gc
import logging
import os
from typing import Optional

import jax
import psutil

from log import clear_prefix, set_prefix


def _proc_mem_mb() -> float:
    """Return current process memory usage in MB.

    Returns
    -------
    float
        Process memory usage in megabytes (USS on Linux/macOS, RSS otherwise).
    """
    # Get the current process
    proc = psutil.Process(os.getpid())

    # USS is the most honest “unique” memory on Linux/macOS; fall back to RSS otherwise
    try:
        return proc.memory_full_info().uss / 1024**2
    except Exception:
        return proc.memory_info().rss / 1024**2


def _device_mem_mb() -> list[tuple[str, Optional[float], Optional[float], dict]]:
    """Return per-device memory usage as a list of tuples.

    Returns
    -------
    list[tuple[str, Optional[float], Optional[float], dict]]
        List of tuples containing (device_name, used_mb, total_mb, stats_dict).
    """
    try:
        backend = jax.lib.xla_bridge.get_backend()
        out = []
        for d in backend.devices():
            try:
                ms = d.memory_stats()
                used = ms.get("bytes_in_use", 0) / 1024**2
                total = ms.get("bytes_limit", 0) / 1024**2
                out.append((str(d), used, total, ms))
            except Exception:
                out.append((str(d), None, None, {}))
        return out
    except Exception:
        return []


def log_mem(prefix: str = "", *, cleanup: bool = False) -> None:
    """Log current host and per-device memory at this instant.

    Parameters
    ----------
    prefix : str, optional
        Prefix for the log message. Default is "".
    cleanup : bool, optional
        Whether to run garbage collection and clear JAX caches. Default is False.
    """
    set_prefix(f"memory {prefix}")
    # If cleanup is requested, run garbage collection and clear JAX caches
    if cleanup:
        gc.collect()
        jax.clear_caches()

    # Get the current memory usage
    host = _proc_mem_mb()
    devs = _device_mem_mb()

    # Log the host stats
    logging.info(f" Host: {host:.1f} MB")

    # If possible, log the device stats
    for name, used, total, ms in devs:
        if used is None:
            continue
        else:
            peak = ms.get("peak_bytes_in_use")
            largest_free = ms.get("largest_free_block_bytes")
            line = f"{name}: {used:.1f} / {total:.1f} MB"
            if peak is not None:
                line += f", peak {peak/1024**2:.1f} MB"
            if largest_free is not None:
                line += f", largest free block {largest_free/1024**2:.1f} MB"
            logging.info(line)
    clear_prefix()
