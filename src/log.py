"""Log set up and handling utilities.

This module provides logging configuration and utilities for the application,
including custom log levels, prefix filtering, and timing messages.
"""

import contextvars
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional

import jax

from json_encoder import format_json_string

TIMING_LEVEL_NUM = 15
log_separator = "-" * 50

logging.addLevelName(TIMING_LEVEL_NUM, "TIMING")


def timing(self: object, message: str, *args, **kwargs) -> None:
    """Log a timing message.

    Parameters
    ----------
    self : object
        Logger instance.
    message : str
        Message to log.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.
    """
    if self.isEnabledFor(TIMING_LEVEL_NUM):
        self._log(TIMING_LEVEL_NUM, message, args, **kws)


def setup_logging(
    title: str,
    config: Dict,
    debug: bool = False,
    keys_to_remove: Optional[List[str]] = None,
) -> None:
    """Set up the logging configuration for the application.

    Parameters
    ----------
    title : str
        Title for the log file.
    config : Dict
        Configuration dictionary.
    debug : bool, optional
        Whether to use debug level logging. Default is False.
    keys_to_remove : Optional[List[str]], optional
        Keys to remove from config before logging.
    """
    logfile_name = f"{title}.log"
    log_format = "%(asctime)s - %(levelname)s %(prefix)s- %(message)s"

    log_level = logging.DEBUG if debug else TIMING_LEVEL_NUM

    if config.output_parameters.save_logs:
        current_path = Path(__file__).parent.resolve()
        log_path = current_path.parent / "output" / "log" / logfile_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=log_level,
            filename=str(log_path),
            filemode="w",
            format=log_format,
            force=True,
        )
    else:
        logging.basicConfig(level=log_level, format=log_format, force=True)

    # Attach the filter to the root logger
    logging.getLogger().addFilter(PrefixFilter())

    logging.info(log_separator)
    logging.info(f"Using {jax.devices()}")
    logging.info("Input dictionary used:")
    # Remove the keys that are not needed for the logging
    input_dict_sanitised = copy.deepcopy(config)
    if keys_to_remove is not None:
        for key in keys_to_remove:
            input_dict_sanitised.pop(key)
    # Dump the formatted JSON to the log
    logging.info(format_json_string(input_dict_sanitised))
    logging.info(log_separator)


# Set up the custom prefix filter
_current_prefix = contextvars.ContextVar("prefix", default="")


class PrefixFilter(logging.Filter):
    """Filter to add the current prefix to log records.

    Attributes
    ----------
    prefix : str
        Prefix to add to log records.
    """

    def filter(self, record: object) -> bool:
        """Set the prefix attribute on the log record.

        Parameters
        ----------
        record : object
            Log record to filter.

        Returns
        -------
        bool
            Always returns True.
        """
        opt = _current_prefix.get()
        record.prefix = f"[{opt}] " if opt else ""
        return True


def set_prefix(name: str) -> None:
    """Set the current prefix label for subsequent logs."""
    _current_prefix.set(name)


def clear_prefix() -> None:
    """Clear the current prefix label."""
    _current_prefix.set("")


# Global record factory to guarantee the field exists on ALL records
_orig_factory = logging.getLogRecordFactory()


def _prefix_record_factory(*args, **kwargs) -> logging.LogRecord:
    record = _orig_factory(*args, **kwargs)
    # Only set if not already present
    if not hasattr(record, "prefix"):
        opt = _current_prefix.get()
        record.prefix = f"[{opt}] " if opt else ""
    return record


# Set the global record factory
logging.setLogRecordFactory(_prefix_record_factory)
