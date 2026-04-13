"""Custom JSON encoder for input file formatting.

This module provides a flexible JSON encoder with scientific notation
and customizable formatting options for arrays, indentation, and
number precision.
"""

import json
from decimal import ROUND_DOWN, Decimal, localcontext
from enum import Enum
from typing import Union

from jaxtyping import Array


class ArrayFormat(Enum):
    """Enum for different array formatting styles.

    Attributes
    ----------
    ALWAYS_MULTILINE : str
        Every array on multiple lines.
    COMPACT : str
        All arrays on single line.
    SMART : str
        Based on length/complexity.
    THRESHOLD : str
        Based on item count.
    HYBRID : str
        Hybrid format.
    """

    # Every array on multiple lines
    ALWAYS_MULTILINE = "always_multiline"
    # All arrays on single line
    COMPACT = "compact"
    # Based on length/complexity
    SMART = "smart"
    # Based on item count
    THRESHOLD = "threshold"
    # Hybrid format
    HYBRID = "hybrid"


class ScientificJSONEncoder:
    """A highly flexible JSON encoder with scientific notation.

    This encoder provides full control over number formatting with
    configurable scientific notation thresholds, array/list formatting
    (single line, multi-line, smart formatting), indentation, spacing,
    and special handling for different data types.

    Parameters
    ----------
    lo : float, optional
        Lower threshold for scientific notation. Default is 0.01.
    hi : float, optional
        Upper threshold for scientific notation. Default is 1000.
    precision : int, optional
        Decimal precision for scientific notation. Default is 12.
    indent : int, optional
        Number of spaces for indentation. Default is 4.
    array_format : ArrayFormat, optional
        How to format arrays. Default is ALWAYS_MULTILINE.
    array_threshold : int, optional
        Max items before multiline (for THRESHOLD mode). Default is 5.
    array_wrap_length : int, optional
        Max line length (for SMART mode). Default is 80.
    compact : bool, optional
        Remove spaces after colons and commas. Default is False.
    trailing_commas : bool, optional
        Add trailing commas (non-standard JSON). Default is False.
    blank_lines_between_top_level : bool, optional
        Add blank lines between top-level entries. Default is False.
    sort_keys : bool, optional
        Sort dictionary keys alphabetically. Default is False.
    ensure_ascii : bool, optional
        Do not escape non-ASCII characters. Default is False.
    """

    def __init__(
        self,
        *,
        lo: float = 0.01,
        hi: float = 1000,
        precision: int = 12,
        # Formatting settings
        indent: int = 4,
        array_format: ArrayFormat = ArrayFormat.ALWAYS_MULTILINE,
        # For THRESHOLD mode: max items before multiline
        array_threshold: int = 5,
        # For SMART mode: max line length
        array_wrap_length: int = 80,
        # Spacing settings
        # Remove spaces after : and ,
        compact: bool = False,
        # Add trailing commas (non-standard JSON)
        trailing_commas: bool = False,
        # Add blank lines between top-level entries
        blank_lines_between_top_level: bool = False,
        # Special formatting
        # Sort dictionary keys alphabetically
        sort_keys: bool = False,
        # Do not escape non-ASCII characters
        ensure_ascii: bool = False,
    ) -> None:
        self.lo = lo
        self.hi = hi
        self.precision = precision
        self.fmt = f".{precision}e"

        self.indent = indent
        self.array_format = array_format
        self.array_threshold = array_threshold
        self.array_wrap_length = array_wrap_length

        self.compact = compact
        self.trailing_commas = trailing_commas
        self.blank_lines_between_top_level = blank_lines_between_top_level

        self.sort_keys = sort_keys
        self.ensure_ascii = ensure_ascii

        # Precompute spacing strings for separators
        self.colon_separator = ":" if compact else ": "
        self.comma_separator = "," if compact else ", "

    def encode(self, obj: object) -> str:
        """Encode a Python object to a formatted JSON string.

        Parameters
        ----------
        obj : object
            Python object to encode.

        Returns
        -------
        str
            Formatted JSON string.
        """
        return self._encode_value(obj, 0)

    def _encode_value(self, obj: object, level: int) -> str:
        """Recursively encode a value with proper formatting.

        Parameters
        ----------
        obj : object
            Value to encode.
        level : int
            Current indentation level.

        Returns
        -------
        str
            Encoded string representation.
        """
        if obj is None:
            return "null"
        elif obj is True:
            return "true"
        elif obj is False:
            return "false"
        elif isinstance(obj, str):
            return self._encode_string(obj)
        elif isinstance(obj, int):
            return str(obj)
        elif isinstance(obj, float):
            return self._format_number(obj)
        elif isinstance(obj, dict):
            return self._encode_dict(obj, level)
        elif isinstance(obj, (list, tuple)):
            return self._encode_array(obj, level)
        elif isinstance(obj, Array) and obj.ndim == 0:
            return self._format_number(float(obj))
        elif isinstance(obj, Array) and obj.ndim == 1:
            return self._encode_array(obj.tolist(), level)
        elif isinstance(obj, Array) and obj.ndim > 1:
            return f"{obj.ndim}-dim_array"
        else:
            # Fallback for other types
            return json.dumps(obj, ensure_ascii=self.ensure_ascii)

    def _encode_string(self, s: str) -> str:
        """Encode a string value.

        Parameters
        ----------
        s : str
            String to encode.

        Returns
        -------
        str
            JSON-encoded string.
        """
        return json.dumps(s, ensure_ascii=self.ensure_ascii)

    def _truncate_to_precision(self, val: float) -> str:
        """Truncate a float to the specified precision decimal places.

        Truncate (not round) a float to `self.precision` decimal places.
        Return a compact string with no unnecessary trailing zeros, but
        at least one decimal place.

        Parameters
        ----------
        val : float
            Float value to truncate.

        Returns
        -------
        str
            Truncated string representation.
        """
        with localcontext() as ctx:
            # A little extra precision internally to avoid artifacts
            ctx.prec = max(self.precision + 5, 20)

            # Step size 10^-precision, e.g. 0.001 for precision=3
            quant = Decimal(1).scaleb(-self.precision)

            # Convert via str() to avoid binary FP surprises (e.g. 0.1)
            d = Decimal(str(val)).quantize(quant, rounding=ROUND_DOWN)

            # Fixed-point string (no exponent)
            s = format(d, "f")

        # Strip unnecessary trailing zeros, keep at least one decimal place
        if "." in s:
            s = s.rstrip("0").rstrip(".")
            if "." not in s:
                s += ".0"
        else:
            # Shouldn't really hit this (we formatted as fixed), but be safe
            s += ".0"

        return s

    def _format_number(self, val: Union[int, float]) -> str:
        """Apply scientific notation formatting to a number.

        Parameters
        ----------
        val : Union[int, float]
            Number to format.

        Returns
        -------
        str
            Formatted number string.
        """
        if val == 0.0:
            return "0.0"

        # Convert to float for processing
        val = float(val)

        # Only sci-format if outside [lo, hi)
        if abs(val) < self.lo or abs(val) >= self.hi:
            raw = format(val, self.fmt)
            mantissa, exp = raw.split("e")

            # Strip trailing zeros in mantissa, but keep at least one decimal
            mantissa = mantissa.rstrip("0").rstrip(".")
            if "." not in mantissa:
                mantissa += ".0"

            # Clean exponent: drop "+" and leading zeros
            sign = ""
            if exp[0] in "+-":
                sign, digits = exp[0], exp[1:]
            else:
                digits = exp
            digits = digits.lstrip("0") or "0"

            return f"{mantissa}e{sign}{digits}"
        else:
            # Within normal range: obey precision by truncation (not rounding)
            return self._truncate_to_precision(val)

    def _encode_dict(self, d: dict, level: int) -> str:
        """Encode a dictionary with proper indentation.

        Parameters
        ----------
        d : dict
            Dictionary to encode.
        level : int
            Current indentation level.

        Returns
        -------
        str
            Encoded JSON string.
        """
        if not d:
            return "{}"

        indent_str = " " * (self.indent * level)
        next_indent_str = " " * (self.indent * (level + 1))

        items = []
        keys = sorted(d.keys()) if self.sort_keys else d.keys()

        for key in keys:
            key_str = self._encode_string(key)
            val_str = self._encode_value(d[key], level + 1)
            items.append(f"{next_indent_str}{key_str}{self.colon_separator}{val_str}")

        # Handle trailing comma
        separator = ",\n"
        if self.trailing_commas and items:
            items[-1] += ","
            separator = "\n"

        return "{\n" + separator.join(items) + "\n" + indent_str + "}"

    def _encode_array(self, arr: list, level: int) -> str:
        """Encode an array with configurable formatting.

        Parameters
        ----------
        arr : list
            Array to encode.
        level : int
            Current indentation level.

        Returns
        -------
        str
            Encoded JSON string.
        """
        if not arr:
            return "[]"

        # Determine whether to use multiline based on format setting
        use_multiline = self._should_use_multiline(arr, level)

        if not use_multiline:
            # Single line format
            items = [self._encode_value(item, level + 1) for item in arr]
            return "[" + self.comma_separator.join(items) + "]"
        else:
            # Multiline format
            indent_str = " " * (self.indent * level)
            next_indent_str = " " * (self.indent * (level + 1))

            items = []
            for item in arr:
                val_str = self._encode_value(item, level + 1)
                items.append(f"{next_indent_str}{val_str}")

            # Handle trailing comma
            separator = ",\n"
            if self.trailing_commas:
                items[-1] += ","
                separator = "\n"

            return "[\n" + separator.join(items) + "\n" + indent_str + "]"

    def _should_use_multiline(self, arr: list, level: int) -> bool:
        """Determine whether an array should be formatted on multiple lines.

        Parameters
        ----------
        arr : list
            Array to check.
        level : int
            Current indentation level.

        Returns
        -------
        bool
            True if array should use multiline formatting.
        """
        if self.array_format == ArrayFormat.ALWAYS_MULTILINE:
            return True
        elif self.array_format == ArrayFormat.COMPACT:
            return False
        elif self.array_format == ArrayFormat.THRESHOLD:
            return len(arr) > self.array_threshold
        elif self.array_format == ArrayFormat.SMART:
            # Check if any item is complex (dict or list)
            has_complex = any(isinstance(item, (dict, list)) for item in arr)
            if has_complex:
                return True

            # Don't wrap arrays that contain only integers
            all_ints = all(isinstance(item, int) for item in arr)
            if all_ints:
                return False

            # For mixed simple types or non-integer arrays, check total length
            # Convert all encoded values to strings first
            encoded_items = [str(self._encode_value(item, level + 1)) for item in arr]
            items_str = self.comma_separator.join(encoded_items)
            total_length = len("[") + len(items_str) + len("]") + (self.indent * level)
            return total_length > self.array_wrap_length

        # Default to multiline
        return True


def format_json_string(
    data: dict,
    # Scientific notation settings
    lo: float = 1e-2,
    hi: float = 1e3,
    precision: int = 12,
    # Formatting settings
    indent: int = 4,
    array_format: Union[ArrayFormat, str] = ArrayFormat.SMART,
    array_threshold: int = 3,
    array_wrap_length: int = 120,
    # Other options
    compact: bool = False,
    sort_keys: bool = True,
) -> str:
    """Format JSON with scientific notation and customizable array formatting.

    Parameters
    ----------
    data : dict
        The data to encode.
    lo : float, optional
        Lower threshold for scientific notation. Default is 1e-2.
    hi : float, optional
        Upper threshold for scientific notation. Default is 1e3.
    precision : int, optional
        Decimal precision for scientific notation. Default is 12.
    indent : int, optional
        Number of spaces for indentation. Default is 4.
    array_format : Union[ArrayFormat, str], optional
        How to format arrays:
        - "always_multiline": Every array on multiple lines
        - "compact": All arrays on single line
        - "smart": Based on content and length
        - "threshold": Based on item count
        Default is SMART.
    array_threshold : int, optional
        For "threshold" mode, max items before going multiline. Default is 3.
    array_wrap_length : int, optional
        For "smart" mode, max line length before wrapping. Default is 120.
    compact : bool, optional
        Remove spaces after colons and commas. Default is False.
    sort_keys : bool, optional
        Sort dictionary keys alphabetically. Default is True.

    Returns
    -------
    str
        Formatted JSON string.
    """
    # Convert string to enum if needed
    if isinstance(array_format, str):
        array_format = ArrayFormat(array_format)

    encoder = ScientificJSONEncoder(
        lo=lo,
        hi=hi,
        precision=precision,
        indent=indent,
        array_format=array_format,
        array_threshold=array_threshold,
        array_wrap_length=array_wrap_length,
        compact=compact,
        sort_keys=sort_keys,
    )

    return encoder.encode(data)
