#!/usr/bin/env python3

"""Optimiser filename checker/renamer (Jinja-powered)."""

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

current_path = Path(__file__).parent.resolve()
src_path = current_path.parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    TemplateSyntaxError,
)
from jsonpath_ng import parse as jsonpath_parse

from json_encoder import format_json_string

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------


def list_json_files(d: Path) -> List[Path]:
    """
    List all .json files in a directory (recursively), sorted by path.
    """
    if not d.exists():
        logging.error(f"Optimiser directory not found: {d}")
        return []
    if not d.is_dir():
        logging.error(f"Optimiser path is not a directory: {d}")
        return []
    return sorted(p for p in d.rglob("*.json") if p.is_file())


def load_config(cfg_path: Path) -> Dict[str, object]:
    """
    Load a JSON config file.
    """
    if not cfg_path.exists():
        logging.error(f"Config file not found: {cfg_path}")
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON config {cfg_path}: {e}")
        return {}


def build_env(template_dir: Path) -> Environment:
    """
    Build a Jinja2 environment for rendering templates.
    """
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        # Fail fast on missing keys
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def apply_mappings(
    mappings: Dict[str, str], data: Dict[str, object]
) -> Dict[str, object]:
    """
    Evaluate JSONPath mappings against a JSON object.
    """
    out: Dict[str, object] = {}
    for var_name, jp_expr in mappings.items():
        try:
            expr = jsonpath_parse(jp_expr)
            matches = expr.find(data)
            out[var_name] = matches[0].value if matches else None
        except Exception as e:
            logging.error(f"Mapping error for '{var_name}' with '{jp_expr}': {e}")
            out[var_name] = None
    return out


def lr_block_from(data: Dict[str, object]) -> str:
    """
    Compute the LR part of the filename based on schedule type.
    """
    lr_type = (data.get("learning_rate_type") or "").strip().lower()
    start = data.get("learning_rate_start")
    end = data.get("learning_rate_end")
    trans = data.get("learning_rate_transition")

    # constant: just LR(start)
    if lr_type == "constant":
        ms = mant_exp(start)
        return f"LR{ms}" if ms else ""

    # cosine_decay: LS(start) + LRCD
    if lr_type == "cosine_decay":
        ms = mant_exp(start)
        return f"LS{ms}_LRCD" if ms else "LRCD"

    # linear or exponential_decay: full block
    if lr_type in ("linear", "exponential_decay"):
        if start is not None and end is not None and trans is not None:
            return f"LS{mant_exp(start)}_LE{mant_exp(end)}_{lr_type_abbr(lr_type)}_TE{trans}"

    # If no usable fields → nothing
    return ""


def render_name(
    env: Environment,
    template_name: str,
    data: Dict[str, object],
    mapped: Dict[str, object],
    file_path: Path,
) -> str:
    """
    Render a filename using a Jinja2 template.
    """
    tmpl = env.get_template(template_name)
    ctx = {
        # Raw JSON keys from the optimiser file
        **data,
        # Variables created via mappings (e.g., 'name')
        **mapped,
        "__file_stem": file_path.stem,
        "__file_name": file_path.name,
        "__dir_name": file_path.parent.name,
        "lr_block": lr_block_from(data),
    }
    return tmpl.render(**ctx)


def iter_files_with_exts(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    """
    Yield files under `root` whose suffix is in `exts`.
    """
    exts = set(e.lower() for e in exts)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def replace_in_text_file(
    fp: Path,
    exact_pairs: List[Tuple[str, str]],
    stem_pairs: List[Tuple[str, str]],
    also_stem: bool,
) -> int:
    """
    Replace occurrences in a generic text file.
    """
    try:
        txt = fp.read_text()
    except UnicodeDecodeError:
        return 0

    total = 0
    # Full filename exact replacements
    for old, new in exact_pairs:
        if old in txt:
            count = txt.count(old)
            txt = txt.replace(old, new)
            total += count

    # Optional stem replacements with word boundaries
    if also_stem and stem_pairs:
        for old_stem, new_stem in stem_pairs:
            # Backslash b doesn't treat underscore as a boundary;
            # allow start/end or non-word on both sides
            # Safer pattern: (?<![\w-])old_stem(?![\w-])
            pattern = re.compile(rf"(?<![\w-]){re.escape(old_stem)}(?![\w-])")
            txt, n = pattern.subn(new_stem, txt)
            total += n

    if total > 0:
        fp.write_text(txt)
    return total


def replace_in_json_file(
    fp: Path, full_map: Dict[str, str], stem_map: Dict[str, str]
) -> int:
    """
    Parse JSON and replace string values that exactly equal.

    Either:
      - A full filename in full_map (e.g., 'adamw_LR-3ME500.json')
      - A stem in stem_map (e.g., 'adamw_LR-3ME500')
    Returns number of replacements performed.
    """
    # Read JSON
    try:
        data = json.loads(fp.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0

    changed = 0

    def visit(node: object) -> object:
        nonlocal changed

        if isinstance(node, dict):
            for k, v in list(node.items()):
                node[k] = visit(v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                node[i] = visit(v)
        elif isinstance(node, str):
            # Exact match on full filename
            if node in full_map:
                changed += 1
                return full_map[node]
            # Exact match on stem
            if node in stem_map:
                changed += 1
                return stem_map[node]
            return node
        return node

    new_data = visit(data)

    if changed > 0:
        # Write back (pretty-logging.infoed;
        # if you want to preserve formatting, plug your formatter here)
        fp.write_text(format_json_string(new_data))
    return changed


def mant_exp(value: Optional[float]) -> str:
    """
    Format positive float as 'M-E' where value = M * 10^-E and 1 <= M < 10.
    Examples: 1e-4 -> '1-4', 2e-3 -> '2-3', 0.5 -> '5-1'.
    Assumes value > 0.0 (learning rates should be positive).
    """
    if value is None or value == "":
        return ""
    f = float(value)
    if not (f > 0.0):
        raise ValueError(f"mant_exp expects positive, got {value!r}")
    # Can be negative/zero
    e = int(math.floor(math.log10(f)))
    # 1 <= m < 10
    m = f / (10**e)
    # Use the leading digit of m (avoid rounding changing the digit)
    m_digit = int(str(m).replace(".", "")[0]) if m < 10 else 9
    # We always emit 'M-abs(e)' as per spec (learning rates are < 1 typically)
    return f"{m_digit}-{abs(e)}"


def lr_type_abbr(s: Optional[str]) -> str:
    """
    Map learning rate type string to abbreviation.
    """
    if s is None:
        return "LR?"
    key = str(s).strip().lower()
    return {
        "linear": "LRL",
        "constant": "LRC",
        "cosine_decay": "LRCD",
        "exponential_decay": "LRED",
    }.get(key, "LR?")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------


def main() -> None:
    """Main entry point for the script."""
    script_dir = Path(__file__).parent.resolve()
    input_dir = script_dir.parent / "input"
    optimiser_dir = input_dir / "optimisers"
    schema_dir = input_dir / "schemas"
    config_path = schema_dir / "optimisers_config.json"

    parser = argparse.ArgumentParser(
        description="Optimiser filename checker/renamer (Jinja-powered)"
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Auto-apply renames and reference updates without prompting",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress info logs")
    parser.add_argument(
        "--ref-exts",
        nargs="*",
        default=[".json", ".yml", ".yaml", ".toml", ".md", ".txt", ".py"],
        help="File extensions to scan in input/ when updating references",
    )
    parser.add_argument(
        "--also-stem",
        action="store_true",
        help="Additionally replace bare stems (without .json) in non-JSON text files using word boundaries",
    )
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    logging.info(f"Script dir:     {script_dir}")
    logging.info(f"Input dir:      {input_dir}")
    logging.info(f"Optimiser dir:  {optimiser_dir}")
    logging.info(f"Schemas dir:    {schema_dir}")
    logging.info(f"Config path:    {config_path}")

    # Load config
    cfg = load_config(config_path)
    if not cfg:
        logging.error("Config missing or invalid; cannot proceed.")
        sys.exit(1)

    mappings: Dict[str, str] = cfg.get("mappings", {})
    template_file: str | None = cfg.get("template_file")
    if not template_file:
        logging.error("Config missing 'template_file'.")
        sys.exit(1)

    template_path = schema_dir / template_file
    logging.info(f"Template path:  {template_path}")
    if not template_path.exists():
        logging.error(f"Template file not found: {template_path}")
        sys.exit(1)

    # Jinja env + compile template
    try:
        env = build_env(template_path.parent)
        env.filters.update(
            {
                "mant_exp": mant_exp,
                "lr_type_abbr": lr_type_abbr,
            }
        )
        # Pre-compile
        env.get_template(template_path.name)

    except TemplateSyntaxError as e:
        logging.error(
            f"Template syntax error in {template_path}:{e.lineno}: {e.message}"
        )
        sys.exit(1)

    # Discover optimiser JSONs
    files = list_json_files(optimiser_dir)
    if not files:
        logging.warning("No optimiser JSON files found.")
        return

    logging.info("\nPlanned changes:\n")
    total_files_renamed = 0
    total_ref_updates = 0
    failures = 0

    # Collect mappings for reference updates (after each rename)
    for fp in files:
        try:
            # Load optimiser JSON
            try:
                data = json.loads(fp.read_text())
            except json.JSONDecodeError as e:
                logging.error(f"[{fp.name}] Invalid JSON: {e}")
                failures += 1
                continue

            # Mappings and expected name
            mapped = apply_mappings(mappings, data)
            required = ["optimiser", "number_of_epochs"]
            missing_required = [
                k for k in required if (mapped.get(k) is None and data.get(k) is None)
            ]
            if missing_required:
                logging.warning(
                    f"[{fp.name}] Missing required mapped values for: {', '.join(missing_required)}"
                )

            try:
                expected_name = render_name(env, template_path.name, data, mapped, fp)
            except TemplateError as e:
                logging.error(f"[{fp.name}] Template rendering failed: {e}")
                failures += 1
                continue

            current_name = fp.name
            relative_path = fp.relative_to(optimiser_dir)
            expected_path = fp.parent / expected_name
            logging.info(
                f" - {relative_path}  ->  {fp.parent.relative_to(optimiser_dir) / expected_name}"
            )

            if expected_name == current_name:
                # Nothing to do
                continue

            new_path = fp.with_name(expected_name)

            # Confirm
            do_apply = args.yes
            if not args.yes:
                ans = (
                    input(
                        f"Rename and update references? [y/N] {relative_path} -> {expected_name}: "
                    )
                    .strip()
                    .lower()
                )
                do_apply = ans == "y"
            if not do_apply:
                continue

            # Rename
            os.rename(fp, new_path)
            total_files_renamed += 1
            logging.info(
                f"Renamed: {relative_path} -> {fp.parent.relative_to(optimiser_dir) / expected_name}"
            )

            # Build replacement pairs
            # E.g., 'adamw_LR-3ME500.json'
            old_full = current_name
            # E.g., 'adamw_LR-3ME500_v2.json'
            new_full = expected_name
            # E.g., 'adamw_LR-3ME500'
            old_stem = Path(current_name).stem
            # E.g., 'adamw_LR-3ME500_v2'
            new_stem = Path(expected_name).stem

            # JSON-aware updates (safe structural replacement)
            json_updates = 0
            for jf in iter_files_with_exts(input_dir, [".json"]):
                # skip the renamed optimiser JSON itself
                if jf == new_path:
                    continue
                json_updates += replace_in_json_file(
                    jf,
                    full_map={old_full: new_full},
                    stem_map={old_stem: new_stem},
                )

            # Other text files (exact filename, optional stems)
            text_updates = 0
            if args.ref_exts:
                others = [e for e in args.ref_exts if e.lower() != ".json"]
                if others:
                    for tf in iter_files_with_exts(input_dir, others):
                        text_updates += replace_in_text_file(
                            tf,
                            exact_pairs=[(old_full, new_full)],
                            stem_pairs=[(old_stem, new_stem)],
                            also_stem=args.also_stem,
                        )

            updates = json_updates + text_updates
            total_ref_updates += updates
            logging.info(f"Updated references: {updates} occurrence(s) in input/")

        except Exception as e:
            logging.error(f"[{fp.name}] Unexpected error: {e}")
            failures += 1

    logging.info("\nSummary:")
    logging.info(f" - Files renamed: {total_files_renamed}")
    logging.info(f" - References updated in input/: {total_ref_updates}")

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
