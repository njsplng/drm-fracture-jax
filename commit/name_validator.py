#!/usr/bin/env python3

"""Name validator and renamer for JSON files using Jinja2 templates. Run with SCAN_ALL_FILES=True to process all files, otherwise will check only git-changed files."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is in path for io_handlers
current_path = Path(__file__).parent.resolve()
src_path = current_path.parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from naming_utils import JsonJinjaRenamer


def main() -> None:
    """Main entry point for the script."""
    script_dir = Path(__file__).parent.resolve()
    input_dir = script_dir.parent / "input"
    schemas_dir = input_dir / "schemas"
    parser = argparse.ArgumentParser(
        description="Jinja2 renamer with extension support"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["parent", "fem", "nn"]:
        p = sub.add_parser(name, help=f"Process {name} directory")
        p.add_argument("-y", "--yes", action="store_true", help="Auto-confirm")
        p.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    allp = sub.add_parser("all", help="Run all directories")
    allp.add_argument("-y", "--yes", action="store_true", help="Auto-confirm")
    allp.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    def run_one(n: str, args: argparse.Namespace) -> None:
        base = input_dir / n
        cfg = schemas_dir / f"{n}_config.json"
        if not base.is_dir() or not cfg.is_file():
            logging.info(f"Missing dir or schema: {n}")
            return
        ren = JsonJinjaRenamer(base, cfg, getattr(args, "yes", False))
        ren.run()

    if args.cmd == "all":
        for n in ["parent", "fem", "nn"]:
            run_one(n, args)
    else:
        run_one(args.cmd, args)


if __name__ == "__main__":
    main()
