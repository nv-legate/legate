#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# A custom implementation of
# https://github.com/breathe-doc/breathe/blob/main/breathe/apidoc.py which
# strips out options we don't need/use, and adds some additional formatting.
import sys
import shutil
from argparse import Action, ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast
from xml.etree import ElementTree as ET

if TYPE_CHECKING:
    from collections.abc import Sequence

# Reference: Doxygen XSD schema file, CompoundKind only
# Only what breathe supports are included
# Translates identifier to English
TYPEDICT: Final = {
    "class": "Class",
    "interface": "Interface",
    "struct": "Struct",
    "union": "Union",
    "file": "File",
    "namespace": "Namespace",
    "group": "Group",
}

# Types that accept the :members: option.
MEMBERS_TYPES: Final = ["class", "group", "interface", "namespace", "struct"]


class Options(Namespace):
    output_dir: Path
    force: bool
    members: bool
    dry_run: bool
    no_toc: bool
    out_types: list[str]
    quite: bool
    rootpath: Path


def print_info(msg: str, args: Options) -> None:
    if not args.quiet:
        print(msg)  # noqa: T201


def write_file(name: Path, text: str, args: Options) -> None:
    """Write the output file for module/package <name>."""
    fname = args.output_dir / f"{name}.rst"
    if args.dry_run:
        print_info(f"Would create file {fname}.", args)
        return

    if not args.force and fname.is_file():
        print_info(f"File {fname} already exists, skipping.", args)
        return

    print_info(f"Creating file {fname}.", args)
    fname.parent.mkdir(parents=True, exist_ok=True)
    try:
        orig = fname.read_text()
        if orig == text:
            print_info(f"File {fname} up to date, skipping.", args)
            return
    except FileNotFoundError:
        # Don't mind if it isn't there
        pass

    fname.write_text(text)


HEADINGS: Final = ("=", "-", "~")


def format_heading(level: int, text: str) -> str:
    """Create a heading of <level> [1, 2 or 3 supported]."""
    assert level in range(len(HEADINGS) + 1)
    underlining = HEADINGS[level - 1] * len(text)
    return f"{text}\n{underlining}\n\n"


def format_directive(package_type: str, package: str, args: Options) -> str:
    """Create the breathe directive and add the options."""
    directive = f".. doxygen{package_type}:: {package}\n"
    if args.members and package_type in MEMBERS_TYPES:
        directive += "   :members:\n"
    return directive


def create_package_file(
    package: str, package_type: str, package_id: str, args: Options
) -> None:
    """Build the text of the file and write the file."""
    # Skip over types that weren't requested
    if package_type not in args.out_types:
        return

    text = format_heading(1, package)
    text += format_directive(package_type, package, args)

    write_file(Path(package_type, package_id), text, args)


def create_modules_toc_file(key: str, value: str, args: Options) -> None:
    """Create the module's index."""
    if not (args.output_dir / key).is_dir():
        return

    text = format_heading(1, f"{value} list")
    text += ".. toctree::\n"
    text += "   :glob:\n"
    text += "   :maxdepth: 2\n"
    text += "\n"
    text += f"   {key}/*\n"

    write_file(Path(f"{key}list"), text, args)


def recurse_tree(args: Options) -> None:
    """
    Look for every file in the directory tree and create the corresponding
    ReST files.
    """
    index = ET.parse(args.rootpath / "index.xml")  # noqa: S314

    # Assuming this is a valid Doxygen XML
    for compound in index.getroot():
        name = compound.findtext("name")
        kind = compound.get("kind")
        refid = compound.get("refid")
        assert name is not None
        assert kind is not None
        assert refid is not None
        create_package_file(name, kind, refid, args)


class TypeAction(Action):
    def __init__(
        self, option_strings: list[str], dest: str, **kwargs: Any
    ) -> None:
        super().__init__(option_strings, dest, **kwargs)
        self.default = TYPEDICT.keys()
        self.metavar = ",".join(TYPEDICT.keys())

    def __call__(
        self,
        parser: ArgumentParser,  # noqa: ARG002
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        assert isinstance(values, str)
        value_list = values.split(",")
        for value in value_list:
            if value not in TYPEDICT:
                m = f"{value} not a valid option"
                raise ValueError(m)
        setattr(namespace, self.dest, value_list)


def parse_args() -> Options:
    parser = ArgumentParser(
        description=(
            "Parse XML created by Doxygen in <rootpath> and create one "
            "reST file with breathe generation directives per definition in "
            "the <DESTDIR>. Note: By default this script will not overwrite "
            "already created files."
        )
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to place all output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        dest="force",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-m",
        "--members",
        action="store_true",
        dest="members",
        help=f"Include members for types: {MEMBERS_TYPES}",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Run the script without creating files",
    )
    parser.add_argument(
        "-T",
        "--no-toc",
        action="store_true",
        help="Don't create a table of contents file",
    )
    parser.add_argument(
        "-g",
        "--generate",
        action=TypeAction,
        dest="out_types",
        help="types of output to generate, comma-separated list",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress informational messages",
    )
    parser.add_argument(
        "rootpath", type=Path, help="The directory contains index.xml"
    )
    args = cast(Options, parser.parse_args())
    args.rootpath = args.rootpath.resolve()

    if not args.rootpath.is_dir():
        print(f"{args.roothpath} is not a directory.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    if not (args.rootpath / "index.xml").is_file():
        print(f"{args.rootpath} does not contain a index.xml", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    return args


def main() -> None:
    """Parse and check the command line arguments."""
    args = parse_args()

    if not args.dry_run:
        print(f"Would clear {args.output_dir}")  # noqa: T201
        shutil.rmtree(args.output_dir)

    if not args.output_dir.is_dir() and not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    recurse_tree(args)

    if args.no_toc:
        return

    for key in args.out_types:
        create_modules_toc_file(key, TYPEDICT[key], args)


if __name__ == "__main__":
    main()
