# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from docutils import nodes
from docutils.statemachine import StringList
from packaging.version import Version
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

if TYPE_CHECKING:
    from docutils.nodes import Node
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata


class Releases(SphinxDirective):
    def parse(self, rst_lines: list[str], annotation: str) -> list[Node]:
        result = StringList()
        for line in rst_lines:
            result.append(line, annotation)
        node = nodes.paragraph()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)
        return node.children

    def run(self) -> list[Node]:
        env = self.env
        cur_file = Path(env.doc2path(env.docname))
        cur_dir = cur_file.parent
        assert cur_dir.is_dir()

        skip_files = {cur_dir / "dev.rst", cur_file}

        all_rst = (p for p in cur_dir.iterdir() if p.suffix == ".rst")
        all_versions = (p for p in all_rst if p not in skip_files)
        versions = [p.stem for p in all_versions]
        versions.sort(key=Version, reverse=True)

        lines = [".. toctree::", "  :maxdepth: 1", "  :caption: Contents:", ""]
        for v in versions:
            prefix = " " * (len(lines[1]) - len(lines[1].lstrip()))
            lines.append(f"{prefix}{v} <{v}.rst>")
        return self.parse(lines, "<legate-releases>")


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("legate-releases", Releases)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
