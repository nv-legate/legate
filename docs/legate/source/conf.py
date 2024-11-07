# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
from os import getenv
from typing import Any

import legate

SWITCHER_PROD = "https://docs.nvidia.com/legate/switcher.json"
SWITCHER_DEV = "http://localhost:8000/legate/switcher.json"
JSON_URL = SWITCHER_DEV if getenv("SWITCHER_DEV") == "1" else SWITCHER_PROD

# -- Project information -----------------------------------------------------

project = "NVIDIA legate"
if "dev" in legate.__version__:
    project += f" ({legate.__version__})"

copyright = "2021-2024, NVIDIA"
author = "NVIDIA Corporation"

# -- General configuration ---------------------------------------------------

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "myst_parser",
    "legate._sphinxext.settings",
]

suppress_warnings = ["ref.myst"]
exclude_patterns = ["BUILD.md"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "switcher": {
        "json_url": JSON_URL,
        "navbar_start": ["navbar-logo", "version-switcher"],
        "version_match": ".".join(legate.__version__.split(".", 2)[:2]),
    }
}

# -- Options for extensions --------------------------------------------------

autosummary_generate = True
templates_path = ["_templates"]
napoleon_include_special_with_doc = True

breathe_default_project = "legate"
breathe_default_members = ("members", "protected-members")

copybutton_prompt_text = ">>> "

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

pygments_style = "sphinx"

myst_heading_anchors = 3


def dont_skip_documented_dunders(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    skip: bool,
    options: dict[str, Any],
) -> bool | None:
    SKIP = True  # definitely skip the value (does not show up in docs)
    KEEP = False  # definitely do NOT skip the value (will show up in docs)
    LET_AUTODOC_DECIDE = None  # fall back to autodoc (might show up in docs)

    if not name.startswith("_"):
        # Not a dunder, defer to default autodoc-skip-member
        return LET_AUTODOC_DECIDE
    if obj is getattr(object, name, None):
        # The function is actually object.__foo__, which, obviously we did not
        # write. Defer to autodoc.
        return LET_AUTODOC_DECIDE
    if inspect.isbuiltin(obj):
        # Some object.__foo__ methods aren't caught by the above, but they are
        # caught by this.
        return LET_AUTODOC_DECIDE
    if not hasattr(obj, "__doc__"):
        return LET_AUTODOC_DECIDE

    if "pyx_vtable" in name or "_cython_" in repr(obj):
        # Cython implementation details, we never want these
        return SKIP

    if any(
        item in name
        for item in (
            "array_interface",
            "legate_data_interface",
            "__init__",
            "__doc__",
        )
    ):
        # We always want these
        return KEEP

    match what:
        case "method":
            return KEEP
        case _:
            print(f"====== LET AUTODOC DECIDE ({what})", name, "->", obj)
            return LET_AUTODOC_DECIDE


def setup(app):
    app.add_css_file("params.css")
    app.connect("autodoc-skip-member", dont_skip_documented_dunders)
