# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# -- Project information -----------------------------------------------------

project = "legate.core"
copyright = "2021-2023, NVIDIA"
author = "NVIDIA"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_parser",
    "legate._sphinxext.settings",
]

suppress_warnings = ["ref.myst"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "footer_start": ["copyright"],
    "github_url": "https://github.com/nv-legate/legate.core",
    # https://github.com/pydata/pydata-sphinx-theme/issues/1220
    "icon_links": [],
    "logo": {
        "text": project,
        "link": "https://nv-legate.github.io/legate.core/",
    },
    "navbar_align": "left",
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "primary_sidebar_end": ["indices.html"],
    "secondary_sidebar_items": ["page-toc"],
    "show_nav_level": 2,
    "show_toc_level": 2,
}

# -- Options for extensions --------------------------------------------------

autosummary_generate = True

copybutton_prompt_text = ">>> "

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

pygments_style = "sphinx"

myst_heading_anchors = 3


def setup(app):
    app.add_css_file("params.css")
