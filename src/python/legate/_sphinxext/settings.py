# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import textwrap
import importlib
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from docutils.statemachine import StringList
from jinja2 import Template
from sphinx.errors import SphinxError
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

from legate.util.settings import EnvOnlySetting, PrioritizedSetting, _Unset

SETTINGS_DETAIL = Template(
    """
{% for setting in settings %}

``{{ setting['name'] }}``
{{ "''''" +  "'" * setting['name']|length }}

:**Type**: {{ setting['type'] }}
:**Env var**: ``{{ setting['env_var'] }}``
:**Default**: {{ setting['default'] }}

{{ setting['help'] }}

{% endfor %}
"""
)


class SettingsDirective(SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 1
    option_spec: ClassVar = {"module": unchanged}

    def run(self):
        obj_name = " ".join(self.arguments)
        module_name = self.options["module"]

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            msg = (
                f"Unable to generate reference docs for {obj_name}: "
                f"couldn't import module {module_name}"
            )
            raise SphinxError(msg)

        obj = getattr(module, obj_name, None)
        if obj is None:
            msg = (
                f"Unable to generate reference docs for {obj_name}: "
                f"no model {obj_name} in module {module_name}"
            )
            raise SphinxError(msg)

        settings = []
        for x in obj.__class__.__dict__.values():
            if isinstance(x, PrioritizedSetting):
                default = "(Unset)" if x.default is _Unset else repr(x.default)
            elif isinstance(x, EnvOnlySetting):
                default = repr(x.default)
                if x.test_default is not _Unset:
                    default += f" (test-mode default: {x.test_default!r})"
            else:
                continue

            settings.append(
                {
                    "name": x.name,
                    "env_var": x.env_var,
                    "type": x.convert_type,
                    "help": textwrap.dedent(x.help),
                    "default": default,
                }
            )

        rst_text = SETTINGS_DETAIL.render(
            name=obj_name, module_name=module_name, settings=settings
        )
        return self.parse(rst_text, "<settings>")

    def parse(self, rst_text, annotation):
        result = StringList()
        for line in rst_text.split("\n"):
            result.append(line, annotation)
        node = nodes.paragraph()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)
        return node.children


def setup(app):
    """Required Sphinx extension setup function."""
    app.add_directive_to_domain("py", "settings", SettingsDirective)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
