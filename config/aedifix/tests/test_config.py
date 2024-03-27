# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

import sys

import pytest

from ..config import ConfigFile
from ..util.exception import LengthError
from .fixtures.dummy_manager import DummyManager


@pytest.fixture
def config_file(manager: DummyManager) -> ConfigFile:
    return ConfigFile(manager)


class TestConfigFile:
    def test_create(self, manager: DummyManager) -> None:
        config = ConfigFile(manager)
        assert config._project_rules == {}
        assert config._project_search_variables == {}
        assert config._project_variables == {}

    def add_rule_common(
        self,
        *,
        config_file: ConfigFile,
        rule_lines: tuple[str, ...],
        phony: bool,
        deps: tuple[str, ...] | None,
        exist_ok: bool,
    ) -> tuple[str, str, tuple[str, ...]]:
        expected_deps: tuple[str, ...]
        if deps is None:
            expected_deps = tuple()
        else:
            expected_deps = deps

        if not rule_lines and not deps:
            pytest.skip(
                "Adding rule with no rule_lines and no dependencies is tested "
                "elsewhere. It is easier to just skip this test rather than "
                "reconfigure the entire mark.parametrize() combo..."
            )

        rule_name_1 = "rule_1"

        assert rule_name_1 not in config_file._project_rules
        config_file.add_rule(
            rule_name_1, *rule_lines, phony=phony, deps=deps, exist_ok=exist_ok
        )
        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {
            rule_name_1: (phony, expected_deps, rule_lines)
        }

        rule_name_2 = "rule_2"
        config_file.add_rule(
            rule_name_2, *rule_lines, phony=phony, deps=deps, exist_ok=exist_ok
        )
        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {
            rule_name_1: (phony, expected_deps, rule_lines),
            rule_name_2: (phony, expected_deps, rule_lines),
        }
        return rule_name_1, rule_name_2, expected_deps

    @pytest.mark.parametrize(
        "rule_lines", (tuple(), ("lorem",), ("lorem", "ipsum", "dolor"))
    )
    @pytest.mark.parametrize("phony", (True, False))
    @pytest.mark.parametrize("deps", (None, tuple(), ("baz",), ("bop", "bar")))
    def test_add_rule_exist_ok(
        self,
        config_file: ConfigFile,
        rule_lines: tuple[str, ...],
        phony: bool,
        deps: tuple[str, ...] | None,
    ) -> None:
        rule_name_1, rule_name_2, expected_deps = self.add_rule_common(
            config_file=config_file,
            rule_lines=rule_lines,
            phony=phony,
            deps=deps,
            exist_ok=True,
        )

        new_rule_lines = rule_lines + ("some", "new", "rules")
        new_phony = not phony
        new_deps = expected_deps + ("some", "new", "deps")
        # should not throw
        config_file.add_rule(
            rule_name_1,
            *new_rule_lines,
            phony=new_phony,
            deps=new_deps,
            exist_ok=True,
        )
        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {
            rule_name_1: (new_phony, new_deps, new_rule_lines),
            rule_name_2: (phony, expected_deps, rule_lines),
        }

    @pytest.mark.parametrize(
        "rule_lines", (tuple(), ("lorem",), ("lorem", "ipsum", "dolor"))
    )
    @pytest.mark.parametrize("phony", (True, False))
    @pytest.mark.parametrize("deps", (None, tuple(), ("baz",), ("bop", "bar")))
    def test_add_rule_exist_not_ok(
        self,
        config_file: ConfigFile,
        rule_lines: tuple[str, ...],
        phony: bool,
        deps: tuple[str, ...] | None,
    ) -> None:
        rule_name_1, rule_name_2, expected_deps = self.add_rule_common(
            config_file=config_file,
            rule_lines=rule_lines,
            phony=phony,
            deps=deps,
            exist_ok=False,
        )
        # should throw
        with pytest.raises(
            ValueError, match=f"Project rule '{rule_name_1}' already exists"
        ):
            config_file.add_rule(
                rule_name_1,
                *rule_lines,
                phony=phony,
                deps=deps,
                exist_ok=False,
            )
        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {
            rule_name_1: (phony, expected_deps, rule_lines),
            rule_name_2: (phony, expected_deps, rule_lines),
        }

    def test_add_rule_bad(self, config_file: ConfigFile) -> None:
        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {}
        with pytest.raises(LengthError, match="Rule name must not be empty"):
            config_file.add_rule("")

        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {}
        with pytest.raises(LengthError, match="Rule name must not be empty"):
            config_file.add_rule("    ")

        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {}
        with pytest.raises(
            ValueError, match="Cannot have an empty rule with empty deps!"
        ):
            config_file.add_rule("asdasd")

        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {}
        with pytest.raises(
            ValueError, match="Cannot have an empty rule with empty deps!"
        ):
            config_file.add_rule("asdasd", deps=tuple())

        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {}

    def add_variable_common(
        self, config_file: ConfigFile, override_ok: bool
    ) -> tuple[str, str]:
        var_name = "foo"
        value = "bar"

        assert config_file._project_search_variables == {}
        assert config_file._project_rules == {}
        assert config_file._project_variables == {}
        config_file.add_variable(var_name, value, override_ok=override_ok)
        assert config_file._project_search_variables == {}
        assert config_file._project_rules == {}
        assert config_file._project_variables == {
            var_name: (override_ok, value)
        }

        return var_name, value

    def test_add_variable_override_ok(self, config_file: ConfigFile) -> None:
        override_ok = True
        var_name, value_1 = self.add_variable_common(
            config_file=config_file, override_ok=override_ok
        )

        value_2 = "asdjasbdoiad"
        assert value_2 != value_1
        # should not throw
        config_file.add_variable(var_name, value_2, override_ok=override_ok)
        assert config_file._project_search_variables == {}
        assert config_file._project_rules == {}
        assert config_file._project_variables == {
            var_name: (override_ok, value_2)
        }

        # should update override_ok as well
        config_file.add_variable(
            var_name, value_2, override_ok=not override_ok
        )
        assert config_file._project_search_variables == {}
        assert config_file._project_rules == {}
        assert config_file._project_variables == {
            var_name: (not override_ok, value_2)
        }

    def test_add_variable_override_not_ok(
        self, config_file: ConfigFile
    ) -> None:
        override_ok = False
        var_name, value_1 = self.add_variable_common(
            config_file=config_file, override_ok=override_ok
        )

        value_2 = "bazz"
        assert value_2 != value_1

        with pytest.raises(
            ValueError,
            match=f"Project variable {var_name} already registered: .*",
        ):
            config_file.add_variable(
                var_name, value_2, override_ok=override_ok
            )
        assert config_file._project_search_variables == {}
        assert config_file._project_rules == {}
        assert config_file._project_variables == {
            var_name: (override_ok, value_1)
        }

    @pytest.mark.parametrize("var_name", ("", "    "))
    def test_add_variable_bad(
        self, config_file: ConfigFile, var_name: str
    ) -> None:
        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {}
        with pytest.raises(
            LengthError, match="Variable name must not be empty"
        ):
            config_file.add_variable(var_name, "")

        assert config_file._project_search_variables == {}
        assert config_file._project_variables == {}
        assert config_file._project_rules == {}

    @pytest.mark.parametrize("project_var_name", (None, "foo_project"))
    @pytest.mark.parametrize("exist_ok", (True, False))
    def test_add_search_variable(
        self,
        config_file: ConfigFile,
        project_var_name: str | None,
        exist_ok: bool,
    ) -> None:
        cmake_name = "foo_cmake"
        if project_var_name is None:
            expected_project_var_name = cmake_name
        else:
            expected_project_var_name = project_var_name

        config_file.add_search_variable(
            cmake_name, project_var_name=project_var_name, exist_ok=exist_ok
        )
        assert config_file._project_search_variables == {
            cmake_name: expected_project_var_name
        }
        assert config_file._project_variables == {}
        assert config_file._project_rules == {}

        if exist_ok:
            config_file.add_search_variable(
                cmake_name,
                project_var_name=project_var_name,
                exist_ok=exist_ok,
            )
            assert config_file._project_search_variables == {
                cmake_name: expected_project_var_name
            }
            assert config_file._project_variables == {}
            assert config_file._project_rules == {}

            new_project_var_name = "new_foo_project"
            config_file.add_search_variable(
                cmake_name,
                project_var_name=new_project_var_name,
                exist_ok=exist_ok,
            )
            assert config_file._project_search_variables == {
                cmake_name: new_project_var_name
            }
            assert config_file._project_variables == {}
            assert config_file._project_rules == {}
        else:
            with pytest.raises(
                ValueError,
                match=(
                    f"Project search variable {cmake_name} already "
                    "registered"
                ),
            ):
                new_project_var_name = "asdjabsdablda"
                assert project_var_name != new_project_var_name
                config_file.add_search_variable(
                    cmake_name,
                    project_var_name=new_project_var_name,
                    exist_ok=exist_ok,
                )
            # the above should have no effect
            assert config_file._project_search_variables == {
                cmake_name: expected_project_var_name
            }
            assert config_file._project_variables == {}
            assert config_file._project_rules == {}

    def test_setup(self, config_file: ConfigFile) -> None:
        # There is a lot this function does, so here we just test it doesn't
        # die.
        config_file.setup()

    @pytest.mark.xfail(reason="TODO")  # TODO
    def test_finalize(self, config_file: ConfigFile) -> None:
        var_file = config_file.project_variables_file
        assert not var_file.exists()
        config_file.finalize()
        assert var_file.exists()
        assert var_file.is_file()


if __name__ == "__main__":
    sys.exit(pytest.main())
