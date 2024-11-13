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

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from shlex import quote
from subprocess import run
from typing import TYPE_CHECKING, Any, Iterable

from rich import print as rich_print
from rich.console import NewLine, group
from rich.rule import Rule

from ..util.system import System
from ..util.types import DataclassMixin
from ..util.ui import env, section
from .command import CMD_PARTS_EXEC, CMD_PARTS_PYTHON
from .config import ConfigProtocol
from .environment import ENV_PARTS_LEGATE
from .launcher import Launcher

if TYPE_CHECKING:
    from rich.console import RenderableType

    from ..util.types import Command, EnvDict

__all__ = ("LegateDriver", "format_verbose")


@dataclass(frozen=True)
class LegateVersions(DataclassMixin):
    """Collect package versions relevant to Legate."""

    legate_version: str


class LegateDriver:
    """Coordinate the system, user-configuration, and launcher to appropriately
    execute the Legate process.

    Parameters
    ----------
        config : Config

        system : System

    """

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        self.config = config
        self.system = system
        self.launcher = Launcher.create(config, system)

    @property
    def cmd(self) -> Command:
        """The full command invocation to use to run the Legate program."""
        config = self.config
        launcher = self.launcher
        system = self.system

        cmd_parts = (
            CMD_PARTS_PYTHON if config.run_mode == "python" else CMD_PARTS_EXEC
        )

        parts = (part(config, system, launcher) for part in cmd_parts)
        return launcher.cmd + sum(parts, ())

    @property
    def env(self) -> EnvDict:
        """The system environment that should be used when starting Legate."""
        env = dict(self.launcher.env)

        # The previous contents of LEGATE_CONFIG can be ignored, because we
        # already spliced it into the command line arguments to the driver
        legate_parts = (part(self.config) for part in ENV_PARTS_LEGATE)
        LEGATE_CONFIG = " ".join(sum(legate_parts, ()))
        env["LEGATE_CONFIG"] = LEGATE_CONFIG.strip()

        return env

    @property
    def custom_env_vars(self) -> set[str]:
        """The names of environment variables that we have explicitly set
        for the system environment.

        """
        return {"LEGATE_CONFIG", *self.launcher.custom_env_vars}

    @property
    def dry_run(self) -> bool:
        """Check verbose and dry run.

        Returns
        -------
            bool : whether dry run is enabled

        """
        if self.config.info.verbose:
            msg = format_verbose(self.system, self)
            self.print_on_head_node(msg, flush=True)

        return self.config.other.dry_run

    def run(self) -> int:
        """Run the Legate process.

        Returns
        -------
            int : process return code

        """
        if self.dry_run:
            return 0

        if self.config.multi_node.nodes > 1 and self.config.console:
            raise RuntimeError("Cannot start console with more than one node.")

        if self.config.other.timing:
            self.print_on_head_node(f"Legate start: {datetime.now()}")

        ret = run(self.cmd, env=self.env).returncode

        if self.config.other.timing:
            self.print_on_head_node(f"Legate end: {datetime.now()}")

        log_dir = self.config.logging.logdir

        if self.config.profiling.profile:
            self.print_on_head_node(
                f"Profiles have been generated under {log_dir}, run "
                f"legion_prof view {log_dir}/legate_*.prof to view them"
            )

        if self.config.debugging.spy:
            self.print_on_head_node(
                f"Legion Spy logs have been generated under {log_dir}, run "
                f"legion_spy.py {log_dir}/legate_*.log to process them"
            )

        return ret

    def print_on_head_node(self, *args: Any, **kw: Any) -> None:
        launcher = self.launcher

        if launcher.kind != "none" or launcher.detected_rank_id == "0":
            rich_print(*args, **kw)


def get_versions() -> LegateVersions:
    from legate import __version__ as lg_version

    return LegateVersions(legate_version=lg_version)


@group()
def format_verbose(
    system: System,
    driver: LegateDriver | None = None,
) -> Iterable[RenderableType]:
    """Print system and driver configuration values.

    Parameters
    ----------
    system : System
        A System instance to obtain Legate and Legion paths from

    driver : Driver or None, optional
        If not None, a Driver instance to obtain command invocation and
        environment from (default: None)

    Returns
    -------
        RenderableType

    """

    yield Rule("Legate Configuration")
    yield NewLine()

    yield section("Legate paths")
    yield system.legate_paths.ui
    yield NewLine()

    yield section("Legion paths")
    yield system.legion_paths.ui
    yield NewLine()

    yield section("Versions")
    yield get_versions().ui
    yield NewLine()

    if driver:
        yield section("Command")
        cmd = " ".join(quote(t) for t in driver.cmd)
        yield f" [dim green]{cmd}[/]"
        yield NewLine()

        if keys := sorted(driver.custom_env_vars):
            yield section("Customized Environment")
            yield env(driver.env, keys=keys)
        yield NewLine()

    yield Rule()
