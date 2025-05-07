# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..util.system import System
    from ..util.types import Command, EnvDict, LauncherType
    from .config import ConfigProtocol

__all__ = ("Launcher",)

RANK_ENV_VARS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)

LAUNCHER_VAR_PREFIXES = (
    "CONDA_",
    "CUTENSOR_",
    "LEGATE_",
    "LEGION_",
    "LG_",
    "REALM_",
    "GASNET_",
    "PYTHON",
    "UCX_",
    "NCCL_",
    "CUPYNUMERIC_",
    "NVIDIA_",
    "LD_",
)


class Launcher:
    """A base class for custom launch handlers for Legate.

    Subclasses should set ``kind`` and ``cmd`` properties during their
    initialization.

    Parameters
    ----------
        config : Config

        system : System

    """

    kind: LauncherType

    cmd: Command

    # base class will attempt to set this
    detected_rank_id: str | None = None

    _config: ConfigProtocol

    _system: System

    _env: EnvDict | None = None

    _custom_env_vars: set[str] | None = None

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        self._config = config
        self._system = system

        if config.multi_node.ranks == 1:
            self.detected_rank_id = "0"
        else:
            for var in RANK_ENV_VARS:
                try:
                    self.detected_rank_id = system.env[var]
                except KeyError:
                    continue
                else:
                    break

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return (
            isinstance(other, type(self))
            and self.kind == other.kind
            and self.cmd == other.cmd
            and self.env == other.env
        )

    @classmethod
    def create(cls, config: ConfigProtocol, system: System) -> Launcher:
        """Factory method for creating appropriate Launcher subclass based on
        user configuration.

        Parameters
        ----------
            config : Config

            system : System

        Returns
        -------
            Launcher

        """
        kind = config.multi_node.launcher
        match kind:
            case "none":
                return SimpleLauncher(config, system)
            case "mpirun":
                return MPILauncher(config, system)
            case "jsrun":
                return JSRunLauncher(config, system)
            case "srun":
                return SRunLauncher(config, system)
            case "dask":
                return DaskLauncher(config, system)

    # Slightly annoying, but it is helpful for testing to avoid importing
    # legate unless necessary, so defined these two as properties since the
    # command env depends on legate/legion paths

    @property
    def env(self) -> EnvDict:
        """A system environment to use with this launcher process."""
        if self._env is None:
            self._env, self._custom_env_vars = self._compute_env()
        return self._env

    @property
    def custom_env_vars(self) -> set[str]:
        """The set of environment variables specifically customized by us."""
        if self._custom_env_vars is None:
            self._env, self._custom_env_vars = self._compute_env()
        return self._custom_env_vars

    @staticmethod
    def is_launcher_var(name: str) -> bool:
        """Whether an environment variable name is important for the
        launcher.
        """
        return name.endswith("PATH") or any(
            name.startswith(prefix) for prefix in LAUNCHER_VAR_PREFIXES
        )

    def _compute_env(self) -> tuple[EnvDict, set[str]]:
        config = self._config
        system = self._system

        env = {}

        # We never want to save python byte code for legate
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        if config.debugging.gasnet_trace:
            env["GASNET_TRACEFILE"] = str(
                config.logging.logdir / "gasnet_%.log"
            )

        custom_env_vars = set(env)

        full_env = dict(system.env)
        full_env.update(env)

        return full_env, custom_env_vars


RANK_ERR_MSG = """\
Could not detect rank ID on multi-rank run with no --launcher provided. If you
want Legate to use a launcher, e.g. mpirun, internally (recommended), then you
need to specify which one to use by passing --launcher. Otherwise you need to
invoke the legate script itself through a launcher.
"""


class SimpleLauncher(Launcher):
    """A Launcher subclass for the "no launcher" case."""

    kind: LauncherType = "none"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config, system)

        # legate-bind.sh handles computing local and global rank id, even in
        # the simple case, just for consistency. But we do still check the
        # known rank env vars below in order to issue RANK_ERR_MSG if needed
        if config.multi_node.ranks > 1 and self.detected_rank_id is None:
            raise RuntimeError(RANK_ERR_MSG)

        self.cmd = ()


class MPILauncher(Launcher):
    """A Launcher subclass to use mpirun [1] for launching Legate processes.

    [1] https://www.open-mpi.org/doc/current/man1/mpirun.1.php

    """

    kind: LauncherType = "mpirun"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config, system)

        ranks = config.multi_node.ranks

        cmd = ["mpirun", "-n", str(ranks), *self._mpirun_commands(config)]

        # hack: the launcher.env does not know about what the driver does with
        # LEGATE_CONFIG, but we do need to make sure it gets forwarded
        env_vars = set(self.env).union({"LEGATE_CONFIG"})

        for var in sorted(env_vars):
            if self.is_launcher_var(var):
                cmd += ["-x", var]

        self.cmd = tuple(cmd + config.multi_node.launcher_extra)

    def _mpirun_commands(self, config: ConfigProtocol) -> list[str]:
        out = subprocess.check_output(["mpirun", "--version"]).decode()
        out_lo = out.casefold()

        if any(sub in out_lo for sub in ("open mpi", "openmpi", "open-mpi")):
            return self._openmpi_mpirun_commands(config)
        if any(sub in out_lo for sub in ("mpich", "hydra")):
            return self._mpich_mpirun_commands(config)

        m = (
            f"Unknown MPI implementation:\n\n{out}\n\nPlease file a bug at "
            "https://github.com/nv-legate/legate showing this error message "
            "along with a description of your system and MPI implementation"
        )
        raise RuntimeError(m)

    def _openmpi_mpirun_commands(self, config: ConfigProtocol) -> list[str]:
        ret = ["--npernode", str(config.multi_node.ranks_per_node)]
        ret += ["--bind-to", "none"]
        ret += ["--mca", "mpi_warn_on_fork", "0"]
        return ret

    def _mpich_mpirun_commands(self, config: ConfigProtocol) -> list[str]:
        ret = ["-ppn", str(config.multi_node.ranks_per_node)]
        ret += ["--bind-to", "none"]
        return ret


class JSRunLauncher(Launcher):
    """A Launcher subclass to use jsrun [1] for launching Legate processes.

    [1] https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=SSWRJV_10.1.0/jsm/jsrun.html
    """

    kind: LauncherType = "jsrun"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config, system)

        ranks = config.multi_node.ranks
        ranks_per_node = config.multi_node.ranks_per_node

        cmd = ["jsrun", "-n", str(ranks // ranks_per_node)]

        cmd += ["-r", "1"]
        cmd += ["-a", str(ranks_per_node)]
        cmd += ["-c", "ALL_CPUS"]
        cmd += ["-g", "ALL_GPUS"]
        cmd += ["-b", "none"]

        self.cmd = tuple(cmd + config.multi_node.launcher_extra)


class SRunLauncher(Launcher):
    """A Launcher subclass to use srun [1] for launching Legate processes.

    [1] https://slurm.schedmd.com/srun.html

    """

    kind: LauncherType = "srun"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config, system)

        ranks = config.multi_node.ranks
        ranks_per_node = config.multi_node.ranks_per_node

        cmd = ["srun", "-n", str(ranks)]

        cmd += ["--ntasks-per-node", str(ranks_per_node)]

        if config.debugging.gdb or config.debugging.cuda_gdb:
            # Execute in pseudo-terminal mode when we need to be interactive
            cmd += ["--pty"]

        self.cmd = tuple(cmd + config.multi_node.launcher_extra)


class DaskLauncher(Launcher):
    """A Launcher subclass to run legate program on a dask cluster."""

    kind: LauncherType = "dask"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config=config, system=system)

        if config.multi_node.nodes > 1:
            msg = "Dask launcher only supports single-node runs"
            raise RuntimeError(msg)

        cmd = ["dask"]
        cmd += ["--workers-per-node", str(config.multi_node.ranks_per_node)]

        self.cmd = tuple(cmd + config.multi_node.launcher_extra)
