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

"""Consolidate driver configuration from command-line and environment.

"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypedDict

from jupyter_client.kernelspec import (
    KernelSpec,
    KernelSpecManager,
    NoSuchKernel,
)

from legate.driver import LegateDriver
from legate.jupyter.config import Config
from legate.util.types import ArgList
from legate.util.ui import error


class LegateMetadata(TypedDict):
    argv: ArgList
    multi_node: dict[str, Any]
    memory: dict[str, Any]
    core: dict[str, Any]


LEGATE_JUPYTER_KERNEL_SPEC_KEY = "__LEGATE_JUPYTER_KERNEL_SPEC__"
LEGATE_JUPYTER_METADATA_KEY: Literal["legate"] = "legate"


def generate_kernel_spec(driver: LegateDriver, config: Config) -> KernelSpec:
    legion_kernel = Path(__file__).parent / "_legion_kernel.py"
    argv = list(driver.cmd) + [str(legion_kernel), "-f", "{connection_file}"]

    env = {k: v for k, v in driver.env.items() if k in driver.custom_env_vars}

    # Inexplicably, there is apparently no reasonable or supported way to
    # determine the name of the currently running/connected Jupyter kernel.
    # Instead, tunnel an env var with the name through, so that our LegateInfo
    # line magic can actually find the right kernel spec to report on.
    assert LEGATE_JUPYTER_KERNEL_SPEC_KEY not in env
    env[LEGATE_JUPYTER_KERNEL_SPEC_KEY] = config.kernel.spec_name

    return KernelSpec(
        display_name=config.kernel.display_name,
        language="python",
        argv=argv,
        env=env,
        metadata={
            LEGATE_JUPYTER_METADATA_KEY: LegateMetadata(
                {
                    "argv": config.argv[1:],
                    "multi_node": asdict(config.multi_node),
                    "memory": asdict(config.memory),
                    "core": asdict(config.core),
                }
            )
        },
    )


def install_kernel_spec(spec: KernelSpec, config: Config) -> None:
    ksm = KernelSpecManager()

    spec_name = config.kernel.spec_name
    display_name = spec.display_name

    try:
        ksm.get_kernel_spec(spec_name)
    except NoSuchKernel:
        pass
    else:
        # inexplicably, install_kernel_spec calls lower on the supplied kernel
        # name before using, so we need to call lower for this advice to work
        msg = error(
            f"kernel spec {spec_name!r} already exists. Remove it by "
            f"running: 'jupyter kernelspec uninstall {spec_name.lower()}', "
            "or choose a new kernel name."
        )
        print(msg)
        sys.exit(1)

    with TemporaryDirectory() as tmpdir:
        os.chmod(tmpdir, 0o755)
        with open(Path(tmpdir).joinpath("kernel.json"), "w") as f:
            out = json.dumps(spec.to_dict(), sort_keys=True, indent=2)
            if config.verbose > 0:
                print(f"Wrote kernel spec file {spec_name}/kernel.json\n")
            if config.verbose > 1:
                print(f"\n{out}\n")
            f.write(out)

        try:
            ksm.install_kernel_spec(
                tmpdir,
                spec_name,
                user=config.kernel.user,
                prefix=config.kernel.prefix,
            )
            print(
                f"Jupyter kernel spec {spec_name} ({display_name}) "
                "has been installed"
            )
        except Exception as e:
            msg = error(
                "Failed to install the Jupyter kernel spec "
                f"{spec_name} ({display_name}) with error: {e}"
            )
            print(msg)
            sys.exit(1)
