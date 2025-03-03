# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

from ipykernel.ipkernel import IPythonKernel  # type: ignore [import]

if TYPE_CHECKING:
    from collections.abc import Iterator

__version__ = "0.1"


@contextmanager
def reset_stdout(stdout: TextIO) -> Iterator[None]:
    _stdout = sys.stdout
    sys.stdout = stdout
    yield
    sys.stdout = _stdout


class LegionKernel(IPythonKernel):  # type: ignore [misc,no-any-unimported]
    implementation = "legion_kernel"
    implementation_version = __version__
    banner = "Legion IPython Kernel for SM"
    language = "python"
    language_version = __version__
    language_info: ClassVar = {
        "name": "legion_kernel",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    }

    def __init__(self, **kwargs: Any) -> None:
        with reset_stdout(Path("/dev/stdout").open(mode="w")):
            print(  # noqa: T201
                "Initializing Legion kernel for single- or multi-node."
            )
        super().__init__(**kwargs)


if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp  # type: ignore [import]

    IPKernelApp.launch_instance(kernel_class=LegionKernel)
