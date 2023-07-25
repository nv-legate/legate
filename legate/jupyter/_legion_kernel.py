#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from contextlib import contextmanager
from typing import Any, Iterator, TextIO

from ipykernel.ipkernel import IPythonKernel  # type: ignore [import]

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
    language_info = {
        "name": "legion_kernel",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    }

    def __init__(self, **kwargs: Any) -> None:
        with reset_stdout(open("/dev/stdout", "w")):
            print("Initializing Legion kernel for single- or multi-node.")
        super().__init__(**kwargs)


if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp  # type: ignore [import]

    IPKernelApp.launch_instance(kernel_class=LegionKernel)
