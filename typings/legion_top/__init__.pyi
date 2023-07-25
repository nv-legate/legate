# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Any, Callable, Tuple

from legion_cffi.lib import legion_context_t, legion_runtime_t

def add_cleanup_item(callback: Callable[[], None]) -> None: ...
def legion_canonical_python_main(sys_argv: Tuple[str, ...]) -> None: ...
def legion_canonical_python_cleanup() -> None: ...

class top_level:
    runtime: list[legion_runtime_t]
    context: list[legion_context_t]

__all__ = (
    "add_cleanup_item",
    "legion_canonical_python_main",
    "legion_canonical_python_cleanup",
    "top_level",
)
