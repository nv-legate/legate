# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

class Trace:
    def __init__(self, trace_id: int): ...
    def __enter__(self) -> None: ...
    def __exit__(self, _: Any, __: Any, ___: Any) -> None: ...
