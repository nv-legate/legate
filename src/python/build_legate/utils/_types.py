# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Literal, Protocol

ConfigSettings = dict[str, list[str] | str]
BuildKind = Literal["editable", "wheel"]


class BuildImpl(Protocol):
    def __call__(
        self,
        wheel_directory: str,
        config_settings: ConfigSettings | None,
        metadata_directory: str | None,
    ) -> str:
        pass
