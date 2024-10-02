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
