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

import functools
import shutil


class _Constants:
    @staticmethod
    @functools.cache
    def _compute_banner_length() -> int:
        try:
            length, _ = shutil.get_terminal_size()
        except Exception:
            # Default if all else fails (the ISO/ANSI screen size is 80x24, as
            # I'm sure we all know).
            length = 80
        else:
            # Leave some buffer room, but if for some reason an absolute madman
            # is configuring on a 1 char wide screen, then we at least don't
            # have negative length.
            length = max(length - 2, 1)
        return length

    @property
    def banner_length(self) -> int:
        return self._compute_banner_length()


Constants = _Constants()
