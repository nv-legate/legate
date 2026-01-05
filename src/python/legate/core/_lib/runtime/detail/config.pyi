# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ...utilities.unconstructable import Unconstructable

class Config(Unconstructable):
    @property
    def profile(self) -> bool: ...
    @property
    def provenance(self) -> bool: ...
