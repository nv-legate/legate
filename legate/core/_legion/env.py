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

import os

assert "LEGATE_MAX_DIM" in os.environ
LEGATE_MAX_DIM = int(os.environ["LEGATE_MAX_DIM"])

assert "LEGATE_MAX_FIELDS" in os.environ
LEGATE_MAX_FIELDS = int(os.environ["LEGATE_MAX_FIELDS"])
