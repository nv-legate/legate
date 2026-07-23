# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NewType

LocalTaskID = NewType("LocalTaskID", int)

LocalRedopID = NewType("LocalRedopID", int)
GlobalRedopID = NewType("GlobalRedopID", int)
