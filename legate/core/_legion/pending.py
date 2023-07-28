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

from typing import Any

# We can't call out to the CFFI from inside of finalizer methods
# because that can risk a deadlock (CFFI's lock is stupid, they
# take it still in python so if a garbage collection is triggered
# while holding it you can end up deadlocking trying to do another
# CFFI call inside a finalizer because the lock is not reentrant).
# Therefore we defer deletions until we end up launching things
# later at which point we know that it is safe to issue deletions
_pending_unordered: dict[Any, Any] = dict()

# We also have some deletion operations which are only safe to
# be done if we know the Legion runtime is still running so we'll
# move them here and only issue the when we know we are inside
# of the execution of a task in some way
_pending_deletions: list[Any] = list()
