# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# This needs to be loaded before anything from ._lib
from ..._libucx_loader import _libucx
from ._lib.has_started import runtime_has_started

__all__ = ["runtime_has_started"]

# runtime_has_started() has to be in a submodule of util and not imported into
# util directly, because
#
# - runtime_has_started() calls liblegate (legate::has_started()),
# - _maybe_import_ucx_module() must run before liblegate is loaded, and
# - _maybe_import_ucx_module() indirectly imports from util
#
# So import runtime_has_started() in util/__init__.py would create a circular
# dependency
