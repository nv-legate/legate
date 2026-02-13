# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# We need to import libucx for Python pip wheel builds. This is due to the
# fact that liblegate.so links to librealm-legate.so which links to libucx.so
# and we need to ensure when MPI is used and the wrappers are dlopened that
# they will ideally use the system UCX libraries they were compiled against.
# This is why this must happen so early on in the import process and there
# are still potential issues. We fallback to the bundled UCX libraries if the
# system ones are not found, and only look for the unversioned SOs at this
# point (libucs.so not libucs.so.0) which does not work for all installations.
#
# TODO(cryos, jfaibussowit)
# Implement the same loading logic from libucx on the C++ side, and remove
# this workaround.


def _maybe_import_ucx_module() -> Any:  # pragma: no cover
    from .install_info import wheel_build  # noqa: PLC0415

    if not wheel_build:
        return None

    # Prefer wheels libraries that should load a consistent set of libraries.
    #
    # See https://github.com/rapidsai/ucx-wheels/blob/main/python/libucx/libucx/load.py#L55
    # for the environment variable check and logic for library loading.
    try:
        import libucx  # type: ignore[import-not-found]  # noqa: PLC0415
    except ModuleNotFoundError:
        return None

    # The handles are returned here in order to ensure the libraries are
    # loaded for the duration of execution.
    return libucx.load_library()


_libucx = _maybe_import_ucx_module()
