# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, TypedDict

from ._lib.data.logical_store import LogicalStore

if TYPE_CHECKING:
    from . import StoreTarget


MIN_DATA_INTERFACE_VERSION: Final = 2
MAX_DATA_INTERFACE_VERSION: Final = 2


class LegateDataInterfaceItem(TypedDict):
    version: int
    data: dict[str, LogicalStore]


class LegateDataInterface(Protocol):
    @property
    def __legate_data_interface__(  # noqa: D105
        self,
    ) -> LegateDataInterfaceItem: ...  # pragma: no cover


LogicalStoreLike: TypeAlias = LogicalStore | LegateDataInterface


def as_logical_store(obj: LegateDataInterface) -> LogicalStore:
    """Extract a LogicalStore from an object that provides the Legate
    data interface.

    Parameters
    ----------
    obj : LegateDataInterface
        An object exposing a legate data interface.

    Returns
    -------
        LogicalStore

    Raises
    ------
        TypeError
            In case obj does not expose a valid Legate Data Interface

        NotImplementedError
            In case the Legate Data Interface specifies unsupported
            features

    """
    if isinstance(obj, LogicalStore):
        return obj

    if not hasattr(obj, "__legate_data_interface__"):
        msg = "object does not provide Legate data interface"
        raise TypeError(msg)

    iface = obj.__legate_data_interface__

    if "version" not in iface:
        msg = "Legate data interface missing a version number"  # type: ignore [unreachable]
        raise TypeError(msg)

    v = iface["version"]

    if not isinstance(v, int):
        msg = f"Legate data interface version expected an integer, got {v!r}"  # type: ignore [unreachable]
        raise TypeError(msg)

    if v < MIN_DATA_INTERFACE_VERSION:
        msg = (
            f"Legate data interface version {v} is below "
            f"{MIN_DATA_INTERFACE_VERSION=}"
        )
        raise TypeError(msg)

    if v > MAX_DATA_INTERFACE_VERSION:
        msg = f"Unsupported Legate data interface version {v}"
        raise NotImplementedError(msg)

    data = iface["data"]

    it = iter(data)

    try:
        field = next(it)
    except StopIteration:
        msg = "Legate data object has no fields"
        raise TypeError(msg)

    try:
        next(it)
    except StopIteration:
        pass
    else:
        msg = (
            "Legate data interface objects with more than "
            "one store are unsupported"
        )
        raise NotImplementedError(msg)

    return data[field]


def offload_to(obj: LogicalStoreLike, *, target: StoreTarget) -> None:
    """Offload a logical store-like object to a particular memory space.

    Parameters
    ----------
    obj: LogicalStoreLike
        The object to offload. A ``LogicalStore`` or object exposing a
        Legate Data Interface.
    target: :class:`~legate.core.StoreTarget`
        The store target to offload to, e.g. StoreTarget.SYSMEM

    """
    store = as_logical_store(obj)
    store.offload_to(target)
