# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, TypedDict

from ._lib.data.logical_array import LogicalArray
from ._lib.data.logical_store import LogicalStore

if TYPE_CHECKING:
    from . import StoreTarget
    from ._lib.type.types import Type


MIN_DATA_INTERFACE_VERSION: Final = 1
MAX_DATA_INTERFACE_VERSION: Final = 1


class LegateDataInterfaceItem(TypedDict):
    version: int
    data: dict[Field, LogicalArray]


class LegateDataInterface(Protocol):
    @property
    def __legate_data_interface__(  # noqa: D105
        self,
    ) -> LegateDataInterfaceItem: ...


LogicalArrayLike: TypeAlias = LogicalArray | LogicalStore | LegateDataInterface


def as_logical_array(obj: LegateDataInterface) -> LogicalArray:
    """Extract a LogicalArray from an object that provides the Legate
    data interface.

    Parameters
    ----------
    obj : LegateDataInterface
        An object exposing a legate data interface.

    Returns
    -------
        LogicalArray

    Raises
    ------
        TypeError
            In case obj does not expose a valid Legate Data Interface

        NotImplementedError
            In case the Legate Data Interface specifies unsupported
            features (e.g. nullable fields)

    """
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


def offload_to(obj: LogicalArrayLike, *, target: StoreTarget) -> None:
    """Offload a logical array-like object to a particular memory space.

    Parameters
    ----------
    obj: LogicalArrayLike
        The object to offload. A ``LogicalArray`` or object exposing a
        Legate Data Interface.
    target: :class:`~legate.core.StoreTarget`
        The store target to offload to, e.g. StoreTarget.SYSMEM

    """
    if isinstance(obj, (LogicalArray, LogicalStore)):
        obj.offload_to(target)

    else:
        array = as_logical_array(obj)
        array.offload_to(target)


class Field:
    def __init__(self, name: str, dtype: Type, *, nullable: bool = False):
        """
        A field is metadata associated with a single array in the legate data
        interface object.

        Parameters
        ----------
        name : str
            Field name
        dtype : Type
            The type of the array
        nullable : bool
            Indicates whether the array is nullable
        """
        if nullable:
            msg = "Nullable array is not yet supported"
            raise NotImplementedError(msg)

        self._name = name
        self._dtype = dtype
        self._nullable = nullable

    @property
    def name(self) -> str:
        """
        Returns the array's name.

        Returns
        -------
        str
            Name of the field
        """
        return self._name

    @property
    def type(self) -> Type:
        """
        Returns the array's data type.

        Returns
        -------
        Type
            Data type of the field
        """
        return self._dtype

    @property
    def nullable(self) -> bool:
        """
        Indicates whether the array is nullable.

        Returns
        -------
        bool
            ``True`` if the array is nullable. ``False`` otherwise.
        """
        return self._nullable

    def __repr__(self) -> str:  # noqa: D105
        return f"Field({self.name!r})"


class Table(LegateDataInterface):
    def __init__(
        self, fields: list[Field], columns: list[LogicalArray]
    ) -> None:
        """
        A Table is a collection of top-level, equal-length LogicalArray
        objects.
        """
        self._fields = fields
        self._columns = columns

    @property
    def __legate_data_interface__(self) -> LegateDataInterfaceItem:
        """
        The Legate data interface allows for different Legate libraries to get
        access to the base Legion primitives that back objects from different
        Legate libraries. It currently requires objects that implement it to
        return a dictionary that contains two members.

        Returns
        -------
        A dictionary with the following entries:

        'version' (required) : int
            An integer showing the version number of this implementation of
            the interface (i.e. 1 for this version)

        'data' (required) : dict[Field, LogicalArray]
            An dictionary mapping ``Field`` objects that represent the
            names and types of the field data to ``LogicalArray`` objects
        """
        result: LegateDataInterfaceItem = {
            "version": 1,
            "data": dict(zip(self._fields, self._columns, strict=True)),
        }
        return result

    @staticmethod
    def from_arrays(names: list[str], arrays: list[LogicalArray]) -> Table:
        """
        Construct a Table from a list of LogicalArrays.

        Parameters
        ----------
        arrays : list[LogicalArray]
            Equal-length arrays that should form the table.
        names : list[str], optional
            Names for the table columns. If not passed, schema must be passed

        Returns
        -------
        Table
        """
        if len(names) != len(arrays):
            msg = (
                f"Length of names ({names}) does not match "
                f"length of arrays ({arrays})"
            )
            raise ValueError(msg)
        fields = [
            Field(name, array.type)
            for name, array in zip(names, arrays, strict=True)
        ]
        return Table(fields, arrays.copy())
