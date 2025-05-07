# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, TypedDict

if TYPE_CHECKING:
    from ._lib.data.logical_array import LogicalArray
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
