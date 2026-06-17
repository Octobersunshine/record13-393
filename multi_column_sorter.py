from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import pandas as pd


class SortOrder(Enum):
    ASC = "asc"
    DESC = "desc"


@dataclass(frozen=True)
class SortColumn:
    column: str
    order: SortOrder = SortOrder.ASC
    na_position: str = "last"

    def __post_init__(self):
        if self.na_position not in ("first", "last"):
            raise ValueError(
                f"na_position must be 'first' or 'last', got '{self.na_position}'"
            )


@dataclass
class SortSpec:
    columns: List[SortColumn] = field(default_factory=list)

    def add(self, column: str, order: SortOrder = SortOrder.ASC, na_position: str = "last") -> "SortSpec":
        self.columns.append(SortColumn(column=column, order=order, na_position=na_position))
        return self

    @classmethod
    def from_list(cls, specs: Sequence[tuple]) -> "SortSpec":
        instance = cls()
        for spec in specs:
            if len(spec) == 1:
                instance.add(spec[0])
            elif len(spec) == 2:
                instance.add(spec[0], spec[1])
            elif len(spec) == 3:
                instance.add(spec[0], spec[1], spec[2])
            else:
                raise ValueError(
                    f"Each sort spec must be a tuple of 1-3 elements, got {len(spec)}"
                )
        return instance


class MultiColumnSorter:
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

    def sort(self, spec: SortSpec) -> pd.DataFrame:
        if not spec.columns:
            return self._df

        self._validate_columns(spec)

        by = [col.column for col in spec.columns]
        ascending = [col.order is SortOrder.ASC for col in spec.columns]
        na_positions = [col.na_position for col in spec.columns]

        result = self._df.sort_values(
            by=by,
            ascending=ascending,
            na_position=na_positions[0] if len(set(na_positions)) == 1 else "last",
            kind="mergesort",
            ignore_index=True,
        )

        if len(set(na_positions)) > 1:
            result = self._sort_with_mixed_na_positions(spec, result)

        return result

    def _validate_columns(self, spec: SortSpec):
        missing = [c.column for c in spec.columns if c.column not in self._df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")

    def _sort_with_mixed_na_positions(self, spec: SortSpec, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for sort_col in reversed(spec.columns):
            col = sort_col.column
            ascending = sort_col.order is SortOrder.ASC
            na_pos = sort_col.na_position

            na_mask = result[col].isna()
            non_na = result[~na_mask]
            na_part = result[na_mask]

            non_na = non_na.sort_values(
                by=col, ascending=ascending, kind="mergesort", ignore_index=False
            )

            if na_pos == "first":
                result = pd.concat([na_part, non_na], ignore_index=True)
            else:
                result = pd.concat([non_na, na_part], ignore_index=True)

        return result


def multi_sort(
    df: pd.DataFrame,
    by: Optional[List[str]] = None,
    order: Optional[List[SortOrder]] = None,
    na_position: str = "last",
) -> pd.DataFrame:
    spec = SortSpec()

    if by is None:
        return df.copy()

    if order is None:
        order = [SortOrder.ASC] * len(by)

    if len(by) != len(order):
        raise ValueError(
            f"Length of 'by' ({len(by)}) and 'order' ({len(order)}) must match"
        )

    for col, ord_ in zip(by, order):
        spec.add(col, ord_, na_position)

    return MultiColumnSorter(df).sort(spec)
