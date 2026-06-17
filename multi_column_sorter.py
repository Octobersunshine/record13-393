from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
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

        result = self._safe_sort(self._df, spec)

        if len(set(col.na_position for col in spec.columns)) > 1:
            result = self._sort_with_mixed_na_positions(spec, result)

        return result

    def _validate_columns(self, spec: SortSpec):
        missing = [c.column for c in spec.columns if c.column not in self._df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")

    def _safe_sort(self, df: pd.DataFrame, spec: SortSpec) -> pd.DataFrame:
        by_columns, ascending, na_positions, temp_key_names = self._build_sort_plan(df, spec)

        sort_df = df
        if temp_key_names:
            sort_df = df.copy()
            for name, key in temp_key_names.items():
                sort_df[name] = key

        result = sort_df.sort_values(
            by=by_columns,
            ascending=ascending,
            na_position=na_positions[0] if len(set(na_positions)) == 1 else "last",
            kind="mergesort",
            ignore_index=True,
        )

        if temp_key_names:
            result = result.drop(columns=list(temp_key_names.keys()))

        return result

    def _build_sort_plan(
        self, df: pd.DataFrame, spec: SortSpec
    ) -> Tuple[List[str], List[bool], List[str], dict]:
        by_columns: List[str] = []
        ascending: List[bool] = []
        na_positions: List[str] = []
        temp_keys: dict = {}

        for i, sort_col in enumerate(spec.columns):
            col = sort_col.column
            series = df[col]

            ascending.append(sort_col.order is SortOrder.ASC)
            na_positions.append(sort_col.na_position)

            if self._is_mixed_type(series):
                key_name = f"__sort_key_{i}_{col}__"
                temp_keys[key_name] = self._make_type_safe_key(series, sort_col)
                by_columns.append(key_name)
            else:
                by_columns.append(col)

        return by_columns, ascending, na_positions, temp_keys

    def _is_mixed_type(self, series: pd.Series) -> bool:
        if series.dtype != object:
            return False

        non_na = series.dropna()
        if non_na.empty:
            return False

        has_numeric = False
        has_string = False

        for v in non_na:
            if isinstance(v, (int, float, np.integer, np.floating)):
                has_numeric = True
            elif isinstance(v, str):
                has_string = True
            if has_numeric and has_string:
                return True

        return has_numeric and has_string

    def _make_type_safe_key(self, series: pd.Series, sort_col: SortColumn) -> pd.Series:
        def _type_group(value: Any) -> int:
            if pd.isna(value):
                return 3
            if isinstance(value, (int, float, np.integer, np.floating)):
                return 0
            if isinstance(value, str):
                return 1
            return 2

        def _normalize_value(value: Any) -> Any:
            if pd.isna(value):
                return np.nan
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
            if isinstance(value, str):
                return value
            return str(value)

        groups = series.apply(_type_group)
        values = series.apply(_normalize_value)

        na_pos = sort_col.na_position
        if na_pos == "first":
            groups = groups.apply(lambda g: -1 if g == 3 else g)

        result = pd.Series(
            list(zip(groups, values)),
            index=series.index,
            name=f"__key_{sort_col.column}__",
        )
        return result

    def _sort_with_mixed_na_positions(self, spec: SortSpec, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for sort_col in reversed(spec.columns):
            col = sort_col.column
            ascending = sort_col.order is SortOrder.ASC
            na_pos = sort_col.na_position

            na_mask = result[col].isna()
            non_na = result[~na_mask]
            na_part = result[na_mask]

            if self._is_mixed_type(non_na[col]):
                sort_spec = SortSpec.from_list([(col, sort_col.order, sort_col.na_position)])
                non_na = self._safe_sort(non_na, sort_spec)
            else:
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
