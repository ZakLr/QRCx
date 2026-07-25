from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target_col_idx: int = 0


def _year_mask(index: pd.DatetimeIndex, years) -> pd.Series:
    """`years` is either a single int (one calendar year, the original
    convention) or a (start, end) tuple (inclusive range, Sprint 6:
    val/test as multi-year ranges, matching train_years' existing tuple
    convention)."""
    if isinstance(years, tuple):
        return (index.year >= years[0]) & (index.year <= years[1])
    return index.year == years


def temporal_split(
    df: pd.DataFrame,
    train_years: tuple[int, int] = (2019, 2022),
    val_year=2023,
    test_year=2024,
) -> DataSplit:
    assert isinstance(df.index, pd.DatetimeIndex)
    train = df[(df.index.year >= train_years[0]) & (df.index.year <= train_years[1])]
    val = df[_year_mask(df.index, val_year)]
    test = df[_year_mask(df.index, test_year)]
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    return DataSplit(train=train, val=val, test=test)
