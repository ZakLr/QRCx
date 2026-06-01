from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target_col_idx: int = 0


def temporal_split(
    df: pd.DataFrame,
    train_years: tuple[int, int] = (2019, 2022),
    val_year: int = 2023,
    test_year: int = 2024,
) -> DataSplit:
    assert isinstance(df.index, pd.DatetimeIndex)
    train = df[(df.index.year >= train_years[0]) & (df.index.year <= train_years[1])]
    val = df[df.index.year == val_year]
    test = df[df.index.year == test_year]
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    return DataSplit(train=train, val=val, test=test)
