import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def download_isd_year(year: int, output_dir: Path = Path("./data/isd"), max_retries: int = 3) -> Path:
    """Download and decompress an ISD-Lite yearly file.

    Args:
        year: Year to download.
        output_dir: Directory to store downloaded files.
        max_retries: Number of download attempts.

    Returns:
        Path to the decompressed .txt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"725300-94846-{year}.txt"
    if txt_path.exists() and txt_path.stat().st_size > 10000:
        return txt_path
    gz_path = output_dir / f"725300-94846-{year}.gz"
    url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{gz_path.name}"
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                with open(gz_path, "wb") as f:
                    f.write(r.read())
            break
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {year}: {e}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(txt_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    return txt_path


def parse_isd_file(txt_path: Path) -> pd.DataFrame:
    """Parse an ISD-Lite whitespace-delimited file.

    Args:
        txt_path: Path to the .txt file.

    Returns:
        DataFrame with DatetimeIndex and columns ['T_db', 'T_dew', 'SLP', 'WS', 'WD', 'RH'].
    """
    names = ["year", "month", "day", "hour", "temp", "dewp", "slp", "wind_dir", "wind_speed"]
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                vals = [int(p) if p != "-9999" else None for p in parts[:9]]
                rows.append(vals)
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=names)
    df[names[4:]] = df[names[4:]].astype(float)
    df[["temp", "dewp", "slp", "wind_speed"]] /= 10.0
    df["wind_dir"] = df["wind_dir"].fillna(-9999).astype(int).replace(-9999, np.nan)

    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    T_db = df["temp"].values
    T_dew = df["dewp"].values
    es_db = 6.112 * np.exp((17.67 * T_db) / (T_db + 243.5))
    es_dew = 6.112 * np.exp((17.67 * T_dew) / (T_dew + 243.5))
    df["RH"] = np.clip(100.0 * es_dew / es_db, 0, 100)

    return df[["temp", "dewp", "slp", "wind_dir", "wind_speed", "RH"]].rename(
        columns={"temp": "T_db", "dewp": "T_dew", "slp": "SLP", "wind_dir": "WD", "wind_speed": "WS"},
    )


def load_isd_range(
    start_year: int = 2019,
    end_year: int = 2024,
    output_dir: Path = Path("./data/isd"),
) -> pd.DataFrame:
    """Load ISD-Lite data for a range of years.

    Args:
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        output_dir: Directory for downloaded files.

    Returns:
        Concatenated hourly DataFrame with full DatetimeIndex (gaps as NaN).
    """
    frames = []
    for year in range(start_year, end_year + 1):
        try:
            txt_path = download_isd_year(year, output_dir)
            frames.append(parse_isd_file(txt_path))
            print(f"  {year}: {len(frames[-1]):,} records")
        except Exception as e:
            print(f"  {year}: skipped ({e})")
    if not frames:
        raise RuntimeError("No data downloaded for any year")
    df = pd.concat(frames).sort_index()
    full_idx = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31 23:00",
        freq="h",
    )
    df = df.reindex(full_idx)
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the loaded DataFrame.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If columns are missing or missing rate exceeds 5%.
    """
    required = ["T_db", "T_dew", "SLP", "WS", "WD", "RH"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_unique
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
        missing_rate = df[col].isna().mean()
        print(f"  {col}: {missing_rate:.2%} missing")
        if missing_rate > 0.05:
            raise ValueError(f"Column {col} has {missing_rate:.2%} missing (>5%)")
