"""OptionsDX loader."""
from pathlib import Path
import pandas as pd
from .paths import RAW_DIR, LOCAL_RAW

def load_optionsdx_year(year: int, raw_dir: Path = None) -> pd.DataFrame:
    """Load one year of OptionsDX SPY EOD data with normalised schema."""
    raw_dir = raw_dir or (LOCAL_RAW if LOCAL_RAW.exists() else RAW_DIR)
    fp = raw_dir / f"spy_eod_{year}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"No raw file for year {year}: {fp}")
    df = pd.read_parquet(fp)
    df.columns = [c.strip("[]") for c in df.columns]
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"])
    df["T_years"] = df["DTE"] / 365.0
    return df
