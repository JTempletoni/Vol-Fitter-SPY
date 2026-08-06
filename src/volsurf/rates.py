"""Risk-free rate curve from FRED Treasury data."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .paths import AUX_DIR

TREASURY_PATH = AUX_DIR / "treasury_curve_history.parquet"

def load_treasury_curve() -> pd.DataFrame:
    """Load the persisted FRED Treasury curve history."""
    return pd.read_parquet(TREASURY_PATH)

def get_rate(date: pd.Timestamp, T_years: float, treasury_df: pd.DataFrame) -> float:
    """Continuously-compounded risk-free rate via linear interpolation in maturity."""
    day = treasury_df[treasury_df.date == pd.Timestamp(date)]
    if day.empty:
        day = treasury_df[(treasury_df.date <= pd.Timestamp(date)) &
                          (treasury_df.date >= pd.Timestamp(date) - pd.Timedelta(days=5))]
        if day.empty:
            return np.nan
        day = day[day.date == day.date.max()]
    day = day.sort_values("maturity_years")
    if len(day) < 2:
        return np.nan
    f = interp1d(day.maturity_years.values, day.yield_pct.values / 100.0,
                 kind="linear", bounds_error=False,
                 fill_value=(day.yield_pct.values[0] / 100.0,
                             day.yield_pct.values[-1] / 100.0))
    return float(f(T_years))
