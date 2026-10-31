import numpy as np
import pandas as pd
from .openmeteo import WX_VARS

HORIZON = 24
LAGS = [0, 1, 2, 3, 6, 12, 24]
_WX = [v for v in WX_VARS if v != "wind_direction_10m"] + ["wind_dir_sin", "wind_dir_cos"]
FEATURES_BASE = ([f"pm_lag{l}" for l in LAGS] + ["pm_mean24", "pm_std24"] + _WX + [f"{v}_f24" for v in _WX]
            + ["lat", "lon_sin", "lon_cos", "elevation", "hour_sin", "hour_cos", "month_sin", "month_cos"])
FEATURES = FEATURES_BASE + ["cams_now", "cams_f24"]  # + CAMS air-quality model PM2.5 now and 24 h ahead


def build_features(df, lat, lon, elev, with_target=True):
    """df: hourly UTC index, columns pm25 + WX_VARS. All lags look backwards only;
    *_f24 columns are the (forecast) weather 24 h ahead. Computed per city by the caller."""
    df = df.asfreq("h")
    d = pd.DataFrame(index=df.index)
    lp = np.log1p(df["pm25"])
    cams = np.log1p(df["cams_pm25"]) if "cams_pm25" in df else pd.Series(np.nan, index=df.index)
    d["cams_now"], d["cams_f24"] = cams, cams.shift(-HORIZON)
    for l in LAGS:
        d[f"pm_lag{l}"] = lp.shift(l)
    d["pm_mean24"] = lp.rolling(24, min_periods=18).mean()
    d["pm_std24"] = lp.rolling(24, min_periods=18).std()
    w = df[WX_VARS].copy()
    rad = np.deg2rad(w.pop("wind_direction_10m"))
    w["wind_dir_sin"], w["wind_dir_cos"] = np.sin(rad), np.cos(rad)
    for c in w.columns:
        d[c] = w[c]
        d[f"{c}_f24"] = w[c].shift(-HORIZON)
    solar_hour = (df.index.hour + lon / 15) % 24
    d["lat"], d["elevation"] = lat, elev
    d["lon_sin"], d["lon_cos"] = np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon))
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * solar_hour / 24), np.cos(2 * np.pi * solar_hour / 24)
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * df.index.month / 12), np.cos(2 * np.pi * df.index.month / 12)
    if with_target:
        d["y"] = lp.shift(-HORIZON)
    return d
