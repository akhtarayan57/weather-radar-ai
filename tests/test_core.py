import numpy as np, pandas as pd
from src.aqi import pm25_category
from src.features import FEATURES, build_features
from src.openmeteo import WX_VARS


def frame(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({v: np.arange(n, dtype=float) for v in WX_VARS}, index=idx)
    df["pm25"] = np.arange(n, dtype=float) + 5
    return df


def test_no_leakage_and_alignment():
    df = frame(); X = build_features(df, 10, 20, 100)
    t = 100
    assert np.isclose(X["y"].iloc[t], np.log1p(df.pm25.iloc[t + 24]))
    assert np.isclose(X["pm_lag1"].iloc[t], np.log1p(df.pm25.iloc[t - 1]))
    assert np.isclose(X["temperature_2m_f24"].iloc[t], df.temperature_2m.iloc[t + 24])
    assert set(FEATURES) <= set(X.columns)


def test_gaps_become_nan_not_shifted():
    df = frame(); df = df.drop(df.index[50:55]); X = build_features(df, 0, 0, 0)
    assert len(X) == 200 and np.isnan(X["pm_lag0"].iloc[52])


def test_aqi():
    assert pm25_category(5) == "Good" and pm25_category(300) == "Hazardous" and pm25_category(float("nan")) == "n/a"


def test_cams_features():
    df = frame(); df["cams_pm25"] = np.arange(200, dtype=float) + 1; X = build_features(df, 0, 0, 0)
    assert np.isclose(X["cams_f24"].iloc[10], np.log1p(df.cams_pm25.iloc[34]))
    assert np.isclose(X["cams_now"].iloc[10], np.log1p(df.cams_pm25.iloc[10]))
