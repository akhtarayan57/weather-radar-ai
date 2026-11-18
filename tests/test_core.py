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


def test_field_regions_are_small_enough_for_free_api():
    from src.fieldmap import REGIONS, grid_axes
    for name in REGIONS:
        la, lo = grid_axes(name)
        assert 100 < len(la) * len(lo) <= 600, (name, len(la) * len(lo))


def test_ui_blocks_have_no_indented_lines_or_blank_lines():
    import pandas as pd
    from src import ui
    t = pd.Timestamp("2026-11-14 10:00"); h = pd.DataFrame({"temperature_2m": [20.0], "weather_code": [61], "precipitation_probability": [40.0],
                                                          "precipitation": [0.2], "is_day": [1], "uv_index": [3.0]}, index=[t])
    s = ui.hourly_strip(h, pd.Series([60.0], index=[t]), t, "US EPA", 1)
    assert "\n" not in s and "Now" in s and "💧 40%" in s


def test_world_field_save_load_and_trim(tmp_path):
    import pandas as pd
    from src import fieldmap as fm
    now = pd.Timestamp.now(tz="UTC").floor("h")
    times = [(now + pd.Timedelta(hours=3 * i - 9)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(12)]  # first 3 are > 3 h old
    d = {"times": times, "u": [[1]] * 12, "v": [[1]] * 12, "layers": {"us_aqi": {"frames": [[i] for i in range(12)]}}}
    fm.save_field(d, tmp_path / "f.json.gz")
    out = fm.load_field(tmp_path / "f.json.gz")
    assert 8 <= len(out["times"]) < 12 and len(out["layers"]["us_aqi"]["frames"]) == len(out["times"]) == len(out["u"])
    assert fm.load_field(tmp_path / "missing.json.gz") is None
    assert "World (fine)" not in fm.REGIONS and fm.grid_axes("World (fine)")[0][0] == -60


def test_pm25_to_aqi_scales():
    from src import aqi
    assert round(aqi.rate(89)["value"]) == 174 and aqi.rate(89)["cat"] == "Unhealthy"            # matches AQI.in Mumbai example
    assert round(aqi.rate(188)["value"]) == 263 and aqi.rate(0)["value"] == 0
    r = aqi.rate(188, "India (CPCB)"); assert round(r["value"]) == 353 and r["cat"] == "Very poor"
    assert aqi.rate(float("nan")) is None and aqi.rate(None) is None and aqi.rate(900, "US")["value"] == 500


def test_satellite_and_map_pages_have_no_unfilled_placeholders():
    import re
    from src import fieldmap as fm
    h = fm.satellite_html(28.6, 77.2, "Delhi", "2026-11-16")
    assert "2026-11-16" in h and "VIIRS_SNPP_CorrectedReflectance_TrueColor" in h and not re.search(r"__[A-Z]+__", h)
    assert "noWrap:true" in fm.BASE_JS and "addBorders" in fm.BASE_JS
