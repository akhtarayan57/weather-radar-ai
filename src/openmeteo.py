import pandas as pd
from .common import ApiError, get_json  # noqa: F401

WX_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
           "precipitation", "surface_pressure", "cloud_cover"]
GEO = "https://geocoding-api.open-meteo.com/v1/search"
ARCH = "https://archive-api.open-meteo.com/v1/archive"
FCST = "https://api.open-meteo.com/v1/forecast"


def _frame(j):
    h = j["hourly"]
    df = pd.DataFrame({k: h[k] for k in WX_VARS}, index=pd.to_datetime(h["time"], utc=True))
    df.index.name = "time"
    return df


def search_city(name, n=5):
    j = get_json(GEO, {"name": name, "count": n, "language": "en"})
    return [{"name": r["name"], "admin": r.get("admin1", ""), "country": r.get("country", ""),
             "lat": r["latitude"], "lon": r["longitude"]} for r in j.get("results", [])]


def archive(lat, lon, start, end):
    """ERA5 reanalysis (hourly, UTC). Returns (DataFrame, elevation_m)."""
    j = get_json(ARCH, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                        "hourly": ",".join(WX_VARS), "timezone": "UTC"})
    return _frame(j), float(j.get("elevation") or 0.0)


def forecast(lat, lon, past_days=3, forecast_days=3):
    j = get_json(FCST, {"latitude": lat, "longitude": lon, "hourly": ",".join(WX_VARS), "timezone": "UTC",
                        "past_days": past_days, "forecast_days": forecast_days})
    return _frame(j), float(j.get("elevation") or 0.0)
