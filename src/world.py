"""Global current conditions + hourly forecasts from Open-Meteo (free, no key)."""
import pandas as pd
from .cities import CITIES
from .common import ApiError, get_json

FC = "https://api.open-meteo.com/v1/forecast"
AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
CUR_W = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "cloud_cover", "surface_pressure"]
CUR_A = ["pm2_5", "us_aqi", "uv_index"]
HOURLY_W = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation_probability",
            "precipitation", "wind_speed_10m", "wind_gusts_10m", "surface_pressure", "cloud_cover", "uv_index"]
HOURLY_A = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "us_aqi"]

EXTRA = [("Istanbul", 41.01, 28.98), ("Moscow", 55.76, 37.62), ("Kyiv", 50.45, 30.52), ("Vienna", 48.21, 16.37),
         ("Athens", 37.98, 23.73), ("Dublin", 53.35, -6.26), ("Lisbon", 38.72, -9.14), ("Oslo", 59.91, 10.75),
         ("Helsinki", 60.17, 24.94), ("Reykjavik", 64.15, -21.94), ("Vancouver", 49.28, -123.12),
         ("San Francisco", 37.77, -122.42), ("Miami", 25.76, -80.19), ("Anchorage", 61.22, -149.90),
         ("Havana", 23.11, -82.37), ("Panama City", 8.98, -79.52), ("Quito", -0.18, -78.47),
         ("Caracas", 10.48, -66.90), ("Rio de Janeiro", -22.91, -43.17), ("La Paz", -16.50, -68.15),
         ("Casablanca", 33.57, -7.59), ("Algiers", 36.75, 3.06), ("Kinshasa", -4.44, 15.27),
         ("Dakar", 14.72, -17.47), ("Luanda", -8.84, 13.23), ("Dar es Salaam", -6.79, 39.21),
         ("Cape Town", -33.92, 18.42), ("Khartoum", 15.50, 32.56), ("Baghdad", 33.31, 44.37),
         ("Dubai", 25.20, 55.27), ("Karachi", 24.86, 67.01), ("Kabul", 34.53, 69.17), ("Tashkent", 41.30, 69.24),
         ("Almaty", 43.24, 76.89), ("Novosibirsk", 55.01, 82.93), ("Vladivostok", 43.12, 131.89),
         ("Hong Kong", 22.32, 114.17), ("Taipei", 25.03, 121.57), ("Manila", 14.60, 120.98),
         ("Singapore", 1.35, 103.82), ("Kuala Lumpur", 3.14, 101.69), ("Ho Chi Minh City", 10.82, 106.63),
         ("Yangon", 16.84, 96.17), ("Colombo", 6.93, 79.86), ("Chennai", 13.08, 80.27), ("Bengaluru", 12.97, 77.59),
         ("Hyderabad", 17.39, 78.49), ("Ahmedabad", 23.02, 72.57), ("Perth", -31.95, 115.86),
         ("Darwin", -12.46, 130.84), ("Hobart", -42.88, 147.33), ("Wellington", -41.29, 174.78),
         ("Honolulu", 21.31, -157.86), ("Nuuk", 64.18, -51.72), ("Lhasa", 29.65, 91.12), ("Urumqi", 43.83, 87.62)]
WORLD = [(n, la, lo) for n, _, la, lo in CITIES] + EXTRA


def _multi(url, pts, cur, chunk=40):
    out = []
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        j = get_json(url, {"latitude": ",".join(str(x[1]) for x in p), "longitude": ",".join(str(x[2]) for x in p),
                           "current": ",".join(cur), "timezone": "UTC"})
        j = j if isinstance(j, list) else [j]
        out += [r.get("current") or {} for r in j]
    return out


def now_for(pts):
    """Current weather + air quality (CAMS model) for [(name, lat, lon), ...]."""
    w = _multi(FC, pts, CUR_W)
    try:
        a = _multi(AQ, pts, CUR_A)
    except ApiError:
        a = [{} for _ in pts]
    df = pd.DataFrame([{"name": n, "lat": la, "lon": lo} for n, la, lo in pts])
    for c in CUR_W:
        df[c] = [r.get(c) for r in w]
    for c in CUR_A:
        df[c] = [r.get(c) for r in a]
    cols = CUR_W + CUR_A
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df


def _hourly(url, lat, lon, cols, **kw):
    j = get_json(url, {"latitude": lat, "longitude": lon, "hourly": ",".join(cols), "timezone": "UTC", **kw})
    h = j["hourly"]
    df = pd.DataFrame({c: h[c] for c in cols if c in h}, index=pd.to_datetime(h["time"], utc=True))
    df.index.name = "time"
    return df


def hourly_forecast(lat, lon, days=7):
    return _hourly(FC, lat, lon, HOURLY_W, forecast_days=days, past_days=1)


def air_forecast(lat, lon, days=5):
    try:
        return _hourly(AQ, lat, lon, HOURLY_A, forecast_days=days, past_days=1)
    except ApiError:
        return pd.DataFrame()


def cams_history(lat, lon, start, end):
    """Hourly CAMS PM2.5 (model) for a past period, as a Series named cams_pm25."""
    j = get_json(AQ, {"latitude": lat, "longitude": lon, "hourly": "pm2_5", "start_date": start,
                      "end_date": end, "timezone": "UTC"})
    h = j["hourly"]
    return pd.Series(h["pm2_5"], index=pd.to_datetime(h["time"], utc=True), name="cams_pm25", dtype=float)


def hourly_many(pts, col, chunk=40):
    """Next ~72 h of one variable for many places. Returns DataFrame: index=time, columns=place names."""
    url = AQ if col in ("pm2_5", "us_aqi") else FC
    cols = {}
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        j = get_json(url, {"latitude": ",".join(str(x[1]) for x in p), "longitude": ",".join(str(x[2]) for x in p),
                           "hourly": col, "forecast_days": 3, "timezone": "UTC"})
        j = j if isinstance(j, list) else [j]
        for (n, _, _), r in zip(p, j):
            h = r["hourly"]
            cols[n] = pd.Series(h[col], index=pd.to_datetime(h["time"], utc=True), dtype=float)
    return pd.DataFrame(cols)
