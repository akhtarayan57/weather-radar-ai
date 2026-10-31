import math
import time
import numpy as np
import pandas as pd
from .common import get_json

BASE = "https://api.openaq.org/v3"
PM25_ID = 2
PAUSE = 1.1  # free tier: 60 requests/min


def _h(key):
    return {"X-API-Key": key}


def _km(la1, lo1, la2, lo2):
    p = math.pi / 180
    a = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))


def nearby_stations(lat, lon, key, radius=25000):
    """PM2.5 stations within radius (m), reference monitors first, then nearest."""
    j = get_json(f"{BASE}/locations", {"coordinates": f"{lat},{lon}", "radius": radius,
                                      "parameters_id": PM25_ID, "limit": 100}, _h(key))
    out = []
    for r in j.get("results", []):
        s = [x for x in r.get("sensors", []) if (x.get("parameter") or {}).get("id") == PM25_ID]
        if not s or not r.get("coordinates"):
            continue
        c = r["coordinates"]
        out.append({"location_id": r["id"], "name": r.get("name"), "sensor_id": s[0]["id"],
                    "station_lat": c["latitude"], "station_lon": c["longitude"],
                    "monitor": bool(r.get("isMonitor")), "provider": (r.get("provider") or {}).get("name"),
                    "first": (r.get("datetimeFirst") or {}).get("utc"),
                    "last": (r.get("datetimeLast") or {}).get("utc"),
                    "dist_km": round(_km(lat, lon, c["latitude"], c["longitude"]), 1)})
    return sorted(out, key=lambda s: (not s["monitor"], s["dist_km"]))


def hours(sensor_id, key, start, end, pause=PAUSE, chunk_days=90):
    """Hourly PM2.5 (ug/m3) for one sensor, fetched in chunks. Index = hour start (UTC). Invalid -> NaN."""
    f = "%Y-%m-%dT%H:%M:%SZ"
    rows, t0 = [], start
    while t0 < end:
        t1 = min(t0 + pd.Timedelta(days=chunk_days), end)
        page = 1
        while True:
            j = get_json(f"{BASE}/sensors/{sensor_id}/hours",
                         {"datetime_from": t0.strftime(f), "datetime_to": t1.strftime(f),
                          "limit": 1000, "page": page}, _h(key))
            res = j.get("results", [])
            rows += [(r["period"]["datetimeFrom"]["utc"], r["value"]) for r in res if r.get("value") is not None]
            time.sleep(pause)
            if len(res) < 1000:
                break
            page += 1
        t0 = t1
    if not rows:
        return pd.Series(dtype=float, name="pm25")
    s = pd.Series([v for _, v in rows], index=pd.to_datetime([t for t, _ in rows], utc=True), name="pm25", dtype=float)
    s = s[~s.index.duplicated()].sort_index()
    s.index = s.index.floor("h")
    s = s[~s.index.duplicated()]
    s[(s < 0) | (s > 1000)] = np.nan  # sensor errors / -999 fill values
    return s.asfreq("h")
