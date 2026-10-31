"""Download real PM2.5 (OpenAQ ground stations) + ERA5 weather (Open-Meteo) for many cities.
Resumable: finished cities are cached in data/raw/. Free OpenAQ key: https://explore.openaq.org
Usage: OPENAQ_API_KEY=xxx python scripts/build_dataset.py --start 2024-10-01 --end 2026-09-30"""
import argparse, json, os, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from dotenv import load_dotenv
from src import openaq as oq, openmeteo as om
from src.cities import CITIES
from src.common import ApiError

MIN_COVERAGE = 0.50
ROOT = pathlib.Path(__file__).resolve().parents[1]


def covers(s, start, end, slack=pd.Timedelta(days=60)):
    ok_first = not s["first"] or pd.Timestamp(s["first"]) <= start + slack
    ok_last = not s["last"] or pd.Timestamp(s["last"]) >= end - slack
    return ok_first and ok_last


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--limit", type=int, default=0, help="only first N cities (testing)")
    a = ap.parse_args()
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        sys.exit("Set OPENAQ_API_KEY (env var or .env file).")
    start = pd.Timestamp(a.start, tz="UTC"); end = pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(hours=23)
    raw = ROOT / "data" / "raw"; raw.mkdir(parents=True, exist_ok=True)
    for name, region, lat, lon in (CITIES[:a.limit] if a.limit else CITIES):
        slug = re.sub(r"\W+", "_", name.lower())
        if (raw / f"{slug}.csv").exists():
            print(f"[skip] {name} (cached)"); continue
        try:
            cands = [s for s in oq.nearby_stations(lat, lon, key) if covers(s, start, end)][:6]
            if not cands:
                print(f"[none] {name}: no PM2.5 station covering the period"); continue
            wx, elev = om.archive(lat, lon, a.start, a.end)
            for s in cands:
                try:
                    pm = oq.hours(s["sensor_id"], key, start, end)
                except ApiError as e:
                    print(f"  {name}: {s['name']} failed ({e})"); continue
                df = wx.join(pm.rename("pm25"), how="left")
                cov = float(df["pm25"].notna().mean())
                print(f"  {name}: {s['name']} ({s['dist_km']} km) coverage {cov:.0%}")
                if cov >= MIN_COVERAGE:
                    df.insert(0, "pm25", df.pop("pm25"))
                    df.to_csv(raw / f"{slug}.csv")
                    info = {"city": name, "region": region, "lat": lat, "lon": lon, "elevation": elev,
                            "station_name": s["name"], "location_id": s["location_id"], "sensor_id": s["sensor_id"],
                            "provider": s["provider"], "reference_monitor": s["monitor"], "station_dist_km": s["dist_km"],
                            "n_hours": int(df["pm25"].notna().sum()), "coverage": round(cov, 3)}
                    (raw / f"{slug}.json").write_text(json.dumps(info))
                    print(f"[ok]   {name}"); break
            else:
                print(f"[low]  {name}: no station reached {MIN_COVERAGE:.0%} coverage")
        except ApiError as e:
            print(f"[err]  {name}: {e}")
    frames, infos = [], []
    for j in sorted(raw.glob("*.json")):
        info = json.loads(j.read_text()); infos.append(info)
        d = pd.read_csv(raw / (j.stem + ".csv"), parse_dates=["time"])
        d.insert(0, "city", info["city"]); frames.append(d)
    if not frames:
        sys.exit("No cities downloaded.")
    out = ROOT / "data" / "processed"; out.mkdir(parents=True, exist_ok=True)
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out / "dataset.csv.gz", index=False)
    full.groupby("city").head(500).to_csv(out / "dataset_sample.csv", index=False)
    pd.DataFrame(infos).to_csv(out / "stations.csv", index=False)
    print(f"Done: {len(infos)} cities, {len(full):,} rows -> {out}")


if __name__ == "__main__":
    main()
