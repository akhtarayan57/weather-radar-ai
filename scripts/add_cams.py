"""Add CAMS model PM2.5 (free Open-Meteo air-quality API) as a column to the dataset. No key needed.
Run after build_dataset.py and before train_model.py. Resumable."""
import pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src import world
from src.common import ApiError

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, RAW = ROOT / "data" / "processed", ROOT / "data" / "raw"


def main():
    d = pd.read_csv(PROC / "dataset.csv.gz"); d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.drop(columns=["cams_pm25"], errors="ignore")
    st = pd.read_csv(PROC / "stations.csv"); parts = []
    for r in st.itertuples():
        slug = re.sub(r"\W+", "_", r.city.lower()); f = RAW / f"{slug}_cams.csv"
        t = d.loc[d.city == r.city, "time"]
        if f.exists():
            s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]; s.index = pd.to_datetime(s.index, utc=True)
        else:
            try:
                chunks, t0 = [], t.min().normalize()
                while t0 <= t.max():
                    t1 = min(t0 + pd.Timedelta(days=180), t.max().normalize())
                    chunks.append(world.cams_history(r.lat, r.lon, t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")))
                    t0 = t1 + pd.Timedelta(days=1)
                s = pd.concat(chunks); s = s[~s.index.duplicated()]
                s.to_frame().to_csv(f)
            except ApiError as e:
                print(f"[err] {r.city}: {e}"); continue
        print(f"[ok]  {r.city}: CAMS coverage {s.reindex(t).notna().mean():.0%}")
        parts.append(s.rename("cams_pm25").reset_index().rename(columns={"index": "time"}).assign(city=r.city))
    if not parts:
        sys.exit("No CAMS data downloaded.")
    c = pd.concat(parts); c.columns = ["time", "cams_pm25", "city"]
    d = d.merge(c, on=["city", "time"], how="left")
    d.to_csv(PROC / "dataset.csv.gz", index=False)
    d.groupby("city").head(500).to_csv(PROC / "dataset_sample.csv", index=False)
    print(f"Done: cams_pm25 added to {len(d):,} rows.")


if __name__ == "__main__":
    main()
