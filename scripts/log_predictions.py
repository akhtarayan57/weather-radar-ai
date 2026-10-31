"""Run on a schedule (GitHub Actions): predict 24 h ahead for every dataset city, store it, and fill in the
measured value once the target hour has passed. Builds the live scoreboard in data/live/predictions.csv."""
import json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import joblib, numpy as np, pandas as pd
from dotenv import load_dotenv
from src import features, openaq as oq, openmeteo as om, world
from src.common import ApiError

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live" / "predictions.csv"
COLS = ["city", "region", "anchor", "target", "pred", "lo", "hi", "persistence", "cams", "observed"]


def main():
    load_dotenv(); key = os.getenv("OPENAQ_API_KEY")
    if not key:
        sys.exit("Set OPENAQ_API_KEY.")
    art = ROOT / "artifacts"; meta = json.load(open(art / "metrics.json")); model = joblib.load(art / "model.joblib")
    q10, q90 = joblib.load(art / "model_q10.joblib"), joblib.load(art / "model_q90.joblib")
    st = pd.read_csv(ROOT / "data" / "processed" / "stations.csv"); now = pd.Timestamp.now(tz="UTC").floor("h")
    old = pd.DataFrame(columns=COLS)
    if LIVE.exists():
        old = pd.read_csv(LIVE)
        for c in ("anchor", "target"):
            old[c] = pd.to_datetime(old[c], utc=True)
    new = []
    for r in st.itertuples():
        try:
            pm = oq.hours(int(r.sensor_id), key, now - pd.Timedelta(hours=72), now)
            wx, elev = om.forecast(r.lat, r.lon); air = world.air_forecast(r.lat, r.lon)
        except ApiError as e:
            print(f"[err] {r.city}: {e}"); continue
        m = (old.city == r.city) & old.observed.isna() & (old.target <= now)
        for i in old[m].index:
            v = pm.get(old.at[i, "target"])
            if v is not None and v == v:
                old.at[i, "observed"] = float(v)
        s = pm.dropna()
        if s.empty or now - s.index[-1] > pd.Timedelta(hours=6):
            print(f"[skip] {r.city}: no fresh data"); continue
        anchor = s.index[-1]
        if ((old.city == r.city) & (old.anchor == anchor)).any():
            continue
        df = wx.join(pm.rename("pm25"), how="left")
        if len(air) and "pm2_5" in air:
            df = df.join(air["pm2_5"].rename("cams_pm25"))
        X = features.build_features(df, r.lat, r.lon, elev, with_target=False)
        row = X.loc[[anchor], meta["features"]]
        if row.isna().any().any():
            print(f"[skip] {r.city}: incomplete features"); continue
        p = float(np.clip(np.expm1(model.predict(row))[0], 0, None))
        lo = float(np.clip(np.expm1(q10.predict(row))[0], 0, None)); hi = float(np.clip(np.expm1(q90.predict(row))[0], 0, None))
        cams = float(np.expm1(X.at[anchor, "cams_f24"])) if X.at[anchor, "cams_f24"] == X.at[anchor, "cams_f24"] else np.nan
        new.append({"city": r.city, "region": r.region, "anchor": anchor, "target": anchor + pd.Timedelta(hours=meta["horizon_h"]),
                    "pred": p, "lo": min(lo, p), "hi": max(hi, p), "persistence": float(s.iloc[-1]), "cams": cams, "observed": np.nan})
        print(f"[ok] {r.city}: {p:.1f}")
    out = pd.concat([old, pd.DataFrame(new, columns=COLS)], ignore_index=True) if new else old
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["anchor", "city"]).to_csv(LIVE, index=False)
    print(f"{len(new)} new predictions; {out.observed.notna().sum()} checked of {len(out)} total.")


if __name__ == "__main__":
    main()
