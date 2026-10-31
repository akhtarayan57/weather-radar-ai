"""Train one global PM2.5 (+24 h) model, with and without the CAMS model feature.
Splits: per-city chronological 70/15/15 on seen cities + whole held-out cities (one per region with >=3 cities).
Also reports category accuracy, 'unhealthy day' precision/recall and 80% prediction-interval coverage."""
import argparse, json, pathlib, random, sys, datetime as dt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.features import FEATURES, FEATURES_BASE, HORIZON, build_features
from src.models import BlendModel, Subset

BINS, EVENT = [9.0, 35.4, 55.4, 125.4, 225.4], 35.4


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return {"rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(np.mean(np.abs(y - p))),
            "r2": float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else None, "n": int(len(y))}


def events(obs, pred):
    o, p = obs > EVENT, pred > EVENT; tp = int((o & p).sum())
    return {"cat_accuracy": float(np.mean(np.digitize(obs, BINS) == np.digitize(pred, BINS))),
            "event_precision": tp / max(int(p.sum()), 1), "event_recall": tp / max(int(o.sum()), 1), "event_rate": float(o.mean())}


def hgb(**kw):
    p = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=100, l2_regularization=1.0,
             early_stopping=True, random_state=42)
    p.update(kw)
    return HGB(**p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset.csv.gz")
    ap.add_argument("--stations", default="data/processed/stations.csv")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    d = pd.read_csv(a.data); d["time"] = pd.to_datetime(d["time"], utc=True)
    st = pd.read_csv(a.stations).set_index("city")
    has_cams = "cams_pm25" in d.columns and d["cams_pm25"].notna().any()
    ALL = FEATURES if has_cams else FEATURES_BASE
    print("CAMS feature:", "yes" if has_cams else "NO (run scripts/add_cams.py to add it)")
    frames = []
    for city, g in d.groupby("city"):
        X = build_features(g.set_index("time").sort_index(), st.at[city, "lat"], st.at[city, "lon"], st.at[city, "elevation"])
        X = X.dropna(subset=ALL + ["y"]).copy(); X["city"], X["region"] = city, st.at[city, "region"]
        if len(X) > 500:
            frames.append(X)
    D = pd.concat(frames); rng = random.Random(42); unseen = []
    for _, cs in D.groupby("region")["city"]:
        names = sorted(cs.unique())
        if len(names) >= 3:
            unseen.append(rng.choice(names))
    pos = D.groupby("city").cumcount() / D.groupby("city")["y"].transform("size")
    D["split"] = np.where(pos < 0.7, "train", np.where(pos < 0.85, "val", "test"))
    D.loc[D["city"].isin(unseen), "split"] = "unseen"
    tr, va = D[D.split == "train"], D[D.split == "val"]
    tests = {"test_seen": D[D.split == "test"], "test_unseen": D[D.split == "unseen"]}
    inv = lambda v: np.clip(np.expm1(np.asarray(v, float)), 0, None)
    results = {}

    def evaluate(n, fn):
        results[n] = {"val": score(inv(va.y), inv(fn(va)))}
        for k, t in tests.items():
            if len(t):
                results[n][k] = score(inv(t.y), inv(fn(t)))

    evaluate("Persistence", lambda df: df.pm_lag0.values)
    if has_cams:
        evaluate("CAMS raw (no ML)", lambda df: df.cams_f24.values)
    models = {"Ridge": Subset(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), FEATURES_BASE),
              "HistGradientBoosting": Subset(hgb(), FEATURES_BASE)}
    if has_cams:
        models["HGB + CAMS"] = Subset(hgb(), FEATURES)
    feats_of = {"Ridge": FEATURES_BASE, "HistGradientBoosting": FEATURES_BASE, "HGB + CAMS": FEATURES}
    for n, m in models.items():
        m.fit(tr[ALL], tr.y); evaluate(n, lambda df, m=m: m.predict(df[ALL]))
    top = min([n for n in models if n.startswith("H")], key=lambda n: results[n]["val"]["rmse"])
    pv = models[top].predict(va[ALL])
    w = min(np.linspace(0, 1, 11), key=lambda w: score(inv(va.y), inv(w * pv + (1 - w) * va.pm_lag0))["rmse"])
    bn = f"{top} + persistence blend"; models[bn] = BlendModel(models[top], w); feats_of[bn] = feats_of[top]
    evaluate(bn, lambda df: models[bn].predict(df[ALL]))
    best = min(models, key=lambda n: results[n]["val"]["rmse"]); feats = feats_of[best]
    T = pd.concat([t for t in tests.values() if len(t)]).copy()
    T["pred"], T["obs"], T["pers"] = inv(models[best].predict(T[ALL])), inv(T.y), inv(T.pm_lag0)
    if has_cams:
        T["cams"] = inv(T.cams_f24)
    qm = {q: Subset(hgb(loss="quantile", quantile=q, max_iter=250), feats).fit(tr[ALL], tr.y) for q in (0.1, 0.9)}
    lo, hi = inv(qm[0.1].predict(T[ALL])), inv(qm[0.9].predict(T[ALL]))
    coverage = float(np.mean((T.obs >= lo) & (T.obs <= hi)))

    def rows(g):
        r = {"n": len(g), "rmse_model": np.sqrt(((g.obs - g.pred) ** 2).mean()), "rmse_persistence": np.sqrt(((g.obs - g.pers) ** 2).mean())}
        if has_cams:
            r["rmse_cams"] = np.sqrt(((g.obs - g.cams) ** 2).mean())
        return pd.Series(r)
    by_city = T.groupby(["city", "region", "split"]).apply(rows, include_groups=False).reset_index()
    by_reg = T.groupby("region").apply(rows, include_groups=False).reset_index()
    for t in (by_city, by_reg):
        t["skill_vs_persistence"] = 1 - t.rmse_model / t.rmse_persistence
    ev = {"Persistence": events(T.obs.values, T.pers.values), best: events(T.obs.values, T.pred.values)}
    if has_cams:
        ev["CAMS raw (no ML)"] = events(T.obs.values, T.cams.values)
    out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
    joblib.dump(models[best], out / "model.joblib", compress=3)
    joblib.dump(qm[0.1], out / "model_q10.joblib", compress=3); joblib.dump(qm[0.9], out / "model_q90.joblib", compress=3)
    by_city.to_csv(out / "city_scores.csv", index=False); by_reg.to_csv(out / "region_scores.csv", index=False)
    meta = {"best_model": best, "horizon_h": HORIZON, "features": ALL, "results": results, "events": ev,
            "interval_80_coverage": coverage, "has_cams": bool(has_cams), "unseen_cities": unseen,
            "n_cities": int(D.city.nunique()), "n_rows": int(len(D)), "period": [str(D.index.min()), str(D.index.max())],
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "target": "log1p(PM2.5) 24 h ahead; reported in ug/m3",
            "sources": "OpenAQ ground stations (PM2.5); Open-Meteo ERA5 weather + CAMS model PM2.5"}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print("\nRMSE ug/m3 (lower is better)")
    print(pd.DataFrame({n: {k: round(v["rmse"], 2) for k, v in r.items()} for n, r in results.items()}).T.to_string())
    print("\nunhealthy-day events:", json.dumps({k: {a: round(b, 2) for a, b in v.items()} for k, v in ev.items()}))
    print(f"80% interval coverage on test data: {coverage:.0%}\nbest: {best} | unseen: {unseen}")


if __name__ == "__main__":
    main()
