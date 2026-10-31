import json, os, pathlib
import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from plotly.subplots import make_subplots
from src import openmeteo as om, openaq as oq, features, aqi, world
from src.common import ApiError

st.set_page_config("WeatherRadar AI", "🌦️", layout="wide")
st.markdown("<style>.stApp{background:#0b1623;color:#e6edf5}[data-testid=stSidebar]{background:#0f2133}"
            "[data-testid=stMetric]{background:#12293f;padding:10px;border-radius:8px}</style>", unsafe_allow_html=True)
PROC, ART, LIVE = pathlib.Path("data/processed"), pathlib.Path("artifacts"), pathlib.Path("data/live/predictions.csv")
S_MAP, S_FC, S_CMP, S_AI, S_BOARD, S_DOC = "🗺️ World Map", "📈 Forecast Charts", "⚖️ Compare Cities", "🤖 AI PM2.5 Prediction", "🏆 Live Scoreboard", "📚 Data & Methods"
LAYERS = {
    "🌫️ Air quality: PM2.5 status": ("pm2_5", "cat_pm"), "🌫️ Air quality: US AQI status": ("us_aqi", "cat_aqi"),
    "🌡️ Temperature (°C)": ("temperature_2m", "RdYlBu_r"), "💧 Humidity (%)": ("relative_humidity_2m", "Blues"),
    "💨 Wind speed (km/h)": ("wind_speed_10m", "Viridis"), "🌧️ Precipitation (mm/h)": ("precipitation", "Blues"),
    "☁️ Cloud cover (%)": ("cloud_cover", "Cividis"), "☀️ UV index": ("uv_index", "YlOrRd"),
    "⏲️ Pressure (hPa)": ("surface_pressure", "Plasma")}
CMP_VARS = {"Temperature (°C)": ("temperature_2m", "w"), "Rain (mm/h)": ("precipitation", "w"), "Wind (km/h)": ("wind_speed_10m", "w"),
            "Humidity (%)": ("relative_humidity_2m", "w"), "Cloud cover (%)": ("cloud_cover", "w"), "UV index": ("uv_index", "w"),
            "PM2.5 (µg/m³, CAMS)": ("pm2_5", "a"), "US AQI (CAMS)": ("us_aqi", "a")}
HEALTH = {"Good": "Air is clean. Enjoy outdoor activities.",
          "Moderate": "Unusually sensitive people should reduce long, heavy outdoor exertion.",
          "Unhealthy for sensitive groups": "Children, older adults and people with heart or lung disease should limit prolonged outdoor exertion.",
          "Unhealthy": "Everyone should reduce prolonged outdoor exertion; sensitive groups should stay indoors.",
          "Very unhealthy": "Avoid outdoor exertion; sensitive groups stay indoors. A well-fitted N95/FFP2 mask helps if you must go out.",
          "Hazardous": "Stay indoors with windows closed, use an air purifier if you have one, and avoid outdoor activity."}


def api_key():
    try:
        return st.secrets["OPENAQ_API_KEY"]
    except Exception:
        return os.getenv("OPENAQ_API_KEY")


def fmt(v, f="{:.1f}"):
    return "n/a" if v is None or v != v else f.format(v)


def advice(cat):
    if cat in HEALTH:
        st.info(f"{HEALTH[cat]}  \n*General guidance from US EPA categories, not medical advice.*")


@st.cache_data(ttl=1800, show_spinner="Loading world conditions…")
def world_now():
    return world.now_for(world.WORLD)


@st.cache_data(ttl=3600, show_spinner="Loading 72 h forecast for all cities…")
def world_grid(col):
    return world.hourly_many(world.WORLD, col)


@st.cache_data(ttl=900, show_spinner=False)
def now_one(name, lat, lon):
    return world.now_for([(name, lat, lon)]).iloc[0]


@st.cache_data(ttl=900, show_spinner="Loading forecast…")
def forecasts(lat, lon):
    return world.hourly_forecast(lat, lon), world.air_forecast(lat, lon)


@st.cache_data(ttl=900, show_spinner="Fetching live data…")
def live(lat, lon, key):
    wx, elev = om.forecast(lat, lon)
    air = world.air_forecast(lat, lon)
    station, pm = None, pd.Series(dtype=float, name="pm25")
    if key:
        now = pd.Timestamp.now(tz="UTC")
        for s in oq.nearby_stations(lat, lon, key)[:3]:
            if s["last"] and now - pd.Timestamp(s["last"]) < pd.Timedelta(hours=6):
                p = oq.hours(s["sensor_id"], key, now - pd.Timedelta(hours=72), now, pause=0)
                if p.notna().sum() >= 18:
                    station, pm = s, p
                    break
    return wx, elev, air, station, pm


def line(d, cols, title):
    f = px.line(d.reset_index(), x="time", y=cols, title=title, template="plotly_dark")
    f.update_layout(legend_title=None, hovermode="x unified")
    return f


ss = st.session_state
ss.setdefault("loc", {"name": "Delhi", "lat": 28.61, "lon": 77.21})
st.sidebar.title("🌦️ WeatherRadar AI")
section = st.sidebar.radio("Section", [S_MAP, S_FC, S_CMP, S_AI, S_BOARD, S_DOC])
q = st.sidebar.text_input("🔍 Search any place")
if q:
    try:
        res = om.search_city(q)
    except ApiError as e:
        st.sidebar.error(str(e)); res = []
    if res:
        i = st.sidebar.selectbox("Result", range(len(res)), format_func=lambda i: f"{res[i]['name']}, {res[i]['admin']}, {res[i]['country']}")
        if st.sidebar.button("Use this place"):
            ss["loc"] = {"name": res[i]["name"], "lat": res[i]["lat"], "lon": res[i]["lon"]}
    else:
        st.sidebar.warning("No match found.")
loc = ss["loc"]
st.sidebar.success(f"📍 {loc['name']}")
now = pd.Timestamp.now(tz="UTC").floor("h")

if section == S_MAP:
    st.title("World weather & air quality")
    try:
        tbl = world_now()
    except ApiError as e:
        st.error(str(e)); st.stop()
    c1, c2 = st.columns([2, 1])
    layer = c1.selectbox("Map layer", list(LAYERS))
    mode = c2.radio("Time", ["Now", "Forecast (72 h)"], horizontal=True)
    col, kind = LAYERS[layer]
    d = tbl.copy()
    if mode != "Now":
        try:
            grid = world_grid(col)
        except ApiError as e:
            st.error(str(e)); st.stop()
        times = [t for t in grid.index if t >= now][::3]
        ti = st.select_slider("Forecast time (UTC)", options=list(range(len(times))), format_func=lambda i: f"{times[i]:%a %d %b %H:%M}")
        d[col] = d["name"].map(grid.loc[times[ti]])
    d = d.dropna(subset=[col]).copy()
    kw = dict(lat="lat", lon="lon", custom_data=["name"], hover_name="name", zoom=1, center={"lat": 20, "lon": 10},
              height=600, map_style="carto-darkmatter", hover_data={"name": False, "lat": False, "lon": False, col: ":.1f"})
    if kind.startswith("cat"):
        d["status"] = d[col].map(aqi.pm25_category if kind == "cat_pm" else aqi.aqi_category)
        fig = px.scatter_map(d, color="status", color_discrete_map=aqi.COLORS, category_orders={"status": list(aqi.COLORS)}, **kw)
    else:
        fig = px.scatter_map(d, color=col, color_continuous_scale=kind, **kw)
    fig.update_traces(marker={"size": 14})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0b1623",
                      legend=dict(bgcolor="rgba(15,33,51,.85)", font_color="#e6edf5", title_text=""))
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="worldmap")
    try:
        pts = ev.selection.points
    except Exception:
        pts = []
    if pts and pts[0].get("customdata") and pts[0]["customdata"][0] != ss.get("last_pick"):
        nm = pts[0]["customdata"][0]; r = tbl[tbl["name"] == nm].iloc[0]
        ss["last_pick"] = nm; ss["loc"] = loc = {"name": nm, "lat": float(r.lat), "lon": float(r.lon)}
        st.rerun()

    def _pick():
        r = tbl[tbl["name"] == ss["pick"]].iloc[0]
        ss["loc"] = {"name": r["name"], "lat": float(r.lat), "lon": float(r.lon)}
    st.selectbox("…or choose a city", sorted(tbl["name"]), index=None, placeholder="Pick a city", key="pick", on_change=_pick)
    st.caption("Weather: Open-Meteo forecast models. PM2.5 / AQI: CAMS atmospheric model estimates (not ground-station readings). "
               "Click a dot, pick a city, or search in the sidebar.")
    row = tbl[tbl["name"] == loc["name"]]
    r = row.iloc[0] if len(row) else now_one(loc["name"], loc["lat"], loc["lon"])
    st.subheader(f"📍 {loc['name']}: right now")
    k = st.columns(5)
    k[0].metric("Temperature", fmt(r.temperature_2m) + " °C"); k[1].metric("Humidity", fmt(r.relative_humidity_2m, "{:.0f}") + " %")
    k[2].metric("Wind", fmt(r.wind_speed_10m) + " km/h"); k[3].metric("Precipitation", fmt(r.precipitation) + " mm")
    k[4].metric("Cloud cover", fmt(r.cloud_cover, "{:.0f}") + " %")
    k = st.columns(4)
    k[0].metric("Pressure", fmt(r.surface_pressure, "{:.0f}") + " hPa"); k[1].metric("UV index", fmt(r.uv_index))
    k[2].metric("PM2.5 µg/m³ (model)", fmt(r.pm2_5), aqi.pm25_category(r.pm2_5), delta_color="off")
    k[3].metric("US AQI (model)", fmt(r.us_aqi, "{:.0f}"), aqi.aqi_category(r.us_aqi), delta_color="off")
    advice(aqi.pm25_category(r.pm2_5))

elif section == S_FC:
    st.title(f"Forecast charts: {loc['name']}")
    try:
        hrs, air = forecasts(loc["lat"], loc["lon"])
    except ApiError as e:
        st.error(str(e)); st.stop()
    H = {"24 h": 24, "48 h": 48, "72 h": 72, "7 days": 168}[st.radio("Time range", ["24 h", "48 h", "72 h", "7 days"], horizontal=True, index=1)]
    d = hrs.loc[now:now + pd.Timedelta(hours=H)]
    t = st.tabs(["🌡️ Temperature", "🌧️ Rain", "💨 Wind", "💧 Humidity & clouds", "🌫️ Air quality", "☀️ UV & pressure"])
    with t[0]:
        st.plotly_chart(line(d, ["temperature_2m", "apparent_temperature"], "Temperature & feels-like (°C)"), width="stretch")
    with t[1]:
        f = make_subplots(specs=[[{"secondary_y": True}]])
        f.add_bar(x=d.index, y=d["precipitation"], name="Rain (mm/h)")
        f.add_scatter(x=d.index, y=d["precipitation_probability"], name="Chance of rain (%)", secondary_y=True)
        f.update_layout(template="plotly_dark", title="Rain amount and probability", hovermode="x unified")
        st.plotly_chart(f, width="stretch")
    with t[2]:
        st.plotly_chart(line(d, ["wind_speed_10m", "wind_gusts_10m"], "Wind speed & gusts (km/h)"), width="stretch")
    with t[3]:
        st.plotly_chart(line(d, ["relative_humidity_2m", "cloud_cover"], "Humidity & cloud cover (%)"), width="stretch")
    with t[4]:
        a = air.loc[now:now + pd.Timedelta(hours=min(H, 120))] if len(air) else air
        if len(a):
            st.plotly_chart(line(a, [c for c in ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"] if c in a], "Pollutants forecast (µg/m³, CAMS model)"), width="stretch")
            st.plotly_chart(line(a, ["us_aqi"], "US AQI forecast (CAMS model)"), width="stretch")
        else:
            st.info("Air-quality forecast not available for this place.")
    with t[5]:
        st.plotly_chart(line(d, ["uv_index"], "UV index"), width="stretch")
        st.plotly_chart(line(d, ["surface_pressure"], "Surface pressure (hPa)"), width="stretch")
    dd = hrs.loc[now:now + pd.Timedelta(days=7)].resample("D").agg(
        min_temp=("temperature_2m", "min"), max_temp=("temperature_2m", "max"), rain_mm=("precipitation", "sum"),
        max_wind=("wind_speed_10m", "max"), max_uv=("uv_index", "max")).round(1)
    st.subheader("Day-by-day summary"); st.dataframe(dd, width="stretch")
    st.download_button("Download hourly forecast (CSV)", hrs.to_csv().encode(), "forecast.csv", "text/csv")
    st.caption("Weather forecast: Open-Meteo numerical weather models (external). Our own ML model predicts PM2.5 only (AI page).")

elif section == S_CMP:
    st.title("Compare places")
    look = {n: (la, lo) for n, la, lo in world.WORLD}; look.setdefault(loc["name"], (loc["lat"], loc["lon"]))
    pick = st.multiselect("Places (up to 5)", sorted(look), default=[loc["name"]], max_selections=5)
    c1, c2 = st.columns(2)
    var = c1.selectbox("Variable", list(CMP_VARS))
    H = {"24 h": 24, "48 h": 48, "72 h": 72}[c2.radio("Time range", ["24 h", "48 h", "72 h"], horizontal=True, index=1)]
    col, src = CMP_VARS[var]; series = {}
    try:
        for n in pick:
            hrs, air = forecasts(*look[n]); s = (hrs if src == "w" else air)
            if col in s:
                series[n] = s[col].loc[now:now + pd.Timedelta(hours=H)]
    except ApiError as e:
        st.error(str(e)); st.stop()
    if series:
        w = pd.DataFrame(series); w.index.name = "time"
        st.plotly_chart(line(w, list(w.columns), f"{var}: forecast comparison"), width="stretch")
        st.dataframe(w.agg(["min", "mean", "max"]).round(1).T, width="stretch")
    else:
        st.info("Pick at least one place with available data.")

elif section == S_AI:
    st.title(f"AI PM2.5 prediction: {loc['name']}")
    key = api_key(); mp, jp = ART / "model.joblib", ART / "metrics.json"
    if not mp.exists():
        st.info("No trained model found. Run the steps in the README first."); st.stop()
    if not key:
        st.info("Set OPENAQ_API_KEY (env var or Streamlit secret) to use ground-station data."); st.stop()
    try:
        wx, elev, air, station, pm = live(loc["lat"], loc["lon"], key)
    except ApiError as e:
        st.error(str(e)); st.stop()
    if not station:
        st.info("No PM2.5 ground station with fresh data within 25 km, so the model cannot predict here. Try a city near a monitor, e.g. Delhi, Berlin or Seoul."); st.stop()
    meta = json.load(open(jp)); model = joblib.load(mp)
    df = wx.join(pm.rename("pm25"), how="left")
    if len(air) and "pm2_5" in air:
        df = df.join(air["pm2_5"].rename("cams_pm25"))
    X = features.build_features(df, loc["lat"], loc["lon"], elev, with_target=False)
    anchor = pm.dropna().index[-1]; row = X.loc[[anchor], meta["features"]]
    if row.isna().any().any():
        st.error("Not enough recent data (need ≥18 of the last 24 hourly readings, a 24 h weather forecast and the CAMS forecast).")
    else:
        inv = lambda m: float(np.clip(np.expm1(m.predict(row))[0], 0, None))
        yhat = inv(model); t = anchor + pd.Timedelta(hours=meta["horizon_h"]); lo = hi = None
        if (ART / "model_q10.joblib").exists():
            lo, hi = min(inv(joblib.load(ART / "model_q10.joblib")), yhat), max(inv(joblib.load(ART / "model_q90.joblib")), yhat)
        cams = X.at[anchor, "cams_f24"]
        k = st.columns(3)
        k[0].metric(f"Predicted PM2.5 at {t:%Y-%m-%d %H:%M} UTC (our ML model)", f"{yhat:.1f} µg/m³", aqi.pm25_category(yhat), delta_color="off")
        k[1].metric(f"Last measured ({anchor:%H:%M} UTC)", f"{float(pm[anchor]):.1f} µg/m³")
        k[2].metric("CAMS model forecast, same hour", fmt(float(np.expm1(cams)) if cams == cams else None) + " µg/m³")
        res = meta["results"][meta["best_model"]]; err = (res.get("test_unseen") or res["test_seen"])["rmse"]
        st.caption(f"Station: {station['name']} ({station['dist_km']} km, {station['provider']}). Model: {meta['best_model']}, {meta['horizon_h']} h ahead. "
                   f"Typical error (test RMSE) ±{err:.1f} µg/m³." + (f" Likely range (80% interval): {lo:.1f} to {hi:.1f} µg/m³." if lo is not None else "")
                   + " This is a prediction, not an observation.")
        advice(aqi.pm25_category(yhat))
        dm = pm.dropna()
        pt = go.Scatter(x=[t], y=[yhat], mode="markers", marker=dict(size=12, color="orange"), name="Predicted")
        if lo is not None:
            pt.update(error_y=dict(type="data", symmetric=False, array=[hi - yhat], arrayminus=[yhat - lo]))
        fig = go.Figure([go.Scatter(x=dm.index, y=dm.values, name="Measured"), pt])
        fig.update_layout(template="plotly_dark", title="Measured PM2.5 and 24 h prediction (µg/m³)")
        st.plotly_chart(fig, width="stretch")

elif section == S_BOARD:
    st.title("Live scoreboard: real predictions vs real measurements")
    if not LIVE.exists():
        st.info("No live predictions logged yet. Turn on the scheduled GitHub Action (see README). Results appear once predictions are 24 h old."); st.stop()
    L = pd.read_csv(LIVE)
    for c in ("anchor", "target"):
        L[c] = pd.to_datetime(L[c], utc=True)
    cols = ["pred", "persistence"] + (["cams"] if L["cams"].notna().any() else [])
    done = L.dropna(subset=["observed"] + cols)
    st.write(f"{len(L)} predictions logged for {L.city.nunique()} cities; {len(done)} already checked against measurements.")
    if done.empty:
        st.info("Waiting for the first predictions to mature."); st.stop()
    names = {"pred": "Our model", "persistence": "Persistence (same as now)", "cams": "CAMS model"}
    err = {names[c]: done[c] - done.observed for c in cols}
    st.dataframe(pd.DataFrame({k: {"MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean()))} for k, e in err.items()}).T.round(2), width="stretch")
    if done.lo.notna().any():
        st.caption(f"80% prediction interval contained the real value {((done.observed >= done.lo) & (done.observed <= done.hi)).mean():.0%} of the time (target 80%).")
    day = done.assign(day=done.target.dt.floor("D"))
    daily = pd.DataFrame({names[c]: (day[c] - day.observed).abs().groupby(day.day).mean() for c in cols})
    daily.index.name = "time"
    st.plotly_chart(line(daily, list(daily.columns), "Daily mean absolute error (µg/m³), lower is better"), width="stretch")
    pc = done.groupby("city").apply(lambda g: pd.Series({"n": len(g), "MAE_model": (g.pred - g.observed).abs().mean(),
                                                         "MAE_persistence": (g.persistence - g.observed).abs().mean()}), include_groups=False)
    pc["better_than_persistence"] = pc.MAE_model < pc.MAE_persistence
    st.dataframe(pc.round(2), width="stretch")
    st.subheader("Latest predictions still waiting for their measurement")
    st.dataframe(L[L.observed.isna()].sort_values("anchor", ascending=False).head(30)[["city", "target", "pred", "lo", "hi", "persistence", "cams"]].round(1), hide_index=True)

else:
    st.title("Data & Methods")
    st.write("**World map & forecasts:** Open-Meteo (free). Weather = numerical weather models; map air quality = CAMS model estimates. "
             "**AI model:** trained on real ground-station PM2.5 from OpenAQ (hourly means, invalid values removed, stations need ≥50% coverage) plus ERA5 weather "
             "and the CAMS model PM2.5, one global model predicting log(1+PM2.5) 24 h ahead, also tested on whole cities never seen in training.")
    st.caption("Limitations: station coverage is uneven worldwide; training uses ERA5 and archived CAMS values in place of true 24 h forecasts, so live errors are larger "
               "(the Live Scoreboard measures exactly that).")
    if (ART / "metrics.json").exists():
        meta = json.load(open(ART / "metrics.json"))
        st.write(f"Trained {meta['trained_at']} on {meta['n_cities']} cities, {meta['n_rows']:,} hourly rows, {meta['period'][0][:10]} to {meta['period'][1][:10]}. "
                 f"Best model: **{meta['best_model']}**. Held-out cities: {', '.join(meta['unseen_cities']) or 'none'}. "
                 f"80% interval coverage on test data: {meta.get('interval_80_coverage', float('nan')):.0%}.")
        st.dataframe(pd.DataFrame([{"model": n, "split": s, **m} for n, r in meta["results"].items() for s, m in r.items()]), hide_index=True)
        if meta.get("events"):
            st.write("Unhealthy-day detection (PM2.5 > 35.4 µg/m³) and category accuracy on test data")
            st.dataframe(pd.DataFrame(meta["events"]).T.round(3))
        for f, label in [("region_scores.csv", "Scores by region"), ("city_scores.csv", "Scores by city")]:
            if (ART / f).exists():
                st.write(label); st.dataframe(pd.read_csv(ART / f), hide_index=True)
    if (PROC / "stations.csv").exists():
        s = pd.read_csv(PROC / "stations.csv"); st.write("Stations used (provenance)"); st.dataframe(s, hide_index=True)
        st.download_button("Download stations.csv", s.to_csv(index=False).encode(), "stations.csv", "text/csv")
    if (PROC / "dataset_sample.csv").exists():
        st.download_button("Download dataset sample (CSV)", (PROC / "dataset_sample.csv").read_bytes(), "dataset_sample.csv", "text/csv")
    if (PROC / "dataset.csv.gz").exists():
        st.download_button("Download full dataset (CSV, gzip)", (PROC / "dataset.csv.gz").read_bytes(), "dataset.csv.gz")
