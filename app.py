import json, os, pathlib
import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src import openmeteo as om, openaq as oq, features, aqi, world, fieldmap, ui, wmo
from src.common import ApiError

st.set_page_config("WeatherRadar AI", "🌦️", layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)
PROC, ART, LIVE = pathlib.Path("data/processed"), pathlib.Path("artifacts"), pathlib.Path("data/live/predictions.csv")
P_TODAY, P_14, P_MAPS, P_PLACES, P_MORE = "🏠 Today", "📅 14 days", "🗺️ Maps", "📍 My places", "⚙️ More"
M_AI, M_BOARD, M_CMP, M_DOC = "🤖 AI PM2.5 prediction", "🏆 Live scoreboard", "⚖️ Compare places", "📚 Data & methods"
CMP_VARS = {"Temperature (°C)": ("temperature_2m", "w"), "Rain (mm/h)": ("precipitation", "w"), "Wind (km/h)": ("wind_speed_10m", "w"),
            "Humidity (%)": ("relative_humidity_2m", "w"), "Cloud cover (%)": ("cloud_cover", "w"), "UV index": ("uv_index", "w"),
            "PM2.5 (µg/m³, CAMS)": ("pm2_5", "a"), "US AQI (CAMS)": ("us_aqi", "a")}
HEALTH = {"Good": "Air is clean. Enjoy outdoor activities.",
          "Moderate": "Unusually sensitive people should reduce long, heavy outdoor exertion.",
          "Unhealthy for sensitive groups": "Children, older adults and people with heart or lung disease should limit prolonged outdoor exertion.",
          "Unhealthy": "Everyone should reduce prolonged outdoor exertion; sensitive groups should stay indoors.",
          "Very unhealthy": "Avoid outdoor exertion; sensitive groups stay indoors. A well-fitted N95/FFP2 mask helps if you must go out.",
          "Hazardous": "Stay indoors with windows closed, use an air purifier if you have one, and avoid outdoor activity."}
DEFAULT_PLACES = [{"name": "Delhi", "lat": 28.61, "lon": 77.21}, {"name": "Lucknow", "lat": 26.85, "lon": 80.95}, {"name": "Mumbai", "lat": 19.08, "lon": 72.88}]


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


def show_iframe(html, height):
    (st.iframe if hasattr(st, "iframe") else __import__("streamlit.components.v1", fromlist=["html"]).html)(html, height=height)


def parse_places(raw):
    out = []
    for part in (raw or "").split(";"):
        try:
            n, la, lo = part.split("|"); out.append({"name": n, "lat": float(la), "lon": float(lo)})
        except ValueError:
            pass
    return out


def dump_places(ps):
    return ";".join(f"{p['name'].replace('|', ' ').replace(';', ' ')}|{p['lat']:.2f}|{p['lon']:.2f}" for p in ps)


@st.cache_data(ttl=900, show_spinner="Loading weather…")
def app_fc(lat, lon):
    return world.app_forecast(lat, lon)


@st.cache_data(ttl=900, show_spinner=False)
def app_air(lat, lon):
    return world.app_air(lat, lon)


@st.cache_data(ttl=3600, show_spinner="Loading forecast grid (10 to 40 seconds the first time)…")
def field_data(region, with_air):
    return fieldmap.fetch_field(region, with_air)


@st.cache_data(ttl=900, show_spinner=False)
def world_file():
    return fieldmap.load_field("data/live/world_field.json.gz")


@st.cache_data(ttl=900, show_spinner=False)
def place_now(name, lat, lon):
    return world.now_for([(name, lat, lon)]).iloc[0]


@st.cache_data(ttl=900, show_spinner=False)
def measured_now(lat, lon, key):
    """Latest real PM2.5 (mean of the last 3 hourly readings) from the nearest fresh OpenAQ station, or None."""
    if not key:
        return None
    try:
        now = pd.Timestamp.now(tz="UTC")
        for s in oq.nearby_stations(lat, lon, key)[:3]:
            if s["last"] and now - pd.Timestamp(s["last"]) < pd.Timedelta(hours=6):
                p = oq.hours(s["sensor_id"], key, now - pd.Timedelta(hours=12), now, pause=0).dropna()
                if len(p) >= 2:
                    return {"pm": float(p.iloc[-3:].mean()), "name": s["name"], "dist": s["dist_km"]}
    except ApiError:
        return None
    return None


def scale_picker(lat, lon):
    return st.radio("AQI scale", ["US EPA", "India (CPCB, PM2.5-based)"], index=0, horizontal=True, key="scale_r",
                    help="The same air gives different numbers on different scales. AQI.in uses the US scale; Google in India shows the India scale.")


def air_view(lat, lon, air, scale):
    """Best available air quality: real station reading if there is one, otherwise the CAMS model. Returns (chip, hourly PM2.5 Series)."""
    mp = air["current"].get("pm2_5") if air else None
    hp = air["hourly"]["pm2_5"] if air else None
    m = measured_now(lat, lon, api_key())
    if m:
        ratio = min(6.0, max(0.4, m["pm"] / mp)) if mp and mp == mp and mp >= 3 else 1.0
        return ({"rate": aqi.rate(m["pm"], scale), "pm": m["pm"], "src": f"measured at {m['name']} ({m['dist']:.0f} km)"},
                None if hp is None else hp * ratio)
    if mp is not None and mp == mp:
        return {"rate": aqi.rate(mp, scale), "pm": mp, "src": "CAMS model estimate (no station nearby)"}, hp
    return None, hp


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
    f = px.line(d.reset_index(), x="time", y=cols, title=title, template="plotly_white")
    f.update_layout(legend_title=None, hovermode="x unified")
    return f


ss = st.session_state
if "places" not in ss:
    ss["places"] = parse_places(st.query_params.get("places")) or [dict(p) for p in DEFAULT_PLACES]
    ss["loc"] = (parse_places(st.query_params.get("loc")) or [ss["places"][0]])[0]


def set_loc(name, lat, lon):
    ss["loc"] = {"name": name, "lat": lat, "lon": lon}
    st.query_params["loc"] = dump_places([ss["loc"]])


def pick_result():
    r = ss["_res"][ss["resi"]]
    set_loc(r["name"], r["lat"], r["lon"])


st.markdown('<div class="topbar">🌦️ WeatherRadar <span>AI</span></div>', unsafe_allow_html=True)
c1, c2 = st.columns([3, 2])
q = c1.text_input("Search", placeholder="🔍  Search any city or place", label_visibility="collapsed")
if q:
    try:
        res = om.search_city(q)
    except ApiError as e:
        c2.error(str(e)); res = []
    ss["_res"] = res
    if res:
        c2.selectbox("Results", range(len(res)), index=None, placeholder="Choose a result…", key="resi", label_visibility="collapsed",
                     format_func=lambda i: f"{res[i]['name']}, {res[i]['admin']}, {res[i]['country']}", on_change=pick_result)
    else:
        c2.warning("No match found.")
loc = ss["loc"]
page = st.radio("Page", [P_TODAY, P_14, P_MAPS, P_PLACES, P_MORE], horizontal=True, label_visibility="collapsed")
now = pd.Timestamp.now(tz="UTC").floor("h")

if page in (P_TODAY, P_14):
    try:
        d = app_fc(loc["lat"], loc["lon"])
    except ApiError as e:
        st.error(str(e)); st.stop()
    air = app_air(loc["lat"], loc["lon"])
    cur, h, dly = d["current"], d["hourly"], d["daily"]
    now_l = pd.Timestamp(cur["time"]).floor("h")
    scale = scale_picker(loc["lat"], loc["lon"]) if page == P_TODAY else "US EPA"
    chip, hpm = air_view(loc["lat"], loc["lon"], air, scale)

if page == P_TODAY:
    st.markdown(ui.hero(loc["name"], cur, dly.iloc[0], chip), unsafe_allow_html=True)
    saved = any(p["name"] == loc["name"] for p in ss["places"])
    if st.button("✓ Saved in My places" if saved else "⭐ Save to My places", disabled=saved):
        ss["places"].append(dict(loc)); st.query_params["places"] = dump_places(ss["places"]); st.rerun()
    st.subheader("Next 24 hours")
    st.markdown(ui.hourly_strip(h, hpm, now_l, scale), unsafe_allow_html=True)
    st.subheader("Details")
    st.markdown(ui.details(cur, h.loc[now_l], chip), unsafe_allow_html=True)
    if chip:
        st.info(f"{chip['rate']['advice']}  \n*General guidance, not medical advice.*")
    st.subheader("Today")
    st.markdown(ui.day_parts(h, hpm, now_l.normalize(), scale), unsafe_allow_html=True)
    st.caption("Weather: Open-Meteo forecast models, shown in the place's local time. Air quality: real station readings (OpenAQ) when a station is within 25 km, otherwise the CAMS model, which can underestimate very polluted cities such as Delhi. The hourly AQI badges are the model forecast scaled to match the latest station reading. AQI here is calculated from PM2.5 only.")

elif page == P_14:
    st.subheader(f"14-day forecast: {loc['name']}")
    x = dly.index.strftime("%a %d")
    fig = go.Figure()
    fig.add_scatter(x=x, y=dly.temperature_2m_max, mode="lines+markers+text", text=[f"{v:.0f}°" for v in dly.temperature_2m_max],
                    textposition="top center", line=dict(color="#e0463c"), name="High")
    fig.add_scatter(x=x, y=dly.temperature_2m_min, mode="lines+markers+text", text=[f"{v:.0f}°" for v in dly.temperature_2m_min],
                    textposition="bottom center", line=dict(color="#1f6a96"), name="Low")
    fig.add_scatter(x=x, y=[dly.temperature_2m_min.min() - 4] * len(x), mode="text", text=[wmo.describe(c, 1)[1] for c in dly.weather_code],
                    textfont=dict(size=18), showlegend=False, hoverinfo="skip")
    fig.update_layout(template="plotly_white", height=380, showlegend=False, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(visible=False))
    st.plotly_chart(fig, width="stretch")
    st.markdown(ui.day_rows(dly), unsafe_allow_html=True)

elif page == P_MAPS:
    mode = st.radio("Map", ["🌧️ Rain radar", "🛰️ Satellite", "😷 Air quality", "🌬️ Wind & weather"], horizontal=True, label_visibility="collapsed")
    if mode.startswith("🌧️"):
        show_iframe(fieldmap.radar_html(loc["lat"], loc["lon"], loc["name"]), 640)
        st.caption("Radar: RainViewer free public data (past 2 hours; no radar forecast; zoom beyond 7 is enlarged). Radar does not cover every country. "
                   "Press ▶: frames load one by one (about 20 seconds) because the free service limits requests.")
    elif mode.startswith("🛰️"):
        today_utc = pd.Timestamp.now(tz="UTC").date()
        day = st.date_input("Date (UTC)", value=today_utc - pd.Timedelta(days=1), max_value=today_utc, help="Yesterday is the most complete. Pick an earlier day to compare, e.g. crop-fire season.")
        show_iframe(fieldmap.satellite_html(loc["lat"], loc["lon"], loc["name"], day.strftime("%Y-%m-%d")), 640)
        st.caption("Real satellite photos from NASA (VIIRS/MODIS via GIBS, free): clouds, smoke and haze, and active fires (the red dots, such as crop burning near Delhi in autumn). "
                   "Imagery is a daily mosaic and has gaps from clouds and satellite swaths. Haze brightness is not ground-level PM2.5.")
    else:
        regs = ["World (all)"] + [r for r in fieldmap.REGIONS if r != "World"]
        region = st.selectbox("Area", regs, help="World shows everything. Pick a region for a more detailed view that is loaded on demand.")
        data, note = None, ""
        try:
            if region == "World (all)":
                data = world_file()
                note = f"World grid updated {data['generated'][:16].replace('T', ' ')} UTC (every 12 h). " if data else \
                       "Coarse 12-degree fallback grid (the detailed world file is not built yet; see README). "
            if region == "World (all)" and not data:
                st.warning("The detailed world file is not built yet, so this is a rough 12-degree map. Build it once with `python scripts/build_world_field.py` (about 10 minutes, see README) for a much better one.")
            data = data or field_data("World" if region == "World (all)" else region, True)
        except ApiError as e:
            st.error(str(e)); st.stop()
        home = [loc["lat"], loc["lon"], loc["name"]]
        data = (fieldmap.prepare(data, ["us_aqi", "pm2_5"], wind=False, home=home, start="us_aqi") if mode.startswith("😷")
                else fieldmap.prepare(data, None, wind=True, home=home, start="wind_speed_10m"))
        show_iframe(fieldmap.build_html(data), 640)
        st.caption("Click the map for local values" + (" and health advice. " if mode.startswith("😷") else ". Moving lines show wind. ") +
                   "Press ▶ to animate 48 h. Fields are smoothed between a coarse forecast grid (Open-Meteo), so they show regional patterns, not street-level detail. "
                   "Air quality here is the CAMS model estimate, not station readings, and it can underestimate very polluted cities such as Delhi; the Today page uses real station readings when available. " + note)

elif page == P_PLACES:
    st.subheader("My places")
    scale = scale_picker(loc["lat"], loc["lon"])
    for i, p in enumerate(list(ss["places"])):
        try:
            r = place_now(p["name"], p["lat"], p["lon"])
        except ApiError:
            r = {}
        m = measured_now(p["lat"], p["lon"], api_key())
        pm_v = m["pm"] if m else r.get("pm2_5")
        st.markdown(ui.place_card(p["name"], r, ui.badge(aqi.rate(pm_v, scale))), unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Open", key=f"o{i}"):
            set_loc(p["name"], p["lat"], p["lon"]); st.toast(f"{p['name']} selected. Open the Today tab."); st.rerun()
        if b2.button("Remove", key=f"r{i}"):
            ss["places"].pop(i); st.query_params["places"] = dump_places(ss["places"]); st.rerun()
    st.caption("Search a city above, open the Today tab and press ⭐ to add places. Your list is stored in this page's web address: bookmark it to keep your places.")

elif page == P_MORE:
    sub = st.selectbox("More", [M_AI, M_BOARD, M_CMP, M_DOC], label_visibility="collapsed")
    if sub == M_AI:
        st.subheader(f"AI PM2.5 prediction: {loc['name']}")
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
            fig.update_layout(template="plotly_white", title="Measured PM2.5 and 24 h prediction (µg/m³)")
            st.plotly_chart(fig, width="stretch")

    elif sub == M_BOARD:
        st.subheader("Live scoreboard: real predictions vs real measurements")
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

    elif sub == M_CMP:
        st.subheader("Compare places")
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

    else:
        st.subheader("Data & Methods")
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
