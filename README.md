# WeatherRadar AI (global)
Predicts PM2.5 24 h ahead for any city that has a nearby ground monitor. One global model trained on real station data.

## Data (all free)
- **PM2.5 target:** OpenAQ v3 ground stations (needs a free key: https://explore.openaq.org). Not model output.
- **Weather:** Open-Meteo (ERA5 archive for training, forecast API live; non-commercial use, no key). Times UTC.

## Run
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
export OPENAQ_API_KEY=your_key                         # Windows: set OPENAQ_API_KEY=your_key  (or put it in a .env file)
python scripts/build_dataset.py --start 2024-10-01 --end 2026-09-30   # ~15-30 min, resumable
python scripts/add_cams.py                              # adds free CAMS model PM2.5 as a feature (no key)
python scripts/train_model.py
streamlit run app.py
```
Quick trial first: add `--limit 5` to build_dataset (you'll need ≥3 cities in a region for a held-out city).

## For your teacher
`data/processed/` holds `dataset.csv.gz` (every row: city, UTC hour, PM2.5, weather), `dataset_sample.csv`, and `stations.csv` (station, provider, distance, coverage). `artifacts/` holds `metrics.json`, `city_scores.csv`, `region_scores.csv`. Same files are downloadable in the app's "Data & Methods" page.

## Method
Features: PM2.5 lags/rolling stats (per city, backward-looking only), current and +24 h weather, location, local solar time, season. Target log(1+PM2.5). Models: Persistence, Ridge, HistGradientBoosting; best by validation RMSE. Splits: per-city chronological 70/15/15, plus whole cities held out (one per region) to test generalisation.

## Deploy (Streamlit Community Cloud)
Commit `artifacts/model.joblib` and `data/processed/` (remove the full dataset if >100 MB). Add `OPENAQ_API_KEY = "..."` under app Settings > Secrets.

## Limitations
Station coverage is uneven (dense in Europe/US/India/China, sparse in Africa/South America). Training uses ERA5 as stand-in for forecast weather, so live errors are larger. Live prediction needs a station with fresh data within 25 km. Free API rate limits apply.

## Honest notes on results
- Cities with no usable station (e.g. Shanghai, Chengdu, Cairo, Lagos, Nairobi, Kampala, Tehran, Bogota) are skipped; Africa and South America are thin in the final dataset. Report this, do not claim equal accuracy everywhere.
- Model predicts the change from current PM2.5 (residual on log scale) and is always compared with the persistence baseline, including on whole cities never seen in training.

## App sections
- **World Map:** ~107 cities coloured by status (air quality categories, temperature, humidity, wind, rain, cloud, UV, pressure). Click a dot, pick a city, or search any place.
- **Forecast Charts:** hourly Open-Meteo forecasts (24 h to 7 days): temperature, rain, wind, humidity/clouds, air quality (CAMS), UV, pressure.
- **AI PM2.5 Prediction:** our ML model (trained on real ground-station data), only where a live station exists.

## Live scoreboard (free, automatic)
`.github/workflows/live_log.yml` runs every 6 h on GitHub Actions: it predicts 24 h ahead for every dataset city, saves it to `data/live/predictions.csv`, and fills in the real measurement once the hour has passed. The app's **Live Scoreboard** page then shows real errors of our model vs persistence vs CAMS.
Setup: push the project to a GitHub repo (commit `artifacts/*.joblib`, `data/processed/stations.csv`), then Settings > Secrets and variables > Actions > add `OPENAQ_API_KEY`. Run it once from the Actions tab ("Run workflow"). Streamlit Cloud redeploys when the CSV is committed.
Use the same Python version on Streamlit Cloud as you trained with; scikit-learn is pinned in requirements.txt.

## Honest notes on the CAMS feature
CAMS PM2.5 from Open-Meteo's archive is a near-analysis value, better than a true 24 h-ahead forecast, so test scores are optimistic. Live scoreboard errors are the honest number to quote.

## Animated map
World Map > "Animated wind & colour map": colour-filled field (temperature, wind, humidity, cloud, rain, pressure, PM2.5, US AQI) with moving wind lines, a 48 h time slider (press play) and click-for-values. Data: Open-Meteo forecast grid (free, ~300-450 points per region, cached 1 h); drawn in the browser with Leaflet + CARTO basemap tiles. Fields are interpolated between grid points, so they are smooth overviews, not street-level detail. PM2.5/AQI layers are CAMS model estimates.

## Phone-app style screens (v2)
Tabs: Today (hero, next 24 h with AQI badges, details, day parts), 14 days, Maps (rain radar, air quality, wind & weather), My places, More (AI prediction, live scoreboard, compare, methods). Light theme is set in `.streamlit/config.toml`. Saved places and the selected place are stored in the page address (bookmark it).
Map tiles: Esri Light Gray (keyless) with OpenStreetMap fallback. CARTO tiles were removed because CARTO now requires an API key (they watermark "API KEY REQUIRED"). Radar: RainViewer free public API (past 2 h only, zoom 7 max, 100 requests/min per IP).

## World map data (detailed, free)
The Air quality and Wind & weather maps show the whole world from `data/live/world_field.json.gz` (6-degree grid, 72 h, ~600 KB), built by `scripts/build_world_field.py` (about 10 minutes, no key) and refreshed twice a day by `.github/workflows/world_field.yml`. Until that file exists the app falls back to a coarse 12-degree grid. Build it once now: `python scripts/build_world_field.py`, commit the file and push, or run the "world-field" workflow from the GitHub Actions tab.
Country borders and place names are drawn above the colours; the Opacity slider controls how strongly the colours cover the map.

## Air quality accuracy (important)
The free CAMS model (map layers, fallback values) is coarse and can badly underestimate very polluted cities such as Delhi (model about 58 ug/m3 vs about 190-250 measured). The Today and My places screens therefore use the latest real PM2.5 from the nearest OpenAQ station within 25 km (mean of the last 3 hourly readings) when a key is set, and say so ("measured at ..."). AQI is calculated from PM2.5 only, on the US EPA scale (default, same as AQI.in) or India's CPCB scale (toggle). Official AQI uses 24 h averages and other pollutants, so numbers can differ a little from official dashboards.

## Satellite tab and map fixes (v3)
Maps > Satellite uses NASA GIBS (free, no key): VIIRS true-colour photo, MODIS aerosol optical depth (haze/smoke/dust) and VIIRS active fires, for any UTC date (default yesterday). Layer ids: VIIRS_SNPP_CorrectedReflectance_TrueColor, MODIS_Combined_Value_Added_AOD, VIIRS_SNPP_Thermal_Anomalies_375m_All. Aerosol optical depth is a column measurement, not ground-level PM2.5.
Field maps (air quality, wind & weather): no repeated world copies, the view fills the width, subtle labels, and country borders from Natural Earth GeoJSON are drawn above the colours.
