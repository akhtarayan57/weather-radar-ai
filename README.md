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
