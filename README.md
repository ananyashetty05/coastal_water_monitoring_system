# CoastalWatch

CoastalWatch is a Streamlit dashboard for coastal water quality monitoring and forecasting in Ireland and England. The app loads monitoring station data, computes CCME Water Quality Index (WQI) ratings, visualizes station distribution on a map, and provides analytics plus a 7-day parameter forecast with quality classification and mitigation recommendations.

## 🚀 Features

- CSV ingest with schema validation and cleaning
- WQI classification pipeline (CCME + rule-based fallback)
- Model benchmarking (scikit-learn): logistic regression, KNN, random forest, gradient boosting, SVM, voting, stacking
- Location map (pydeck) with coloured WQI markers and station-side summary
- Analytics: parameter trends, trending scores, country-level summary and charts
- 7-day OLS forecast per metric + forecast WQI quality prediction
- Automated recommendations for degraded/marginal water quality
- CSS styling and responsive page sections

## 📁 Project structure

- `app.py` - landing page with intro and nav cards
- `pages/01_upload.py` - dataset upload, preview, and model benchmark
- `pages/02_map.py` - geospatial map and per-station metrics
- `pages/03_analytics.py` - time-series charts and CCME breakdowns
- `pages/04_predictions.py` - forecast engine output
- `core/processor.py` - data parsing, cleaning, and aggregations
- `core/classifier.py` - rule-based + ML classification + recommendations
- `core/predictor.py` - time-series forecasting and forecast quality predictions
- `core/state.py` - Streamlit session-state utilities
- `components/` - reusable UI components
- `assets/style.css` - app theme styles

## 📋 Data schema

Expected CSV headers:

- `Country` (Ireland / England)
- `Area` (station / waterbody)
- `Waterbody Type` (Coastal / Transitional / Estuarine / Sea Water)
- `Date` (DD-MM-YYYY)
- `Ammonia (mg/l)`
- `Biochemical Oxygen Demand (mg/l)`
- `Dissolved Oxygen (mg/l)`
- `Orthophosphate (mg/l)`
- `pH (ph units)`
- `Temperature (cel)`
- `Nitrogen (mg/l)`
- `Nitrate (mg/l)`
- `CCME_Values` (0-100)
- `CCME_WQI` (Excellent / Good / Marginal / Fair / Poor)

## ⚙️ Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit:

```bash
streamlit run app.py
```

4. Open the app in browser (usually `http://localhost:8501`).

## 🧪 Usage

1. Go to **Upload Data** and upload the dataset file (e.g., `data.csv`).
2. Monitor data loading stats, and view preview + WQI distribution.
3. Visit **Map View** to inspect stations geographically, filter by country/type.
4. Visit **Analytics** to view time-series trending and CCME ranking.
5. Visit **Predictions** for 7-day forecast of metrics + predicted WQI.

## 🛠️ Notes

- `core/processor.py` includes location centroid estimation for mapping from `Area`.
- `core/classifier.py` uses ML benchmarking when scikit-learn is installed, else rule-based class only.
- `core/predictor.py` uses OLS slope forecasting (default `horizon=7`) and classifies predicted rows.
- `components/summary_table.py`, `metric_row.py`, `quality_badge.py` provide Streamlit block components.

## 🔮 Extending

- Add more robust forecasting models (ARIMA, Prophet, LSTM).
- Add custom map tokens/styles (Mapbox) and region-clustering.
- Persist dataset in DB and implement user-based saved sessions.

## 🧾 license

Project has no explicit license file; add one if publishing (e.g., MIT).