# Costco Route & Weather Delay Forecasting

A data pipeline and ML model for **Costco delivery routes**: from depot to warehouses, with **segment-level weather** and a **weather-based delay prediction model**.

---

## Project structure (Frontend / Backend)

```
Costco_forecasting/
├── backend/          # API, models, data pipeline
│   ├── app.py        # Flask: serves frontend + POST /api/predict
│   ├── predict_delay.py
│   ├── build_delay_model.py
│   ├── requirements.txt
│   ├── data/         # routes, weather, source data
│   ├── cache/        # pipeline caches
│   ├── delay_model_*.joblib
│   ├── package.json  # Node pipeline
│   ├── get_routes.js, add_weather_to_routes.js, ...
│   └── README.md
├── frontend/         # Web UI for the demo
│   ├── index.html
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── README.md
├── README.md         # this file
└── PROJECT_SUMMARY.md
```

- **Backend**: Python API (Flask), delay models (Ridge, Random Forest, XGBoost), data pipeline (Node.js + Google Routes API), `data/` and `cache/`.
- **Frontend**: Static UI (HTML/CSS/JS) served by the backend; **source** = Tracy Depot (fixed), **destination** = Costco US locations from `costco_warehouses_full.json`; **Get routes & predict delay** calls Google Directions API and shows per-route delay (Ridge, Random Forest, XGBoost) and optional map.

---

## What This Project Does

- **Routes**: Depot → each Costco warehouse, with up to 3 alternative paths per destination.
- **Segments**: Each path is split into ~20 km “pitstops” for segment-level analysis.
- **Weather**: Every pitstop gets weather from the nearest station (weekly/daily).
- **Travel time**: Google Routes API provides distance and duration between consecutive pitstops.
- **Delay models**: Three regression models (**Ridge**, **Random Forest**, **XGBoost**) predict **delay %** vs baseline using weather.  
  **adjusted_duration = baseline_duration × (1 + delay_pct / 100)**.

---

## How to Run

1. **Backend (API + models + pipeline)**  
   See **`backend/README.md`**.  
   - `cd backend`  
   - `pip install -r requirements.txt`  
   - `python build_delay_model.py` (once)  
   - `python app.py` → serves API and frontend at **http://127.0.0.1:5000**

2. **Frontend**  
   No separate run; it is served by the backend. Open http://127.0.0.1:5000 after starting `backend/app.py`.

3. **Node pipeline (routes/weather)**  
   From `backend/`: `npm install`, then run `get_routes.js`, `add_weather_to_routes.js`, `add_segment_time_to_routes.js` as needed. See `backend/README.md`.

---

## Key Files Reference

| Purpose | Location |
|--------|----------|
| Flask API + serve frontend | `backend/app.py` |
| Delay models (Ridge, RF, XGBoost) | `backend/delay_model_*.joblib` |
| Predict delay % | `backend/predict_delay.py` |
| Build models | `backend/build_delay_model.py` |
| Routes + weather data | `backend/data/` |
| Demo UI | `frontend/index.html`, `frontend/static/` |
| Technical summary | `PROJECT_SUMMARY.md` |
