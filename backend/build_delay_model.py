"""
Build a regression model to predict delay % from baseline using segment data from
routes_with_weather_and_substation_time.json and weather from point_weekly_weather.json.

For each segment we look up weather_key in point_weekly_weather, aggregate 52-week
stats to features, define a synthetic delay % target from weather severity, and fit
a model. The saved model can be used later to predict delay_pct for any segment
given its weather features.

Usage:
  pip install -r requirements.txt
  python build_delay_model.py

Output:
  - delay_model_ridge.joblib
  - delay_model_random_forest.joblib
  - delay_model_xgboost.joblib  (each: model + scaler + feature_names)
"""

import json
import os
import numpy as np
from joblib import dump

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = os.path.dirname(os.path.abspath(__file__))
POINT_WEEKLY_WEATHER_PATH = os.path.join(ROOT, "data", "weather", "point_weekly_weather.json")
ROUTES_PATH = os.path.join(ROOT, "data", "routes", "routes_with_weather_and_substation_time.json")
MODEL_FILES = {
    "ridge": os.path.join(ROOT, "delay_model_ridge.joblib"),
    "random_forest": os.path.join(ROOT, "delay_model_random_forest.joblib"),
    "xgboost": os.path.join(ROOT, "delay_model_xgboost.joblib"),
}
MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MAX_SEGMENTS = 120_000

FEATURE_NAMES = [
    "temp_min_mean",
    "temp_max_mean",
    "temp_mean_mean",
    "snow_depth_mean",
    "prcp_total_mean",
    "visibility_mean",
    "wind_speed_mean",
    "wind_speed_max_mean",
    "wind_gust_max_mean",
    "journey_start_hour",
    "journey_start_day_of_week",
    "journey_start_month",
    "week_of_year",
]


def aggregate_weekly_weather(weekly_list):
    """
    Reduce 52-week list to mean features matching point_weekly_weather.json schema.
    Each week has: week, temp_mean_avg, temp_max_avg, temp_min_avg, snow_depth_avg,
    prcp_total_avg, wind_speed_mean_avg, wind_speed_max_avg, wind_gust_max_avg, visibility_avg.
    """
    if not weekly_list or len(weekly_list) == 0:
        return None
    keys_map = [
        ("temp_min_avg", "temp_min_mean"),
        ("temp_max_avg", "temp_max_mean"),
        ("temp_mean_avg", "temp_mean_mean"),
        ("snow_depth_avg", "snow_depth_mean"),
        ("prcp_total_avg", "prcp_total_mean"),
        ("visibility_avg", "visibility_mean"),
        ("wind_speed_mean_avg", "wind_speed_mean"),
        ("wind_speed_max_avg", "wind_speed_max_mean"),
        ("wind_gust_max_avg", "wind_gust_max_mean"),
    ]
    out = {}
    for src, dst in keys_map:
        vals = [w[src] for w in weekly_list if isinstance(w, dict) and src in w and w[src] is not None]
        if not vals:
            return None
        out[dst] = float(np.mean(vals))
    return out


def synthetic_delay_pct(agg):
    """
    Synthetic delay % from weather (no observed delays in data).
    Uses same fields as point_weekly_weather.json aggregation.
    """
    if agg is None:
        return 0.0
    delay = 0.0
    delay += min(20.0, 2.0 * (agg.get("snow_depth_mean") or 0))
    delay += min(15.0, 3.0 * (agg.get("prcp_total_mean") or 0))
    vis = agg.get("visibility_mean") or 10
    if vis < 10:
        delay += min(15.0, (10 - vis) * 2.0)
    wg = agg.get("wind_gust_max_mean")
    delay += min(10.0, 0.3 * (wg or 0))
    if (agg.get("temp_min_mean") or 40) < 32:
        delay += 12.0
    return min(50.0, max(0.0, delay))


def build_weather_aggregates(path):
    """Load point_weekly_weather.json and return dict key -> aggregated features."""
    print("Loading", path, "...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    agg = {}
    for key, val in data.items():
        if not isinstance(val, dict) or "weekly_weather" not in val:
            continue
        a = aggregate_weekly_weather(val["weekly_weather"])
        if a is not None:
            agg[key] = a
    print("  Aggregated weather for", len(agg), "points")
    return agg


def collect_segments(routes_path, weather_agg, max_segments):
    """Yield (features_vec, delay_pct) for each segment with valid weather.
    Augments each segment with random journey start hour/day/month so models are trained with datetime inputs.
    """
    print("Loading", routes_path, "...")
    with open(routes_path, "r", encoding="utf-8") as f:
        warehouses = json.load(f)
    np.random.seed(42)
    n = 0
    for wh in warehouses:
        for route in wh.get("routes") or []:
            path = route.get("path_coordinates") or []
            for i in range(len(path) - 1):
                if max_segments and n >= max_segments:
                    print("  Collected", n, "segments (capped)")
                    return
                pt = path[i]
                wk = pt.get("weather_key")
                if not wk or wk not in weather_agg:
                    continue
                feats = weather_agg[wk]
                # Random journey start time and week of year for training (no real timestamps in segment data)
                hour = np.random.randint(0, 24)
                day_of_week = np.random.randint(0, 7)
                month = np.random.randint(1, 13)
                week_of_year = np.random.randint(1, 53)
                vec = np.array([
                    feats.get("temp_min_mean", 50),
                    feats.get("temp_max_mean", 70),
                    feats.get("temp_mean_mean", 60),
                    feats.get("snow_depth_mean", 0),
                    feats.get("prcp_total_mean", 0),
                    feats.get("visibility_mean", 8),
                    feats.get("wind_speed_mean", 8),
                    feats.get("wind_speed_max_mean", 14),
                    feats.get("wind_gust_max_mean", 20),
                    float(hour),
                    float(day_of_week),
                    float(month),
                    float(week_of_year),
                ], dtype=np.float64)
                delay_pct = synthetic_delay_pct(feats)
                n += 1
                yield vec, delay_pct
    print("  Collected", n, "segments")


def main():
    weather_agg = build_weather_aggregates(POINT_WEEKLY_WEATHER_PATH)
    rows = list(collect_segments(ROUTES_PATH, weather_agg, MAX_SEGMENTS))
    if not rows:
        raise SystemExit("No segments with weather found.")
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    print("Training set:", X.shape[0], "samples,", X.shape[1], "features")
    print("Delay % (synthetic): min={:.2f} max={:.2f} mean={:.2f}".format(y.min(), y.max(), y.mean()))

    if not HAS_SKLEARN:
        raise SystemExit("scikit-learn is required for Ridge and Random Forest. pip install scikit-learn")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

    # 1. Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    print("Ridge          R² train={:.4f} val={:.4f}".format(ridge.score(X_train, y_train), ridge.score(X_val, y_val)))
    ridge.fit(X_scaled, y)
    dump({"model": ridge, "scaler": scaler, "feature_names": FEATURE_NAMES, "model_display_name": "Ridge"}, MODEL_FILES["ridge"])
    print("  Saved", MODEL_FILES["ridge"])

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("Random Forest  R² train={:.4f} val={:.4f}".format(rf.score(X_train, y_train), rf.score(X_val, y_val)))
    rf.fit(X_scaled, y)
    dump({"model": rf, "scaler": scaler, "feature_names": FEATURE_NAMES, "model_display_name": "Random Forest"}, MODEL_FILES["random_forest"])
    print("  Saved", MODEL_FILES["random_forest"])

    # 3. XGBoost (optional)
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        print("XGBoost        R² train={:.4f} val={:.4f}".format(xgb_model.score(X_train, y_train), xgb_model.score(X_val, y_val)))
        xgb_model.fit(X_scaled, y)
        dump({"model": xgb_model, "scaler": scaler, "feature_names": FEATURE_NAMES, "model_display_name": "XGBoost"}, MODEL_FILES["xgboost"])
        print("  Saved", MODEL_FILES["xgboost"])
    else:
        print("XGBoost        skipped (pip install xgboost to include)")


if __name__ == "__main__":
    main()
