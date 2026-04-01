"""
core/predictor.py
Hybrid forecasting + ML quality interpretation.

Core ideas:
- Parameter forecasting is still time-series based (OLS baseline)
- Forecasted parameter vectors are then classified through the supervised
  model stack from core.classifier (domain-integrated + ML integration)
- Degraded predictions trigger recommendations
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from core.classifier import classify, generate_recommendations
from core.processor import METRICS

# Physical clamp ranges (same as processor.py)
CLAMP = {
    "do": (0, 20),
    "ph": (0, 14),
    "ammonia": (0, 500),
    "bod": (0, 1000),
    "orthophosphate": (0, 200),
    "temp": (-2, 40),
    "nitrogen": (0, 500),
    "nitrate": (0, 500),
    "ccme_values": (0, 100),
}


def _ols_forecast(x: np.ndarray, y: np.ndarray, steps: int) -> np.ndarray:
    """Fit OLS on (x, y) and return `steps` future predictions."""
    if len(x) < 2 or x.max() == x.min():
        return np.full(steps, y[-1] if len(y) else np.nan)
    try:
        coeffs = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return np.full(steps, y[-1])
    future_x = x[-1] + np.arange(1, steps + 1, dtype=float)
    return np.polyval(coeffs, future_x)


def _build_forecast_rows(
    d: pd.DataFrame,
    predictions: dict[str, list[float]],
    horizon: int,
    future_dates: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    """Convert forecast vectors into per-day feature rows for ML classification."""
    rows: list[dict[str, Any]] = []
    station_meta = d.iloc[-1].to_dict()

    for i in range(horizon):
        row = {
            "date": future_dates[i],
            "location": station_meta.get("location"),
            "country": station_meta.get("country"),
            "waterbody_type": station_meta.get("waterbody_type"),
        }
        for m in METRICS:
            vals = predictions.get(m, [])
            row[m] = vals[i] if i < len(vals) else np.nan
        rows.append(row)

    return rows


def predict(
    df: pd.DataFrame,
    location: str,
    horizon: int = 7,
    classifier_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run OLS forecast for all METRICS, then classify each forecast day with ML.

    Returns
    -------
    {
      "predictions": {metric: [float x horizon]},
      "forecast_series": {metric: pd.DataFrame(Historical, Forecast)},
      "quality_forecast": [ {day, date, wqi_label, status, confidence, recommendations} ... ],
      "recommendations": [str...],  # deduplicated summary recommendations
      "benchmark": pd.DataFrame or None,
      "best_model_name": str | None
    }
    """
    d = df[df["location"] == location].sort_values("date").copy()
    if d.empty:
        raise ValueError(f"No data found for location: {location}")

    # Use the cached classifier bundle if available; otherwise fall back to
    # rule-based classification so forecasting stays responsive.
    model_bundle = classifier_bundle

    # Numeric day index from first observation
    d["_day"] = (d["date"] - d["date"].min()).dt.days.astype(float)

    last_date = d["date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]

    predictions: dict[str, list[float]] = {}
    forecast_series: dict[str, pd.DataFrame] = {}

    for metric in METRICS:
        if metric not in d.columns:
            continue
        metric_frame = (
            d[["date", "_day", metric]]
            .dropna()
            .groupby("date", as_index=False)
            .agg({"_day": "mean", metric: "mean"})
            .sort_values("date")
        )
        col = metric_frame[["_day", metric]]
        if col.empty:
            continue

        x = col["_day"].values
        y = col[metric].values
        raw_forecast = _ols_forecast(x, y, horizon)

        # Clamp to physical range
        lo, hi = CLAMP.get(metric, (-np.inf, np.inf))
        raw_forecast = np.clip(raw_forecast, lo, hi)

        predictions[metric] = [round(float(v), 4) for v in raw_forecast]

        # Build historical + forecast series for chart using one value per date.
        hist = metric_frame.set_index("date")[metric].rename("Historical")
        fcast = pd.Series(raw_forecast, index=pd.DatetimeIndex(future_dates), name="Forecast")
        forecast_series[metric] = pd.concat([hist, fcast], axis=1)

    # Forecast-day quality classification and recommendations.
    quality_forecast = []
    all_recommendations: list[str] = []
    for i, row in enumerate(_build_forecast_rows(d, predictions, horizon, future_dates), start=1):
        cls = classify(row, model_bundle=model_bundle)
        recs = cls.get("messages", []) or generate_recommendations(row, predicted_wqi=cls.get("wqi_label"))
        all_recommendations.extend(recs)

        quality_forecast.append(
            {
                "day": i,
                "date": row["date"],
                "wqi_label": cls.get("wqi_label", cls.get("status")),
                "status": cls.get("status"),
                "confidence": cls.get("confidence"),
                "method": cls.get("method"),
                "recommendations": recs,
            }
        )

    dedup_recs = list(dict.fromkeys([r for r in all_recommendations if r]))

    return {
        "predictions": predictions,
        "forecast_series": forecast_series,
        "quality_forecast": quality_forecast,
        "recommendations": dedup_recs,
        "benchmark": (model_bundle or {}).get("benchmark") if isinstance(model_bundle, dict) else None,
        "best_model_name": (model_bundle or {}).get("best_model_name") if isinstance(model_bundle, dict) else None,
    }
