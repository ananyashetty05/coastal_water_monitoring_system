"""
core/predictor.py  —  FINAL FIXED
Key fixes vs previous version:
  1. OLS trend is the ANCHOR — forecast always follows the actual slope
  2. RF/ARIMA provide residual corrections ON TOP of OLS trend
  3. CI based on recent 12-month residual volatility — not all-time std
     so it stays narrow and doesn't form a trumpet/triangle shape
  4. Sequential (no threads) — safe on Windows + Streamlit
  5. Monthly-native — resamples to monthly before fitting
"""
from __future__ import annotations
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    RF_OK = True
except ImportError:
    RF_OK = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_OK = True
except ImportError:
    ARIMA_OK = False


# ── resample to monthly ────────────────────────────────────────────────────────

def _to_monthly(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    monthly = series.resample("MS").mean().dropna()
    return monthly if not monthly.empty else series


# ── OLS trend (the anchor — always used) ──────────────────────────────────────

def _ols_trend(y: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      trend_pred  — OLS extrapolation for future steps
      trend_full  — OLS fit on all y (for residual computation)
      mae         — in-sample MAE
    """
    n = len(y)
    x = np.arange(n, dtype=float)

    if n < 2:
        flat = np.full(steps, float(y[-1]))
        return flat, np.full(n, float(y[-1])), 999.0

    c          = np.polyfit(x, y, 1)
    trend_full = np.polyval(c, x)
    future_x   = n + np.arange(steps, dtype=float)
    trend_pred = np.polyval(c, future_x)
    mae        = float(np.mean(np.abs(y - trend_full)))
    return trend_pred, trend_full, mae


# ── RF on residuals (correction on top of OLS trend) ──────────────────────────

def _rf_residual(resid: np.ndarray, steps: int) -> tuple[np.ndarray | None, float]:
    """Forecast residuals using RF, then add back to OLS trend."""
    if not RF_OK or len(resid) < 6:
        return None, 999.0
    try:
        lags = min(3, len(resid) // 2)
        X, tgt = [], []
        for i in range(lags, len(resid)):
            X.append(np.concatenate([[float(i)], resid[i-lags:i][::-1]]))
            tgt.append(resid[i])
        X, tgt = np.array(X), np.array(tgt)
        sp  = max(1, int(len(X) * 0.8))
        mdl = RandomForestRegressor(n_estimators=50, max_depth=4,
                                    random_state=42, n_jobs=1)
        mdl.fit(X[:sp], tgt[:sp])
        mae = (float(mean_absolute_error(tgt[sp:], mdl.predict(X[sp:])))
               if sp < len(X) else float(np.std(tgt) + 1e-6))
        mdl.fit(X, tgt)
        history = list(resid[-lags:])
        out = []
        for s in range(steps):
            feat = np.concatenate([[float(len(resid)+s)],
                                   np.array(history[-lags:])[::-1]])
            p = float(mdl.predict(feat.reshape(1,-1))[0])
            out.append(p); history.append(p)
        return np.array(out), mae
    except Exception:
        return None, 999.0


# ── ARIMA on residuals ────────────────────────────────────────────────────────

def _arima_residual(resid: np.ndarray, steps: int) -> tuple[np.ndarray | None, float]:
    """Forecast residuals using ARIMA."""
    if not ARIMA_OK or len(resid) < 24:
        return None, 999.0
    try:
        sp  = max(1, int(len(resid) * 0.8))
        m   = ARIMA(resid[:sp], order=(1, 0, 1)).fit()  # AR on stationary residuals
        val = m.forecast(steps=len(resid)-sp)
        mae = float(np.mean(np.abs(resid[sp:] - val)))
        full_resid_fc = np.array(ARIMA(resid, order=(1,0,1)).fit().forecast(steps=steps))
        return full_resid_fc, mae
    except Exception:
        return None, 999.0


# ── combine: OLS trend + weighted residual corrections ────────────────────────

def _ensemble_forecast(y: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Final forecast = OLS_trend + weighted_residual_correction.
    CI = ±1.96 × recent_residual_std (last 12 months or all if fewer).
    Returns (forecast, ci_halfwidth, info_dict).
    """
    trend_pred, trend_full, ols_mae = _ols_trend(y, steps)
    resid = y - trend_full

    rf_resid,    rf_mae    = _rf_residual(resid, steps)
    arima_resid, arima_mae = _arima_residual(resid, steps)

    # build weighted correction
    candidates = {}
    if rf_resid is not None and not np.any(np.isnan(rf_resid)):
        candidates["RF"]    = (rf_resid,    rf_mae)
    if arima_resid is not None and not np.any(np.isnan(arima_resid)):
        candidates["ARIMA"] = (arima_resid, arima_mae)

    correction = np.zeros(steps)
    weights    = {}
    maes       = {"OLS": round(ols_mae, 4)}
    models_used = ["OLS"]

    if candidates:
        inv = {k: 1.0/max(v[1], 1e-6) for k,v in candidates.items()}
        tot = sum(inv.values())
        w   = {k: inv[k]/tot for k in inv}
        for k, (arr, _) in candidates.items():
            correction += w[k] * arr
        weights = {k: round(v, 3) for k,v in w.items()}
        maes.update({k: round(candidates[k][1], 4) for k in candidates})
        models_used = ["OLS"] + list(candidates.keys())

    forecast = trend_pred + correction

    # CI: based on recent volatility of residuals (last 12mo or all)
    window     = min(12, len(resid))
    recent_std = float(np.std(resid[-window:])) if window > 1 else float(np.std(resid))
    recent_std = max(recent_std, 1e-6)

    # slight widening with horizon but capped at 3× recent_std
    horizon_factor = np.sqrt(np.arange(1, steps+1) / max(steps, 1))
    ci_half        = np.clip(1.96 * recent_std * (1 + horizon_factor * 0.3),
                             1e-6, recent_std * 3)

    info = {
        "weights":  weights,
        "maes":     maes,
        "models":   models_used,
        "std":      ci_half,
        "ols_slope_per_month": round(float(
            np.polyfit(np.arange(len(y)), y, 1)[0]), 6
        ),
    }
    return forecast, ci_half, info


# ── forecast one metric ────────────────────────────────────────────────────────

def _forecast_one(series: pd.Series, metric: str, steps: int, schema: dict) -> dict | None:
    monthly = _to_monthly(series)
    y       = np.array(monthly.values, dtype=float)
    y       = y[~np.isnan(y)]
    if len(y) == 0:
        return None

    clamp  = schema.get("metrics", {}).get(metric, {}).get("clamp") or [None, None]
    lo, hi = clamp[0], clamp[1]

    # future month-start dates
    last_month   = pd.Timestamp(monthly.index[-1]).to_period("M")
    future_dates = [(last_month + i).to_timestamp() for i in range(1, steps+1)]

    forecast, ci_half, info = _ensemble_forecast(y, steps)

    if lo is not None or hi is not None:
        forecast = np.clip(forecast, lo, hi)

    lower = forecast - ci_half
    upper = forecast + ci_half
    if lo is not None or hi is not None:
        lower = np.clip(lower, lo, hi)
        upper = np.clip(upper, lo, hi)

    n = len(y)
    if   n < 2:  method = "Flat"
    elif n < 6:  method = f"OLS (n={n}mo)"
    elif n < 24: method = f"OLS+RF (n={n}mo)"
    else:        method = f"OLS+RF+ARIMA (n={n}mo)"

    current = float(y[-1])
    end_val = float(forecast[-1])

    return {
        "historical":     monthly,
        "historical_raw": series,
        "forecast":       [round(float(v), 4) for v in forecast],
        "lower":          [round(float(v), 4) for v in lower],
        "upper":          [round(float(v), 4) for v in upper],
        "future_dates":   future_dates,
        "ensemble_info":  info,
        "current":        round(current, 4),
        "end_value":      round(end_val, 4),
        "change":         round(end_val - current, 4),
        "day7":           round(end_val, 4),
        "n_points":       n,
        "n_raw":          len(series),
        "method":         method,
    }


# ── public API ────────────────────────────────────────────────────────────────

def forecast(
    df: pd.DataFrame,
    location: str,
    schema: dict,
    steps: int = 12,
    metrics_filter: list[str] | None = None,
    progress_cb=None,
) -> dict:
    """Monthly ensemble forecast. steps = months ahead."""
    loc_df = df[df["location"] == location].sort_values("date")
    if loc_df.empty:
        return {}

    loc_df = loc_df.copy()
    try:
        loc_df["date"] = pd.to_datetime(loc_df["date"])
    except Exception:
        pass

    all_keys = list(schema.get("metrics", {}).keys())
    keys     = [k for k in (metrics_filter or all_keys) if k in all_keys]

    jobs = [
        (metric, loc_df.set_index("date")[metric].dropna())
        for metric in keys
        if metric in loc_df.columns and loc_df[metric].dropna().shape[0] >= 1
    ]

    if not jobs:
        return {}

    results = {}
    for i, (metric, series) in enumerate(jobs):
        try:
            result = _forecast_one(series, metric, steps, schema)
            if result is not None:
                results[metric] = result
        except Exception:
            pass
        if progress_cb:
            progress_cb(metric, i + 1, len(jobs))

    return results