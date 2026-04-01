"""
Handles all data loading, cleaning, and aggregation for the CoastalWatch dataset.

Actual CSV schema:
  Country, Area, Waterbody Type, Date,
  Ammonia (mg/l), Biochemical Oxygen Demand (mg/l), Dissolved Oxygen (mg/l),
  Orthophosphate (mg/l), pH (ph units), Temperature (cel),
  Nitrogen (mg/l), Nitrate (mg/l),
  CCME_Values, CCME_WQI
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# ── Internal column names (short aliases used throughout the app) ─────────────
COL_MAP = {
    "Country":                              "country",
    "Area":                                 "location",
    "Waterbody Type":                       "waterbody_type",
    "Date":                                 "date",
    "Ammonia (mg/l)":                       "ammonia",
    "Biochemical Oxygen Demand (mg/l)":     "bod",
    "Dissolved Oxygen (mg/l)":              "do",
    "Orthophosphate (mg/l)":               "orthophosphate",
    "pH (ph units)":                        "ph",
    "Temperature (cel)":                    "temp",
    "Nitrogen (mg/l)":                      "nitrogen",
    "Nitrate (mg/l)":                       "nitrate",
    "CCME_Values":                          "ccme_values",
    "CCME_WQI":                             "ccme_wqi",
}

NUMERIC_COLS = ["ammonia", "bod", "do", "orthophosphate", "ph", "temp",
                "nitrogen", "nitrate", "ccme_values"]

# WQI label order for sorting / colouring
WQI_ORDER = ["Excellent", "Good", "Marginal", "Fair", "Poor"]

# Approximate lat/lon centroids for known areas.
# Used to place map markers since the CSV has no coordinates.
AREA_COORDS: dict[str, tuple[float, float]] = {
    # Ireland
    "Ballyteigue":      (52.19, -6.82),
    "Cork":             (51.90, -8.47),
    "Dublin":           (53.33, -6.25),
    "Galway":           (53.27, -9.05),
    "Kerry":            (52.15, -9.57),
    "Louth":            (53.93, -6.49),
    "Mayo":             (53.85, -9.30),
    "Sligo":            (54.27, -8.47),
    "Waterford":        (52.26, -7.11),
    "Wexford":          (52.33, -6.46),
    "Wicklow":          (52.98, -6.04),
    "Clare":            (52.90, -9.00),
    "Donegal":          (54.65, -8.11),
    "Limerick":         (52.66, -8.63),
    # England
    "Colne":            (51.88,  0.93),
    "Crouch":           (51.63,  0.79),
    "Humber":           (53.72, -0.28),
    "Nene":             (52.68,  0.16),
    "Norfolk":          (52.74,  0.91),
    "Orwell":           (51.97,  1.22),
    "Thames":           (51.48,  0.32),
    "Wash":             (52.87,  0.33),
    "Blackwater":       (51.72,  0.82),
    "Deben":            (52.07,  1.34),
    "Stour":            (51.95,  1.07),
    "Bure":             (52.72,  1.55),
    "Hamford":          (51.89,  1.16),
    "Brightlingsea":    (51.80,  1.01),
    "Roach":            (51.60,  0.79),
    "Folkestone":       (51.08,  1.17),
    "Dover":            (51.13,  1.31),
    "Brighton":         (50.82, -0.14),
    "Portsmouth":       (50.80, -1.09),
    "Southampton":      (50.90, -1.40),
    "Plymouth":         (50.37, -4.14),
    "Bristol":          (51.45, -2.59),
    "Liverpool":        (53.41, -2.99),
    "Mersey":           (53.40, -3.00),
    "Tyne":             (54.97, -1.61),
    "Tees":             (54.57, -1.23),
    "Exe":              (50.62, -3.52),
    "Medway":           (51.40,  0.54),
    "Swale":            (51.37,  0.87),
    # USA – coastal monitoring stations
    "Chesapeake":       (37.50, -76.10),
    "Delaware":         (39.45, -75.35),
    "Long Island":      (40.80, -73.10),
    "New York":         (40.71, -74.01),
    "Boston":           (42.36, -71.06),
    "Miami":            (25.77, -80.19),
    "Tampa":            (27.95, -82.46),
    "Gulf":             (29.00, -90.00),
    "Mississippi":      (29.15, -89.25),
    "Mobile":           (30.69, -88.04),
    "Galveston":        (29.30, -94.80),
    "Houston":          (29.76, -95.37),
    "San Francisco":    (37.77,-122.42),
    "Los Angeles":      (33.94,-118.41),
    "San Diego":        (32.72,-117.16),
    "Seattle":          (47.61,-122.33),
    "Portland":         (45.52,-122.68),
    "Puget":            (47.50,-122.40),
    "Monterey":         (36.60,-121.90),
    "Santa Barbara":    (34.42,-119.70),
    "Narragansett":     (41.49, -71.42),
    "Buzzards":         (41.60, -70.80),
    "Cape Cod":         (41.85, -70.00),
    "Penobscot":        (44.50, -68.80),
    "Puget Sound":      (47.60,-122.35),
    "Apalachicola":     (29.73, -85.00),
    "Charlotte Harbor": (26.90, -82.10),
    "Indian River":     (27.80, -80.45),
    "Pamlico":          (35.40, -76.50),
    "Albemarle":        (36.05, -76.40),
    "Waquoit":          (41.55, -70.52),
    # China – coastal monitoring stations
    "Bohai":            (39.00, 120.00),
    "Yellow Sea":       (35.00, 122.00),
    "East China":       (30.00, 125.00),
    "South China":      (20.00, 115.00),
    "Pearl River":      (22.50, 113.60),
    "Yangtze":          (31.40, 121.50),
    "Shanghai":         (31.23, 121.47),
    "Guangzhou":        (23.13, 113.26),
    "Shenzhen":         (22.54, 114.06),
    "Tianjin":          (39.08, 117.20),
    "Qingdao":          (36.07, 120.38),
    "Dalian":           (38.91, 121.60),
    "Xiamen":           (24.48, 118.09),
    "Zhoushan":         (30.00, 122.10),
    "Hangzhou":         (30.25, 120.16),
    "Ningbo":           (29.87, 121.54),
    "Wenzhou":          (28.02, 120.67),
    "Fuzhou":           (26.07, 119.30),
    "Hainan":           (20.02, 110.33),
    "Beibu":            (21.50, 109.00),
    "Liaodong":         (40.50, 122.00),
    "Jiaozhou":         (36.28, 120.18),
}

# Country-level fallback centroids
_COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    "ireland":       (53.41, -8.24),
    "england":       (52.50, -1.50),
    "uk":            (54.00, -2.00),
    "united kingdom":(54.00, -2.00),
    "usa":           (37.09,-95.71),
    "united states": (37.09,-95.71),
    "us":            (37.09,-95.71),
    "china":         (35.86, 104.20),
    "france":        (46.23,  2.21),
    "germany":       (51.17, 10.45),
    "australia":     (-25.27, 133.78),
    "india":         (20.59, 78.96),
    "japan":         (36.20, 138.25),
}

_DEFAULT_COORD = (20.0, 0.0)  # World centre


def _guess_coord(area: str, country: str = "") -> tuple[float, float]:
    """Return an approximate lat/lon using area keyword matching, then country fallback."""
    area_upper = area.upper()
    for key, coord in AREA_COORDS.items():
        if key.upper() in area_upper:
            return coord
    # fallback to country centroid
    country_lower = country.lower().strip()
    for key, coord in _COUNTRY_COORDS.items():
        if key in country_lower:
            return coord
    return _DEFAULT_COORD


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_csv(file) -> pd.DataFrame:
    """
    Read and clean the CoastalWatch CSV.

    Accepts a file-like object (st.file_uploader) or a file path string.
    Returns a clean DataFrame with short internal column names.
    Raises ValueError on schema mismatch.
    """
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    missing = [c for c in COL_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
    df = df.rename(columns=COL_MAP)

    # Parse dates (format: DD-MM-YYYY)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Numeric coercion
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add approximate lat/lon from area name + country
    coords = df.apply(lambda r: _guess_coord(str(r["location"]), str(r["country"])), axis=1)
    df["lat"] = coords.apply(lambda c: c[0])
    df["lon"] = coords.apply(lambda c: c[1])

    # Clip physically implausible values
    clamp = {
        "do":             (0, 20),
        "ph":             (0, 14),
        "ammonia":        (0, 500),
        "bod":            (0, 1000),
        "orthophosphate": (0, 200),
        "temp":           (-2, 40),
        "nitrogen":       (0, 500),
        "nitrate":        (0, 500),
        "ccme_values":    (0, 100),
    }
    for col, (lo, hi) in clamp.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    return df


# ── Per-location statistics ───────────────────────────────────────────────────

METRICS = ["do", "ph", "ammonia", "bod", "orthophosphate",
           "temp", "nitrogen", "nitrate", "ccme_values"]

METRIC_LABELS = {
    "do":             "Dissolved Oxygen (mg/L)",
    "ph":             "pH",
    "ammonia":        "Ammonia (mg/L)",
    "bod":            "BOD (mg/L)",
    "orthophosphate": "Orthophosphate (mg/L)",
    "temp":           "Temperature (°C)",
    "nitrogen":       "Nitrogen (mg/L)",
    "nitrate":        "Nitrate (mg/L)",
    "ccme_values":    "CCME Score",
}


def get_stats(df: pd.DataFrame, location: str) -> dict:
    """
    Return a statistics dict for a single location.

    Per metric: { mean, min, max, std, latest, trend }
    Plus '_meta': record count, date range, country, waterbody type, CCME label.
    """
    d = df[df["location"] == location].copy().sort_values("date")
    if d.empty:
        return {}

    stats: dict = {}
    for m in METRICS:
        col = d[m].dropna()
        if col.empty:
            continue
        stats[m] = {
            "mean":   round(float(col.mean()),   4),
            "min":    round(float(col.min()),    4),
            "max":    round(float(col.max()),    4),
            "std":    round(float(col.std()),    4),
            "latest": round(float(col.iloc[-1]), 4),
            "trend":  _calc_trend(d, m),
        }

    latest_row = d.iloc[-1]
    stats["_meta"] = {
        "n":              len(d),
        "date_from":      d["date"].min().strftime("%Y-%m-%d"),
        "date_to":        d["date"].max().strftime("%Y-%m-%d"),
        "location":       location,
        "country":        latest_row["country"],
        "waterbody_type": latest_row["waterbody_type"],
        "ccme_wqi":       latest_row["ccme_wqi"],
        "ccme_values":    round(float(latest_row["ccme_values"]), 2)
                          if pd.notna(latest_row["ccme_values"]) else None,
    }
    return stats


def _calc_trend(d: pd.DataFrame, col: str, window_days: int = 90) -> float:
    """OLS slope (units/day) over the most recent `window_days`."""
    cutoff = d["date"].max() - timedelta(days=window_days)
    recent = d[d["date"] >= cutoff][["date", col]].dropna()
    if len(recent) < 2:
        return 0.0
    x = (recent["date"] - recent["date"].min()).dt.days.values.astype(float)
    y = recent[col].values.astype(float)
    # Guard: if all x values are identical polyfit will fail (SVD error)
    if x.max() == x.min():
        return 0.0
    try:
        slope = float(np.polyfit(x, y, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    return round(slope, 6)


# ── Location summaries (map page) ─────────────────────────────────────────────

def get_location_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """One row per location with latest readings, mean coords, and WQI label."""
    rows = []
    for loc, grp in df.groupby("location"):
        grp = grp.sort_values("date")
        last = grp.iloc[-1]

        def _safe(val):
            return round(float(val), 3) if pd.notna(val) else None

        rows.append({
            "location":       loc,
            "country":        last["country"],
            "waterbody_type": last["waterbody_type"],
            "lat":            grp["lat"].mean(),
            "lon":            grp["lon"].mean(),
            "n":              len(grp),
            "date":           last["date"].strftime("%Y-%m-%d"),
            "do":             _safe(last["do"]),
            "ph":             _safe(last["ph"]),
            "ammonia":        _safe(last["ammonia"]),
            "bod":            _safe(last["bod"]),
            "temp":           _safe(last["temp"]),
            "nitrogen":       _safe(last["nitrogen"]),
            "nitrate":        _safe(last["nitrate"]),
            "ccme_values":    _safe(last["ccme_values"]),
            "ccme_wqi":       last["ccme_wqi"],
        })
    return pd.DataFrame(rows)


# ── Time-series helper ────────────────────────────────────────────────────────

def get_timeseries(df: pd.DataFrame, location: str, metric: str) -> pd.Series:
    """Date-indexed Series for one location + metric (sorted, NaN dropped)."""
    return (
        df[df["location"] == location]
        .set_index("date")[metric]
        .sort_index()
        .dropna()
    )


# ── WQI distribution helper ───────────────────────────────────────────────────

def get_wqi_distribution(df: pd.DataFrame, location: str | None = None) -> pd.Series:
    """
    Counts of each CCME_WQI label, ordered Excellent → Poor.
    Pass location=None to aggregate across all locations.
    """
    subset = df if location is None else df[df["location"] == location]
    counts = subset["ccme_wqi"].value_counts()
    return counts.reindex([l for l in WQI_ORDER if l in counts.index], fill_value=0)


# ── Filter / subset helpers ───────────────────────────────────────────────────

def get_countries(df: pd.DataFrame) -> list[str]:
    return sorted(df["country"].dropna().unique().tolist())


def get_waterbody_types(df: pd.DataFrame) -> list[str]:
    return sorted(df["waterbody_type"].dropna().unique().tolist())


def filter_df(
    df: pd.DataFrame,
    country: str | None = None,
    waterbody_type: str | None = None,
    wqi_labels: list[str] | None = None,
    date_from=None,
    date_to=None,
) -> pd.DataFrame:
    """
    General-purpose filter used by analytics and map pages.
    All parameters are optional; pass None to skip.
    """
    mask = pd.Series(True, index=df.index)
    if country:
        mask &= df["country"] == country
    if waterbody_type:
        mask &= df["waterbody_type"] == waterbody_type
    if wqi_labels:
        mask &= df["ccme_wqi"].isin(wqi_labels)
    if date_from is not None:
        mask &= df["date"] >= pd.to_datetime(date_from)
    if date_to is not None:
        mask &= df["date"] <= pd.to_datetime(date_to)
    return df[mask].reset_index(drop=True)