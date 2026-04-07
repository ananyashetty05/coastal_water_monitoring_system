"""
core/processor.py  —  FINAL
Key fix: structural columns (date, location, depth) are extracted BEFORE
any metric renaming, so they can never be accidentally overwritten.
Also: smart location building — combines Zone + Station if both present,
falls back to any likely column if schema detection missed location_col.
"""
from __future__ import annotations
import re
from datetime import timedelta
import numpy as np
import pandas as pd

WQI_ORDER = ["Excellent", "Good", "Marginal", "Fair", "Poor"]


def _clean_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    s = s.astype(str).str.strip()
    s = s.replace(
        ["N/A","n/a","NA","na","-","--","","ND","nd",
         "BDL","<BDL","bdl","NaN","nan","<0"],
        np.nan,
    )
    s = s.str.replace(r"^[<>≤≥~]\s*", "", regex=True)
    s = s.str.replace(r"[^\d.\-eE]", "", regex=True)
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors="coerce")


_COORD_LOOKUP = {
    "hong kong": (22.32, 114.17), "tolo":       (22.48, 114.22),
    "victoria":  (22.28, 114.18), "ireland":    (53.41,  -8.24),
    "england":   (52.50,  -1.50), "usa":         (37.09, -95.71),
    "china":     (35.86, 104.20), "australia":  (-25.27, 133.78),
    "japan":     (36.20, 138.25), "india":       (20.59,  78.96),
    "uk":        (54.00,  -2.00), "france":      (46.23,   2.21),
    "mediterranean": (35.00, 18.00), "pacific":  (0.0,  -160.0),
    "atlantic":  (0.0,   -30.0),  "north sea":   (56.00,   3.00),
    "korea":     (36.50, 127.50), "taiwan":      (23.70, 120.96),
    "malaysia":  (4.21,  101.97), "singapore":   (1.35,  103.82),
}


def _coord(text: str) -> tuple[float, float]:
    t = text.lower()
    for k, v in _COORD_LOOKUP.items():
        if k in t:
            return v
    return (0.0, 0.0)


def _best_location_col(df: pd.DataFrame, exclude: set) -> str | None:
    """Find the best location-like column not already used."""
    candidates = [
        c for c in df.columns
        if c not in exclude and any(k in c.lower() for k in
           ["station","area","site","location","place","name","point","id"])
    ]
    return candidates[0] if candidates else None


def _best_zone_col(df: pd.DataFrame, loc_col: str, exclude: set) -> str | None:
    """Find a zone/area column to combine with station."""
    candidates = [
        c for c in df.columns
        if c not in exclude and c != loc_col
        and any(k in c.lower() for k in ["zone","area","region","district"])
    ]
    return candidates[0] if candidates else None


def parse(file, schema: dict) -> pd.DataFrame:
    """
    Parse any water quality CSV using schema from detector.
    Structural columns are locked in FIRST before any metric renaming.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Cannot read CSV: {e}")

    df.columns = df.columns.str.strip()
    metrics    = schema.get("metrics", {})

    # ── Step 0: identify all structural column names UP FRONT ────────────────
    # These must never be touched by metric renaming
    date_col    = schema.get("date_col")
    location_col = schema.get("location_col")
    country_col  = schema.get("country_col")
    waterbody_col = schema.get("waterbody_col")

    structural_originals = set(filter(None, [
        date_col, location_col, country_col, waterbody_col
    ]))

    # ── Step 1: extract structural columns into safe temp names ───────────────
    # Save them before any rename so metrics can't overwrite them
    _date_data     = df[date_col].copy()      if date_col      and date_col      in df.columns else None
    _location_data = df[location_col].copy()  if location_col  and location_col  in df.columns else None
    _country_data  = df[country_col].copy()   if country_col   and country_col   in df.columns else None
    _waterbody_data= df[waterbody_col].copy() if waterbody_col and waterbody_col in df.columns else None

    # also save zone column if it exists (for combining with station)
    _zone_data = None
    zone_col   = _best_zone_col(df, location_col or "", structural_originals)
    if zone_col and zone_col in df.columns:
        _zone_data = df[zone_col].copy()

    # ── Step 2: clean and rename metric columns ───────────────────────────────
    rename = {}
    for internal, meta in metrics.items():
        orig = meta.get("original_col", "")
        if not orig or orig not in df.columns:
            continue
        if orig in structural_originals:
            continue   # never rename a structural column to a metric key
        df[orig] = _clean_numeric(df[orig])
        clamp = meta.get("clamp") or [None, None]
        lo, hi = clamp[0], clamp[1]
        if lo is not None or hi is not None:
            df[orig] = df[orig].clip(lo, hi)
        rename[orig] = internal

    df = df.rename(columns=rename)

    # ── Step 3: restore structural columns from saved copies ─────────────────
    if _date_data is not None:
        df["__date_raw__"] = _date_data
    if _location_data is not None:
        df["__location_raw__"] = _location_data
    if _country_data is not None:
        df["__country_raw__"] = _country_data
    if _waterbody_data is not None:
        df["__waterbody_raw__"] = _waterbody_data
    if _zone_data is not None:
        df["__zone_raw__"] = _zone_data

    # ── Step 4: build DATE ────────────────────────────────────────────────────
    if "__date_raw__" in df.columns:
        df["date"] = pd.to_datetime(df["__date_raw__"], dayfirst=True, errors="coerce")
        df.drop(columns=["__date_raw__"], inplace=True)
    else:
        # last-ditch: scan all non-metric columns for date-like values
        for col in df.columns:
            if col.startswith("__") or col in rename.values():
                continue
            try:
                parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    df["date"] = parsed
                    break
            except Exception:
                pass

    if "date" not in df.columns:
        raise ValueError(
            f"No date column found. Columns: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        raise ValueError("All date values failed to parse.")

    # ── Step 5: build LOCATION ────────────────────────────────────────────────
    if "__location_raw__" in df.columns:
        station = df["__location_raw__"].astype(str)
        df.drop(columns=["__location_raw__"], inplace=True)

        if "__zone_raw__" in df.columns:
            zone = df["__zone_raw__"].astype(str)
            df.drop(columns=["__zone_raw__"], inplace=True)
            # combine: "Tolo Harbour and Channel — TM2"
            df["location"] = zone + " — " + station
        else:
            df["location"] = station
    else:
        # schema missed location_col — try to find it ourselves
        fallback_loc = _best_location_col(df, set(rename.values()) | {"date"})
        if fallback_loc:
            # also look for zone to combine
            fallback_zone = _best_zone_col(df, fallback_loc, set(rename.values()) | {"date"})
            if fallback_zone and fallback_zone in df.columns:
                df["location"] = df[fallback_zone].astype(str) + " — " + df[fallback_loc].astype(str)
            else:
                df["location"] = df[fallback_loc].astype(str)
        else:
            df["location"] = "Unknown Station"

    # clean up any remaining raw columns
    for tmp in ["__zone_raw__"]:
        if tmp in df.columns:
            df.drop(columns=[tmp], inplace=True)

    # ── Step 6: build COUNTRY ─────────────────────────────────────────────────
    if "__country_raw__" in df.columns:
        df["country"] = df["__country_raw__"].astype(str)
        df.drop(columns=["__country_raw__"], inplace=True)
    else:
        df["country"] = schema.get("detected_location", "Unknown")

    # ── Step 7: build WATERBODY / DEPTH ──────────────────────────────────────
    if "__waterbody_raw__" in df.columns:
        df["waterbody_type"] = df["__waterbody_raw__"].astype(str)
        df.drop(columns=["__waterbody_raw__"], inplace=True)
    elif "waterbody_type" not in df.columns:
        # try to find a depth-like column
        depth_fallback = next(
            (c for c in df.columns if any(k in c.lower()
             for k in ["depth","layer","level","surface","middle","bottom"])),
            None
        )
        df["waterbody_type"] = df[depth_fallback].astype(str) if depth_fallback else "Unknown"

    # ── Step 8: coordinates ───────────────────────────────────────────────────
    hint = schema.get("detected_location", "")
    df["lat"] = df.apply(
        lambda r: _coord(f"{r['location']} {r['country']} {hint}")[0], axis=1
    )
    df["lon"] = df.apply(
        lambda r: _coord(f"{r['location']} {r['country']} {hint}")[1], axis=1
    )

    return df.sort_values(["location", "date"]).reset_index(drop=True)


# ── Stats ──────────────────────────────────────────────────────────────────────

def get_stats(df: pd.DataFrame, location: str, schema: dict) -> dict:
    d = df[df["location"] == location].sort_values("date")
    if d.empty:
        return {}

    stats: dict = {}
    for key in schema.get("metrics", {}):
        if key not in d.columns:
            continue
        col = d[key].dropna()
        if len(col) < 2:
            continue

        cutoff = d["date"].max() - timedelta(days=90)
        recent = d[d["date"] >= cutoff][["date", key]].dropna()
        slope  = 0.0
        if len(recent) >= 2:
            x = (recent["date"] - recent["date"].min()).dt.days.values.astype(float)
            y = recent[key].values.astype(float)
            if x.max() != x.min():
                try:
                    slope = float(np.polyfit(x, y, 1)[0])
                except Exception:
                    pass

        stats[key] = {
            "mean":   round(float(col.mean()), 4),
            "min":    round(float(col.min()),  4),
            "max":    round(float(col.max()),  4),
            "std":    round(float(col.std()),  4),
            "latest": round(float(col.iloc[-1]), 4),
            "trend":  round(slope, 6),
        }

    last = d.iloc[-1]
    stats["_meta"] = {
        "n":        len(d),
        "date_from": d["date"].min().strftime("%Y-%m-%d"),
        "date_to":   d["date"].max().strftime("%Y-%m-%d"),
        "location":  location,
        "country":   str(last.get("country", "?")),
        "waterbody": str(last.get("waterbody_type", "?")),
    }
    return stats


def param_status(metric: str, value: float, schema: dict) -> tuple[str, str]:
    m    = schema.get("metrics", {}).get(metric, {})
    safe = m.get("safe_range") or [None, None]
    poor = m.get("poor_range") or [None, None]

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown", "#94a3b8"

    lo_p, hi_p = poor[0], poor[1]
    lo_s, hi_s = safe[0], safe[1]

    is_poor = (lo_p is not None and value <= lo_p) or (hi_p is not None and value >= hi_p)
    is_safe = (lo_s is None or value >= lo_s) and (hi_s is None or value <= hi_s)

    if is_poor: return "Poor",     "#ef4444"
    if is_safe: return "Safe",     "#22c55e"
    return              "Moderate", "#f59e0b"


# ── Filter helpers ─────────────────────────────────────────────────────────────

def get_countries(df: pd.DataFrame) -> list[str]:
    return sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []


def get_waterbody_types(df: pd.DataFrame) -> list[str]:
    return sorted(df["waterbody_type"].dropna().unique().tolist()) if "waterbody_type" in df.columns else []


def filter_df(df, country=None, waterbody_type=None, date_from=None, date_to=None):
    mask = pd.Series(True, index=df.index)
    if country:
        mask &= df["country"] == country
    if waterbody_type:
        mask &= df["waterbody_type"] == waterbody_type
    if date_from is not None:
        mask &= df["date"] >= pd.to_datetime(date_from)
    if date_to is not None:
        mask &= df["date"] <= pd.to_datetime(date_to)
    return df[mask].reset_index(drop=True)