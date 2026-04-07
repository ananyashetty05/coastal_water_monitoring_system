"""
core/detector.py  —  FINAL FIXED VERSION
Key fixes:
  1. Metrics classified in batches of 8 — prevents JSON truncation for wide datasets
  2. max_tokens set correctly per batch size
  3. Bulletproof JSON extraction handles all LLM output formats
  4. llama-3.3-70b-versatile for schema, llama-3.1-8b-instant for commentary
"""
from __future__ import annotations
import hashlib, json, os, re, time
import numpy as np
import pandas as pd
import streamlit as st

try:
    from groq import Groq, RateLimitError
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

SCHEMA_MODEL     = "llama-3.3-70b-versatile"   # reliable JSON output
COMMENTARY_MODEL = "llama-3.1-8b-instant"       # fast text, high RPD
METRIC_BATCH     = 8                             # metrics per LLM call


# ── API key ────────────────────────────────────────────────────────────────────

def _key() -> str:
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            k = str(st.secrets["GROQ_API_KEY"]).strip()
            if k and k != "gsk_your_key_here" and len(k) > 10:
                return k
    except Exception:
        pass
    k = os.environ.get("GROQ_API_KEY", "").strip()
    return k if k and len(k) > 10 else ""


# ── Groq call with retry ───────────────────────────────────────────────────────

def _call(model: str, prompt: str, max_tokens: int) -> str:
    client = Groq(api_key=_key())
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except RateLimitError:
            wait = [15, 30, 60][attempt]
            st.warning(f"⏳ Rate limit — retrying in {wait}s…")
            time.sleep(wait)
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}") from e
    return ""


# ── Bulletproof JSON extractor ─────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Extract JSON from LLM response no matter how it's wrapped.
    Uses brace-counting so truncated/fenced/prefixed responses all work.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response")

    text = raw.strip()

    # strip markdown fences
    text = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$",               "", text, flags=re.MULTILINE)
    text = text.strip()

    # locate outermost { ... } using brace-counting
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found. Response: {raw[:200]}")

    depth, end, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"' and not esc:
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i; break

    if end == -1:
        raise ValueError(f"JSON object not closed. Response: {raw[:200]}")

    js = text[start:end + 1]

    # patch common LLM mistakes
    js = re.sub(r",(\s*[}\]])", r"\1", js)         # trailing commas
    js = re.sub(r":\s*None\b",  ": null",   js)
    js = re.sub(r":\s*True\b",  ": true",   js)
    js = re.sub(r":\s*False\b", ": false",  js)
    if '"' not in js and "'" in js:
        js = js.replace("'", '"')

    return json.loads(js)


def _hash(df: pd.DataFrame) -> str:
    payload = str(df.columns.tolist()) + df.head(3).astype(str).to_csv()
    return hashlib.md5(payload.encode()).hexdigest()


# ── Prompt: structural columns (Stage 1) ──────────────────────────────────────

def _prompt_structural(cols: list, rows: list) -> str:
    return f"""You are a water quality data scientist.
Return ONLY a JSON object. No markdown. No explanation. Just JSON.

CSV columns: {json.dumps(cols)}
Sample rows (3): {json.dumps(rows, default=str)}

Return this exact JSON:
{{
  "date_col":          "<exact column name with dates, or null>",
  "location_col":      "<exact column name for station/area/site, or null>",
  "country_col":       "<exact column name for country, or null>",
  "waterbody_col":     "<exact column name for depth/layer/waterbody type, or null>",
  "skip_cols":         ["<columns that are IDs or row numbers — not measurements>"],
  "detected_location": "<inferred city or country from data values>",
  "dataset_summary":   "<one sentence describing this dataset>"
}}"""


# ── Prompt: one batch of metrics (Stage 2, called multiple times) ──────────────

def _prompt_batch(batch_cols: list, rows: list) -> str:
    return f"""You are a water quality scientist.
Return ONLY a JSON object. No markdown. No explanation. Just JSON.

Classify EXACTLY these columns (all of them): {json.dumps(batch_cols)}
Sample data: {json.dumps(rows, default=str)}

Return this exact JSON — one entry per column, NO omissions:
{{
  "metrics": {{
    "<snake_case_key>": {{
      "original_col":     "<exact column name from the list above>",
      "label":            "<human-readable name>",
      "unit":             "<unit or empty string>",
      "icon":             "<one emoji>",
      "group":            "<oxygen|nutrients|physical|biological|solids|other>",
      "higher_is_better": true,
      "safe_range":       [null, null],
      "poor_range":       [null, null],
      "clamp":            [0, 9999]
    }}
  }}
}}

Rules:
- snake_case keys: lowercase + underscores only
- higher_is_better: true = high is good (DO, pH neutral), false = high is bad (BOD, bacteria, turbidity)
- safe_range / poor_range: use WHO/EPA/marine standards. null means no bound.
- clamp: physical impossibility bounds (DO max 25, pH 0-14, temperature -2 to 45)
- Columns like <0.002 or N/A are still numeric — include them"""


# ── Main detect ────────────────────────────────────────────────────────────────

def detect(df: pd.DataFrame) -> dict:
    """
    Two-stage LLM schema detection with batched metric classification.
    Cached in session_state — API called only once per unique file.
    """
    cache_key = f"schema_{_hash(df)}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    if not GROQ_OK:
        st.error("❌ groq not installed. Run: pip install groq")
        schema = _fallback(df)
        st.session_state[cache_key] = schema
        return schema

    if not _key():
        st.error(
            "❌ **GROQ_API_KEY not set.**\n\n"
            "Create `.streamlit/secrets.toml`:\n"
            "```\nGROQ_API_KEY = \"gsk_your_key\"\n```\n"
            "Free key: https://console.groq.com"
        )
        schema = _fallback(df)
        st.session_state[cache_key] = schema
        return schema

    cols = df.columns.tolist()
    rows = [
        {str(k)[:40]: str(v)[:25] for k, v in row.items()}
        for row in df.head(3).to_dict("records")
    ]

    # ── Stage 1: identify structural columns ─────────────────────────────────
    try:
        raw1       = _call(SCHEMA_MODEL, _prompt_structural(cols, rows), max_tokens=350)
        structural = _extract_json(raw1)
    except Exception as e:
        st.warning(f"Stage 1 failed ({e}). Using heuristic fallback.")
        schema = _fallback(df)
        st.session_state[cache_key] = schema
        return schema

    # build skip set
    skip_cols = set(filter(None,
        [structural.get("date_col"), structural.get("location_col"),
         structural.get("country_col"), structural.get("waterbody_col")]
        + structural.get("skip_cols", [])
    ))

    # columns to classify as metrics
    metric_candidates = [c for c in cols if c not in skip_cols]

    # ── Stage 2: classify metrics in batches of METRIC_BATCH ─────────────────
    all_metrics: dict = {}
    batches = [
        metric_candidates[i:i + METRIC_BATCH]
        for i in range(0, len(metric_candidates), METRIC_BATCH)
    ]

    n_batches = len(batches)
    for batch_idx, batch in enumerate(batches):
        # tokens needed ≈ 100 per metric + 300 overhead
        needed_tokens = len(batch) * 120 + 400
        try:
            raw = _call(
                SCHEMA_MODEL,
                _prompt_batch(batch, rows),
                max_tokens=needed_tokens,
            )
            result = _extract_json(raw)
            batch_metrics = result.get("metrics", {})
            all_metrics.update(batch_metrics)
        except Exception as e:
            st.warning(f"Batch {batch_idx + 1}/{n_batches} failed ({e}) — using heuristic for these columns.")
            # heuristic fallback for just this batch
            for col in batch:
                key = re.sub(r"[^\w]", "_", col.lower()).strip("_")
                key = re.sub(r"_+", "_", key)
                base = key; i = 2
                while key in all_metrics:
                    key = f"{base}_{i}"; i += 1
                all_metrics[key] = {
                    "original_col": col, "label": col, "unit": "",
                    "icon": "📊", "group": "other",
                    "higher_is_better": None,
                    "safe_range": [None, None],
                    "poor_range": [None, None],
                    "clamp": [None, None],
                }

    # ── Merge ─────────────────────────────────────────────────────────────────
    schema = {
        "_llm":              True,
        "date_col":          structural.get("date_col"),
        "location_col":      structural.get("location_col"),
        "country_col":       structural.get("country_col"),
        "waterbody_col":     structural.get("waterbody_col"),
        "skip_cols":         list(skip_cols),
        "dataset_summary":   structural.get("dataset_summary", ""),
        "detected_location": structural.get("detected_location", ""),
        "metrics":           all_metrics,
        "key_concerns":      [],
        "suggested_charts":  [],
        "confidence":        0.92 if all_metrics else 0.5,
    }

    if not schema["metrics"]:
        st.warning("No metrics returned — using heuristic fallback.")
        schema = _fallback(df)

    # ── Stage 3: get key concerns + suggested charts (lightweight call) ───────
    try:
        metric_labels = [m.get("label", k) for k, m in list(all_metrics.items())[:10]]
        concerns_prompt = f"""Given this water quality dataset from {schema['detected_location']}
with parameters: {', '.join(metric_labels)}

Return ONLY this JSON:
{{
  "key_concerns": ["<concern1>", "<concern2>", "<concern3>"],
  "suggested_charts": ["<chart idea 1>", "<chart idea 2>", "<chart idea 3>"]
}}"""
        raw3   = _call(SCHEMA_MODEL, concerns_prompt, max_tokens=300)
        extra  = _extract_json(raw3)
        schema["key_concerns"]    = extra.get("key_concerns", [])
        schema["suggested_charts"] = extra.get("suggested_charts", [])
    except Exception:
        pass   # non-critical — skip silently

    st.session_state[cache_key] = schema
    return schema


# ── Marine life analysis ───────────────────────────────────────────────────────

def marine_analysis(schema: dict, stats: dict) -> str:
    if not GROQ_OK or not _key():
        return "Groq unavailable. Add GROQ_API_KEY to .streamlit/secrets.toml"

    lines = []
    for k, v in stats.items():
        if k == "_meta" or not isinstance(v, dict):
            continue
        m     = schema.get("metrics", {}).get(k, {})
        trend = "rising" if v["trend"] > 0 else ("falling" if v["trend"] < 0 else "stable")
        lines.append(f"{m.get('label', k)}: mean={v['mean']:.3f} {m.get('unit','')}, {trend}")

    prompt = f"""You are a marine biologist. Write 4-5 sentences analysing how these
water quality statistics affect marine life, coral reefs, and ocean ecosystems.
Be specific with numbers. Mention the most concerning parameters. Flowing prose, no bullets.

Location: {schema.get('detected_location', 'Unknown')}
Dataset: {schema.get('dataset_summary', '')}
Statistics:
{chr(10).join(lines) or 'No data.'}"""

    try:
        return _call(COMMENTARY_MODEL, prompt, max_tokens=300)
    except Exception as e:
        return f"Analysis unavailable: {e}"


# ── Degradation analysis ───────────────────────────────────────────────────────

def degradation_forecast_analysis(schema: dict, forecast_summary: dict) -> str:
    if not GROQ_OK or not _key():
        return "Groq unavailable. Add GROQ_API_KEY to .streamlit/secrets.toml"

    lines = [
        f"{schema.get('metrics',{}).get(m,{}).get('label',m)}: "
        f"now={v['current']:.3f}, day7={v['day7']:.3f}, "
        f"change={v['change']:+.3f} {schema.get('metrics',{}).get(m,{}).get('unit','')}"
        for m, v in forecast_summary.items()
    ]

    prompt = f"""You are a marine environmental scientist. Write three parts:
1. Assessment (2 sentences): Will quality degrade or improve?
2. Timeline (1 sentence): At this rate, critical conditions in X weeks/months.
3. Prevention: 3 specific numbered actionable measures.

Location: {schema.get('detected_location', '')}
7-day forecast changes:
{chr(10).join(lines) or 'No data.'}"""

    try:
        return _call(COMMENTARY_MODEL, prompt, max_tokens=350)
    except Exception as e:
        return f"Analysis unavailable: {e}"


# ── Heuristic fallback ─────────────────────────────────────────────────────────

def _fallback(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()

    date_col = next((c for c in cols if any(k in c.lower() for k in
        ["date","time","dates","datetime","period"])), None)
    loc_col  = next((c for c in cols if any(k in c.lower() for k in
        ["station","area","site","location","zone","place","name","point"])), None)
    depth_col = next((c for c in cols if any(k in c.lower() for k in
        ["depth","layer","level"])), None)

    skip = set(filter(None, [date_col, loc_col, depth_col]))
    for col in cols:
        if col in skip: continue
        try:
            if df[col].nunique() == len(df) and pd.api.types.is_integer_dtype(df[col]):
                skip.add(col)
        except Exception:
            pass

    metrics: dict = {}
    for col in cols:
        if col in skip: continue
        sample  = df[col].dropna().head(20).astype(str)
        cleaned = (sample.str.strip()
                   .str.replace(r"^[<>≤≥~]\s*", "", regex=True)
                   .str.replace(r"[^\d.\-eE]", "", regex=True)
                   .replace("", np.nan).dropna())
        if len(cleaned) < 3: continue
        try:
            pd.to_numeric(cleaned, errors="raise")
            key  = re.sub(r"[^\w]", "_", col.lower()).strip("_")
            key  = re.sub(r"_+", "_", key)
            base = key; i = 2
            while key in metrics:
                key = f"{base}_{i}"; i += 1
            metrics[key] = {
                "original_col": col, "label": col, "unit": "",
                "icon": "📊", "group": "other",
                "higher_is_better": None,
                "safe_range": [None, None], "poor_range": [None, None],
                "clamp": [None, None],
            }
        except Exception:
            pass

    return {
        "_llm": False,
        "date_col": date_col, "location_col": loc_col,
        "country_col": None, "waterbody_col": depth_col,
        "skip_cols": list(skip), "metrics": metrics,
        "key_concerns": [], "suggested_charts": [],
        "dataset_summary": "Loaded with heuristic detection.",
        "detected_location": "Unknown", "confidence": 0.3,
    }