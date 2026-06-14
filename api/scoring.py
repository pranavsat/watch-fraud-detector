"""Feature engineering and scoring for a single listing.

This mirrors the feature logic from the training notebook so that scores produced
by the API match what was learned offline. All lookup tables and the trained
Isolation Forest are loaded once at import time.
"""
import os
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def _load(name):
    return joblib.load(os.path.join(MODELS_DIR, name))


# --- artifacts produced by the notebook save cell ---
iso_price = _load("iso_price.pkl")
model_stats = pd.read_pickle(os.path.join(MODELS_DIR, "model_stats.pkl"))
brand_stats = pd.read_pickle(os.path.join(MODELS_DIR, "brand_stats.pkl"))
auto_median = pd.read_pickle(os.path.join(MODELS_DIR, "auto_median.pkl"))
brand_quantiles = _load("brand_price_quantiles.pkl")
spec_range = _load("spec_score_range.pkl")
constants = _load("constants.pkl")

# column order the Isolation Forest was trained on - must not change
PRICE_FEATURES = ["price_zscore", "price_pct_in_brand",
                  "mvmt_price_flag", "cond_price_flag", "age_price_flag"]
PREMIUM_CONDITIONS = {"Unworn", "New"}
CURRENT_YEAR = 2026


def _extract_year(yop):
    if not yop:
        return np.nan
    import re
    m = re.search(r"(\d{4})", str(yop))
    return float(m.group(1)) if m else np.nan


def _extract_size_mm(size):
    if not size:
        return np.nan
    import re
    m = re.search(r"(\d+\.?\d*)\s*mm", str(size))
    return float(m.group(1)) if m else np.nan


def _price_zscore(brand, model, price):
    """Log-price z-score vs brand+model, with brand-level fallback."""
    log_price = np.log1p(price)
    key = (brand, model)
    if (key in model_stats.index
            and model_stats.loc[key, "count"] >= 10
            and model_stats.loc[key, "std"] > 0):
        med, std = model_stats.loc[key, "median"], model_stats.loc[key, "std"]
    elif brand in brand_stats.index:
        med, std = brand_stats.loc[brand, "median"], brand_stats.loc[brand, "std"]
    else:
        return 0.0
    if std is None or std == 0 or np.isnan(std):
        return 0.0
    return float((log_price - med) / std)


def _price_pct_in_brand(brand, price):
    """Approximate percentile rank of price within its brand, via saved quantiles."""
    if brand not in brand_quantiles:
        return 0.5
    q = brand_quantiles[brand]                 # 101 percentile points (0..100)
    pct = np.interp(price, q, np.linspace(0, 1, len(q)))
    return float(np.clip(pct, 0.0, 1.0))


def compute_features(listing: dict) -> dict:
    """Recreate the model features for one listing."""
    brand = listing.get("brand")
    model = listing.get("model")
    price = float(listing["price"])

    yop_year = _extract_year(listing.get("yop"))
    size_mm = _extract_size_mm(listing.get("size"))
    ref = listing.get("ref")
    mvmt = listing.get("mvmt")
    condition = listing.get("condition")

    z = _price_zscore(brand, model, price)
    pct = _price_pct_in_brand(brand, price)

    # null_count: missing specs + missing reference
    spec_values = [mvmt, listing.get("casem"), listing.get("bracem"),
                   yop_year, size_mm]
    null_count = sum(1 for v in spec_values if v is None or (isinstance(v, float) and np.isnan(v)))
    ref_missing = 1 if (ref is None or str(ref).strip() == "" or str(ref) == "UNKNOWN") else 0
    null_count += ref_missing

    # quartz priced like an automatic
    mvmt_flag = 0
    if mvmt == "Quartz" and brand in auto_median.index:
        mvmt_flag = int(price > auto_median[brand])

    # premium condition but suspiciously cheap
    cond_flag = int(condition in PREMIUM_CONDITIONS and pct < 0.05)

    # old but priced in the brand's top quartile
    watch_age = CURRENT_YEAR - yop_year if not np.isnan(yop_year) else np.nan
    age_flag = int((not np.isnan(watch_age)) and watch_age > 30 and pct > 0.75)

    return {
        "price_zscore": z,
        "price_pct_in_brand": pct,
        "mvmt_price_flag": mvmt_flag,
        "cond_price_flag": cond_flag,
        "age_price_flag": age_flag,
        "null_count": int(null_count),
    }


def _spec_anomaly_score(feats: dict) -> float:
    """Isolation Forest score for one listing, normalized to 0-100."""
    row = np.array([[feats[c] for c in PRICE_FEATURES]], dtype=float)
    raw = float(iso_price.score_samples(row)[0])
    lo, hi = spec_range["min"], spec_range["max"]
    if hi == lo:
        return 0.0
    score = (hi - raw) / (hi - lo) * 100
    return float(np.clip(score, 0, 100))


def score_listing(listing: dict) -> dict:
    """Full scoring pipeline for one listing."""
    feats = compute_features(listing)

    underpriced = float(np.clip(max(-feats["price_zscore"], 0.0) / 4 * 100, 0, 100))
    spec_anomaly = _spec_anomaly_score(feats)
    completeness = float(feats["null_count"] / constants["max_null"] * 100)

    risk = round(0.60 * underpriced + 0.25 * spec_anomaly + 0.15 * completeness, 1)

    band = "low" if risk < 30 else ("medium" if risk < 60 else "high")

    reasons = []
    if underpriced >= 50:
        reasons.append(f"Priced well below market for its brand/model (z={feats['price_zscore']:.1f})")
    if feats["mvmt_price_flag"]:
        reasons.append("Quartz movement priced like a high-end automatic")
    if feats["cond_price_flag"]:
        reasons.append("Premium condition but unusually low price")
    if feats["age_price_flag"]:
        reasons.append("Older watch priced in the brand's top quartile")
    if completeness >= 60:
        reasons.append(f"Listing missing several spec fields ({feats['null_count']} of 6)")
    if not reasons:
        reasons.append("No strong anomaly signals detected")

    return {
        "risk_score": risk,
        "risk_band": band,
        "reasons": reasons,
        "breakdown": {
            "price_zscore": round(feats["price_zscore"], 3),
            "underpriced_score": round(underpriced, 1),
            "spec_anomaly_score": round(spec_anomaly, 1),
            "completeness_score": round(completeness, 1),
            "mvmt_price_flag": feats["mvmt_price_flag"],
            "cond_price_flag": feats["cond_price_flag"],
            "age_price_flag": feats["age_price_flag"],
            "null_count": feats["null_count"],
            "price_pct_in_brand": round(feats["price_pct_in_brand"], 3),
        },
    }
