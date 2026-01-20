#scripts/predictor.py

import sys
from pathlib import Path
from scripts.utils import apply_optional_scenarios, compute_revenue_per_ha

import pandas as pd
import joblib
from typing import List


COEF_IRRIGATION_HG_HA = 12000
COEF_FERTILIZATION_HG_HA = 15000


# ========================================================
# Import Model
# ========================================================

# chemin absolu vers ce fichier
BASE_DIR = Path(__file__).resolve().parent
# remonter d'un niveau puis aller dans models/
MODEL_PATH = BASE_DIR.parent / "model" / "hgb_optimized.joblib"
# chargement du modèle
model = joblib.load(MODEL_PATH)

# ========================================================
# Moteur de Prediction + 2 recommenders (yield vs revenue)
# ========================================================
def predict_yield_hg_ha(
    model, *,
    area: str, item: str, year: int,
    avg_rain_mm: float, pesticides_tonnes: float, avg_temp: float,
    irrigation: bool = False, fertilizer: bool = False ) -> float:
    X_in = pd.DataFrame([{
        "area": area,
        "item": item,
        "year": year,
        "avg_rain_mm": avg_rain_mm,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": avg_temp
        }])
    base_pred = float(model.predict(X_in)[0])
    return apply_optional_scenarios(base_pred, irrigation=irrigation, fertilizer=fertilizer)

# ========================================================
# Moteur de Recommendation ( Hg/ha yield & rentabilité)
# ========================================================
def recommend_by_yield(
    model, *,
    area: str, year: int,
    avg_rain_mm: float, pesticides_tonnes: float, avg_temp: float,
    candidate_items: list[str],
    irrigation: bool = False, fertilizer: bool = False,
    top_k: int = 5) -> pd.DataFrame:
    X_in = pd.DataFrame([{
        "area": area,
        "item": it,
        "year": year,
        "avg_rain_mm": avg_rain_mm,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": avg_temp
    } for it in candidate_items])

    base_preds = model.predict(X_in).astype(float)
    adj = (COEF_IRRIGATION_HG_HA if irrigation else 0.0) + (COEF_FERTILIZATION_HG_HA if fertilizer else 0.0)
    preds = base_preds + adj

    out = pd.DataFrame({
        "item": candidate_items,
        "pred_yield_hg_ha": preds,
        "pred_yield_t_ha": preds / 10000,
        "irrigation": irrigation,
        "fertilizer": fertilizer
    }).sort_values("pred_yield_hg_ha", ascending=False)

    return out.head(top_k).reset_index(drop=True)

def recommend_by_revenue(
    model, *,
    area: str, year: int,
    avg_rain_mm: float, pesticides_tonnes: float, avg_temp: float,
    candidate_items: list[str],
    prices: dict[str, float],
    price_unit: str = "eur_per_t",
    irrigation: bool = False, fertilizer: bool = False,
    top_k: int = 5
) -> pd.DataFrame:
    # garder uniquement les items dont l'agriculteur a fourni le prix
    items = [it for it in candidate_items if it in prices]
    if len(items) == 0:
        raise ValueError("No candidate items have a provided price. Provide prices like {'maize': 180, ...}.")

    X_in = pd.DataFrame([{
        "area": area,
        "item": it,
        "year": year,
        "avg_rain_mm": avg_rain_mm,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": avg_temp
    } for it in items])

    base_preds = model.predict(X_in).astype(float)
    adj = (COEF_IRRIGATION_HG_HA if irrigation else 0.0) + (COEF_FERTILIZATION_HG_HA if fertilizer else 0.0)
    preds = base_preds + adj

    out = pd.DataFrame({
        "item": items,
        "pred_yield_hg_ha": preds,
        "pred_yield_t_ha": preds / 10_000,
        "price_value": [prices[it] for it in items],
        "price_unit": price_unit,
        "irrigation": irrigation,
        "fertilizer": fertilizer
    })

    out["revenue_per_ha"] = out.apply(
        lambda r: compute_revenue_per_ha(r["pred_yield_hg_ha"], r["price_value"], r["price_unit"]),
        axis=1)

    out = out.sort_values("revenue_per_ha", ascending=False)
    return out.head(top_k).reset_index(drop=True)