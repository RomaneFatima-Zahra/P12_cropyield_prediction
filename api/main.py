from __future__ import annotations
import json
from typing import Dict, List, Optional, Literal
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
from pathlib import Path

from scripts.predictor import (
    model,
    predict_yield_hg_ha,
    recommend_by_yield,
    recommend_by_revenue,
)
from scripts.utils import compute_revenue_per_ha


# ---------------------------------------------------------
# Load candidate items
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CANDIDATE_ITEMS_PATH = BASE_DIR.parent / "inputs" / "candidate_items.json"

def load_candidate_items() -> List[str]:
    if not CANDIDATE_ITEMS_PATH.exists():
        raise FileNotFoundError(
            f"candidate_items.json not found at {CANDIDATE_ITEMS_PATH}. "
            f"Create it with the list of crops (items)."
        )
    with open(CANDIDATE_ITEMS_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("candidate_items.json must be a JSON list of strings.")
    return items

CANDIDATE_ITEMS = load_candidate_items()


# ---------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------

PriceUnit = Literal["eur_per_t", "eur_per_kg", "eur_per_hg"]

class PredictRequest(BaseModel):
    area: str = Field(..., description="Nom du pays")
    item: str = Field(..., description="Type de culture")
    year: int =  Field(..., ge=1900, le=2100, description="Année")
    avg_rain_mm: float = Field(..., ge=0, description="Précipitations moyennes en mm")
    pesticides_tonnes: float = Field(..., ge=0, description="Pesticides en tonnes")
    avg_temp: float = Field(..., description="Température moyenne en °C")
    irrigation: bool = Field(default=False, description="Usage de l'irrigation")
    fertilizer: bool = Field(default=False, description="Usage de la fertilisation")
    # Prix facultatif (pour calculer le revenu)
    price_value: Optional[float] = Field(default=None, description="Prix par culture (optionnel)")
    price_unit: PriceUnit = Field(default="eur_per_t", description="Unité de prix")

    class Config:
        json_schema_extra = {
            "example": {
                "area": "France",
                "item": "maize",
                "year": 2026,
                "avg_rain_mm": 650.0,
                "pesticides_tonnes": 5000.0,
                "avg_temp": 12.5,
                "irrigation": False,
                "fertilizer": False,
                "price_value": 0,
                "price_unit": "eur_per_t"
                
            }
        }

class PredictResponse(BaseModel):
    item: str
    pred_yield_hg_ha: float
    pred_yield_t_ha: float
    revenue_per_ha: Optional[float] = None


class RecommendBaseRequest(BaseModel):
    area: str = Field(..., description="Nom du pays")
    year: int =  Field(..., ge=1900, le=2100, description="Année")
    avg_rain_mm: float = Field(..., ge=0, description="Précipitations moyennes en mm")
    pesticides_tonnes: float = Field(..., ge=0, description="Pesticides en tonnes")
    avg_temp: float = Field(..., description="Température moyenne en °C")
    irrigation: bool = Field(default=False, description="Usage de l'irrigation")
    fertilizer: bool = Field(default=False, description="Usage de la fertilisation")
    top_k: int = Field(default=5, ge=1, le=20, description="Nombre de recommandations")
    prices: Optional[dict] = Field(default=None, description="Prix par culture (optionnel)")
    price_unit: str = Field(default="eur_per_t", description="Unité de prix")


class RecommendYieldRequest(RecommendBaseRequest):
    """Recommandation triée par rendement (pas besoin de prix)."""
    class Config:
        json_schema_extra = {
            "example": {
                "area": "France",
                "year": 2026,
                "avg_rain_mm": 650.0,
                "pesticides_tonnes": 5000.0,
                "avg_temp": 15,
                "irrigation": False,
                "fertilizer": False,
                "top_k": 5,
                }
            }
    pass


class RecommendRevenueRequest(RecommendBaseRequest):
    """Recommandation triée par revenu (prix requis)."""
    prices: Dict[str, float] = Field(
        ...,
        description="Dictionnaire {item: prix} saisi par l'agriculteur"
    )
    price_unit: PriceUnit = Field(default="eur_per_t", description="Unité de prix")

    class Config:
        json_schema_extra = {
            "example": {
                "area": "France",
                "year": 2026,
                "avg_rain_mm": 650.0,
                "pesticides_tonnes": 5000.0,
                "avg_temp": 15,
                "irrigation": False,
                "fertilizer": False,
                "top_k": 5,
                "price_unit": "eur_per_t",
                "prices": {
                    "maize": 0,
                    "rice, paddy": 0,
                    "wheat": 0,
                    "cassava": 0,
                    "sorghum": 0,
                    "potatoes": 0,
                    "soybeans":0,
                    "yams": 0,
                    "sweet potatoes": 0,
                    "plantains and others": 0
                    }
                }
            }


class RecommendRow(BaseModel):
    item: str
    pred_yield_hg_ha: float
    pred_yield_t_ha: float
    revenue_per_ha: Optional[float] = None
    price_value: Optional[float] = None
    price_unit: Optional[str] = None


class RecommendResponse(BaseModel):
    results: List[RecommendRow]

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(
    title="Crop Yield PREDICTION API",
    version="1.0.0",
    description="API de prédiction de rendement et recommandation de cultures à destinationd des agriculteurs")

# ---------------------------------------------------------
# GET / Endpoint santé : 
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "running",
            "message": "Agricultural Yield Prediction API",
            "endpoints": ["/predict", "/recommend", "/docs"]}

# ---------------------------------------------------------
# POST /predict
# ---------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        pred_hg_ha = predict_yield_hg_ha(
            model,
            area=req.area,
            item=req.item,
            year=req.year,
            avg_rain_mm=req.avg_rain_mm,
            pesticides_tonnes=req.pesticides_tonnes,
            avg_temp=req.avg_temp,
            irrigation=req.irrigation,
            fertilizer=req.fertilizer
        )
        resp = {
            "item" : req.item,
            "pred_yield_hg_ha": float(pred_hg_ha),
            "pred_yield_t_ha": float(pred_hg_ha) / 10000,
            "revenue_per_ha": None
        }

        if req.price_value is not None:
            revenue = compute_revenue_per_ha(float(pred_hg_ha), float(req.price_value), req.price_unit)
            resp["revenue_per_ha"] = float(revenue)

        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# POST /recommend/yield
# ---------------------------------------------------------

@app.post("/recommend/yield", response_model=RecommendResponse)
def recommend_yield(req: RecommendYieldRequest):
    try:
        df_out = recommend_by_yield(
            model,
            area=req.area,
            year=req.year,
            avg_rain_mm=req.avg_rain_mm,
            pesticides_tonnes=req.pesticides_tonnes,
            avg_temp=req.avg_temp,
            candidate_items=CANDIDATE_ITEMS,
            irrigation=req.irrigation,
            fertilizer=req.fertilizer,
            top_k=req.top_k
        )

        results = [
            RecommendRow(
                item=str(r["item"]),
                pred_yield_hg_ha=float(r["pred_yield_hg_ha"]),
                pred_yield_t_ha=float(r["pred_yield_t_ha"]),
            )
            for _, r in df_out.iterrows()
        ]
        return RecommendResponse(results=results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# POST /recommend/revenue
# ---------------------------------------------------------
@app.post("/recommend/revenue", response_model=RecommendResponse)
def recommend_revenue(req: RecommendRevenueRequest):
    try:
        if not req.prices:
            raise HTTPException(status_code=400, detail="prices must be a non-empty dict {item: price}")

        # validation simple : prix > 0
        bad = [k for k, v in req.prices.items() if v is None or float(v) <= 0]
        if bad:
            raise HTTPException(status_code=400, detail=f"All prices must be > 0. Invalid items: {bad}")

        df_out = recommend_by_revenue(
            model,
            area=req.area,
            year=req.year,
            avg_rain_mm=req.avg_rain_mm,
            pesticides_tonnes=req.pesticides_tonnes,
            avg_temp=req.avg_temp,
            candidate_items=CANDIDATE_ITEMS,
            prices=req.prices,
            price_unit=req.price_unit,
            irrigation=req.irrigation,
            fertilizer=req.fertilizer,
            top_k=req.top_k
        )

        results = [
            RecommendRow(
                item=str(r["item"]),
                pred_yield_hg_ha=float(r["pred_yield_hg_ha"]),
                pred_yield_t_ha=float(r["pred_yield_t_ha"]),
                revenue_per_ha=float(r["revenue_per_ha"]),
                price_value=float(r["price_value"]),
                price_unit=str(r["price_unit"])
            )
            for _, r in df_out.iterrows()
        ]
        return RecommendResponse(results=results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


