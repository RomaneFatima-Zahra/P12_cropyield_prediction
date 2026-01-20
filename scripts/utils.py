"""
Script de définition de fonction utiles pour le moteur de prédiction et de recommandation : 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Artefacts (plots) pour MLFLOW
# ----------------------------
def save_residual_plot(y_true, y_pred, out_path):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, s=8)
    plt.axhline(0)
    plt.xlabel("Predictions")
    plt.ylabel("Residuals (true - pred)")
    plt.title("Residual plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_pred_vs_true_plot(y_true, y_pred, out_path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    plt.xlabel("True yield")
    plt.ylabel("Predicted yield")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

#================================================================================
# Fonction pour ajouter l'effet de l'irrigation et fertilization sur le rendement 
#================================================================================

def apply_optional_scenarios(yield_hg_ha: float, irrigation: bool = False, fertilizer: bool = False) -> float:
    """
    Post-ajustement additif (what-if) pour irrigation/fertilisation.
    Hypothèse: effet moyen constant, additif.
    """
    adj = 0.0
    if irrigation:
        adj += 12000
    if fertilizer:
        adj += 15000
    return yield_hg_ha + adj

#================================================================================
# Fonction pour ajuster le prix vs unité de rendement
#================================================================================

def compute_revenue_per_ha(yield_hg_ha: float, price_value: float, price_unit: str = "eur_per_t") -> float:
    """
    Convertit un rendement (hg/ha) en revenu/ha selon une unité de prix.
    - eur_per_t : €/tonne  -> revenue = yield_hg_ha * price/10_000
    - eur_per_kg: €/kg     -> revenue = yield_hg_ha * price/10
    - eur_per_hg: €/hg     -> revenue = yield_hg_ha * price
    """
    u = price_unit.lower().strip()
    if u in ["eur_per_t", "€/t", "euro_per_tonne"]:
        return yield_hg_ha * (price_value / 10_000)
    if u in ["eur_per_kg", "€/kg", "euro_per_kg"]:
        return yield_hg_ha * (price_value / 10)
    if u in ["eur_per_hg", "€/hg", "euro_per_hg"]:
        return yield_hg_ha * price_value
    raise ValueError(f"Unsupported price_unit: {price_unit}")