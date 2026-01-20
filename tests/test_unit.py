# tests/test_unit.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from pathlib import Path
import sys
from api.main import (app, PredictRequest, RecommendRevenueRequest, load_candidate_items, CANDIDATE_ITEMS )
from scripts.predictor import model
from scripts.utils import compute_revenue_per_ha


# Création du client de test
client = TestClient(app)

# ---------------------------------------------------------
# Tests unitaires pour les fonctions utilitaires
# ---------------------------------------------------------

def test_compute_revenue_per_ha():
    """Test de la fonction compute_revenue_per_ha avec différentes unités"""
    
    # Test avec eur_per_t
    yield_hg_ha = 10000  # 1 tonne/hectare
    price = 200  # 200€ par tonne
    revenue = compute_revenue_per_ha(yield_hg_ha, price, "eur_per_t")
    assert revenue == 200.0  # 1 * 200 = 200€/ha
    
    # Test avec eur_per_kg
    revenue = compute_revenue_per_ha(yield_hg_ha, price, "eur_per_kg")
    assert revenue == 200000.0  # 1000 * 200 = 200_000€/ha
    
    # Test avec eur_per_hg
    revenue = compute_revenue_per_ha(yield_hg_ha, price, "eur_per_hg")
    assert revenue == 2000000.0  # 10000 * 200 = 2_000_000€/ha
    
    # Test avec prix zéro
    revenue = compute_revenue_per_ha(yield_hg_ha, 0, "eur_per_t")
    assert revenue == 0.0
    
    # Test avec rendement zéro
    revenue = compute_revenue_per_ha(0, price, "eur_per_t")
    assert revenue == 0.0
    
    # Test avec unité invalide (devrait lever une exception)
    with pytest.raises(ValueError):
        compute_revenue_per_ha(yield_hg_ha, price, "invalid_unit")

# ---------------------------------------------------------
# Tests unitaires pour les modèles Pydantic
# ---------------------------------------------------------

def test_predict_request_model():
    """Test de validation du modèle PredictRequest"""

    # Test avec données valides
    valid_data = {
        "area": "France",
        "item": "maize",
        "year": 2026,
        "avg_rain_mm": 650.0,
        "pesticides_tonnes": 5000.0,
        "avg_temp": 12.5,
        "irrigation": False,
        "fertilizer": False,
        "price_value": 200,
        "price_unit": "eur_per_t"
    }
    
    request = PredictRequest(**valid_data)
    assert request.area == "France"
    assert request.item == "maize"
    assert request.year == 2026
    assert request.price_value == 200

    
    # Test avec année invalide
    invalid_year = valid_data.copy()
    invalid_year["year"] = 2200  # > 2100
    with pytest.raises(ValueError):
        PredictRequest(**invalid_year)
    
    # Test avec précipitations négatives
    invalid_rain = valid_data.copy()
    invalid_rain["avg_rain_mm"] = -10.0
    with pytest.raises(ValueError):
        PredictRequest(**invalid_rain)
    
    # Test avec unité de prix valide
    for unit in ["eur_per_t", "eur_per_kg", "eur_per_hg"]:
        valid_data["price_unit"] = unit
        request = PredictRequest(**valid_data)
        assert request.price_unit == unit

def test_recommend_revenue_request_model():
    """Test de validation du modèle RecommendRevenueRequest"""
    
    # Test avec données valides
    valid_data = {
        "area": "France",
        "year": 2026,
        "avg_rain_mm": 650.0,
        "pesticides_tonnes": 5000.0,
        "avg_temp": 15.0,
        "irrigation": False,
        "fertilizer": False,
        "top_k": 5,
        "price_unit": "eur_per_t",
        "prices": {
            "maize": 200,
            "wheat": 180
        }
    }
    
    request = RecommendRevenueRequest(**valid_data)
    assert request.area == "France"
    assert request.prices["maize"] == 200
    assert request.top_k == 5
    
    # Test avec top_k hors limites
    invalid_top_k = valid_data.copy()
    invalid_top_k["top_k"] = 25  # > 10
    with pytest.raises(ValueError):
        RecommendRevenueRequest(**invalid_top_k)
    
# ---------------------------------------------------------
# Tests d'intégration avec mock des dépendances
# ---------------------------------------------------------

class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_endpoint(self):
        """Test du endpoint de santé"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "endpoints" in data
        assert "/predict" in data["endpoints"]

class TestPredictEndpoint:
    """Tests pour l'endpoint /predict"""
    
    @patch('api.main.predict_yield_hg_ha')
    def test_predict_with_price(self, mock_predict):
        """Test de prédiction avec prix"""
        # Mock de la prédiction
        mock_predict.return_value = 15000.0  # 1.5 tonnes/ha
        
        request_data = {
            "area": "France",
            "item": "maize",
            "year": 2026,
            "avg_rain_mm": 650.0,
            "pesticides_tonnes": 5000.0,
            "avg_temp": 12.5,
            "irrigation": False,
            "fertilizer": False,
            "price_value": 200,
            "price_unit": "eur_per_t"
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["item"] == "maize"
        assert data["pred_yield_hg_ha"] == 15000.0
        assert data["pred_yield_t_ha"] == 1.5
        assert data["revenue_per_ha"] == 300.0  # 1.5 * 200 = 300
        
        # Vérification que la fonction mockée a été appelée avec les bons paramètres
        mock_predict.assert_called_once_with(
            model,
            area="France",
            item="maize",
            year=2026,
            avg_rain_mm=650.0,
            pesticides_tonnes=5000.0,
            avg_temp=12.5,
            irrigation=False,
            fertilizer=False
        )
    
    @patch('api.main.predict_yield_hg_ha')
    def test_predict_without_price(self, mock_predict):
        """Test de prédiction sans prix"""
        mock_predict.return_value = 10000.0
        
        request_data = {
            "area": "France",
            "item": "wheat",
            "year": 2025,
            "avg_rain_mm": 600.0,
            "pesticides_tonnes": 4500.0,
            "avg_temp": 14.0,
            "irrigation": True,
            "fertilizer": True
            # Pas de price_value
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["item"] == "wheat"
        assert data["revenue_per_ha"] is None
    
    def test_predict_invalid_data(self):
        """Test avec données invalides"""
        invalid_data = {
            "area": "France",
            "item": "maize",
            "year": 2200,  # Année invalide
            "avg_rain_mm": -10.0,  # Précipitations négatives
            "pesticides_tonnes": 5000.0,
            "avg_temp": 12.5
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # FastAPI devrait retourner 422 pour validation échouée
        assert response.status_code == 422
    
    @patch('api.main.predict_yield_hg_ha')
    def test_predict_model_error(self, mock_predict):
        """Test quand le modèle échoue"""
        mock_predict.side_effect = Exception("Erreur du modèle")
        
        request_data = {
            "area": "France",
            "item": "maize",
            "year": 2026,
            "avg_rain_mm": 650.0,
            "pesticides_tonnes": 5000.0,
            "avg_temp": 12.5,
            "irrigation": False,
            "fertilizer": False
        }
        
        response = client.post("/predict", json=request_data)
        
        # Devrait retourner 500 pour erreur interne
        assert response.status_code == 500

class TestRecommendYieldEndpoint:
    """Tests pour l'endpoint /recommend/yield"""
    
    @patch('api.main.recommend_by_yield')
    def test_recommend_yield_success(self, mock_recommend):
        """Test de recommandation par rendement"""
        # Mock du résultat
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, {"item": "maize", "pred_yield_hg_ha": 15000.0, "pred_yield_t_ha": 1.5}),
            (1, {"item": "wheat", "pred_yield_hg_ha": 12000.0, "pred_yield_t_ha": 1.2})
        ]
        mock_recommend.return_value = mock_df
        
        request_data = {
            "area": "France",
            "year": 2026,
            "avg_rain_mm": 650.0,
            "pesticides_tonnes": 5000.0,
            "avg_temp": 15.0,
            "irrigation": False,
            "fertilizer": False,
            "top_k": 2
        }
        
        response = client.post("/recommend/yield", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 2
        assert data["results"][0]["item"] == "maize"
        assert data["results"][0]["pred_yield_t_ha"] == 1.5
        assert data["results"][1]["item"] == "wheat"
        
        # Vérification des paramètres
        mock_recommend.assert_called_once()
        call_kwargs = mock_recommend.call_args[1]
        assert call_kwargs["top_k"] == 2
        assert call_kwargs["area"] == "France"
    
    @patch('api.main.recommend_by_yield')
    def test_recommend_yield_error(self, mock_recommend):
        """Test d'erreur dans la recommandation"""
        mock_recommend.side_effect = Exception("Erreur de recommandation")
        
        request_data = {
            "area": "France",
            "year": 2026,
            "avg_rain_mm": 650.0,
            "pesticides_tonnes": 5000.0,
            "avg_temp": 15.0,
            "irrigation": False,
            "fertilizer": False,
            "top_k": 5
        }
        
        response = client.post("/recommend/yield", json=request_data)
        
        assert response.status_code == 500

class TestRecommendRevenueEndpoint:
    """Tests pour l'endpoint /recommend/revenue"""
    
    @patch('api.main.recommend_by_revenue')
    def test_recommend_revenue_success(self, mock_recommend):
        """Test de recommandation par revenu"""
        # Mock du résultat
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, {
                "item": "maize", 
                "pred_yield_hg_ha": 15000.0, 
                "pred_yield_t_ha": 1.5,
                "revenue_per_ha": 300.0,
                "price_value": 200.0,
                "price_unit": "eur_per_t"
            })
        ]
        mock_recommend.return_value = mock_df
        
        request_data = {
            "area": "France",
            "year": 2026,
            "avg_rain_mm": 650.0,
            "pesticides_tonnes": 5000.0,
            "avg_temp": 15.0,
            "irrigation": False,
            "fertilizer": False,
            "top_k": 1,
            "price_unit": "eur_per_t",
            "prices": {
                "maize": 200,
                "wheat": 180
            }
        }
        
        response = client.post("/recommend/revenue", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 1
        assert data["results"][0]["revenue_per_ha"] == 300.0
        assert data["results"][0]["price_value"] == 200.0
    
    def test_recommend_revenue_no_prices(self):
        """Test sans prix fournis"""
        request_data = {
            "area": "France",
            "year": 2026,
            "avg_rain_mm": 650.0,
            "pesticides_tonnes": 5000.0,
            "avg_temp": 15.0,
            "irrigation": False,
            "fertilizer": False,
            "top_k": 5,
            "prices": {}  # Dictionnaire vide
        }
        
        response = client.post("/recommend/revenue", json=request_data)
        
        # Devrait retourner 400 pour prix manquants
        assert response.status_code == 400
    
    def test_recommend_revenue_invalid_prices(self):
        """Test avec prix invalides"""
        request_data = {
            "area": "France",
            "year": 2026,
            "avg_rain_mm": 650.0,
            "pesticides_tonnes": 5000.0,
            "avg_temp": 15.0,
            "irrigation": False,
            "fertilizer": False,
            "top_k": 5,
            "prices": {
                "maize": -100,  # Prix négatif
                "wheat": 0      # Prix zéro
            }
        }
        
        response = client.post("/recommend/revenue", json=request_data)
        
        # Devrait retourner 400 pour prix invalides
        assert response.status_code == 400
    
    @patch('api.main.recommend_by_revenue')
    def test_recommend_revenue_model_error(self, mock_recommend):
        """Test d'erreur du modèle"""
        mock_recommend.side_effect = Exception("Erreur de calcul")
        
        request_data = {
            "area": "France",
            "year": 2026,
            "avg_rain_mm": 650.0,
            "pesticides_tonnes": 5000.0,
            "avg_temp": 15.0,
            "irrigation": False,
            "fertilizer": False,
            "top_k": 5,
            "prices": {
                "maize": 200,
                "wheat": 180
            }
        }
        
        response = client.post("/recommend/revenue", json=request_data)
        
        assert response.status_code == 500


# ---------------------------------------------------------
# Configuration pytest
# ---------------------------------------------------------

if __name__ == "__main__":
    # Pour exécuter directement avec pytest
    pytest.main([__file__, "-v", "--tb=short"])