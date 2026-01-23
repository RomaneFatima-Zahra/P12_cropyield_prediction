[![CICD](https://github.com/RomaneFatima-Zahra/P12_cropyield_prediction/actions/workflows/ci_cd.yaml/badge.svg)](https://github.com/RomaneFatima-Zahra/P12_cropyield_prediction/actions/workflows/ci_cd.yaml)
![Python](https://img.shields.io/badge/python-3.12%20|%20CPython-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker&logoColor=white)

# 🌾 Système de Prédiction de Rendement Agricole

Système complet de prédiction de rendements agricoles et de recommandation de cultures rentables basé sur l'apprentissage automatique, utilisant un modèle **HistGradientBoostingRegressor** optimisé. Les prédictions sont faites à partir des données et variables agronomiques et climatiques disponibles sur le site du FAO ( Food and Agriculture Organization of the united nations).

---

## 📋 Table des Matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modèle ML](#modèle-ml)
- [Performance du modèle](#performance-du-modèle)
- [API Endpoints](#api-endpoints)
- [Technologies](#technologies)
- [Tests](#tests)
- [Contribution](#contribution)

---

## 🎯 Aperçu

Ce projet fournit un système de bout en bout pour :
- **Prédire** les rendements agricoles (hg/ha et t/ha)
- **Recommander** les cultures les plus productives
- **Optimiser** la rentabilité en fonction des prix de marché
- **Simuler** l'impact de l'irrigation et de la fertilisation sur le rendement

---

## ✨ Fonctionnalités

### 1. Prédiction de Rendement
- Prédiction pour une culture spécifique
- Prise en compte des conditions environnementales (température, précipitations, pesticides)
- Simulation d'options agricoles (irrigation, fertilisation)
- Calcul optionnel de la rentabilité financière

### 2. Recommandation par Rendement
- Classement des cultures par rendement prédit
- Top-K recommandations personnalisables
- Visualisation comparative

### 3. Recommandation par Rentabilité
- Estimation des rendements financiers basés sur les prix de marché
- Calcul du revenu par hectare
- Support de différentes unités de prix (€/t, €/kg, €/hg)

### 4. Options Agricoles
- **Irrigation** : +12,000 hg/ha
- **Fertilisation** : +15,000 hg/ha
- **Impact combiné** : +27,000 hg/ha (+2.7 t/ha)

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │ ← Interface utilisateur web
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI API   │ ← API REST
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Predictor   │ ← Moteur de prédiction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HGB Model      │ ← Modèle HistGradientBoostingRegressor
└─────────────────┘
```
**Pipeline ML :** 

```
┌───────────────────────────────────────────┐
│      Raw Data (6 fichiers CSV)            │ 
└────────────────┬──────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────┐
│  Data Cleaning & Feature Engineering      │ 
│   - Fusion par (area x year)              │
│   - 28,242 observations finales           │     
└────────────────┬──────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│        Preprocessing Pipeline            │
│          - OneHotEncoder                 │
│         - StandardScaler                 │
│          - SimpleImputer                 │
│          - Split Temporel (2010)         │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│  Trainning , Evaluation & optimization   │
│      - 5 modèles testés                  │
│     - Cross-validation 5-fold            │
│     - RandomizedSearchCV 50 iterations   │ 
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│    HistGradientBoostingRegressor         │ 
│     R² = 95.76% | MAE = 1.07 t/ha        │ 
└──────────────────────────────────────────┘

```
---

## 📁 Structure du Projet

```
p12/
├── api/
│   └── main.py                 # API FastAPI
│   └── app.py                  # Interface Streamlit
│
├── inputs/
│   ├──raw_data                 # 6 Datasets de base
│   ├── processed
│   │      └── clean_data.csv    # Dataset nettoyé
│   └── candidate_items.json    # Liste des cultures
│
├── model/
│   └── hgb_optimized.joblib    # Modèle entraîné
│
├── scripts/
│   ├── predictor.py            # Moteur de prédiction ML
│   ├── utils.py                # Fonctions utilitaires
│   ├── modelisation.ipynb      # Notebook de modélisation
│   ├── exploration.ipynb       # Notebook EDA
│   └── artifacts/              # Screenshots tracking MLFlow   
│  
├── tests/
│   └── test_unit.py            # Tests unitaires
│
├── pyproject.toml              # Configuration Poetry
├── poetry.lock                 # Configuration Poetry
├── Dockerfile                  # Configuration Docker Image
└── README.md                   # Ce fichier
```
---

## 🚀 Installation

### Prérequis
- Python >= 3.12
- Poetry (gestionnaire de dépendances)

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/RomaneFatima-Zahra/P12_cropyield_prediction.git
cd p12
```

2. **Installer les dépendances**
```bash
poetry install
```

3. **Activer l'environnement virtuel**
```bash
poetry shell
```

4. **Vérifier l'installation**
```bash
python --version  # Devrait afficher Python 3.12.x
```

---

## 💻 Utilisation

### 1. Démarrer l'API FastAPI

```bash
# Depuis le dossier api/
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
**Accès :**
- L'API sera accessible à : `http://localhost:8000`
- Documentation interactive (Swagger) : `http://localhost:8000/docs`
- Documentation alternative (ReDoc) : `http://localhost:8000/redoc`

### 2. Lancer l'interface Streamlit

```bash
# Depuis le dossier streamlit_interface/
streamlit run api/app.py
```

L'interface sera accessible à : `http://localhost:8501`

### 3. Utilisation via Python

```python
from scripts.predictor import model, predict_yield_hg_ha, recommend_by_yield, recommend_by_revenue

# Prédiction simple
yield_pred = predict_yield_hg_ha(
    model,
    area="France",
    item="maize",
    year=2026,
    avg_rain_mm=650.0,
    pesticides_tonnes=5000.0,
    avg_temp=15.0,
    irrigation=True,
    fertilizer=False)
print(f"Rendement prédit : {yield_pred:.2f} hg/ha")

# Recommandation de culture : 
candidate_items = ["maize", "potatoes", "rice, paddy", "soybeans", "sorghum"]

yield_ranking = recommend_by_yield(
    model=best_hgb,
    area=area,
    year=year,
    avg_rain_mm=avg_rain_mm,
    pesticides_tonnes=pesticides_tonnes,
    avg_temp=avg_temp,
    candidate_items=candidate_items,
    irrigation=False,
    fertilizer=True,
    top_k=5)

print(yield_ranking)

# Recommandation de culture et de rentabilité : 
prices = { # Prix fictifs €/t pour chaque culture
    "maize": 180,
    "potatoes": 50,
    "rice, paddy": 220,
    "soybeans": 300,
    "sorghum": 200
}

revenue_ranking = recommend_by_revenue(
    model=best_hgb,
    area=area,
    year=year,
    avg_rain_mm=avg_rain_mm,
    pesticides_tonnes=pesticides_tonnes,
    avg_temp=avg_temp,
    candidate_items=candidate_items,
    prices=prices,
    price_unit="eur_per_t",
    irrigation=True,
    fertilizer=True,
    top_k=5)
print(revenue_ranking)
```
---

## 🤖 Modèle ML

### Algorithme
**HistGradientBoostingRegressor** (scikit-learn)

### Performance
- **R² Test** : 0.9576
- **R² Train** : 0.9943
- **RMSE Test** : 19,665 hg/ha
- **MAE Test** : 10,715 hg/ha
- **Overfitting** : 0.037 (très faible)


### Hyperparamètres Optimisés
```python
{
    'learning_rate': 0.1,          # Taux d'apprentissage
    'max_iter': 600,               # Nombre d'arbres
    'max_depth': None,             # Profondeur illimitée
    'min_samples_leaf': 5,         # Minimum d'échantillons par feuille
    'l2_regularization': 1.0,      # Régularisation L2
    'max_bins': 255                # Nombre de bins pour histogrammes
}

```

### Features Importance ( Permutation Importance / MAE)

Evaluation de l'importance des variables calculée par permutation en utilisant la MAE comme métrique.

1. **Type de culture** (item) : 71,189
2. **Pays** (area) : 15,450
3. **Température moyenne** : 13,185
4. **Pesticides** : 8,320
5. **Précipitations** : 6,305
6. **Année** : 0

**Insight clé :** Le type de culture compte près de **5× plus** que le pays, et **11× plus** que la température.

### Pipeline de Preprocessing
- **Variables catégorielles** : OneHotEncoder + SimpleImputer (most_frequent)
- **Variables numériques** : StandardScaler + SimpleImputer (median)

---

## 📊 Performance du modèle

### Comparaison des Modèles (avant optimisation)

| Modèle | R² Test | RMSE Test | MAE Test | Overfitting |
|--------|---------|-----------|----------|-------------|
| **Random Forest** | 0.9508 | 21,182 | 10,652 | 0.0480 |
| **XGBoost** | 0.9504 | 21,264 | 12,320 | 0.0355 |
| **HGB** | 0.9338 | 24,561 | 14,593 | 0.0339 |
| Ridge | 0.7253 | 50,035 | 33,355 | 0.0376 |
| Dummy | -0.0214 | 96,481 | 69,103 | 0.0214 |


### Modèle Final (HGB Optimisé)

| Métrique | Train | Test |
|----------|-------|------|
| **R²** | 0.9943 | 0.9576 |
| **RMSE** | 6,212 hg/ha | 19,665 hg/ha |
| **MAE** | 3,662 hg/ha | 10,715 hg/ha |


**Conclusion :** 
- Le modèle explique **95.76%** de la variance du rendement
- Erreur moyenne de ±1.07 t/ha (±10,715 hg/ha)
- Excellente généralisation (faible overfitting)

**Avantages du modèle** : 

- ✅ **Meilleure précision** : R² = 95.76% (vs 95.08% pour Random Forest)
- ✅ **Erreur minimale** : MAE = 10,715 hg/ha (vs 10,652 pour RF)
- ✅ **Faible overfitting** : Écart train/test = 3.7% (meilleur équilibre)
- ✅ **Rapidité** : Prédictions en <10ms
- ✅ **Robustesse** : Gestion native des valeurs manquantes

---

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

### Prédiction
```http
POST /predict
Content-Type: application/json

{
  "area": "France",
  "item": "maize",
  "year": 2026,
  "avg_rain_mm": 650.0,
  "pesticides_tonnes": 5000.0,
  "avg_temp": 15.0,
  "irrigation": false,
  "fertilizer": false,
  "price_value": 180.0,
  "price_unit": "eur_per_t"
}
```

### Recommandation par Rendement
```http
POST /recommend/yield
Content-Type: application/json

{
  "area": "France",
  "year": 2026,
  "avg_rain_mm": 650.0,
  "pesticides_tonnes": 5000.0,
  "avg_temp": 15.0,
  "irrigation": false,
  "fertilizer": false,
  "top_k": 5
}

```

### Recommandation par Rentabilité
```http
POST /recommend/revenue
Content-Type: application/json

{
  "area": "France",
  "year": 2026,
  "avg_rain_mm": 650.0,
  "pesticides_tonnes": 5000.0,
  "avg_temp": 15.0,
  "irrigation": true,
  "fertilizer": true,
  "top_k": 5,
  "prices": {
    "maize": 180,
    "rice, paddy": 220,
    "wheat": 200
  },
  "price_unit": "eur_per_t"
}
```
---

## 🛠️ Technologies

### Backend
- **FastAPI** : Framework API REST rapide et moderne
- **Pydantic** : Validation automatique des données
- **Uvicorn** : Serveur ASGI pour Fastapi de haute performance

### Frontend
- **Streamlit** : Interface web interactive
- **Plotly** : Visualisations interactives

### Machine Learning
- **Scikit-learn** : Modèles et preprocessing
- **XGBoost** : Modèle alternatif testé
- **MLflow** : Tracking des expériences et versioning des modèles
- **Pandas** : Manipulation et analyse de données
- **Numpy** : Calculs numériques

### DevOps
- **Poetry** : Gestion des dépendances et environnements virtuels
- **Pytest** : Tests unitaires
- **Joblib** : Sérialisation du modèle
- **Docker** : Conteneurisation et déploiement

---

## 🧪 Tests

Lancer les tests unitaires :

```bash
pytest tests/ -v
```

Tester l'API manuellement :

```bash
# Test du health endpoint
curl http://localhost:8000/health

# Test de prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": "France",
    "item": "maize",
    "year": 2026,
    "avg_rain_mm": 650,
    "pesticides_tonnes": 5000,
    "avg_temp": 15,
    "irrigation": false,
    "fertilizer": false
  }'

  # Test de recommandation par rendement
  curl -X 'POST' \ 'http://localhost:8000/recommend/yield' \
  -H 'Content-Type: application/json' \
  -d '{
  "area": "France",
  "avg_rain_mm": 650,
  "avg_temp": 15,
  "fertilizer": false,
  "irrigation": false,
  "pesticides_tonnes": 5000,
  "top_k": 5,
  "year": 2026
}' 

 # Test de recommandation par rentabilité 
 curl -X 'POST' \ 'http://localhost:8000/recommend/revenue' \
  -H 'Content-Type: application/json' \
  -d '{
  "area": "France",
  "avg_rain_mm": 650,
  "avg_temp": 15,
  "fertilizer": false,
  "irrigation": false,
  "pesticides_tonnes": 5000,
  "price_unit": "eur_per_t",
  "prices": {
    "cassava": 580,
    "maize": 100,
    "plantains and others": 130,
    "potatoes": 200,
    "rice, paddy": 300,
    "sorghum": 310,
    "soybeans": 410,
    "sweet potatoes": 370,
    "wheat": 350,
    "yams": 400
  },
  "top_k": 5,
  "year": 2026
}'
```
---

## 🤝🏻 Contribution

### 👥 Auteur

- **Fatima-Zahra BARHOU** 

### Sources 

- Données : FAO (Food and Agriculture Organization) : https://www.fao.org

---

**Note** : Ce projet est développé à des fins pédagogiques. Les prédictions doivent être utilisées comme aide à la décision et non comme unique source d'information pour des décisions agricoles.
