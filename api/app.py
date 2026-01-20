import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ========================================================
# Configuration de la page
# ========================================================
st.set_page_config(
    page_title="Prédiction de Rendement Agricole",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================
# Configuration API
# ========================================================
API_URL = st.sidebar.text_input(
    "URL de l'API",
    value="http://localhost:8000",
    help="Adresse de votre API FastAPI"
)

# Test de connexion API
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API connectée")
    else:
        st.sidebar.warning("⚠️ API répond mais avec erreur")
except:
    st.sidebar.error("❌ API non accessible")

# ========================================================
# Titre et description
# ========================================================
st.title("🌾 Prédiction de Rendement Agricole")
st.markdown("""
Cette application vous permet de prédire les rendements agricoles et de recevoir des recommandations 
de cultures basées sur vos conditions environnementales.
""")

# ========================================================
# Sidebar - Choix du mode
# ========================================================
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.radio(
    "Mode",
    ["Prédiction", "Recommandation par Rendement", "Recommandation par Rentabilité"],
    help="Choisissez le type d'analyse souhaité"
)

# ========================================================
# Paramètres communs
# ========================================================
st.sidebar.subheader("📊 Contexte Environnemental")

area = st.sidebar.text_input(
    "Pays",
    value="France",
    help="Nom du pays"
)

year = st.sidebar.number_input(
    "Année",
    min_value=1990,
    max_value=2100,
    value=2026,
    step=1
)

avg_temp = st.sidebar.slider(
    "Température moyenne (°C)",
    min_value=-15.0,
    max_value=50.0,
    value=15.0,
    step=0.5,
    help="Température moyenne annuelle"
)

avg_rain_mm = st.sidebar.slider(
    "Précipitations (mm)",
    min_value=0.0,
    max_value=5000.0,
    value=650.0,
    step=10.0,
    help="Précipitations moyennes annuelles"
)

pesticides_tonnes = st.sidebar.number_input(
    "Pesticides (tonnes)",
    min_value=0.0,
    max_value=50000.0,
    value=5000.0,
    step=100.0,
    help="Quantité de pesticides utilisés"
)

st.sidebar.subheader("🚜 Options Agricoles")

irrigation = st.sidebar.checkbox(
    "Irrigation",
    value=False,
    help="Utilisation de l'irrigation (+12,000 hg/ha)"
)

fertilizer = st.sidebar.checkbox(
    "Fertilisation",
    value=False,
    help="Utilisation de la fertilisation (+15,000 hg/ha)"
)

# ========================================================
# Mode Prédiction
# ========================================================
if mode == "Prédiction":
    st.header("🔮 Prédiction de Rendement")
    
    col1, col2 = st.columns([2, 1])

    cultures = ["maize", "wheat", "rice, paddy", "potatoes",
                "sorghum", "soybeans", "cassava", 
                "yams", "sweet potatoes", "plantains and others"]
    
    with col1:
        item = st.selectbox(
            "Culture",
            options=cultures,
            index=0,  # culture par défaut
            help="Sélectionnez le type de culture"
        )
        

    with col2:
        calculate_revenue = st.checkbox(
            "Calculer la rentabilité",
            value=False,
            help="Ajouter le calcul du revenu par hectare"
        )
        
        if calculate_revenue:
            price_value = st.number_input(
                "Prix (€/tonne)",
                min_value=0.1,
                value=500.0,
                step=10.0,
                help="Prix de vente par tonne"
            )
            price_unit = st.selectbox(
                "Unité de prix",
                ["eur_per_t", "eur_per_kg", "eur_per_hg"],
                index=0
            )
        else:
            price_value = None
            price_unit = "eur_per_t"
    
    if st.button("🚀 Prédire le rendement", type="primary", width="stretch"):
        with st.spinner("Prédiction en cours..."):
            try:
                # Préparer la requête
                payload = {
                    "area": area,
                    "item": item,
                    "year": year,
                    "avg_rain_mm": avg_rain_mm,
                    "pesticides_tonnes": pesticides_tonnes,
                    "avg_temp": avg_temp,
                    "irrigation": irrigation,
                    "fertilizer": fertilizer
                }
                
                if calculate_revenue and price_value is not None:
                    payload["price_value"] = price_value
                    payload["price_unit"] = price_unit
                
                # Envoyer la requête à l'API
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Afficher les résultats
                    st.success("✅ Prédiction réussie !")
                    
                    # Métriques principales
                    if calculate_revenue and result.get("revenue_per_ha"):
                        col1, col2, col3 = st.columns(3)
                    else:
                        col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Culture",
                            result["item"].capitalize()
                        )
                    
                    with col2:
                        st.metric(
                            "Rendement",
                            f"{result['pred_yield_t_ha']:.3f} t/ha",
                            help="Tonnes par hectare"
                        )
                    
                    if calculate_revenue and result.get("revenue_per_ha"):
                        with col3:
                            st.metric(
                                "Rentabilité",
                                f"{result['revenue_per_ha']:.3f} €/ha",
                                help="Revenu estimé par hectare"
                            )
                    
                    # Détails supplémentaires
                    with st.expander("📊 Détails complets"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Rendement détaillé:**")
                            st.write(f"- {result['pred_yield_hg_ha']:.0f} hg/ha")
                            st.write(f"- {result['pred_yield_t_ha']:.3f} t/ha")
                        with col2:
                            st.write(f"**Options appliquées:**")
                            st.write(f"- 🚰 Irrigation : {'✅ Oui' if irrigation else '❌ Non'}")
                            st.write(f"- 🌱 Fertilisation : {'✅ Oui' if fertilizer else '❌ Non'}")
                    
                else:
                    st.error(f"❌ Erreur {response.status_code}: {response.json().get('detail', 'Erreur inconnue')}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Impossible de se connecter à l'API à l'adresse {API_URL}")
                st.info("Vérifiez que votre API est bien démarrée et accessible.")
            except requests.exceptions.Timeout:
                st.error("❌ Timeout : l'API met trop de temps à répondre")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ========================================================
# Mode Recommandation par Rendement
# ========================================================
elif mode == "Recommandation par Rendement":
    st.header("📋 Recommandation par Rendement")
    st.info("💡 Ce mode recommande les cultures avec les meilleurs rendements prévus (t/ha)")
    
    # Options de recommandation
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider(
            "Nombre de recommandations",
            min_value=1,
            max_value=10,
            value=5,
            help="Nombre de cultures à recommander"
        )
    
    with col2:
        st.write("**Cultures analysées**")
        st.caption("L'API utilise automatiquement la liste des cultures fournie")
    
    if st.button("🚀 Obtenir des recommandations", type="primary", width="stretch"):
        with st.spinner("Calcul des recommandations..."):
            try:
                # Préparer la requête
                payload = {
                    "area": area,
                    "year": year,
                    "avg_rain_mm": avg_rain_mm,
                    "pesticides_tonnes": pesticides_tonnes,
                    "avg_temp": avg_temp,
                    "irrigation": irrigation,
                    "fertilizer": fertilizer,
                    "top_k": top_k
                }
                
                # Envoyer la requête à l'API
                response = requests.post(f"{API_URL}/recommend/yield", json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    recommendations = result["results"]
                    
                    st.success(f"✅ {len(recommendations)} recommandation(s) générée(s) !")
                    
                    # Créer le DataFrame pour l'affichage
                    df = pd.DataFrame(recommendations)
                    
                    # Graphique à barres
                    st.subheader("📊 Visualisation des rendements")
                    
                    fig = px.bar(
                        df,
                        x="item",
                        y="pred_yield_t_ha",
                        title="Rendement prédit par culture (t/ha)",
                        labels={"item": "Culture", "pred_yield_t_ha": "Rendement (t/ha)"},
                        color="pred_yield_t_ha",
                        color_continuous_scale="YlGn",
                        text="pred_yield_t_ha"
                    )
                    
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False,
                        height=500
                    )
                    st.plotly_chart(fig, width="stretch")
                    
                    # Tableau détaillé
                    st.subheader("📋 Détails des recommandations")
                    
                    # Formater le DataFrame pour l'affichage
                    display_df = df.copy()
                    display_df["Culture"] = display_df["item"].str.capitalize()
                    display_df["Rendement (t/ha)"] = display_df["pred_yield_t_ha"].round(2)
                    display_df["Rendement (hg/ha)"] = display_df["pred_yield_hg_ha"].round(0)
                    
                    st.dataframe(
                        display_df[["Culture", "Rendement (t/ha)", "Rendement (hg/ha)"]],
                        width="stretch",
                        hide_index=True
                    )
                    
                    # Options appliquées
                    st.caption(f"🚰 Irrigation : {'✅ Oui' if irrigation else '❌ Non'} | 🌱 Fertilisation : {'✅ Oui' if fertilizer else '❌ Non'}")
                    
                else:
                    st.error(f"❌ Erreur {response.status_code}: {response.json().get('detail', 'Erreur inconnue')}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Impossible de se connecter à l'API à l'adresse {API_URL}")
                st.info("Vérifiez que votre API est bien démarrée et accessible.")
            except requests.exceptions.Timeout:
                st.error("❌ Timeout : l'API met trop de temps à répondre")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ========================================================
# Mode Recommandation par Rentabilité
# ========================================================
else:  # Recommandation par Rentabilité
    st.header("💰 Recommandation par Rentabilité")
    st.info("💡 Ce mode recommande les cultures les plus rentables en €/ha selon les prix que vous indiquez")
    
    # Options de recommandation
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider(
            "Nombre de recommandations",
            min_value=1,
            max_value=10,
            value=5,
            help="Nombre de cultures à recommander"
        )
    
    with col2:
        price_unit = st.selectbox(
            "Unité de prix",
            ["eur_per_t", "eur_per_kg", "eur_per_hg"],
            index=0,
            help="Unité pour saisir les prix"
        )
    
    # Saisie des prix
    st.subheader("💰 Saisie des prix de vente")
    st.write("Entrez les prix de vente pour les cultures que vous souhaitez comparer :")
    
    # Liste prédéfinie de cultures courantes
    common_crops = [
        "maize", "rice, paddy", "wheat", "cassava", "sorghum", 
        "potatoes", "soybeans", "yams", "sweet potatoes", "plantains and others"
    ]
    
    # Créer un formulaire pour les prix
    prices = {}
    
    # Diviser en 2 colonnes pour l'affichage
    col1, col2 = st.columns(2)
    
    for idx, crop in enumerate(common_crops):
        with col1 if idx % 2 == 0 else col2:
            price = st.number_input(
                f"💵 {crop.capitalize()}",
                min_value=0.0,
                value=0.0,
                step=10.0,
                key=f"price_{crop}",
                help=f"Prix de vente pour {crop} (0 = ignorer cette culture)"
            )
            if price > 0:
                prices[crop] = price
        
    # Afficher le résumé des prix
    if prices:
        st.success(f"✅ {len(prices)} culture(s) avec prix défini(s)")
    else:
        st.warning("⚠️ Aucun prix défini. Entrez au moins un prix > 0 pour obtenir des recommandations.")
    
    if st.button("🚀 Obtenir des recommandations", type="primary", width="stretch", disabled=len(prices) == 0):
        with st.spinner("Calcul des recommandations..."):
            try:
                # Préparer la requête
                payload = {
                    "area": area,
                    "year": year,
                    "avg_rain_mm": avg_rain_mm,
                    "pesticides_tonnes": pesticides_tonnes,
                    "avg_temp": avg_temp,
                    "irrigation": irrigation,
                    "fertilizer": fertilizer,
                    "top_k": top_k,
                    "prices": prices,
                    "price_unit": price_unit
                }
                
                # Envoyer la requête à l'API
                response = requests.post(f"{API_URL}/recommend/revenue", json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    recommendations = result["results"]
                    
                    st.success(f"✅ {len(recommendations)} recommandation(s) générée(s) !")
                    
                    # Créer le DataFrame pour l'affichage
                    df = pd.DataFrame(recommendations)
                    
                    # Graphique à barres
                    st.subheader("📊 Visualisation de la rentabilité")
                    
                    fig = px.bar(
                        df,
                        x="item",
                        y="revenue_per_ha",
                        title="Rentabilité par culture (€/ha)",
                        labels={"item": "Culture", "revenue_per_ha": "Rentabilité (€/ha)"},
                        color="revenue_per_ha",
                        color_continuous_scale="YlGn",
                        text="revenue_per_ha"
                    )
                    
                    fig.update_traces(texttemplate='%{text:.2f}€', textposition='outside')
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False,
                        height=500
                    )
                    st.plotly_chart(fig, width="stretch")
                    
                    # Tableau détaillé
                    st.subheader("📋 Détails des recommandations")
                    
                    # Formater le DataFrame pour l'affichage
                    display_df = df.copy()
                    display_df["Culture"] = display_df["item"].str.capitalize()
                    display_df["Rendement (t/ha)"] = display_df["pred_yield_t_ha"].round(2)
                    display_df[f"Prix ({price_unit})"] = display_df["price_value"].round(2)
                    display_df["Rentabilité (€/ha)"] = display_df["revenue_per_ha"].round(2)
                    
                    st.dataframe(
                        display_df[["Culture", "Rendement (t/ha)", f"Prix ({price_unit})", "Rentabilité (€/ha)"]],
                        width="stretch",
                        hide_index=True
                    )
                    
                    # Options appliquées
                    st.caption(f"🚰 Irrigation : {'✅ Oui' if irrigation else '❌ Non'} | 🌱 Fertilisation : {'✅ Oui' if fertilizer else '❌ Non'}")
                    
                    # Insights
                    with st.expander("💡 Insights"):
                        best_crop = df.iloc[0]
                        st.write(f"🏆 **Meilleure culture** : {best_crop['item'].capitalize()}")
                        st.write(f"- Rentabilité : {best_crop['revenue_per_ha']:.2f} €/ha")
                        st.write(f"- Rendement : {best_crop['pred_yield_t_ha']:.2f} t/ha")
                        st.write(f"- Prix : {best_crop['price_value']:.2f} {price_unit}")
                    
                else:
                    error_detail = response.json().get('detail', 'Erreur inconnue')
                    st.error(f"❌ Erreur {response.status_code}: {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Impossible de se connecter à l'API à l'adresse {API_URL}")
                st.info("Vérifiez que votre API est bien démarrée et accessible.")
            except requests.exceptions.Timeout:
                st.error("❌ Timeout : l'API met trop de temps à répondre")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ========================================================
# Footer
# ========================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌾 <strong>Agricultural Yield Prediction System</strong> | Using HistGradientBoosting model </p>
    <p><small>Pour obtenir de l'aide, consultez la documentation de l'API : <a href='http://localhost:8000/docs' target='_blank'>API Docs</a></small></p>
</div>
""", unsafe_allow_html=True)