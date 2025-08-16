import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    layout="wide", 
    page_title="Dashboard Marketing M8 ", 
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .segment-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

## ----------------------------
## CHARGEMENT DES DONNÉES AVANCÉ (M8 Improved)
## ----------------------------

@st.cache_data
def charger_donnees_avancees():
    """Chargement et préparation avancée des données avec segmentation M7"""
    try:
        # 1. Chargement des fichiers (avec gestion d'erreurs robuste)
        fichiers = {
            'clients': './data/customers_data.csv',
            'produits': './data/products_data.csv', 
            'ventes': './data/sales_data.csv',
            'marketing': './data/marketing_data.csv'
        }
        
        dfs = {}
        for nom, path in fichiers.items():
            try:
                dfs[nom] = pd.read_csv(path)
            except FileNotFoundError:
                # Génération de données de démonstration si fichiers manquants
                if nom == 'clients':
                    dfs[nom] = generer_donnees_demo_clients()
                elif nom == 'produits':
                    dfs[nom] = generer_donnees_demo_produits()
                elif nom == 'ventes':
                    dfs[nom] = generer_donnees_demo_ventes()
                else:
                    dfs[nom] = generer_donnees_demo_marketing()
        
        # 2. Standardisation des colonnes
        clients = dfs['clients']
        produits = dfs['produits'] 
        ventes = dfs['ventes']
        marketing = dfs['marketing']
        
        # Renommage intelligent des colonnes
        rename_maps = {
            'ventes': {
                'Product_ID': 'product_id', 'Customer_ID': 'customer_id',
                'Date': 'date', 'Quantity': 'quantity', 'Sale_Price': 'sale_price'
            },
            'produits': {
                'Product_ID': 'product_id', 'Category': 'product_category',
                'Price': 'unit_price', 'Product_Name': 'product_name'
            },
            'clients': {
                'Customer_ID': 'customer_id', 'Name': 'customer_name',
                'Age': 'age', 'Gender': 'gender', 'Location': 'location'
            }
        }
        
        for df_name, rename_map in rename_maps.items():
            if df_name in ['ventes', 'produits', 'clients']:
                df = ventes if df_name == 'ventes' else (produits if df_name == 'produits' else clients)
                available_renames = {k: v for k, v in rename_map.items() if k in df.columns}
                df.rename(columns=available_renames, inplace=True)
        
        # 3. Fusion complète et intelligente des données
        df = pd.merge(ventes, produits, on='product_id', how='left')
        df = pd.merge(df, clients, on='customer_id', how='left')
        
        # 4. Traitement des dates et périodes
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Supprimer les dates invalides
        
        df['mois'] = df['date'].dt.to_period('M').astype(str)
        df['annee'] = df['date'].dt.year
        df['trimestre'] = df['date'].dt.to_period('Q').astype(str)
        df['semaine'] = df['date'].dt.isocalendar().week
        df['jour_semaine'] = df['date'].dt.day_name()
        
        # 5. Calculs financiers avancés
        df['chiffre_affaires'] = df['quantity'] * df.get('unit_price', df.get('sale_price', 50))
        df['marge_brute'] = df['chiffre_affaires'] * 0.35  # Marge 35%
        df['cout_acquisition'] = 15  # CAC moyen fixe pour demo
        
        # 6. SEGMENTATION CLIENTS AVANCÉE (selon M7)
        client_metrics = df.groupby('customer_id').agg({
            'chiffre_affaires': ['sum', 'mean'],
            'date': ['count', 'max', 'min'],
            'quantity': 'sum'
        }).round(2)
        
        # Aplatir les colonnes multi-niveaux
        client_metrics.columns = ['_'.join(col).strip() for col in client_metrics.columns]
        client_metrics = client_metrics.rename(columns={
            'chiffre_affaires_sum': 'clv_total',
            'chiffre_affaires_mean': 'panier_moyen', 
            'date_count': 'frequence',
            'date_max': 'derniere_commande',
            'date_min': 'premiere_commande',
            'quantity_sum': 'quantite_totale'
        })
        
        # Calcul de la récence
        date_reference = df['date'].max()
        client_metrics['recence'] = (date_reference - client_metrics['derniere_commande']).dt.days
        
        # Segmentation selon M7: Champions, Potentiels, Standards
        def segmentation_m7(row):
            clv = row['clv_total']
            freq = row['frequence'] 
            recence = row['recence']
            
            # Champions (25%) : Haute valeur, très actifs
            if clv >= 800 and freq >= 8 and recence <= 30:
                return 'Champions'
            # Potentiels (35%) : Valeur moyenne avec potentiel
            elif clv >= 300 and freq >= 4 and recence <= 90:
                return 'Potentiels'
            # Standards (40%) : Clients occasionnels
            else:
                return 'Standards'
        
        client_metrics['segment_m7'] = client_metrics.apply(segmentation_m7, axis=1)
        
        # Ajout du segment au dataframe principal
        df = pd.merge(df, client_metrics[['segment_m7']], left_on='customer_id', right_index=True)
        
        # 7. Intégration données marketing avec ROI
        if not marketing.empty:
            # Normalisation des colonnes marketing
            marketing_cols = {
                'Start_Date': 'date', 'Budget': 'marketing_spend',
                'Conversions': 'conversions', 'Channel': 'canal_marketing'
            }
            available_marketing_renames = {k: v for k, v in marketing_cols.items() if k in marketing.columns}
            marketing.rename(columns=available_marketing_renames, inplace=True)
            
            if 'date' in marketing.columns:
                marketing['date'] = pd.to_datetime(marketing['date'], errors='coerce')
                
                # Agrégation marketing par date
                marketing_agg = marketing.groupby('date').agg({
                    'marketing_spend': 'sum',
                    'conversions': 'sum'
                }).reset_index()
                
                df = pd.merge(df, marketing_agg, on='date', how='left')
                
                # Calcul ROI et ROAS
                df['marketing_spend'] = df['marketing_spend'].fillna(0)
                df['roas'] = np.where(df['marketing_spend'] > 0, 
                                    df['chiffre_affaires'] / df['marketing_spend'], 0)
                df['roi'] = np.where(df['marketing_spend'] > 0,
                                   (df['marge_brute'] - df['marketing_spend']) / df['marketing_spend'], 0)
        
        return df, client_metrics
        
    except Exception as e:
        st.error(f"Erreur de chargement avancée: {str(e)}")
        # Retourner des données de démonstration
        return generer_dataset_complet_demo()

def generer_donnees_demo_clients():
    """Génère des données clients de démonstration"""
    np.random.seed(42)
    n_clients = 500
    
    return pd.DataFrame({
        'customer_id': range(1, n_clients + 1),
        'customer_name': [f'Client_{i}' for i in range(1, n_clients + 1)],
        'age': np.random.normal(40, 15, n_clients).astype(int),
        'gender': np.random.choice(['M', 'F'], n_clients),
        'location': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'], n_clients)
    })

def generer_donnees_demo_produits():
    """Génère des données produits de démonstration"""
    categories = ['Électronique', 'Mode', 'Maison', 'Sport', 'Beauté']
    n_produits = 100
    
    return pd.DataFrame({
        'product_id': range(1, n_produits + 1),
        'product_name': [f'Produit_{i}' for i in range(1, n_produits + 1)],
        'product_category': np.random.choice(categories, n_produits),
        'unit_price': np.random.uniform(20, 500, n_produits).round(2)
    })

def generer_donnees_demo_ventes():
    """Génère des données de ventes de démonstration"""
    np.random.seed(42)
    n_ventes = 2000
    
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    return pd.DataFrame({
        'customer_id': np.random.randint(1, 501, n_ventes),
        'product_id': np.random.randint(1, 101, n_ventes),
        'date': np.random.choice(dates, n_ventes),
        'quantity': np.random.randint(1, 5, n_ventes),
        'sale_price': np.random.uniform(20, 500, n_ventes).round(2)
    })

def generer_donnees_demo_marketing():
    """Génère des données marketing de démonstration"""
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
    canaux = ['Google Ads', 'Facebook', 'Email', 'Instagram', 'LinkedIn']
    
    return pd.DataFrame({
        'date': np.repeat(dates, len(canaux))[:500],  # Limiter à 500 entrées
        'canal_marketing': np.tile(canaux, len(dates))[:500],
        'marketing_spend': np.random.uniform(100, 2000, 500),
        'conversions': np.random.randint(5, 50, 500)
    })

def generer_dataset_complet_demo():
    """Génère un dataset complet de démonstration en cas d'erreur"""
    clients = generer_donnees_demo_clients()
    produits = generer_donnees_demo_produits()
    ventes = generer_donnees_demo_ventes()
    
    # Fusion basique
    df = pd.merge(ventes, produits, on='product_id', how='left')
    df = pd.merge(df, clients, on='customer_id', how='left')
    
    # Ajout de colonnes de base
    df['chiffre_affaires'] = df['quantity'] * df['unit_price']
    df['marge_brute'] = df['chiffre_affaires'] * 0.35
    df['segment_m7'] = np.random.choice(['Champions', 'Potentiels', 'Standards'], len(df), p=[0.25, 0.35, 0.40])
    
    client_metrics = pd.DataFrame()
    
    return df, client_metrics

# Chargement des données
df_complet, client_metrics = charger_donnees_avancees()

## ----------------------------
## HEADER MODERNE ET NAVIGATION
## ----------------------------

st.markdown("""
<div class="main-header">
    <h1>🚀 Dashboard Marketing M8 - Advanced Analytics</h1>
    <p>Analyse avancée basée sur la stratégie M7 avec segmentation Champions/Potentiels/Standards</p>
</div>
""", unsafe_allow_html=True)

# Navigation par onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'Ensemble", 
    "🎯 Segments M7", 
    "📈 Performance Canaux",
    "🔮 Prédictions",
    "📋 Action Plan"
])

## ----------------------------
## SIDEBAR AVANCÉ AVEC FILTRES INTELLIGENTS
## ----------------------------

st.sidebar.markdown("### 🎛️ **Contrôles Avancés**")

# Métriques rapides sidebar
if not df_complet.empty:
    total_ca = df_complet['chiffre_affaires'].sum()
    total_clients = df_complet['customer_id'].nunique()
    
    st.sidebar.markdown(f"""
    **📈 Aperçu Rapide:**
    - **CA Total:** {total_ca:,.0f} €
    - **Clients:** {total_clients:,}
    - **Commandes:** {len(df_complet):,}
    """)

# Filtres temporels avancés
st.sidebar.markdown("#### 📅 **Période d'Analyse**")

if not df_complet.empty and 'date' in df_complet.columns:
    min_date = df_complet['date'].min().date()
    max_date = df_complet['date'].max().date()
    
    # Présélections rapides
    preset_periods = st.sidebar.selectbox(
        "Période prédéfinie",
        ['Personnalisée', 'Derniers 30 jours', 'Dernier trimestre', 'Dernière année', 'Tout']
    )
    
    if preset_periods == 'Derniers 30 jours':
        start_date = max_date - timedelta(days=30)
        end_date = max_date
    elif preset_periods == 'Dernier trimestre':
        start_date = max_date - timedelta(days=90)
        end_date = max_date
    elif preset_periods == 'Dernière année':
        start_date = max_date - timedelta(days=365)
        end_date = max_date
    elif preset_periods == 'Tout':
        start_date = min_date
        end_date = max_date
    else:
        date_range = st.sidebar.date_input(
            "Sélection manuelle",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    
    # Filtrage des données
    mask_date = (df_complet['date'].dt.date >= start_date) & (df_complet['date'].dt.date <= end_date)
    df_filtered = df_complet[mask_date]
else:
    df_filtered = df_complet

# Filtres de segmentation M7
st.sidebar.markdown("#### 🎯 **Segmentation M7**")

if 'segment_m7' in df_filtered.columns:
    segments_disponibles = df_filtered['segment_m7'].unique()
    selected_segments = st.sidebar.multiselect(
        "Segments clients",
        segments_disponibles,
        default=segments_disponibles
    )
    df_filtered = df_filtered[df_filtered['segment_m7'].isin(selected_segments)]

# Autres filtres
if 'product_category' in df_filtered.columns:
    categories = ['Toutes'] + list(df_filtered['product_category'].unique())
    selected_category = st.sidebar.selectbox("Catégorie produit", categories)
    if selected_category != 'Toutes':
        df_filtered = df_filtered[df_filtered['product_category'] == selected_category]

## ----------------------------
## TAB 1: VUE D'ENSEMBLE AVANCÉE
## ----------------------------

with tab1:
    if df_filtered.empty:
        st.warning("Aucune donnée disponible pour les filtres sélectionnés")
    else:
        # KPI Cards améliorés
        st.markdown("### 📊 **KPI Principaux**")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculs KPI
        ca_total = df_filtered['chiffre_affaires'].sum()
        ca_precedent = df_complet['chiffre_affaires'].sum() - ca_total  # Simulation
        delta_ca = ((ca_total - ca_precedent) / ca_precedent * 100) if ca_precedent > 0 else 0
        
        nb_clients = df_filtered['customer_id'].nunique()
        nb_commandes = len(df_filtered)
        panier_moyen = ca_total / nb_commandes if nb_commandes > 0 else 0
        
        if 'marge_brute' in df_filtered.columns:
            marge_totale = df_filtered['marge_brute'].sum()
            marge_pct = (marge_totale / ca_total * 100) if ca_total > 0 else 0
        else:
            marge_totale = ca_total * 0.35
            marge_pct = 35
        
        with col1:
            st.metric("💰 CA Total", f"{ca_total:,.0f} €", f"{delta_ca:+.1f}%")
        with col2:
            st.metric("👥 Clients", f"{nb_clients:,}", f"CLV: {ca_total/nb_clients:.0f}€" if nb_clients > 0 else "N/A")
        with col3:
            st.metric("🛒 Commandes", f"{nb_commandes:,}", f"{panier_moyen:.0f}€/cmd")
        with col4:
            st.metric("💹 Marge", f"{marge_totale:,.0f} €", f"{marge_pct:.1f}%")
        with col5:
            if 'roas' in df_filtered.columns and df_filtered['roas'].sum() > 0:
                roas_moyen = df_filtered['roas'].mean()
                st.metric("📈 ROAS", f"{roas_moyen:.1f}x")
            else:
                taux_conversion = nb_clients / df_filtered['customer_id'].count() * 100 if len(df_filtered) > 0 else 0
                st.metric("🎯 Conv.", f"{taux_conversion:.1f}%")
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            # Évolution temporelle avec tendance
            if 'mois' in df_filtered.columns:
                evolution_data = df_filtered.groupby('mois').agg({
                    'chiffre_affaires': 'sum',
                    'customer_id': 'nunique',
                    'marge_brute': 'sum'
                }).reset_index()
                
                fig_evolution = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Chiffre d\'Affaires', 'Nombre de Clients'),
                    vertical_spacing=0.08
                )
                
                fig_evolution.add_trace(
                    go.Scatter(
                        x=evolution_data['mois'],
                        y=evolution_data['chiffre_affaires'],
                        mode='lines+markers',
                        name='CA',
                        line=dict(color='#667eea', width=3)
                    ), row=1, col=1
                )
                
                fig_evolution.add_trace(
                    go.Scatter(
                        x=evolution_data['mois'],
                        y=evolution_data['customer_id'],
                        mode='lines+markers',
                        name='Clients',
                        line=dict(color='#764ba2', width=3)
                    ), row=2, col=1
                )
                
                fig_evolution.update_layout(
                    title="📈 Évolution Temporelle",
                    height=500,
                    showlegend=False
                )
                fig_evolution.update_xaxes(tickangle=-45)
                
                st.plotly_chart(fig_evolution, use_container_width=True)
        
        with col2:
            # Performance par segment M7
            if 'segment_m7' in df_filtered.columns:
                segment_perf = df_filtered.groupby('segment_m7').agg({
                    'chiffre_affaires': 'sum',
                    'customer_id': 'nunique',
                    'marge_brute': 'sum'
                }).reset_index()
                
                # Graphique en barres empilées
                fig_segments = go.Figure()
                
                fig_segments.add_trace(go.Bar(
                    name='CA',
                    x=segment_perf['segment_m7'],
                    y=segment_perf['chiffre_affaires'],
                    text=segment_perf['chiffre_affaires'].apply(lambda x: f"{x:,.0f}€"),
                    textposition='auto',
                    marker_color='#667eea'
                ))
                
                fig_segments.update_layout(
                    title="💎 Performance par Segment M7",
                    xaxis_title="Segments",
                    yaxis_title="Chiffre d'Affaires (€)",
                    height=500
                )
                
                st.plotly_chart(fig_segments, use_container_width=True)
        
        # Analyse comportementale
        st.markdown("### 🔍 **Analyse Comportementale**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Distribution des paniers
            if nb_commandes > 0:
                bins = [0, 50, 100, 200, 500, float('inf')]
                labels = ['<50€', '50-100€', '100-200€', '200-500€', '>500€']
                df_filtered['panier_range'] = pd.cut(df_filtered['chiffre_affaires'], bins=bins, labels=labels, include_lowest=True)
                
                panier_dist = df_filtered['panier_range'].value_counts().reset_index()
                panier_dist.columns = ['range', 'count']
                
                fig_paniers = px.pie(
                    panier_dist,
                    values='count',
                    names='range',
                    title="🛒 Distribution des Paniers",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_paniers, use_container_width=True)
        
        with col2:
            # Top catégories
            if 'product_category' in df_filtered.columns:
                cat_perf = df_filtered.groupby('product_category')['chiffre_affaires'].sum().sort_values(ascending=True).tail(8)
                
                fig_categories = px.bar(
                    x=cat_perf.values,
                    y=cat_perf.index,
                    orientation='h',
                    title="🏆 Top Catégories",
                    color=cat_perf.values,
                    color_continuous_scale='viridis'
                )
                fig_categories.update_layout(showlegend=False)
                st.plotly_chart(fig_categories, use_container_width=True)
        
        with col3:
            # Saisonnalité
            if 'jour_semaine' in df_filtered.columns:
                jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                jour_perf = df_filtered.groupby('jour_semaine')['chiffre_affaires'].sum().reindex(jours_ordre)
                
                fig_jours = px.bar(
                    x=[j[:3] for j in jour_perf.index],
                    y=jour_perf.values,
                    title="📅 Performance par Jour",
                    color=jour_perf.values,
                    color_continuous_scale='blues'
                )
                fig_jours.update_layout(showlegend=False)
                st.plotly_chart(fig_jours, use_container_width=True)

## ----------------------------
## TAB 2: ANALYSE DES SEGMENTS M7
## ----------------------------

with tab2:
    st.markdown("""
    <div class="segment-header">
        <h2>🎯 Analyse Détaillée des Segments M7</h2>
        <p>Champions (25%) • Potentiels (35%) • Standards (40%)</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df_filtered.empty or 'segment_m7' not in df_filtered.columns:
        st.warning("Données de segmentation non disponibles")
    else:
        # Analyse comparative des segments
        segment_analysis = df_filtered.groupby('segment_m7').agg({
            'customer_id': 'nunique',
            'chiffre_affaires': ['sum', 'mean'],
            'quantity': 'sum',
            'marge_brute': 'sum'
        }).round(2)
        
        segment_analysis.columns = ['nb_clients', 'ca_total', 'ca_moyen', 'qty_total', 'marge_total']
        segment_analysis['part_clients'] = segment_analysis['nb_clients'] / segment_analysis['nb_clients'].sum() * 100
        segment_analysis['part_ca'] = segment_analysis['ca_total'] / segment_analysis['ca_total'].sum() * 100
        
        st.markdown("### 📊 **Vue d'Ensemble des Segments**")
        
        # Métriques par segment
        segments = segment_analysis.index.tolist()
        
        for segment in segments:
            data = segment_analysis.loc[segment]
            
            # Couleurs par segment
            colors = {
                'Champions': '#FFD700',
                'Potentiels': '#32CD32', 
                'Standards': '#4682B4'
            }
            segment_color = colors.get(segment, '#808080')
            
            with st.container():
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {segment_color}20, {segment_color}10); 
                           padding: 1rem; border-radius: 10px; border-left: 4px solid {segment_color}; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(f"👥 {segment}", f"{data['nb_clients']:,}", f"{data['part_clients']:.1f}% des clients")
                with col2:
                    st.metric("💰 CA Total", f"{data['ca_total']:,.0f} €", f"{data['part_ca']:.1f}% du CA")
                with col3:
                    st.metric("💳 CA Moyen", f"{data['ca_moyen']:,.0f} €")
                with col4:
                    st.metric("📦 Quantité", f"{data['qty_total']:,}")
                with col5:
                    st.metric("💹 Marge", f"{data['marge_total']:,.0f} €")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analyses approfondies par segment
        col1, col2 = st.columns(2)
        
        with col1:
            # Matrice valeur/volume
            fig_matrice = px.scatter(
                segment_analysis.reset_index(),
                x='nb_clients',
                y='ca_moyen',
                size='ca_total',
                color='segment_m7',
                title="💎 Matrice Valeur/Volume par Segment",
                labels={
                    'nb_clients': 'Nombre de Clients',
                    'ca_moyen': 'CA Moyen par Client (€)'
                }
            )
            fig_matrice.update_traces(textposition="middle center")
            st.plotly_chart(fig_matrice, use_container_width=True)
        
        with col2:
            # Evolution temporelle par segment
            if 'mois' in df_filtered.columns:
                segment_evolution = df_filtered.groupby(['mois', 'segment_m7'])['chiffre_affaires'].sum().reset_index()
                
                fig_segment_evolution = px.line(
                    segment_evolution,
                    x='mois',
                    y='chiffre_affaires',
                    color='segment_m7',
                    title="📈 Évolution CA par Segment",
                    markers=True
                )
                fig_segment_evolution.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_segment_evolution, use_container_width=True)
        
        # Analyse comportementale détaillée par segment
        st.markdown("### 🔍 **Comportements par Segment**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Fréquence d'achat par segment
            if not client_metrics.empty and 'frequence' in client_metrics.columns:
                freq_by_segment = client_metrics.groupby('segment_m7')['frequence'].mean().reset_index()
                
                fig_freq = px.bar(
                    freq_by_segment,
                    x='segment_m7',
                    y='frequence',
                    title="🔄 Fréquence d'Achat Moyenne",
                    color='segment_m7'
                )
                st.plotly_chart(fig_freq, use_container_width=True)
        
        with col2:
            # Récence par segment
            if not client_metrics.empty and 'recence' in client_metrics.columns:
                recence_by_segment = client_metrics.groupby('segment_m7')['recence'].mean().reset_index()
                
                fig_recence = px.bar(
                    recence_by_segment,
                    x='segment_m7',
                    y='recence',
                    title="⏰ Récence Moyenne (jours)",
                    color='segment_m7'
                )
                st.plotly_chart(fig_recence, use_container_width=True)
        
        with col3:
            # Panier moyen par segment
            panier_by_segment = df_filtered.groupby('segment_m7')['chiffre_affaires'].mean().reset_index()
            
            fig_panier = px.bar(
                panier_by_segment,
                x='segment_m7',
                y='chiffre_affaires',
                title="🛒 Panier Moyen",
                color='segment_m7'
            )
            st.plotly_chart(fig_panier, use_container_width=True)
        
        # Recommandations stratégiques par segment
        st.markdown("### 🎯 **Stratégies Recommandées M7**")
        
        recommendations = {
            'Champions': {
                'emoji': '🏆',
                'color': '#FFD700',
                'strategies': [
                    "Programme VIP exclusif avec benefits digitaux",
                    "Campagnes de fidélisation automatisées", 
                    "Early access aux nouveaux produits",
                    "Personal shopper virtuel",
                    "Canaux prioritaires: Email premium + Push"
                ]
            },
            'Potentiels': {
                'emoji': '⭐',
                'color': '#32CD32',
                'strategies': [
                    "Campagnes d'upselling ciblées",
                    "Recommandations produits personnalisées",
                    "Gamification (points, badges, défis)",
                    "Retargeting display intelligent", 
                    "Canaux: Social media + Email automation"
                ]
            },
            'Standards': {
                'emoji': '📊',
                'color': '#4682B4',
                'strategies': [
                    "Campagnes promotionnelles agressives",
                    "Win-back automations pour inactifs",
                    "Contenus éducatifs sur la valeur produit",
                    "Référral programs",
                    "Canaux: Google Ads + Social media ads"
                ]
            }
        }
        
        for segment, info in recommendations.items():
            if segment in segments:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {info['color']}20, {info['color']}10); 
                           padding: 1.5rem; border-radius: 10px; border-left: 4px solid {info['color']}; margin: 1rem 0;">
                    <h4>{info['emoji']} Stratégie {segment}</h4>
                    <ul>
                        {''.join([f'<li>{strategy}</li>' for strategy in info['strategies']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

## ----------------------------
## TAB 3: PERFORMANCE DES CANAUX
## ----------------------------

with tab3:
    st.markdown("### 📈 **Performance des Canaux Marketing**")
    
    # Simuler des données de canaux si pas disponibles
    if 'canal_marketing' not in df_filtered.columns:
        # Simulation de données de canaux
        canaux = ['Google Ads', 'Facebook', 'Email', 'Instagram', 'LinkedIn']
        df_filtered['canal_marketing'] = np.random.choice(canaux, len(df_filtered))
        df_filtered['marketing_spend'] = np.random.uniform(50, 500, len(df_filtered))
        df_filtered['roas'] = df_filtered['chiffre_affaires'] / df_filtered['marketing_spend']
    
    # Analyse par canal
    canal_performance = df_filtered.groupby('canal_marketing').agg({
        'chiffre_affaires': 'sum',
        'marketing_spend': 'sum',
        'customer_id': 'nunique',
        'marge_brute': 'sum'
    }).round(2)
    
    canal_performance['roas'] = canal_performance['chiffre_affaires'] / canal_performance['marketing_spend']
    canal_performance['cpa'] = canal_performance['marketing_spend'] / canal_performance['customer_id']
    canal_performance['roi'] = (canal_performance['marge_brute'] - canal_performance['marketing_spend']) / canal_performance['marketing_spend']
    
    # KPI par canal
    st.markdown("#### 📊 **KPI par Canal**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # ROAS par canal
        fig_roas = px.bar(
            canal_performance.reset_index(),
            x='canal_marketing',
            y='roas',
            title="📈 ROAS par Canal",
            color='roas',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_roas, use_container_width=True)
    
    with col2:
        # CPA par canal
        fig_cpa = px.bar(
            canal_performance.reset_index(),
            x='canal_marketing',
            y='cpa',
            title="💰 CPA par Canal",
            color='cpa',
            color_continuous_scale='reds_r'
        )
        st.plotly_chart(fig_cpa, use_container_width=True)
    
    with col3:
        # ROI par canal
        fig_roi = px.bar(
            canal_performance.reset_index(),
            x='canal_marketing',
            y='roi',
            title="📊 ROI par Canal",
            color='roi',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col4:
        # Part de budget
        budget_share = canal_performance['marketing_spend'] / canal_performance['marketing_spend'].sum() * 100
        
        fig_budget = px.pie(
            values=budget_share.values,
            names=budget_share.index,
            title="💳 Répartition Budget"
        )
        st.plotly_chart(fig_budget, use_container_width=True)
    
    # Tableau détaillé des performances
    st.markdown("#### 📋 **Tableau de Performance Détaillé**")
    
    performance_display = canal_performance.copy()
    performance_display['chiffre_affaires'] = performance_display['chiffre_affaires'].apply(lambda x: f"{x:,.0f} €")
    performance_display['marketing_spend'] = performance_display['marketing_spend'].apply(lambda x: f"{x:,.0f} €")
    performance_display['roas'] = performance_display['roas'].apply(lambda x: f"{x:.2f}x")
    performance_display['cpa'] = performance_display['cpa'].apply(lambda x: f"{x:.0f} €")
    performance_display['roi'] = performance_display['roi'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(performance_display, use_container_width=True)
    
    # Attribution multi-canal
    st.markdown("#### 🔄 **Attribution Multi-Canal**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Matrice de corrélation des canaux
        if len(canal_performance) > 1:
            correlation_data = df_filtered.pivot_table(
                values='chiffre_affaires',
                index='date',
                columns='canal_marketing',
                aggfunc='sum',
                fill_value=0
            )
            
            correlation_matrix = correlation_data.corr()
            
            fig_correlation = px.imshow(
                correlation_matrix,
                title="🔗 Corrélation entre Canaux",
                aspect="auto",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_correlation, use_container_width=True)
    
    with col2:
        # Journey client simulé
        customer_journey = pd.DataFrame({
            'Étape': ['Découverte', 'Considération', 'Conversion', 'Rétention'],
            'Canal_Principal': ['Google Ads', 'Facebook', 'Email', 'Instagram'],
            'Contribution': [35, 25, 30, 10]
        })
        
        fig_journey = px.funnel(
            customer_journey,
            x='Contribution',
            y='Étape',
            title="🛤️ Journey Client Type",
            color='Canal_Principal'
        )
        st.plotly_chart(fig_journey, use_container_width=True)

## ----------------------------
## TAB 4: PRÉDICTIONS ET TENDANCES
## ----------------------------

with tab4:
    st.markdown("### 🔮 **Prédictions et Analyse Prédictive**")
    
    # Prédiction simple basée sur les tendances
    if 'mois' in df_filtered.columns:
        monthly_data = df_filtered.groupby('mois')['chiffre_affaires'].sum().reset_index()
        monthly_data['mois_num'] = range(len(monthly_data))
        
        # Régression linéaire simple
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        if len(monthly_data) > 2:
            X = monthly_data['mois_num'].values.reshape(-1, 1)
            y = monthly_data['chiffre_affaires'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Prédiction pour les 3 prochains mois
            future_months = np.arange(len(monthly_data), len(monthly_data) + 3).reshape(-1, 1)
            predictions = model.predict(future_months)
            
            # Graphique de prédiction
            col1, col2 = st.columns(2)
            
            with col1:
                fig_prediction = go.Figure()
                
                # Données historiques
                fig_prediction.add_trace(go.Scatter(
                    x=monthly_data['mois'],
                    y=monthly_data['chiffre_affaires'],
                    mode='lines+markers',
                    name='Historique',
                    line=dict(color='#667eea', width=3)
                ))
                
                # Prédictions
                future_dates = ['Mois+1', 'Mois+2', 'Mois+3']
                fig_prediction.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Prédiction',
                    line=dict(color='#f093fb', width=3, dash='dash')
                ))
                
                fig_prediction.update_layout(
                    title="🔮 Prédiction CA - 3 mois",
                    xaxis_title="Période",
                    yaxis_title="Chiffre d'Affaires (€)"
                )
                
                st.plotly_chart(fig_prediction, use_container_width=True)
            
            with col2:
                # Métriques de prédiction
                r2_score = model.score(X, y)
                trend_slope = model.coef_[0]
                
                st.markdown("#### 📊 **Métriques de Prédiction**")
                
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("🎯 R² Score", f"{r2_score:.3f}")
                with col2_2:
                    trend_direction = "📈" if trend_slope > 0 else "📉"
                    st.metric(f"{trend_direction} Tendance", f"{trend_slope:+,.0f}€/mois")
                
                st.markdown("#### 🔮 **Prédictions**")
                for i, pred in enumerate(predictions, 1):
                    st.metric(f"Mois +{i}", f"{pred:,.0f} €")
    
    # Analyse des risques et opportunités
    st.markdown("### ⚠️ **Analyse des Risques & Opportunités**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🚨 **Risques Identifiés**
        
        - **Concentration segment Champions**: Dépendance élevée
        - **Saisonnalité**: Fluctuations importantes
        - **Acquisition coût**: Tendance à la hausse
        - **Concurrence**: Pression sur les marges
        """)
        
        # Alerte automatique sur les KPI
        if not df_filtered.empty:
            # Simulation d'alertes
            alerts = []
            
            if 'roas' in df_filtered.columns:
                roas_moyen = df_filtered['roas'].mean()
                if roas_moyen < 2:
                    alerts.append("🚨 ROAS en baisse < 2x")
            
            if 'segment_m7' in df_filtered.columns:
                segment_counts = df_filtered['segment_m7'].value_counts()
                if 'Champions' in segment_counts and segment_counts['Champions'] / len(df_filtered) < 0.15:
                    alerts.append("⚠️ Segment Champions < 15%")
            
            if alerts:
                st.markdown("#### 🚨 **Alertes Automatiques**")
                for alert in alerts:
                    st.warning(alert)
    
    with col2:
        st.markdown("""
        #### 🚀 **Opportunités Détectées**
        
        - **Segment Potentiels**: Fort potentiel d'upselling
        - **Nouveaux canaux**: TikTok, Pinterest
        - **Personnalisation**: IA/ML pour recommandations
        - **International**: Expansion géographique
        """)
        
        # Calcul du potentiel d'optimisation
        if 'segment_m7' in df_filtered.columns:
            potentiel_data = df_filtered.groupby('segment_m7')['chiffre_affaires'].sum()
            
            # Simulation d'optimisation
            optimizations = {
                'Champions': {'current': potentiel_data.get('Champions', 0), 'potential': 1.15},
                'Potentiels': {'current': potentiel_data.get('Potentiels', 0), 'potential': 1.25},
                'Standards': {'current': potentiel_data.get('Standards', 0), 'potential': 1.10}
            }
            
            total_current = sum([opt['current'] for opt in optimizations.values()])
            total_potential = sum([opt['current'] * opt['potential'] for opt in optimizations.values()])
            
            st.markdown("#### 💎 **Potentiel d'Optimisation**")
            st.metric(
                "💰 CA Potentiel", 
                f"{total_potential:,.0f} €",
                f"+{total_potential - total_current:,.0f} € vs actuel"
            )

## ----------------------------
## TAB 5: PLAN D'ACTION
## ----------------------------

with tab5:
    st.markdown("### 📋 **Plan d'Action Stratégique M8**")
    
    # Tableau de bord des actions prioritaires
    actions_df = pd.DataFrame({
        'Action': [
            'Implémentation segmentation M7',
            'Automation email par segment', 
            'Optimisation Google Ads',
            'Campagnes Facebook Lookalike',
            'Programme fidélité Champions',
            'Win-back automation Standards',
            'A/B test créatives',
            'Dashboard temps réel'
        ],
        'Priorité': ['Très Haute', 'Haute', 'Haute', 'Moyenne', 'Haute', 'Moyenne', 'Moyenne', 'Basse'],
        'Timeline': ['S1-S2', 'S2-S3', 'S1-S4', 'S3-S4', 'S2-S4', 'S4-S6', 'S1-S8', 'S6-S8'],
        'Impact_CA': ['+20%', '+15%', '+12%', '+8%', '+10%', '+5%', '+3%', '+2%'],
        'Budget': ['2K€', '3K€', '5K€', '4K€', '2.5K€', '1.5K€', '1K€', '3K€'],
        'Responsable': ['Marketing', 'CRM', 'SEM', 'Social', 'CRM', 'Automation', 'Créatif', 'IT']
    })
    
    # Filtre par priorité
    priority_filter = st.selectbox(
        "Filtrer par priorité",
        ['Toutes', 'Très Haute', 'Haute', 'Moyenne', 'Basse']
    )
    
    if priority_filter != 'Toutes':
        actions_display = actions_df[actions_df['Priorité'] == priority_filter]
    else:
        actions_display = actions_df
    
    # Affichage du tableau avec couleurs
    def highlight_priority(row):
        colors = {
            'Très Haute': 'background-color: #ffebee',
            'Haute': 'background-color: #fff3e0', 
            'Moyenne': 'background-color: #f3e5f5',
            'Basse': 'background-color: #e8f5e8'
        }
        return [colors.get(row['Priorité'], '')] * len(row)
    
    st.dataframe(
        actions_display.style.apply(highlight_priority, axis=1),
        use_container_width=True
    )
    
    # Timeline visuelle
    st.markdown("#### 📅 **Timeline des Actions**")
    
    # Diagramme de Gantt simplifié
    timeline_data = []
    for _, action in actions_df.iterrows():
        timeline_data.append({
            'Action': action['Action'],
            'Start': f"2024-{action['Timeline'].split('-')[0][1:].zfill(2)}-01",
            'End': f"2024-{action['Timeline'].split('-')[1][1:].zfill(2)}-01",
            'Priorité': action['Priorité']
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df['Start'] = pd.to_datetime(timeline_df['Start'])
    timeline_df['End'] = pd.to_datetime(timeline_df['End'])
    
    fig_timeline = px.timeline(
        timeline_df,
        x_start='Start',
        x_end='End',
        y='Action',
        color='Priorité',
        title="📅 Timeline des Actions Prioritaires"
    )
    fig_timeline.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Budget allocation
    st.markdown("#### 💰 **Allocation Budget**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Budget par action
        budget_values = [float(b.replace('K€', '').replace(',', '.')) for b in actions_df['Budget']]
        actions_df['Budget_Numeric'] = budget_values
        
        fig_budget_actions = px.pie(
            actions_df,
            values='Budget_Numeric',
            names='Action',
            title="💳 Répartition Budget par Action"
        )
        st.plotly_chart(fig_budget_actions, use_container_width=True)
    
    with col2:
        # Budget par département
        budget_by_dept = actions_df.groupby('Responsable')['Budget_Numeric'].sum().reset_index()
        
        fig_budget_dept = px.bar(
            budget_by_dept,
            x='Responsable',
            y='Budget_Numeric',
            title="🏢 Budget par Département",
            color='Budget_Numeric',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_budget_dept, use_container_width=True)
    
    # ROI Projections
    st.markdown("#### 📈 **Projections ROI**")
    
    # Calcul des projections
    total_budget = sum(budget_values)
    current_ca = df_filtered['chiffre_affaires'].sum() if not df_filtered.empty else 100000
    
    # Impacts cumulés (simulation)
    impact_percentages = [20, 15, 12, 8, 10, 5, 3, 2]
    projected_increases = [current_ca * (p/100) for p in impact_percentages]
    total_projected_increase = sum(projected_increases)
    
    roi_projection = (total_projected_increase - total_budget*1000) / (total_budget*1000)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Budget Total", f"{total_budget:.1f}K€")
    with col2:
        st.metric("📈 CA Additionnel", f"{total_projected_increase:,.0f}€")
    with col3:
        st.metric("🎯 ROI Projeté", f"{roi_projection:.1%}")
    with col4:
        payback_months = (total_budget*1000) / (total_projected_increase/12)
        st.metric("⏰ Payback", f"{payback_months:.1f} mois")
    
    # Export du plan d'action
    st.markdown("#### 📥 **Export du Plan**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Exporter Plan Action"):
            csv_plan = actions_df.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv_plan,
                file_name=f"plan_action_m8_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Exporter Données Dashboard"):
            if not df_filtered.empty:
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                    label="Télécharger Données",
                    data=csv_data,
                    file_name=f"dashboard_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

## ----------------------------
## FOOTER ET INFORMATIONS
## ----------------------------

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p> <strong>Dashboard M8 </strong> | Basé sur la stratégie M7</p>
    <p>Dernière mise à jour: """ + datetime.now().strftime("%d/%m/%Y %H:%M") + """</p>
    <p>Données analysées: """ + (f"{len(df_filtered):,} transactions" if not df_filtered.empty else "Données de démonstration") + """</p>
</div>
""", unsafe_allow_html=True)