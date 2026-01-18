"""
⚽ MATCH PROBABILITY PREDICTOR
==============================
Applicazione Streamlit per calcolare le probabilità di un match di calcio.

Funzionalità:
- Selezione lega e squadre
- Calcolo probabilità 1X2, Over/Under, BTTS
- Visualizzazione matrice punteggi esatti
- Indicatore qualità previsione
- Refresh dati

Autore: Sistema sviluppato con Peppe
Versione: 2.0 (Gennaio 2025)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

# Import moduli locali
from probability_engine import (
    calculate_match_probabilities,
    assess_prediction_quality
)
from data_fetcher import (
    LEAGUES,
    fetch_teams_for_league,
    get_match_stats,
    get_head_to_head,
    get_api_predictions,
    get_injuries,
    get_team_shots_avg,
    get_current_season
)
from team_logos import get_logo_path
from config import API_FOOTBALL_KEY, DEFAULT_LEAGUE
from auth import (
    init_session,
    is_authenticated,
    is_admin,
    show_login_page,
    show_admin_panel,
    show_user_info_sidebar
)
from database import init_database


def get_logo_base64_simple(logo_path):
    """Converte un file immagine in base64 string."""
    if not logo_path:
        return None
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================
st.set_page_config(
    page_title="BetEngine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inizializza database (dopo che Streamlit è pronto)
try:
    init_database()
except Exception as e:
    st.error(f"Errore connessione database: {e}")

# CSS personalizzato
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
    /* Import font Audiowide */
    @import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
    
    /* Fix viewport per mobile */
    html, body, [data-testid="stAppViewContainer"] {
        width: 100% !important;
        overflow-x: hidden !important;
    }
    
    /* Sfondo generale blu scuro */
    .stApp {
        background: linear-gradient(180deg, #1a3a52 0%, #0d2137 100%);
    }
    
    /* Sidebar ancora più scura */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #051221 100%);
    }
    
    /* Sidebar - gestione chiusura su mobile/tablet */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        transform: translateX(-100%) !important;
    }
    
    /* Fix sidebar collapsed - nascondere completamente */
    [data-testid="stSidebar"][aria-expanded="false"] > div {
        display: none !important;
    }
    
    /* Sidebar aperta - larghezza fissa */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 280px;
        max-width: 320px;
    }
    
    /* Rimuovi qualsiasi sfondo bianco dai markdown nella sidebar */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown div,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] .element-container {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Tutto il testo nella sidebar bianco */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown strong,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Checkbox nella sidebar - testo bianco */
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stCheckbox span,
    section[data-testid="stSidebar"] .stCheckbox p {
        color: #ffffff !important;
    }
    
    /* Bottoni nella sidebar - sfondo bianco, testo NERO */
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] .stButton button,
    section[data-testid="stSidebar"] .stButton > button {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] .stButton button p,
    section[data-testid="stSidebar"] .stButton button span {
        color: #000000 !important;
    }
    
    /* Testo principale chiaro */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #e8f4fc;
    }
    
    /* Label dei form - bianche */
    .stTextInput label,
    .stTextInput label p,
    .stTextInput label span,
    .stTextInput > label,
    .stNumberInput label,
    .stNumberInput label p,
    .stNumberInput label span,
    .stNumberInput > label,
    .stTextArea label,
    .stTextArea label p,
    .stTextArea label span,
    .stSelectbox label,
    .stSelectbox label p,
    .stSelectbox label span,
    .stCheckbox label,
    .stCheckbox label p,
    .stCheckbox label span,
    .stCheckbox > label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] small,
    .stMarkdown small,
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] label p,
    div[data-testid="stForm"] label span,
    div[data-testid="stForm"] label small,
    div[data-testid="stForm"] small {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* FORZA tutte le label bianche CON SFONDO TRASPARENTE */
    label, label p, label span, label small {
        color: #ffffff !important;
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Widget label container - sfondo trasparente */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] > div,
    .stSelectbox label,
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Markdown headers - sfondo trasparente */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        background: transparent !important;
        color: #ffffff !important;
    }
    
    /* Container elementi - sfondo trasparente nella main area */
    .main .element-container,
    .main [data-testid="stVerticalBlock"],
    .main [data-testid="column"] {
        background: transparent !important;
    }
    
    /* === INPUT, SELECT, CASELLE === */
    /* Selectbox e dropdown */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }
    
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #1a1a2e !important;
    }
    
    /* Input text */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }
    
    /* Dropdown menu aperto */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] div {
        color: #1a1a2e !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] li {
        color: #1a1a2e !important;
    }
    
    /* === TESTO NERO NELLE BOX BIANCHE === */
    /* Alert boxes (success, warning, error, info) */
    .stAlert, .stAlert p, .stAlert span, .stAlert div,
    .stAlert strong, .stAlert small {
        color: #1a1a2e !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader, 
    .streamlit-expanderContent,
    .streamlit-expanderContent p,
    .streamlit-expanderContent span,
    .streamlit-expanderContent div {
        color: #1a1a2e !important;
    }
    
    [data-testid="stExpander"] {
        background: #ffffff;
        border-radius: 10px;
    }
    
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] div,
    [data-testid="stExpander"] strong,
    [data-testid="stExpander"] small {
        color: #1a1a2e !important;
    }
    
    /* Form e contenitori bianchi */
    .stForm input, .stForm textarea {
        color: #1a1a2e !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #ffffff !important;
    }
    
    /* DataFrame/Tabelle - tutto nero */
    .stDataFrame, .stDataFrame div, .stDataFrame span,
    .stDataFrame td, .stDataFrame th,
    [data-testid="stDataFrame"] div,
    [data-testid="stDataFrame"] span {
        color: #1a1a2e !important;
    }
    
    /* Caption e help text - più chiaro per leggibilità */
    .stCaption, small, .stCaption p {
        color: #a0b4c4 !important;
    }
    
    /* Fix contrasto per qualsiasi testo su sfondo blu scuro */
    .stApp p, .stApp span, .stApp div {
        color: #e8f4fc;
    }
    
    /* Testo bianco garantito su sfondi scuri */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em {
        color: #ffffff !important;
    }
    
    /* Eccezione: testo scuro negli elementi con sfondo chiaro */
    .stAlert p, .stAlert span, .stAlert div,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] div,
    table td, table th {
        color: #1a1a2e !important;
    }
    
    /* Header tabella - sempre bianco */
    table th {
        color: #ffffff !important;
    }
    
    /* Stile generale */
    .main-header {
        font-family: 'Audiowide', sans-serif;
        font-size: 3.5rem;
        text-align: center;
        color: #4fc3f7 !important;
        text-shadow: 0 0 10px rgba(79, 195, 247, 0.6),
                     0 0 20px rgba(79, 195, 247, 0.4),
                     0 0 30px rgba(79, 195, 247, 0.3);
        letter-spacing: 5px;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #b0c4d8 !important;
        margin-bottom: 2rem;
    }
    
    /* Card probabilità */
    .prob-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    .prob-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .prob-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Statistiche squadra */
    .team-stat {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Metriche Streamlit */
    [data-testid="stMetricValue"] {
        color: #4fc3f7 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a8d4f0 !important;
    }
    
    /* DataFrame/Tabelle - sfondo chiaro per leggibilità */
    .stDataFrame {
        background: #ffffff;
        border-radius: 10px;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: #ffffff !important;
    }
    
    /* Celle tabella */
    .stDataFrame td, .stDataFrame th {
        color: #1a1a2e !important;
        background: #ffffff !important;
    }
    
    /* Grafici Plotly - sfondo trasparente */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7fb3d3 !important;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #2c5282;
    }
    
    .footer p {
        color: #7fb3d3 !important;
    }
    
    /* Nascondi elementi Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Forza visibilità pulsante apertura sidebar */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[kind="header"] {
        display: block !important;
        visibility: visible !important;
    }
    
    /* Nascondi pulsanti fullscreen/expand sulle immagini */
    button[title="View fullscreen"] {display: none;}
    [data-testid="StyledFullScreenButton"] {display: none;}
    
    /* Card statistiche squadre - testo nero per label e valori */
    .team-stats-card h4,
    .team-stats-card small,
    .team-stats-card div {
        color: #000000 !important;
    }
    
    /* ============================================================
       RESPONSIVE - MOBILE STYLES
       ============================================================ */
    
    /* Tablet e Mobile (< 992px) */
    @media screen and (max-width: 992px) {
        /* Riduci padding generale */
        .main .block-container {
            padding: 1rem 0.5rem !important;
            max-width: 100% !important;
        }
        
        /* Font più piccoli */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        
        /* Colonne Streamlit - forza stack verticale */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        
        /* Container colonne - direzione verticale */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }
        
        /* Sidebar su tablet - comportamento migliorato */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        
        section[data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 260px !important;
            max-width: 300px !important;
        }
    }
    
    /* Mobile (< 768px) */
    @media screen and (max-width: 768px) {
        /* Header partita - più compatto */
        .match-header-container {
            flex-direction: column !important;
            padding: 10px !important;
        }
        
        /* Sidebar su mobile - overlay invece di push */
        section[data-testid="stSidebar"] {
            position: fixed !important;
            z-index: 999999 !important;
            height: 100vh !important;
            top: 0 !important;
            left: 0 !important;
        }
        
        /* Sidebar chiusa su mobile - nascosta completamente */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            transform: translateX(-100%) !important;
            width: 0 !important;
            min-width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            visibility: hidden !important;
        }
        
        /* Sidebar aperta su mobile - full width */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 85vw !important;
            max-width: 320px !important;
            transform: translateX(0) !important;
            visibility: visible !important;
        }
        
        /* Main content quando sidebar è aperta */
        .main .block-container {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Font ancora più piccoli */
        h1 { font-size: 1.3rem !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1rem !important; }
        p, span, div { font-size: 0.9rem !important; }
        
        /* Metriche Streamlit più compatte */
        [data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.7rem !important;
        }
        
        /* Padding ridotto nelle card */
        [data-testid="stMetric"] {
            padding: 0.5rem !important;
        }
        
        /* Tabelle scrollabili orizzontalmente */
        table {
            display: block !important;
            overflow-x: auto !important;
            white-space: nowrap !important;
            -webkit-overflow-scrolling: touch !important;
            font-size: 0.8rem !important;
        }
        
        th, td {
            padding: 6px 8px !important;
            font-size: 0.75rem !important;
        }
        
        /* Grafici Plotly - dimensioni adattive */
        .js-plotly-plot, .plotly {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Card 1X2 più compatte */
        .stMarkdown div[style*="border-radius:15px"] {
            padding: 12px !important;
        }
        
        .stMarkdown div[style*="font-size:2.5rem"] {
            font-size: 1.8rem !important;
        }
        
        /* Bottoni full width su mobile */
        .stButton button {
            width: 100% !important;
            padding: 0.6rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Input fields più compatti */
        .stTextInput input,
        .stSelectbox select,
        .stNumberInput input {
            font-size: 16px !important; /* Previene zoom su iOS */
            padding: 0.5rem !important;
        }
        
        /* Expander più compatto */
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
            padding: 0.5rem !important;
        }
        
        /* Immagini loghi ridimensionate */
        img {
            max-width: 45px !important;
            height: auto !important;
        }
    }
    
    /* Mobile piccolo (< 480px) */
    @media screen and (max-width: 480px) {
        /* Padding minimo */
        .main .block-container {
            padding: 0.5rem 0.3rem !important;
        }
        
        /* Font minimi leggibili */
        h1 { font-size: 1.2rem !important; }
        h2 { font-size: 1rem !important; }
        h3 { font-size: 0.9rem !important; }
        p, span { font-size: 0.85rem !important; }
        
        /* Metriche ancora più compatte */
        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        
        /* Nascondi testo meno importante */
        .stCaption {
            font-size: 0.7rem !important;
        }
        
        /* Tabelle font minimo */
        th, td {
            padding: 4px 6px !important;
            font-size: 0.7rem !important;
        }
        
        /* VS badge più piccolo */
        .vs-badge {
            padding: 5px 10px !important;
            font-size: 0.9rem !important;
        }
        
        /* Loghi più piccoli */
        img {
            max-width: 35px !important;
        }
    }
    
    /* Fix per iOS Safari - previeni zoom su input */
    @supports (-webkit-touch-callout: none) {
        input, select, textarea {
            font-size: 16px !important;
        }
    }
    
    /* Scrollbar più sottile su mobile */
    @media screen and (max-width: 768px) {
        ::-webkit-scrollbar {
            width: 4px;
            height: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.3);
            border-radius: 2px;
        }
    }
    
    /* ============================================================
       FIX CONTRASTO GLOBALI
       ============================================================ */
    
    /* Testo principale su sfondo scuro - sempre chiaro */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff !important;
    }
    
    /* Markdown container - testo bianco e sfondo trasparente */
    .main [data-testid="stMarkdownContainer"] {
        color: #e8f4fc !important;
        background: transparent !important;
    }
    
    /* IMPORTANTE: Tutti i container nella main area - sfondo trasparente */
    .main .stMarkdown,
    .main .element-container,
    .main [data-testid="stVerticalBlock"] > div,
    .main [data-testid="column"] > div,
    .main [data-testid="stHorizontalBlock"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Fix specifico: card con sfondo chiaro -> testo scuro */
    div[style*="background:#f"], 
    div[style*="background: #f"],
    div[style*="background:white"],
    div[style*="background: white"],
    div[style*="background:#fff"],
    div[style*="background: #fff"] {
        color: #1a1a2e !important;
    }
    
    /* Fix specifico: card con sfondo colorato -> testo bianco */
    div[style*="background:linear-gradient"],
    div[style*="background: linear-gradient"],
    div[style*="background:#3498db"],
    div[style*="background:#27ae60"],
    div[style*="background:#e74c3c"],
    div[style*="background:#9b59b6"],
    div[style*="background:#34495e"],
    div[style*="background:#1a1a2e"],
    div[style*="background:#f39c12"] {
        color: #ffffff !important;
    }
    
    /* Metriche delta - verde/rosso leggibile */
    [data-testid="stMetricDelta"] svg {
        stroke: currentColor !important;
    }
    [data-testid="stMetricDelta"][data-testid*="Up"] {
        color: #00c853 !important;
    }
    [data-testid="stMetricDelta"][data-testid*="Down"] {
        color: #ff5252 !important;
    }
    
    /* Info/Warning/Success/Error boxes - testo scuro */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #1a1a2e !important;
    }
    
    /* Tooltip e popover - testo leggibile */
    [data-baseweb="tooltip"] {
        background: #ffffff !important;
        color: #1a1a2e !important;
    }
    
    /* Caption sotto i loghi - testo chiaro */
    .stImage figcaption,
    [data-testid="stImage"] figcaption {
        color: #a8d4f0 !important;
    }
    
    /* Write/text normale */
    .stApp .stMarkdown div:not([style]) {
        color: #e8f4fc !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNZIONI DI VISUALIZZAZIONE
# ============================================================

def create_probability_bar(probs: dict, labels: list, colors: list, title: str) -> go.Figure:
    """Crea un grafico a barre orizzontali per le probabilità."""
    values = [probs.get(k, 0) * 100 for k in labels]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=[l.upper() for l in labels],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='inside',
        textfont=dict(size=14, color='white'),
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#e8f4fc')),
        xaxis=dict(range=[0, 100], title="Probabilità %"),
        yaxis=dict(title=""),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f4fc'),
        xaxis_gridcolor='rgba(255,255,255,0.1)'
    )
    
    return fig


def create_ou_comparison(probs: dict) -> go.Figure:
    """Crea grafico comparativo Over/Under per diverse linee."""
    lines = [1.5, 2.5, 3.5, 4.5]
    over_vals = [probs.get(f"over_{l}", 0) * 100 for l in lines]
    under_vals = [probs.get(f"under_{l}", 0) * 100 for l in lines]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='OVER',
        x=[f'{l}' for l in lines],
        y=over_vals,
        marker_color='#27ae60',
        text=[f'{v:.1f}%' for v in over_vals],
        textposition='outside',
    ))
    
    fig.add_trace(go.Bar(
        name='UNDER',
        x=[f'{l}' for l in lines],
        y=under_vals,
        marker_color='#e74c3c',
        text=[f'{v:.1f}%' for v in under_vals],
        textposition='outside',
    ))
    
    fig.update_layout(
        title="Over/Under per Linea",
        xaxis_title="Linea Gol",
        yaxis_title="Probabilità %",
        barmode='group',
        height=350,
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f4fc'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis_gridcolor='rgba(255,255,255,0.1)'
    )
    
    return fig


def create_cards_comparison(probs: dict) -> go.Figure:
    """Crea grafico comparativo Over/Under per i cartellini."""
    lines = [2.5, 3.5, 4.5, 5.5, 6.5]
    over_vals = [probs.get(f"cards_over_{l}", 0) * 100 for l in lines]
    under_vals = [probs.get(f"cards_under_{l}", 0) * 100 for l in lines]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='OVER',
        x=[f'{l}' for l in lines],
        y=over_vals,
        marker_color='#f39c12',  # Giallo/arancio per cartellini
        text=[f'{v:.1f}%' for v in over_vals],
        textposition='outside',
    ))
    
    fig.add_trace(go.Bar(
        name='UNDER',
        x=[f'{l}' for l in lines],
        y=under_vals,
        marker_color='#9b59b6',  # Viola per under
        text=[f'{v:.1f}%' for v in under_vals],
        textposition='outside',
    ))
    
    fig.update_layout(
        title="Cartellini Over/Under per Linea",
        xaxis_title="Linea Cartellini",
        yaxis_title="Probabilità %",
        barmode='group',
        height=350,
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f4fc'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis_gridcolor='rgba(255,255,255,0.1)'
    )
    
    return fig


def create_score_matrix_heatmap(matrix: np.ndarray, home_name: str, away_name: str) -> go.Figure:
    """Crea heatmap della matrice punteggi esatti."""
    # Limita a 6x6 per visualizzazione
    M = matrix[:7, :7] * 100  # Converti in percentuale
    
    fig = go.Figure(data=go.Heatmap(
        z=M,
        x=[str(i) for i in range(7)],
        y=[str(i) for i in range(7)],
        colorscale='Blues',
        text=[[f'{M[i,j]:.1f}%' if M[i,j] > 0.5 else '' for j in range(7)] for i in range(7)],
        texttemplate='%{text}',
        textfont={"size": 10, "color": "white"},
        hovertemplate=f'{home_name} %{{y}} - %{{x}} {away_name}<br>Prob: %{{z:.2f}}%<extra></extra>',
    ))
    
    fig.update_layout(
        xaxis_title=f"Gol {away_name}",
        yaxis_title=f"Gol {home_name}",
        height=400,
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f4fc'),
    )
    
    return fig


def display_exact_scores(scores: list) -> None:
    """Mostra i punteggi esatti più probabili."""
    cols = st.columns(5)
    for i, (home_g, away_g, prob) in enumerate(scores[:10]):
        col_idx = i % 5
        with cols[col_idx]:
            color = "#3498db" if home_g > away_g else "#e74c3c" if away_g > home_g else "#95a5a6"
            st.markdown(f"""
            <div style="background:{color}; color:white; padding:10px; border-radius:10px; 
                        text-align:center; margin:5px 0;">
                <div style="font-size:1.5rem; font-weight:bold;">{home_g} - {away_g}</div>
                <div style="font-size:0.9rem;">{prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)


def display_team_stats(stats: dict, team_name: str, is_home: bool) -> None:
    """Mostra le statistiche di una squadra."""
    pos = "Casa" if is_home else "Trasferta"
    att_key = "attack_home" if is_home else "attack_away"
    def_key = "defense_home" if is_home else "defense_away"
    
    att = stats.get(att_key, 1.0)
    def_ = stats.get(def_key, 1.0)
    form = stats.get("form_factor", 1.0)
    rank = stats.get("rank", "N/A")
    form_str = stats.get("form_string", "")
    
    # Colori basati sui valori
    att_color = "#27ae60" if att > 1.1 else "#e74c3c" if att < 0.9 else "#f39c12"
    def_color = "#27ae60" if def_ < 0.9 else "#e74c3c" if def_ > 1.1 else "#f39c12"
    
    st.markdown(f"""
    <div class="team-stats-card" style="background:#f8f9fa; padding:15px; border-radius:10px; border-left:4px solid {'#3498db' if is_home else '#e74c3c'};">
        <h4 style="margin:0 0 10px 0; color:#000000 !important;">{team_name}</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div>
                <small style="color:#000000 !important; font-size:0.75rem;">Att. ({pos})</small>
                <div style="font-size:1.2rem; color:{att_color} !important; font-weight:bold;">{att:.2f}</div>
            </div>
            <div>
                <small style="color:#000000 !important; font-size:0.75rem;">Dif. ({pos})</small>
                <div style="font-size:1.2rem; color:{def_color} !important; font-weight:bold;">{def_:.2f}</div>
            </div>
            <div>
                <small style="color:#000000 !important; font-size:0.75rem;">Forma</small>
                <div style="font-size:1.1rem; color:#000000 !important;">{form:.2f} <small style="color:#000000 !important;">({form_str[-5:] if form_str else 'N/A'})</small></div>
            </div>
            <div>
                <small style="color:#000000 !important; font-size:0.75rem;">Posizione</small>
                <div style="font-size:1.2rem; color:#000000 !important;">{rank}°</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def calculate_top_predictions(probabilities: dict, home_team: str, away_team: str) -> list:
    """
    Calcola i top 3 pronostici tra:
    - 1X2
    - BTTS (Gol/NoGol)
    - Over/Under 2.5 gol
    - Cartellini Over/Under 2.5
    Ordinati per probabilità più alta.
    """
    predictions = []
    
    p_home = probabilities['p_home'] * 100
    p_draw = probabilities['p_draw'] * 100
    p_away = probabilities['p_away'] * 100
    
    # === MERCATI 1X2 ===
    predictions.append({
        'name': f'Vittoria {home_team}',
        'short': '1',
        'prob': p_home,
        'icon': '🏠'
    })
    predictions.append({
        'name': 'Pareggio',
        'short': 'X',
        'prob': p_draw,
        'icon': '🤝'
    })
    predictions.append({
        'name': f'Vittoria {away_team}',
        'short': '2',
        'prob': p_away,
        'icon': '✈️'
    })
    
    # === OVER/UNDER 2.5 GOL ===
    predictions.append({
        'name': 'Over 2.5',
        'short': 'O2.5',
        'prob': probabilities.get('over_2.5', 0) * 100,
        'icon': '⬆️'
    })
    predictions.append({
        'name': 'Under 2.5',
        'short': 'U2.5',
        'prob': probabilities.get('under_2.5', 0) * 100,
        'icon': '⬇️'
    })
    
    # === BTTS (GOL/NOGOL) ===
    predictions.append({
        'name': 'Gol (BTTS Sì)',
        'short': 'GG',
        'prob': probabilities.get('btts_yes', 0) * 100,
        'icon': '✅'
    })
    predictions.append({
        'name': 'NoGol (BTTS No)',
        'short': 'NG',
        'prob': probabilities.get('btts_no', 0) * 100,
        'icon': '❌'
    })
    
    # === CARTELLINI OVER/UNDER 3.5 ===
    cards_over_35 = probabilities.get('cards_over_3.5', 0) * 100
    cards_under_35 = probabilities.get('cards_under_3.5', 0) * 100
    
    if cards_over_35 > 0:  # Solo se abbiamo dati cartellini
        predictions.append({
            'name': 'Cart. Over 3.5',
            'short': 'CO3.5',
            'prob': cards_over_35,
            'icon': '🟨⬆️'
        })
        predictions.append({
            'name': 'Cart. Under 3.5',
            'short': 'CU3.5',
            'prob': cards_under_35,
            'icon': '🟨⬇️'
        })
    
    # === ORDINA PER PROBABILITÀ E CALCOLA STELLE ===
    for pred in predictions:
        prob = pred['prob']
        # Stelle basate sulla probabilità (semplice)
        if prob >= 70:
            pred['stars'] = 5
        elif prob >= 60:
            pred['stars'] = 4
        elif prob >= 55:
            pred['stars'] = 3
        elif prob >= 50:
            pred['stars'] = 2
        else:
            pred['stars'] = 1
    
    # Filtra solo probabilità > 50% e ordina per probabilità
    valid_predictions = [p for p in predictions if p['prob'] > 50]
    valid_predictions.sort(key=lambda x: x['prob'], reverse=True)
    
    return valid_predictions[:3]


def display_top_predictions(predictions: list) -> None:
    """Visualizza i top 3 pronostici consigliati."""
    
    st.subheader("🏆 Pronostici Consigliati")
    
    if not predictions:
        st.info("Nessun pronostico con confidence sufficiente")
        return
    
    cols = st.columns(3)
    medals = ['🥇', '🥈', '🥉']
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Oro, Argento, Bronzo
    
    for i, pred in enumerate(predictions[:3]):
        with cols[i]:
            stars = '⭐' * pred['stars'] + '☆' * (5 - pred['stars'])
            confidence_text = "Molto Alta" if pred['stars'] >= 5 else "Alta" if pred['stars'] >= 4 else "Media-Alta" if pred['stars'] >= 3 else "Media"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {colors[i]}22, {colors[i]}44); 
                        border: 2px solid {colors[i]}; 
                        border-radius: 15px; 
                        padding: 20px; 
                        text-align: center;
                        min-height: 180px;">
                <div style="font-size: 2rem;">{medals[i]}</div>
                <div style="font-size: 1.1rem; font-weight: bold; color: #e8f4fc; margin: 10px 0;">
                    {pred['icon']} {pred['name']}
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: #4fc3f7;">
                    {pred['prob']:.1f}%
                </div>
                <div style="font-size: 0.9rem; color: #ffd700; margin: 5px 0;">
                    {stars}
                </div>
                <div style="font-size: 0.75rem; color: #a8d4f0;">
                    {confidence_text}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# AUTENTICAZIONE
# ============================================================

# Inizializza sessione
init_session()

# Se non autenticato, mostra pagina login
if not is_authenticated():
    show_login_page()
    st.stop()

# Info utente e logout nella sidebar
show_user_info_sidebar()

# Pannello Admin (solo per admin) - in una pagina separata
if is_admin():
    admin_tab = st.sidebar.checkbox("👑 Mostra Pannello Admin", value=False)
    if admin_tab:
        show_admin_panel()
        st.markdown("---")
        st.stop()  # Non mostrare l'app principale quando si è nel pannello admin

# ============================================================
# SIDEBAR - CONFIGURAZIONE
# ============================================================

with st.sidebar:
    st.markdown("---")
    
    # Usa API Key dal config (nascosta)
    api_key = API_FOOTBALL_KEY
    
    # Selezione Lega
    st.markdown("### 🏆 Seleziona Lega")
    league_options = {v["name"]: k for k, v in LEAGUES.items()}
    # Trova l'indice della lega default
    league_names = list(league_options.keys())
    default_league_name = next((name for name, lid in league_options.items() if lid == DEFAULT_LEAGUE), league_names[0])
    default_index = league_names.index(default_league_name) if default_league_name in league_names else 0
    
    selected_league_name = st.selectbox(
        "Campionato",
        options=league_names,
        index=default_index
    )
    selected_league_id = league_options[selected_league_name]
    
    # Stagione fissa 2025/2026
    selected_season = 2025
    
    st.markdown("---")
    
    # Pulsante Refresh
    if st.button("🔄 Refresh Dati", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# MAIN CONTENT
# ============================================================

# Header con logo centrato
col_logo1, col_logo2, col_logo3 = st.columns([1, 1, 1])
with col_logo2:
    try:
        import base64
        with open("logo.png", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_data}" style="width: 220px;">
        </div>
        """, unsafe_allow_html=True)
    except:
        st.image("logo.png", width=220)

st.markdown('<p class="sub-header">Trasforma i dati in probabilità vincenti</p>', unsafe_allow_html=True)

# Spazio dopo il logo
st.markdown("<br>", unsafe_allow_html=True)

if not api_key or api_key == "INSERISCI_QUI_LA_TUA_API_KEY":
    st.warning("⚠️ API Key non configurata!")
    st.markdown("""
    ### Come configurare:
    1. Apri il file `config.py`
    2. Sostituisci `INSERISCI_QUI_LA_TUA_API_KEY` con la tua chiave
    3. Salva e riavvia l'app
    
    ### Come ottenere una API Key:
    1. Vai su [api-football.com](https://www.api-football.com/)
    2. Registrati per un account gratuito
    3. Vai su "My Account" > "API Keys"
    4. Copia la tua chiave
    
    Il piano gratuito include 100 chiamate/giorno, sufficienti per testare!
    """)
    st.stop()

# Carica squadre
with st.spinner(f"Caricamento squadre {selected_league_name}..."):
    teams = fetch_teams_for_league(api_key, selected_league_id, selected_season)

if not teams:
    st.error("Impossibile caricare le squadre. Verifica la tua API key.")
    st.stop()

team_names = {t["name"]: t["id"] for t in teams}

# Selezione squadre
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown("### 🏠 Squadra Casa")
    home_team_name = st.selectbox(
        "Seleziona squadra casa",
        options=list(team_names.keys()),
        key="home_team"
    )

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>VS</h2>", unsafe_allow_html=True)

with col3:
    st.markdown("### ✈️ Squadra Trasferta")
    # Filtra per non permettere stessa squadra
    away_options = [t for t in team_names.keys() if t != home_team_name]
    away_team_name = st.selectbox(
        "Seleziona squadra trasferta",
        options=away_options,
        key="away_team"
    )

# Input opzionale arbitro
st.markdown("#### 👨‍⚖️ Arbitro (opzionale)")
col_ref1, col_ref2 = st.columns([2, 1])
with col_ref1:
    referee_name = st.text_input(
        "Nome arbitro",
        placeholder="Es: Orsato, Guida, Mariani...",
        help="Inserisci anche solo il cognome! Es: 'Feliciani' trova 'Luca Feliciani'",
        key="referee_input"
    )
with col_ref2:
    st.markdown("<br>", unsafe_allow_html=True)
    if referee_name:
        try:
            from fetch_referee_stats import get_referee_adjustment
            ref_info = get_referee_adjustment(referee_name, selected_league_name)
            if ref_info.get("found"):
                severity = ref_info.get("severity_factor", 1.0)
                ref_full_name = ref_info.get("name", referee_name)
                if severity > 1.1:
                    st.warning(f"🔴 {ref_full_name} ({severity:.2f}x)")
                elif severity < 0.9:
                    st.success(f"🟢 {ref_full_name} ({severity:.2f}x)")
                else:
                    st.info(f"🟡 {ref_full_name} ({severity:.2f}x)")
            else:
                st.caption("❌ Non trovato nel database")
        except Exception as e:
            st.caption(f"⚠️ Errore: {e}")

# Pulsante calcola
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    calculate_btn = st.button("🔮 CALCOLA PROBABILITÀ", type="primary", use_container_width=True)

st.markdown("---")

# ============================================================
# CALCOLO E VISUALIZZAZIONE RISULTATI
# ============================================================

if calculate_btn:
    home_team_id = team_names[home_team_name]
    away_team_id = team_names[away_team_name]
    
    # Recupera statistiche
    with st.spinner("📊 Recupero statistiche squadre..."):
        home_stats, away_stats, league_info = get_match_stats(
            api_key, home_team_id, away_team_id, selected_league_id, selected_season
        )
    
    # Recupera dati aggiuntivi
    with st.spinner("🔍 Recupero dati avanzati (H2H, Tiri, Infortuni)..."):
        # Head to Head
        h2h_data = get_head_to_head(api_key, home_team_id, away_team_id, last_n=10)
        
        # Media tiri ultime 5 partite
        home_shots = get_team_shots_avg(api_key, home_team_id, selected_league_id, selected_season, last_n=5)
        away_shots = get_team_shots_avg(api_key, away_team_id, selected_league_id, selected_season, last_n=5)
    
    # Calcola probabilità
    with st.spinner("🧮 Calcolo probabilità..."):
        probabilities = calculate_match_probabilities(
            home_stats, away_stats, selected_league_id,
            h2h_data=h2h_data,
            home_shots=home_shots,
            away_shots=away_shots,
            referee_name=referee_name if referee_name else None,
            league_name=selected_league_name
        )
    
    # Valuta qualità previsione
    quality = assess_prediction_quality(home_stats, away_stats)
    
    # === SEZIONE RISULTATI ===
    # Header con loghi squadre - layout fisso su una riga
    try:
        home_logo_path = get_logo_path(home_team_name)
        away_logo_path = get_logo_path(away_team_name)
        
        st.markdown("---")
        
        # Prepara HTML loghi con fallback
        home_b64 = get_logo_base64_simple(home_logo_path) if home_logo_path else None
        away_b64 = get_logo_base64_simple(away_logo_path) if away_logo_path else None
        
        home_logo_html = f'<img src="data:image/png;base64,{home_b64}" style="width:40px; height:40px; object-fit:contain;">' if home_b64 else '<span style="font-size:1.5rem;">🏠</span>'
        away_logo_html = f'<img src="data:image/png;base64,{away_b64}" style="width:40px; height:40px; object-fit:contain;">' if away_b64 else '<span style="font-size:1.5rem;">✈️</span>'
        
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:10px; flex-wrap:nowrap; padding:10px 0;">
            <div style="display:flex; align-items:center; gap:8px; flex-shrink:0;">
                {home_logo_html}
                <span style="font-size:clamp(0.9rem, 3vw, 1.3rem); font-weight:700; color:#ffffff; white-space:nowrap;">{home_team_name}</span>
            </div>
            <span style="font-size:clamp(1rem, 3vw, 1.5rem); font-weight:700; color:#f39c12; margin:0 10px;">VS</span>
            <div style="display:flex; align-items:center; gap:8px; flex-shrink:0;">
                <span style="font-size:clamp(0.9rem, 3vw, 1.3rem); font-weight:700; color:#ffffff; white-space:nowrap;">{away_team_name}</span>
                {away_logo_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
                
    except Exception as e:
        # Fallback semplice
        st.subheader(f"🏠 {home_team_name}  VS  {away_team_name} ✈️")
    
    # Indicatore qualità
    st.markdown(f"""
    <div style="background:#f0f2f6; padding:10px; border-radius:10px; margin-bottom:20px; color:#1a1a2e;">
        <strong>Affidabilità Previsione:</strong> {quality['level']} ({quality['score']}%) - {quality['message']}
    </div>
    """, unsafe_allow_html=True)
    
    # Metriche principali (3 colonne: xG Casa - xG Trasferta - Tot Gol)
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("⚽ xG Casa", f"{probabilities['mu_home']:.2f}")
    with metric_col2:
        st.metric("⚽ xG Trasferta", f"{probabilities['mu_away']:.2f}")
    with metric_col3:
        st.metric("📈 Tot. Gol Attesi", f"{probabilities['total_expected_goals']:.2f}")
    
    st.markdown("---")
    
    # === PRONOSTICI CONSIGLIATI ===
    top_predictions = calculate_top_predictions(probabilities, home_team_name, away_team_name)
    display_top_predictions(top_predictions)
    
    st.markdown("---")
    
    # === 1X2 ===
    st.subheader("🎲 Esito Finale (1X2)")
    
    col_1x2_1, col_1x2_2, col_1x2_3 = st.columns(3)
    
    with col_1x2_1:
        p_home = probabilities['p_home'] * 100
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #3498db, #2980b9); color:white; 
                    padding: clamp(10px, 3vw, 20px); border-radius:15px; text-align:center; min-height: 100px;">
            <div style="font-size: clamp(0.7rem, 2.5vw, 0.9rem);">🏠 {home_team_name}</div>
            <div style="font-size: clamp(1.5rem, 6vw, 2.5rem); font-weight:bold;">{p_home:.1f}%</div>
            <div style="font-size: clamp(0.6rem, 2vw, 0.8rem); opacity:0.8;">Quota: {100/p_home:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_1x2_2:
        p_draw = probabilities['p_draw'] * 100
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #95a5a6, #7f8c8d); color:white; 
                    padding: clamp(10px, 3vw, 20px); border-radius:15px; text-align:center; min-height: 100px;">
            <div style="font-size: clamp(0.7rem, 2.5vw, 0.9rem);">🤝 Pareggio</div>
            <div style="font-size: clamp(1.5rem, 6vw, 2.5rem); font-weight:bold;">{p_draw:.1f}%</div>
            <div style="font-size: clamp(0.6rem, 2vw, 0.8rem); opacity:0.8;">Quota: {100/p_draw:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_1x2_3:
        p_away = probabilities['p_away'] * 100
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #e74c3c, #c0392b); color:white; 
                    padding: clamp(10px, 3vw, 20px); border-radius:15px; text-align:center; min-height: 100px;">
            <div style="font-size: clamp(0.7rem, 2.5vw, 0.9rem);">✈️ {away_team_name}</div>
            <div style="font-size: clamp(1.5rem, 6vw, 2.5rem); font-weight:bold;">{p_away:.1f}%</div>
            <div style="font-size: clamp(0.6rem, 2vw, 0.8rem); opacity:0.8;">Quota: {100/p_away:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Verifica che sommino a 100%
    st.caption(f"✓ Somma: {p_home + p_draw + p_away:.1f}%")
    
    st.markdown("---")
    
    # === BTTS e Over/Under ===
    col_btts, col_ou = st.columns(2)
    
    with col_btts:
        st.subheader("⚽ Gol / No Gol (BTTS)")
        
        p_btts_yes = probabilities['p_btts_yes'] * 100
        p_btts_no = probabilities['p_btts_no'] * 100
        
        col_yes, col_no = st.columns(2)
        
        with col_yes:
            st.markdown(f"""
            <div style="background:#27ae60; color:white; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-size:1rem;">GOL (Sì)</div>
                <div style="font-size:2rem; font-weight:bold;">{p_btts_yes:.1f}%</div>
                <div style="font-size:0.8rem;">Quota: {100/p_btts_yes:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_no:
            st.markdown(f"""
            <div style="background:#e74c3c; color:white; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-size:1rem;">NO GOL (No)</div>
                <div style="font-size:2rem; font-weight:bold;">{p_btts_no:.1f}%</div>
                <div style="font-size:0.8rem;">Quota: {100/p_btts_no:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"✓ Somma: {p_btts_yes + p_btts_no:.1f}%")
    
    with col_ou:
        st.subheader("📊 Over/Under 2.5")
        
        p_over = probabilities['over_2.5'] * 100
        p_under = probabilities['under_2.5'] * 100
        
        col_over, col_under = st.columns(2)
        
        with col_over:
            st.markdown(f"""
            <div style="background:#9b59b6; color:white; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-size:1rem;">OVER 2.5</div>
                <div style="font-size:2rem; font-weight:bold;">{p_over:.1f}%</div>
                <div style="font-size:0.8rem;">Quota: {100/p_over:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_under:
            st.markdown(f"""
            <div style="background:#34495e; color:white; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-size:1rem;">UNDER 2.5</div>
                <div style="font-size:2rem; font-weight:bold;">{p_under:.1f}%</div>
                <div style="font-size:0.8rem;">Quota: {100/p_under:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"✓ Somma: {p_over + p_under:.1f}%")
    
    st.markdown("---")
    
    # === GRAFICO OVER/UNDER TUTTE LE LINEE ===
    st.subheader("📈 Over/Under - Tutte le Linee")
    fig_ou = create_ou_comparison(probabilities)
    st.plotly_chart(fig_ou, use_container_width=True, config={'displayModeBar': False})
    
    # Tabella dettagliata O/U con header colorati
    lines_data = [1.5, 2.5, 3.5, 4.5]
    
    table_html = '''<table style="width:100%; border-collapse: collapse; border-radius: 10px; overflow: hidden;">
        <tr>
            <th style="background:#3498db; color:white; padding:12px; text-align:left;">Linea</th>
            <th style="background:#27ae60; color:white; padding:12px; text-align:left;">OVER %</th>
            <th style="background:#e74c3c; color:white; padding:12px; text-align:left;">UNDER %</th>
            <th style="background:#27ae60; color:white; padding:12px; text-align:left;">Quota OVER</th>
            <th style="background:#e74c3c; color:white; padding:12px; text-align:left;">Quota UNDER</th>
        </tr>'''
    
    for l in lines_data:
        over_p = probabilities[f'over_{l}'] * 100
        under_p = probabilities[f'under_{l}'] * 100
        table_html += f'''<tr style="background:white;">
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{l}</td>
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{over_p:.1f}%</td>
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{under_p:.1f}%</td>
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{100/over_p:.2f}</td>
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{100/under_p:.2f}</td>
        </tr>'''
    
    table_html += '</table>'
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === HEAT MAP RISULTATI ESATTI ===
    st.subheader("🎯 Heat Map Probabili Risultati Esatti")
    st.markdown("*Visualizza la distribuzione delle probabilità per ogni possibile risultato finale*")
    
    fig_matrix = create_score_matrix_heatmap(
        probabilities['matrix'], 
        home_team_name, 
        away_team_name
    )
    st.plotly_chart(fig_matrix, use_container_width=True, config={'displayModeBar': False})
    
    # Caselle risultati più probabili (senza titolo)
    display_exact_scores(probabilities['exact_scores'])
    
    st.markdown("---")
    
    # === CARTELLINI ===
    st.subheader("🟨 Cartellini - Tutte le Linee")
    
    # Info arbitro se disponibile
    if probabilities.get("referee_found"):
        severity = probabilities.get("referee_severity", 1.0)
        adj = probabilities.get("referee_adjustment", 1.0)
        ref_avg = probabilities.get("referee_avg_cards")
        ref_matches = probabilities.get("referee_matches", 0)
        ref_name = probabilities.get("referee_name", "")
        
        if severity > 1.1:
            ref_color = "#e74c3c"  # Rosso
            ref_emoji = "🔴"
            ref_label = "SEVERO"
        elif severity < 0.9:
            ref_color = "#27ae60"  # Verde
            ref_emoji = "🟢"
            ref_label = "PERMISSIVO"
        else:
            ref_color = "#f39c12"  # Giallo
            ref_emoji = "🟡"
            ref_label = "NELLA MEDIA"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {ref_color}22, {ref_color}11); 
                    border-left: 4px solid {ref_color}; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <strong style="font-size: 1.1em; color: #ffffff;">{ref_emoji} Arbitro: {ref_name}</strong>
                    <span style="background: {ref_color}; color: white; padding: 2px 8px; border-radius: 12px; 
                                 font-size: 0.75em; margin-left: 10px;">{ref_label}</span>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.9em; color: #e0e0e0;">
                        📊 {ref_avg:.1f} cart/partita | 
                        ⚖️ Fattore: {severity:.2f}x | 
                        🎮 {ref_matches} partite analizzate
                    </span>
                </div>
            </div>
            <div style="font-size: 0.85em; margin-top: 8px; color: #c0c0c0;">
                <em>Le probabilità cartellini sono state aggiustate del {((adj-1)*100):+.0f}% rispetto alla media</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif referee_name:
        st.info(f"👨‍⚖️ Arbitro '{referee_name}' non trovato nel database. Usando probabilità standard.")
    
    # Metriche cartellini
    cards_col1, cards_col2, cards_col3, cards_col4 = st.columns(4)
    with cards_col1:
        st.metric("🏠 Media Casa", f"{probabilities.get('home_cards_avg', 0):.2f}")
    with cards_col2:
        st.metric("✈️ Media Trasferta", f"{probabilities.get('away_cards_avg', 0):.2f}")
    with cards_col3:
        base_cards = probabilities.get('expected_cards_base', probabilities.get('expected_cards', 0))
        st.metric("📊 Cartellini Base", f"{base_cards:.2f}")
    with cards_col4:
        st.metric("📊 Cartellini Attesi", f"{probabilities.get('expected_cards', 0):.2f}",
                  delta=f"{probabilities.get('expected_cards', 0) - base_cards:+.2f}" if probabilities.get("referee_found") else None)
    
    # Grafico cartellini
    fig_cards = create_cards_comparison(probabilities)
    st.plotly_chart(fig_cards, use_container_width=True, config={'displayModeBar': False})
    
    # Tabella cartellini con header colorati
    cards_table = '''<table style="width:100%; border-collapse: collapse; border-radius: 10px; overflow: hidden;">
        <tr>
            <th style="background:#3498db; color:white; padding:12px; text-align:left;">Linea</th>
            <th style="background:#f39c12; color:white; padding:12px; text-align:left;">OVER %</th>
            <th style="background:#9b59b6; color:white; padding:12px; text-align:left;">UNDER %</th>
            <th style="background:#f39c12; color:white; padding:12px; text-align:left;">Quota OVER</th>
            <th style="background:#9b59b6; color:white; padding:12px; text-align:left;">Quota UNDER</th>
        </tr>'''
    
    for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
        over_p = probabilities.get(f"cards_over_{line}", 0) * 100
        under_p = probabilities.get(f"cards_under_{line}", 0) * 100
        quota_over = f"{100/over_p:.2f}" if over_p > 0 else "N/A"
        quota_under = f"{100/under_p:.2f}" if under_p > 0 else "N/A"
        cards_table += f'''<tr style="background:white;">
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{line}</td>
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{over_p:.1f}%</td>
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{under_p:.1f}%</td>
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{quota_over}</td>
            <td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{quota_under}</td>
        </tr>'''
    
    cards_table += '</table>'
    st.markdown(cards_table, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === HEAD TO HEAD ===
    st.subheader("⚔️ Scontri Diretti (Ultimi 10)")
    
    if h2h_data.get("matches", 0) > 0:
        h2h_col1, h2h_col2, h2h_col3, h2h_col4 = st.columns(4)
        
        with h2h_col1:
            st.metric("Partite", h2h_data["matches"])
        with h2h_col2:
            st.metric(f"Vinte {home_team_name}", h2h_data["team1_wins"])
        with h2h_col3:
            st.metric("Pareggi", h2h_data["draws"])
        with h2h_col4:
            st.metric(f"Vinte {away_team_name}", h2h_data["team2_wins"])
        
        st.write(f"**Media gol negli scontri diretti:** {h2h_data['avg_goals']}")
    else:
        st.info("Nessuno scontro diretto trovato negli ultimi anni")
    
    st.markdown("---")
    
    # === TIRI ===
    st.subheader("🎯 Media Tiri (Ultime 5 partite)")
    
    shots_col1, shots_col2 = st.columns(2)
    
    with shots_col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #1a5f2a, #2d8a4e); padding:15px; border-radius:10px; text-align:center;">
            <h4 style="margin:0; color:white;">{home_team_name}</h4>
            <div style="font-size:2rem; font-weight:bold; color:white;">{home_shots.get('shots_avg', 'N/A')}</div>
            <div style="font-size:0.9rem; color:#ccc;">Tiri totali/partita</div>
            <div style="font-size:1.5rem; font-weight:bold; color:#4ade80; margin-top:5px;">{home_shots.get('shots_on_target_avg', 'N/A')}</div>
            <div style="font-size:0.9rem; color:#ccc;">In porta/partita</div>
        </div>
        """, unsafe_allow_html=True)
    
    with shots_col2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #8b1538, #c41e3a); padding:15px; border-radius:10px; text-align:center;">
            <h4 style="margin:0; color:white;">{away_team_name}</h4>
            <div style="font-size:2rem; font-weight:bold; color:white;">{away_shots.get('shots_avg', 'N/A')}</div>
            <div style="font-size:0.9rem; color:#ccc;">Tiri totali/partita</div>
            <div style="font-size:1.5rem; font-weight:bold; color:#f87171; margin-top:5px;">{away_shots.get('shots_on_target_avg', 'N/A')}</div>
            <div style="font-size:0.9rem; color:#ccc;">In porta/partita</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === STATISTICHE SQUADRE ===
    st.subheader("📋 Statistiche Squadre Utilizzate")
    
    col_stats1, col_stats2 = st.columns(2)
    
    with col_stats1:
        display_team_stats(home_stats, home_team_name, is_home=True)
    
    with col_stats2:
        display_team_stats(away_stats, away_team_name, is_home=False)
    
    # === INFO LEGA ===
    with st.expander("ℹ️ Info Lega"):
        st.write(f"**Lega:** {league_info['name']}")
        st.write(f"**Squadre totali:** {league_info['total_teams']}")
        st.write(f"**Media gol casa:** {league_info.get('avg_gf_home', 'N/A'):.2f}")
        st.write(f"**Media gol trasferta:** {league_info.get('avg_gf_away', 'N/A'):.2f}")

# Footer
st.markdown("""
<div class="footer">
    <p>⚽ BetEngine v2.0 | 📊 Modello statistico calibrato su 50k+ partite</p>
</div>
""", unsafe_allow_html=True)
