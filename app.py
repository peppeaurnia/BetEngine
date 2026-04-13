"""
⚽ BETENGINE - Analisi Probabilistica Partite
==============================================
Pagina principale: partite del giorno con analisi dettagliata on-click.

Versione: 3.0 (Aprile 2026)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import base64
import os

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================
st.set_page_config(
    page_title="BetEngine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# IMPORT MODULI
# ============================================================
from probability_engine import (
    calculate_match_probabilities,
    assess_prediction_quality
)
from data_fetcher import (
    LEAGUES,
    fetch_teams_for_league,
    get_match_stats,
    get_head_to_head,
    get_team_shots_avg,
    get_current_season,
    fetch_todays_fixtures,
)
from team_logos import get_logo_path
from config import API_FOOTBALL_KEY

# The Odds API (opzionale)
try:
    from odds_api import fetch_match_odds
    ODDS_API_AVAILABLE = True
except ImportError:
    ODDS_API_AVAILABLE = False

# Referee stats (opzionale)
try:
    from fetch_referee_stats import get_referee_adjustment
    REFEREE_AVAILABLE = True
except ImportError:
    REFEREE_AVAILABLE = False


# ============================================================
# NOMI ITALIANIZZATI
# ============================================================
DISPLAY_NAMES = {
    "bayern münchen": "Bayern Monaco", "bayern munich": "Bayern Monaco",
    "fc bayern münchen": "Bayern Monaco", "borussia dortmund": "Borussia Dortmund",
    "borussia mönchengladbach": "B. Mönchengladbach",
    "eintracht frankfurt": "Eintracht Francoforte",
    "1. fc köln": "Colonia", "fc koln": "Colonia", "köln": "Colonia",
    "werder bremen": "Werder Brema", "hamburger sv": "Amburgo",
    "rb leipzig": "Lipsia", "vfb stuttgart": "Stoccarda",
    "sc freiburg": "Friburgo", "fc augsburg": "Augusta",
    "fsv mainz 05": "Magonza", "mainz 05": "Magonza",
    "union berlin": "Union Berlino", "1. fc union berlin": "Union Berlino",
    "vfl wolfsburg": "Wolfsburg", "1899 hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim", "1. fc heidenheim": "Heidenheim",
    "fc st. pauli": "St. Pauli", "vfl bochum": "Bochum",
    "bayer leverkusen": "Bayer Leverkusen", "holstein kiel": "Holstein Kiel",
    "paris saint germain": "Paris Saint-Germain", "paris saint-germain": "Paris Saint-Germain",
    "psg": "Paris Saint-Germain",
    "olympique marseille": "Olympique Marsiglia", "marseille": "Marsiglia",
    "olympique lyon": "Olympique Lione", "olympique lyonnais": "Olympique Lione",
    "lyon": "Lione", "lille": "Lilla", "nice": "Nizza", "toulouse": "Tolosa",
    "strasbourg": "Strasburgo", "stade brestois 29": "Brest",
    "rc lens": "Lens", "stade rennais": "Rennes", "fc nantes": "Nantes",
    "angers": "Angers", "auxerre": "Auxerre", "le havre": "Le Havre",
    "metz": "Metz", "lorient": "Lorient", "paris fc": "Paris FC",
    "as monaco": "Monaco", "monaco": "Monaco",
    "psv eindhoven": "PSV", "psv": "PSV", "az alkmaar": "AZ", "az": "AZ",
    "fc utrecht": "Utrecht", "fc twente": "Twente",
    "fc groningen": "Groningen", "sc heerenveen": "Heerenveen",
    "sparta rotterdam": "Sparta Rotterdam", "nec nijmegen": "NEC",
    "fortuna sittard": "Fortuna Sittard", "rkc waalwijk": "RKC Waalwijk",
    "go ahead eagles": "Go Ahead Eagles", "pec zwolle": "PEC Zwolle",
    "heracles almelo": "Heracles", "ajax": "Ajax", "feyenoord": "Feyenoord",
}

def get_display_name(team_name: str) -> str:
    if not team_name:
        return team_name
    return DISPLAY_NAMES.get(team_name.lower().strip(), team_name)


# ============================================================
# HELPER
# ============================================================
def get_logo_base64_simple(logo_path):
    if not logo_path:
        return None
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

@st.cache_data
def get_background_image():
    paths = ["sfondo_betengine.jpg", os.path.join(os.path.dirname(__file__), "sfondo_betengine.jpg")]
    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None


# ============================================================
# CSS
# ============================================================
bg_image = get_background_image()
if bg_image:
    st.markdown(f"""<style>
    .stApp {{ background-image: url("data:image/jpeg;base64,{bg_image}"); background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed; }}
    @media (max-width: 768px), (hover: none) {{ .stApp {{ background-attachment: scroll !important; }} .stApp::before {{ content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(13, 27, 42, 0.85); z-index: 0; pointer-events: none; }} }}
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp { background: linear-gradient(180deg, #1a3a52 0%, #0d2137 100%); }</style>", unsafe_allow_html=True)

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
    html, body, [data-testid="stAppViewContainer"] { width: 100% !important; overflow-x: hidden !important; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #051221 100%); }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; text-shadow: 0 1px 3px rgba(0,0,0,0.6); }
    label, label p, label span, label small,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span { color: #ffffff !important; background: transparent !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; text-shadow: 0 1px 3px rgba(0,0,0,0.8); }
    [data-testid="stMetricLabel"] { color: #d0d0d0 !important; }
    .main .block-container { background: rgba(13, 27, 42, 0.8) !important; border-radius: 12px; padding: 1rem 2rem !important; }
    @media (max-width: 768px) { .main .block-container { background: rgba(13, 27, 42, 0.92) !important; padding: 0.5rem 1rem !important; } }
    .stSelectbox > div > div, [data-baseweb="select"] > div { background-color: #ffffff !important; color: #1a1a2e !important; }
    [data-baseweb="select"] span, [data-baseweb="select"] div { color: #1a1a2e !important; }
    [data-baseweb="popover"], [data-baseweb="popover"] li, [data-baseweb="popover"] div,
    [data-baseweb="menu"], [data-baseweb="menu"] li { background-color: #ffffff !important; color: #1a1a2e !important; }
    .stTextInput > div > div > input, .stNumberInput > div > div > input { background-color: #ffffff !important; color: #1a1a2e !important; }
    .stDateInput > div > div > input { background-color: #ffffff !important; color: #1a1a2e !important; }
    .stAlert, .stAlert p, .stAlert span, .stAlert div { color: #1a1a2e !important; }
    table th { color: #ffffff !important; }
    table td { color: #1a1a2e !important; background: #ffffff !important; }
    .stTabs [data-baseweb="tab-list"] button { color: #ffffff !important; }
    .js-plotly-plot .plotly .bg { fill: transparent !important; }
    .league-header { display: flex; align-items: center; gap: 10px; padding: 10px 0; margin-top: 15px; border-bottom: 2px solid rgba(79, 195, 247, 0.3); }
    .league-header h3 { margin: 0 !important; font-size: 1.2rem !important; }
    .sub-header { font-size: 1.1rem; text-align: center; color: #d0d0d0 !important; text-shadow: 0 1px 3px rgba(0,0,0,0.6); margin-bottom: 1rem; }
    .footer { text-align: center; color: #7fb3d3 !important; font-size: 0.8rem; margin-top: 3rem; padding: 1rem; border-top: 1px solid #2c5282; }
    .footer p { color: #7fb3d3 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    button[title="View fullscreen"] {display: none;}
    [data-testid="StyledFullScreenButton"] {display: none;}
    .team-stats-card, .team-stats-card * { color: #000000 !important; }
    @media screen and (max-width: 768px) { h1 { font-size: 1.5rem !important; } h2 { font-size: 1.3rem !important; } }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNZIONI VISUALIZZAZIONE
# ============================================================

def create_probability_bar(probs, labels, colors, title):
    values = [probs.get(k, 0) * 100 for k in labels]
    fig = go.Figure(go.Bar(x=values, y=[l.upper() for l in labels], orientation='h',
        marker_color=colors, text=[f'{v:.1f}%' for v in values], textposition='inside', textfont=dict(size=14, color='white')))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color='#e8f4fc')),
        xaxis=dict(range=[0, 100], title="Probabilità %"), yaxis=dict(title=""), height=200,
        margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'),
        xaxis_gridcolor='rgba(255,255,255,0.1)')
    return fig

def create_ou_comparison(probs):
    lines = [1.5, 2.5, 3.5, 4.5]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='OVER', x=[f'{l}' for l in lines], y=[probs.get(f"over_{l}", 0)*100 for l in lines],
        marker_color='#27ae60', text=[f'{probs.get(f"over_{l}", 0)*100:.1f}%' for l in lines], textposition='outside'))
    fig.add_trace(go.Bar(name='UNDER', x=[f'{l}' for l in lines], y=[probs.get(f"under_{l}", 0)*100 for l in lines],
        marker_color='#e74c3c', text=[f'{probs.get(f"under_{l}", 0)*100:.1f}%' for l in lines], textposition='outside'))
    fig.update_layout(title="Over/Under per Linea", xaxis_title="Linea Gol", yaxis_title="Probabilità %",
        barmode='group', height=350, yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'), yaxis_gridcolor='rgba(255,255,255,0.1)')
    return fig

def create_cards_comparison(probs):
    lines = [2.5, 3.5, 4.5, 5.5, 6.5]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='OVER', x=[f'{l}' for l in lines], y=[probs.get(f"cards_over_{l}", 0)*100 for l in lines],
        marker_color='#f39c12', text=[f'{probs.get(f"cards_over_{l}", 0)*100:.1f}%' for l in lines], textposition='outside'))
    fig.add_trace(go.Bar(name='UNDER', x=[f'{l}' for l in lines], y=[probs.get(f"cards_under_{l}", 0)*100 for l in lines],
        marker_color='#9b59b6', text=[f'{probs.get(f"cards_under_{l}", 0)*100:.1f}%' for l in lines], textposition='outside'))
    fig.update_layout(title="Cartellini Over/Under per Linea", xaxis_title="Linea Cartellini", yaxis_title="Probabilità %",
        barmode='group', height=350, yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'), yaxis_gridcolor='rgba(255,255,255,0.1)')
    return fig

def create_score_matrix_heatmap(matrix, home_name, away_name):
    M = matrix[:7, :7] * 100
    fig = go.Figure(data=go.Heatmap(z=M, x=[str(i) for i in range(7)], y=[str(i) for i in range(7)],
        colorscale='Blues', text=[[f'{M[i,j]:.1f}%' if M[i,j] > 0.5 else '' for j in range(7)] for i in range(7)],
        texttemplate='%{text}', textfont={"size": 10, "color": "white"},
        hovertemplate=f'{home_name} %{{y}} - %{{x}} {away_name}<br>Prob: %{{z:.2f}}%<extra></extra>'))
    fig.update_layout(xaxis_title=f"Gol {away_name}", yaxis_title=f"Gol {home_name}", height=400,
        yaxis=dict(autorange='reversed'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
    return fig

def display_exact_scores(scores):
    cols = st.columns(5)
    for i, (hg, ag, prob) in enumerate(scores[:10]):
        with cols[i % 5]:
            color = "#3498db" if hg > ag else "#e74c3c" if ag > hg else "#95a5a6"
            st.markdown(f'<div style="background:{color}; color:white; padding:10px; border-radius:10px; text-align:center; margin:5px 0;"><div style="font-size:1.5rem; font-weight:bold;">{hg} - {ag}</div><div style="font-size:0.9rem;">{prob:.1f}%</div></div>', unsafe_allow_html=True)

def display_team_stats(stats, team_name, is_home):
    pos = "Casa" if is_home else "Trasferta"
    att = stats.get("attack_home" if is_home else "attack_away", 1.0)
    def_ = stats.get("defense_home" if is_home else "defense_away", 1.0)
    form = stats.get("form_factor", 1.0)
    rank = stats.get("rank", "N/A")
    form_str = stats.get("form_string", "")
    att_color = "#27ae60" if att > 1.1 else "#e74c3c" if att < 0.9 else "#f39c12"
    def_color = "#27ae60" if def_ < 0.9 else "#e74c3c" if def_ > 1.1 else "#f39c12"
    st.markdown(f'''<div class="team-stats-card" style="background:#f8f9fa; padding:15px; border-radius:10px; border-left:4px solid {'#3498db' if is_home else '#e74c3c'};">
        <h4 style="margin:0 0 10px 0;">{team_name}</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div><small style="font-size:0.75rem;">Att. ({pos})</small><div style="font-size:1.2rem; color:{att_color}; font-weight:bold;">{att:.2f}</div></div>
            <div><small style="font-size:0.75rem;">Dif. ({pos})</small><div style="font-size:1.2rem; color:{def_color}; font-weight:bold;">{def_:.2f}</div></div>
            <div><small style="font-size:0.75rem;">Forma</small><div style="font-size:1.1rem;">{form:.2f} <small>({form_str[-5:] if form_str else "N/A"})</small></div></div>
            <div><small style="font-size:0.75rem;">Posizione</small><div style="font-size:1.2rem;">{rank}°</div></div>
        </div></div>''', unsafe_allow_html=True)


def calculate_top_predictions(probabilities, home_team, away_team, odds_data=None):
    predictions = []
    p_home = probabilities.get('p_home', 0) * 100
    p_draw = probabilities.get('p_draw', 0) * 100
    p_away = probabilities.get('p_away', 0) * 100
    predictions.append({'name': f'Vittoria {home_team}', 'short': '1', 'prob': p_home, 'icon': '🏠', 'market_type': '1X2'})
    predictions.append({'name': 'Pareggio', 'short': 'X', 'prob': p_draw, 'icon': '🤝', 'market_type': '1X2'})
    predictions.append({'name': f'Vittoria {away_team}', 'short': '2', 'prob': p_away, 'icon': '✈️', 'market_type': '1X2'})
    predictions.append({'name': 'Over 2.5', 'short': 'O2.5', 'prob': probabilities.get('over_2.5', 0) * 100, 'icon': '⬆️', 'market_type': 'Over/Under'})
    predictions.append({'name': 'Under 2.5', 'short': 'U2.5', 'prob': probabilities.get('under_2.5', 0) * 100, 'icon': '⬇️', 'market_type': 'Over/Under'})
    co35 = probabilities.get('cards_over_3.5', 0) * 100
    cu35 = probabilities.get('cards_under_3.5', 0) * 100
    if co35 > 0:
        predictions.append({'name': 'Cart. Over 3.5', 'short': 'CO3.5', 'prob': co35, 'icon': '🟨⬆️', 'is_cards': True, 'market_type': 'Cards'})
        predictions.append({'name': 'Cart. Under 3.5', 'short': 'CU3.5', 'prob': cu35, 'icon': '🟨⬇️', 'is_cards': True, 'market_type': 'Cards'})
    
    has_real_odds = odds_data and len(odds_data) > 0
    FILTERS = {'1X2': {'min_prob': 60, 'min_ev': 0.001, 'min_odds': 1.30}, 'Over/Under': {'min_prob': 75, 'min_ev': None, 'min_odds': 1.30}, 'Cards': {'min_prob': 70, 'min_ev': None, 'min_odds': None}}
    
    for p in predictions:
        prob_dec = p['prob'] / 100.0
        short = p['short'].replace(" ", "")
        if has_real_odds and short in odds_data:
            p['odds'] = float(odds_data[short]); p['odds_source'] = 'real'
        else:
            p['odds'] = round(1.0 / max(prob_dec, 0.01), 2) if prob_dec > 0 else 100.0; p['odds_source'] = 'model'
        p['ev'] = prob_dec * p['odds'] - 1.0; p['ev_pct'] = p['ev'] * 100
        is_cards = p.get('is_cards', False)
        if is_cards and p['odds_source'] == 'model':
            prob = p['prob']
            p['stars'] = 5 if prob >= 80 else 4 if prob >= 75 else 3 if prob >= 70 else 2
        else:
            ev = p['ev_pct']
            p['stars'] = 5 if ev >= 15 else 4 if ev >= 10 else 3 if ev >= 5 else 2 if ev >= 2 else 1 if ev > 0 else 0
    
    value_preds = []
    for p in predictions:
        f = FILTERS.get(p.get('market_type', ''))
        if not f: continue
        if p['prob'] < f['min_prob']: continue
        if f['min_odds'] and p['odds'] < f['min_odds']: continue
        if f['min_ev'] is not None:
            if p.get('is_cards') and p['odds_source'] == 'model': pass
            elif p['ev'] < f['min_ev']: continue
        value_preds.append(p)
    
    value_preds.sort(key=lambda x: (-0.001, -x['prob']) if x.get('is_cards') and x['odds_source'] == 'model' else (-x['ev'], -x['prob']))
    return value_preds[:3]


def display_top_predictions(predictions):
    st.subheader("🏆 Pronostici Consigliati")
    if not predictions:
        st.info("⚠️ Nessun pronostico supera i filtri. Soglie: 1X2 ≥60% + EV+, O/U ≥75%, Cards ≥70%. BTTS escluso.")
        return
    medals = ['🥇', '🥈', '🥉']; medal_colors = ['#FFD700', '#C0C0C0', '#CD7F32']
    n = min(len(predictions), 3)
    cols = [st.columns([1,2,1])[1]] if n == 1 else st.columns(n)
    for i in range(n):
        p = predictions[i]
        with cols[i]:
            stars = '⭐' * p['stars'] + '☆' * (5 - p['stars'])
            is_cards_no = p.get('is_cards') and p.get('odds_source') == 'model'
            if is_cards_no:
                prob = p['prob']
                ev_color = "#2ecc71" if prob >= 75 else "#27ae60" if prob >= 65 else "#f1c40f"
                vlabel = "ALTA PROB." if prob >= 75 else "BUONA PROB." if prob >= 65 else "PROB. OK"
                badge = f"{prob:.0f}% — {vlabel}"; odds_line = '<div style="font-size:0.85rem;color:#d0d0d0;margin:4px 0;">📊 Quote non disponibili</div>'
            else:
                ev = p.get('ev_pct', 0)
                ev_color = "#2ecc71" if ev >= 10 else "#27ae60" if ev >= 5 else "#f1c40f"
                vlabel = "STRONG VALUE" if ev >= 10 else "VALUE" if ev >= 5 else "EDGE"
                badge = f"EV: +{ev:.1f}% — {vlabel}"
                oi = "🎰" if p.get('odds_source') == 'real' else "📊"
                odds_line = f'<div style="font-size:0.85rem;color:#d0d0d0;margin:4px 0;">{oi} Quota: <strong style="color:#ffffff;">{p["odds"]:.2f}</strong></div>'
            st.markdown(f'''<div style="background:linear-gradient(135deg,#1a2a3a,#0f1f2f); border:2px solid {medal_colors[i]}; border-radius:15px; padding:15px; text-align:center; min-height:180px;">
                <div style="font-size:1.8rem;">{medals[i]}</div>
                <div style="font-size:0.9rem; font-weight:bold; color:#ffffff; margin:8px 0;">{p["icon"]} {p["name"]}</div>
                <div style="font-size:1.5rem; font-weight:bold; color:#ffffff;">{p["prob"]:.1f}%</div>
                {odds_line}
                <div style="background:{ev_color}; color:white; display:inline-block; padding:3px 10px; border-radius:10px; font-size:0.8rem; font-weight:bold; margin:4px 0;">{badge}</div>
                <div style="font-size:0.8rem; color:#ffd700; margin:4px 0;">{stars}</div>
            </div>''', unsafe_allow_html=True)


# ============================================================
# ANALISI COMPLETA DI UNA PARTITA
# ============================================================

def display_match_analysis(fixture, league_id, league_name):
    api_key = API_FOOTBALL_KEY
    season = get_current_season()
    home_name = fixture["home_name"]; away_name = fixture["away_name"]
    home_id = fixture["home_id"]; away_id = fixture["away_id"]
    referee = fixture.get("referee", "")
    home_disp = get_display_name(home_name); away_disp = get_display_name(away_name)
    
    with st.spinner("📊 Recupero statistiche..."):
        home_stats, away_stats, league_info = get_match_stats(api_key, home_id, away_id, league_id, season)
    with st.spinner("🔍 Dati avanzati..."):
        h2h = get_head_to_head(api_key, home_id, away_id, last_n=10)
        home_shots = get_team_shots_avg(api_key, home_id, league_id, season, last_n=5)
        away_shots = get_team_shots_avg(api_key, away_id, league_id, season, last_n=5)
    with st.spinner("🧮 Calcolo probabilità..."):
        probs = calculate_match_probabilities(home_stats, away_stats, league_id, h2h_data=h2h,
            home_shots=home_shots, away_shots=away_shots,
            referee_name=referee if referee else None, league_name=league_name)
    
    quality = assess_prediction_quality(home_stats, away_stats)
    
    # Affidabilità
    st.markdown(f'<div style="background:#f0f2f6; padding:10px; border-radius:10px; margin-bottom:15px; color:#1a1a2e;"><strong>Affidabilità:</strong> {quality["level"]} ({quality["score"]}%) - {quality["message"]}</div>', unsafe_allow_html=True)
    
    # xG
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("⚽ xG Casa", f"{probs['mu_home']:.2f}")
    with c2: st.metric("⚽ xG Trasferta", f"{probs['mu_away']:.2f}")
    with c3: st.metric("📈 Tot. Gol Attesi", f"{probs['total_expected_goals']:.2f}")
    st.markdown("---")
    
    # Quote reali + pronostici
    real_odds = {}
    if ODDS_API_AVAILABLE:
        try: real_odds = fetch_match_odds(home_name, away_name, league_id)
        except: pass
    top_preds = calculate_top_predictions(probs, home_disp, away_disp, odds_data=real_odds)
    display_top_predictions(top_preds)
    st.markdown("---")
    
    # 1X2
    st.subheader("📊 Probabilità 1X2")
    st.plotly_chart(create_probability_bar(probs, ['p_home', 'p_draw', 'p_away'], ['#27ae60', '#f39c12', '#e74c3c'], f"{home_disp} vs {away_disp}"), use_container_width=True, config={'displayModeBar': False})
    st.markdown("---")
    
    # BTTS + O/U
    st.subheader("⚽ BTTS & Over/Under 2.5")
    bc1, bc2 = st.columns(2)
    btts_yes = probs.get('btts_yes', 0) * 100; btts_no = 100 - btts_yes
    over25 = probs.get('over_2.5', 0) * 100; under25 = probs.get('under_2.5', 0) * 100
    with bc1:
        st.markdown(f'<div style="background:linear-gradient(135deg,#2c3e50,#3498db); padding:15px; border-radius:12px; text-align:center;"><div style="font-size:0.9rem; color:#ccc;">BTTS</div><div style="display:flex; justify-content:center; gap:20px; margin-top:8px;"><div><div style="font-size:1.8rem; font-weight:bold; color:#2ecc71;">✅ {btts_yes:.1f}%</div><div style="color:#ccc; font-size:0.8rem;">GG</div></div><div><div style="font-size:1.8rem; font-weight:bold; color:#e74c3c;">❌ {btts_no:.1f}%</div><div style="color:#ccc; font-size:0.8rem;">NG</div></div></div></div>', unsafe_allow_html=True)
    with bc2:
        st.markdown(f'<div style="background:linear-gradient(135deg,#2c3e50,#e74c3c); padding:15px; border-radius:12px; text-align:center;"><div style="font-size:0.9rem; color:#ccc;">Over/Under 2.5</div><div style="display:flex; justify-content:center; gap:20px; margin-top:8px;"><div><div style="font-size:1.8rem; font-weight:bold; color:#2ecc71;">⬆️ {over25:.1f}%</div><div style="color:#ccc; font-size:0.8rem;">Over</div></div><div><div style="font-size:1.8rem; font-weight:bold; color:#e74c3c;">⬇️ {under25:.1f}%</div><div style="color:#ccc; font-size:0.8rem;">Under</div></div></div></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # O/U tutte linee
    st.subheader("📈 Over/Under - Tutte le Linee")
    st.plotly_chart(create_ou_comparison(probs), use_container_width=True, config={'displayModeBar': False})
    lines_data = [1.5, 2.5, 3.5, 4.5]
    tbl = '<table style="width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden;"><tr><th style="background:#3498db; color:white; padding:12px; text-align:left;">Linea</th><th style="background:#27ae60; color:white; padding:12px;">OVER %</th><th style="background:#e74c3c; color:white; padding:12px;">UNDER %</th><th style="background:#27ae60; color:white; padding:12px;">Quota OVER</th><th style="background:#e74c3c; color:white; padding:12px;">Quota UNDER</th></tr>'
    for l in lines_data:
        op = probs[f'over_{l}'] * 100; up = probs[f'under_{l}'] * 100
        tbl += f'<tr style="background:white;"><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{l}</td><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{op:.1f}%</td><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{up:.1f}%</td><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{100/op:.2f}</td><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{100/up:.2f}</td></tr>'
    tbl += '</table>'
    st.markdown(tbl, unsafe_allow_html=True)
    st.markdown("---")
    
    # Heatmap
    st.subheader("🎯 Heat Map Risultati Esatti")
    st.plotly_chart(create_score_matrix_heatmap(probs['matrix'], home_disp, away_disp), use_container_width=True, config={'displayModeBar': False})
    display_exact_scores(probs['exact_scores'])
    st.markdown("---")
    
    # Cartellini
    st.subheader("🟨 Cartellini")
    if probs.get("referee_found"):
        sev = probs.get("referee_severity", 1.0); adj = probs.get("referee_adjustment", 1.0)
        ref_avg = probs.get("referee_avg_cards"); ref_m = probs.get("referee_matches", 0); rn = probs.get("referee_name", "")
        if sev > 1.1: rc, re, rl = "#e74c3c", "🔴", "SEVERO"
        elif sev < 0.9: rc, re, rl = "#27ae60", "🟢", "PERMISSIVO"
        else: rc, re, rl = "#f39c12", "🟡", "NELLA MEDIA"
        st.markdown(f'<div style="background:rgba(13,27,42,0.95); border-left:4px solid {rc}; padding:15px; border-radius:8px; margin-bottom:15px;"><strong style="font-size:1.1em; color:#ffffff;">{re} Arbitro: {rn}</strong> <span style="background:{rc}; color:white; padding:2px 8px; border-radius:12px; font-size:0.75em; margin-left:10px;">{rl}</span><br><span style="font-size:0.9em; color:#d0d0d0;">📊 {ref_avg:.1f} cart/partita | ⚖️ {sev:.2f}x | 🎮 {ref_m} partite</span></div>', unsafe_allow_html=True)
    
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1: st.metric("🏠 Media Casa", f"{probs.get('home_cards_avg', 0):.2f}")
    with cc2: st.metric("✈️ Media Trasf.", f"{probs.get('away_cards_avg', 0):.2f}")
    base_c = probs.get('expected_cards_base', probs.get('expected_cards', 0))
    with cc3: st.metric("📊 Base", f"{base_c:.2f}")
    with cc4: st.metric("📊 Attesi", f"{probs.get('expected_cards', 0):.2f}", delta=f"{probs.get('expected_cards', 0)-base_c:+.2f}" if probs.get("referee_found") else None)
    st.plotly_chart(create_cards_comparison(probs), use_container_width=True, config={'displayModeBar': False})
    
    ctbl = '<table style="width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden;"><tr><th style="background:#3498db; color:white; padding:12px;">Linea</th><th style="background:#f39c12; color:white; padding:12px;">OVER %</th><th style="background:#9b59b6; color:white; padding:12px;">UNDER %</th><th style="background:#f39c12; color:white; padding:12px;">Quota OVER</th><th style="background:#9b59b6; color:white; padding:12px;">Quota UNDER</th></tr>'
    for ln in [2.5, 3.5, 4.5, 5.5, 6.5]:
        cop = probs.get(f"cards_over_{ln}", 0)*100; cup = probs.get(f"cards_under_{ln}", 0)*100
        qo = f"{100/cop:.2f}" if cop > 0 else "N/A"; qu = f"{100/cup:.2f}" if cup > 0 else "N/A"
        ctbl += f'<tr style="background:white;"><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{ln}</td><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{cop:.1f}%</td><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{cup:.1f}%</td><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{qo}</td><td style="padding:10px 12px; border-bottom:1px solid #eee; color:#1a1a2e;">{qu}</td></tr>'
    ctbl += '</table>'
    st.markdown(ctbl, unsafe_allow_html=True)
    st.markdown("---")
    
    # H2H
    st.subheader("⚔️ Scontri Diretti (Ultimi 10)")
    if h2h.get("matches", 0) > 0:
        h1, h2_, h3, h4 = st.columns(4)
        with h1: st.metric("Partite", h2h["matches"])
        with h2_: st.metric(f"Vinte {home_disp}", h2h["team1_wins"])
        with h3: st.metric("Pareggi", h2h["draws"])
        with h4: st.metric(f"Vinte {away_disp}", h2h["team2_wins"])
        st.markdown(f"<span style='color:#ffffff;'>**Media gol:** {h2h['avg_goals']}</span>", unsafe_allow_html=True)
    else:
        st.info("Nessuno scontro diretto trovato")
    st.markdown("---")
    
    # Tiri
    st.subheader("🎯 Media Tiri (Ultime 5)")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f'<div style="background:linear-gradient(135deg,#1a5f2a,#2d8a4e); padding:15px; border-radius:10px; text-align:center;"><h4 style="margin:0; color:white;">{home_disp}</h4><div style="font-size:2rem; font-weight:bold; color:white;">{home_shots.get("shots_avg","N/A")}</div><div style="font-size:0.9rem; color:#ccc;">Tiri/partita</div><div style="font-size:1.5rem; font-weight:bold; color:#4ade80; margin-top:5px;">{home_shots.get("shots_on_target_avg","N/A")}</div><div style="font-size:0.9rem; color:#ccc;">In porta</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div style="background:linear-gradient(135deg,#8b1538,#c41e3a); padding:15px; border-radius:10px; text-align:center;"><h4 style="margin:0; color:white;">{away_disp}</h4><div style="font-size:2rem; font-weight:bold; color:white;">{away_shots.get("shots_avg","N/A")}</div><div style="font-size:0.9rem; color:#ccc;">Tiri/partita</div><div style="font-size:1.5rem; font-weight:bold; color:#f87171; margin-top:5px;">{away_shots.get("shots_on_target_avg","N/A")}</div><div style="font-size:0.9rem; color:#ccc;">In porta</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Stats squadre
    st.subheader("📋 Statistiche Squadre")
    t1, t2 = st.columns(2)
    with t1: display_team_stats(home_stats, home_disp, is_home=True)
    with t2: display_team_stats(away_stats, away_disp, is_home=False)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 📅 Data")
    selected_date = st.date_input("Data partite", value=date.today(),
        min_value=date.today() - timedelta(days=3), max_value=date.today() + timedelta(days=7))
    st.markdown("---")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.markdown('<div style="color:#888; font-size:0.8rem; text-align:center; margin-top:20px;"><p>⚽ BetEngine v3.0</p><p>Dixon-Coles calibrato</p></div>', unsafe_allow_html=True)


# ============================================================
# HEADER + MAIN
# ============================================================

col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
with col_l2:
    try:
        with open("logo.png", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.markdown(f'<div style="display:flex; justify-content:center; margin-bottom:1rem;"><img src="data:image/png;base64,{logo_data}" style="width:220px;"></div>', unsafe_allow_html=True)
    except:
        st.markdown('<h1 style="font-family:Audiowide; text-align:center; color:#ffffff;">BETENGINE</h1>', unsafe_allow_html=True)

st.markdown('<p class="sub-header">Trasforma i dati in probabilità vincenti</p>', unsafe_allow_html=True)

date_str = selected_date.strftime("%Y-%m-%d")
date_display_str = selected_date.strftime("%d/%m/%Y")
giorni = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
if selected_date == date.today(): day_label = "Oggi"
elif selected_date == date.today() + timedelta(days=1): day_label = "Domani"
elif selected_date == date.today() - timedelta(days=1): day_label = "Ieri"
else: day_label = giorni[selected_date.weekday()]

st.markdown(f'<div style="text-align:center; margin-bottom:1.5rem;"><span style="font-size:1.3rem; color:#4fc3f7; font-weight:bold;">📅 {day_label}, {date_display_str}</span></div>', unsafe_allow_html=True)

# API check
if not API_FOOTBALL_KEY or API_FOOTBALL_KEY == "INSERISCI_QUI_LA_TUA_API_KEY":
    st.warning("⚠️ API Key non configurata! Apri `config.py` e inserisci la tua chiave API-Football.")
    st.stop()

# Partite del giorno
with st.spinner("⚽ Caricamento partite..."):
    todays_fixtures = fetch_todays_fixtures(API_FOOTBALL_KEY, date_str)

if not todays_fixtures:
    st.markdown(f'<div style="text-align:center; padding:40px 20px;"><div style="font-size:3rem; margin-bottom:10px;">😴</div><div style="font-size:1.2rem; color:#d0d0d0;">Nessuna partita in programma per {day_label.lower()}</div><div style="font-size:0.9rem; color:#888; margin-top:10px;">Leghe: Serie A, Premier League, LaLiga, Bundesliga, Ligue 1, Primeira Liga, Eredivisie</div></div>', unsafe_allow_html=True)
else:
    total = sum(len(f) for f in todays_fixtures.values())
    st.markdown(f'<div style="text-align:center; margin-bottom:15px;"><span style="color:#d0d0d0; font-size:0.95rem;">{total} partite in programma</span></div>', unsafe_allow_html=True)
    
    for league_id, fixtures in todays_fixtures.items():
        ln = LEAGUES.get(league_id, {}).get("name", f"Lega {league_id}")
        st.markdown(f'<div class="league-header"><h3>{ln}</h3><span style="color:#888; font-size:0.85rem;">({len(fixtures)} partite)</span></div>', unsafe_allow_html=True)
        
        for fix in fixtures:
            hd = get_display_name(fix["home_name"]); ad = get_display_name(fix["away_name"])
            status = fix["status"]; t = fix["time"]
            if status == "NS": badge = f"🕐 {t}"
            elif status in ("1H", "2H", "HT", "ET", "P", "LIVE"): badge = f"🔴 LIVE {fix.get('score_home',0)}-{fix.get('score_away',0)}"
            elif status == "FT": badge = f"✅ {fix.get('score_home',0)}-{fix.get('score_away',0)}"
            else: badge = f"🕐 {t}"
            
            label = f"{badge}  |  {hd}  vs  {ad}"
            with st.expander(label, expanded=False):
                fk = f"analysis_{fix['fixture_id']}"
                if fk not in st.session_state: st.session_state[fk] = False
                
                if not st.session_state[fk]:
                    st.markdown(f"**{hd}** vs **{ad}** — ⏰ {t} — 👨‍⚖️ {fix.get('referee','N/D') or 'N/D'}")
                    if st.button("🔮 ANALIZZA PARTITA", key=f"btn_{fix['fixture_id']}", use_container_width=True):
                        st.session_state[fk] = True; st.rerun()
                else:
                    display_match_analysis(fix, league_id, ln)

# Footer
st.markdown('<div class="footer"><p>⚽ BetEngine v3.0 | 📊 Dixon-Coles calibrato | 🧮 Probability Engine v3</p></div>', unsafe_allow_html=True)
