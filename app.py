"""
⚽ BETENGINE v3.1 — Analisi Probabilistica Partite
====================================================
Design pulito, sfondo solido, partite del giorno con analisi on-click.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import base64, os, io

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="BetEngine",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Init tracker in session_state
if "tracked_preds" not in st.session_state:
    st.session_state["tracked_preds"] = []

from probability_engine import calculate_match_probabilities, assess_prediction_quality, apply_market_anchoring
from data_fetcher import (LEAGUES, get_match_stats, get_head_to_head, get_team_shots_avg,
                          get_current_season, fetch_todays_fixtures)
from team_logos import get_logo_path
from config import API_FOOTBALL_KEY

try:
    from odds_api import fetch_match_odds
    ODDS_API_AVAILABLE = True
except ImportError:
    ODDS_API_AVAILABLE = False

# ============================================================
# NOMI ITALIANIZZATI
# ============================================================
DISPLAY_NAMES = {
    "bayern münchen": "Bayern Monaco", "bayern munich": "Bayern Monaco", "fc bayern münchen": "Bayern Monaco",
    "borussia dortmund": "Borussia Dortmund", "borussia mönchengladbach": "B. Mönchengladbach",
    "eintracht frankfurt": "Eintracht Francoforte",
    "1. fc köln": "Colonia", "fc koln": "Colonia", "köln": "Colonia",
    "werder bremen": "Werder Brema", "hamburger sv": "Amburgo",
    "rb leipzig": "Lipsia", "vfb stuttgart": "Stoccarda",
    "sc freiburg": "Friburgo", "fc augsburg": "Augusta",
    "fsv mainz 05": "Magonza", "mainz 05": "Magonza",
    "union berlin": "Union Berlino", "1. fc union berlin": "Union Berlino",
    "vfl wolfsburg": "Wolfsburg", "1899 hoffenheim": "Hoffenheim", "tsg hoffenheim": "Hoffenheim",
    "1. fc heidenheim": "Heidenheim", "fc st. pauli": "St. Pauli", "vfl bochum": "Bochum",
    "bayer leverkusen": "Bayer Leverkusen", "holstein kiel": "Holstein Kiel",
    "paris saint germain": "PSG", "paris saint-germain": "PSG", "psg": "PSG",
    "olympique marseille": "Marsiglia", "marseille": "Marsiglia",
    "olympique lyon": "Lione", "olympique lyonnais": "Lione", "lyon": "Lione",
    "lille": "Lilla", "nice": "Nizza", "toulouse": "Tolosa", "strasbourg": "Strasburgo",
    "stade brestois 29": "Brest", "rc lens": "Lens", "stade rennais": "Rennes",
    "fc nantes": "Nantes", "angers": "Angers", "auxerre": "Auxerre", "le havre": "Le Havre",
    "metz": "Metz", "lorient": "Lorient", "paris fc": "Paris FC",
    "as monaco": "Monaco", "monaco": "Monaco",
    "psv eindhoven": "PSV", "psv": "PSV", "az alkmaar": "AZ", "az": "AZ",
    "fc utrecht": "Utrecht", "fc twente": "Twente", "fc groningen": "Groningen",
    "sc heerenveen": "Heerenveen", "sparta rotterdam": "Sparta Rotterdam", "nec nijmegen": "NEC",
    "fortuna sittard": "Fortuna Sittard", "rkc waalwijk": "RKC Waalwijk",
    "go ahead eagles": "Go Ahead Eagles", "pec zwolle": "PEC Zwolle",
    "heracles almelo": "Heracles", "ajax": "Ajax", "feyenoord": "Feyenoord",
}
def dn(name):
    return DISPLAY_NAMES.get(name.lower().strip(), name) if name else name

def logo_b64(path):
    if not path: return None
    try:
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')
    except: return None


# ============================================================
# CSS — DESIGN PULITO, SFONDO SOLIDO
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Outfit:wght@600;700;800&display=swap');

    /* === SFONDO SOLIDO SCURO === */
    .stApp {
        background: #0d1117 !important;
    }

    /* === RESET GENERALE === */
    html, body, [data-testid="stAppViewContainer"] {
        width: 100% !important;
        overflow-x: hidden !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* === CONTAINER PRINCIPALE === */
    .main .block-container {
        background: transparent !important;
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem 1.5rem !important;
    }

    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #c9d1d9 !important;
    }

    /* === TESTI === */
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important;
        font-family: 'Outfit', sans-serif !important;
        text-shadow: none !important;
    }
    p, span, div, li {
        color: #c9d1d9;
    }
    label, label p, label span, label small,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: #c9d1d9 !important;
        background: transparent !important;
    }

    /* === METRICHE === */
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-family: 'Outfit', sans-serif !important;
        text-shadow: none !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    /* === INPUT / SELECT === */
    .stSelectbox > div > div, [data-baseweb="select"] > div {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }
    [data-baseweb="select"] span, [data-baseweb="select"] div { color: #c9d1d9 !important; }
    [data-baseweb="popover"], [data-baseweb="popover"] li,
    [data-baseweb="popover"] div, [data-baseweb="menu"],
    [data-baseweb="menu"] li {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }

    /* === EXPANDER — CARD PARTITA === */
    [data-testid="stExpander"] {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        color: #e6edf3 !important;
        font-weight: 500 !important;
        padding: 14px 18px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: #1c2433 !important;
    }
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary div {
        color: #e6edf3 !important;
    }
    /* Contenuto espanso */
    [data-testid="stExpander"] > div > div {
        background: #0d1117 !important;
        border-top: 1px solid #21262d !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
        color: #c9d1d9 !important;
    }

    /* === ALERTS === */
    .stAlert { border-radius: 8px !important; }
    .stAlert p, .stAlert span, .stAlert div { color: #1a1a2e !important; }

    /* === TABELLE === */
    table { border-collapse: collapse; width: 100%; }
    table th {
        background: #238636 !important;
        color: #ffffff !important;
        padding: 10px 14px !important;
        font-size: 0.85rem;
    }
    table td {
        background: #161b22 !important;
        color: #c9d1d9 !important;
        padding: 10px 14px !important;
        border-bottom: 1px solid #21262d !important;
        font-size: 0.85rem;
    }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] button { color: #c9d1d9 !important; }

    /* === GRAFICI === */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }

    /* === NASCONDI UI STREAMLIT === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    button[title="View fullscreen"] {display: none;}
    [data-testid="StyledFullScreenButton"] {display: none;}

    /* === TEAM STATS CARD === */
    .team-stats-card {
        background: #161b22 !important;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 15px;
    }
    .team-stats-card * { color: #c9d1d9 !important; }
    .team-stats-card h4 { color: #e6edf3 !important; }

    /* === CUSTOM CLASSES === */
    .be-date-pill {
        display: inline-block;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 6px 18px;
        color: #c9d1d9;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .be-date-pill.active {
        background: #238636;
        border-color: #238636;
        color: #ffffff;
    }
    .be-league {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 0 8px 0;
        margin-top: 20px;
        border-bottom: 1px solid #21262d;
    }
    .be-league h3 {
        margin: 0 !important;
        font-size: 1.05rem !important;
        color: #e6edf3 !important;
    }
    .be-league .count {
        color: #8b949e;
        font-size: 0.8rem;
        font-weight: 400;
    }
    .be-empty {
        text-align: center;
        padding: 60px 20px;
    }
    .be-empty .icon { font-size: 2.5rem; margin-bottom: 12px; }
    .be-empty .msg { font-size: 1.1rem; color: #8b949e; }
    .be-empty .sub { font-size: 0.85rem; color: #484f58; margin-top: 8px; }
    .be-footer {
        text-align: center;
        color: #484f58;
        font-size: 0.75rem;
        margin-top: 40px;
        padding: 15px 0;
        border-top: 1px solid #21262d;
    }
    .be-total {
        text-align: center;
        margin-bottom: 20px;
        color: #8b949e;
        font-size: 0.9rem;
    }

    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .main .block-container { padding: 0.5rem 0.75rem !important; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
    }
    
    /* === CALENDAR BUTTONS === */
    /* Rendi i bottoni del calendario più compatti */
    [data-testid="stHorizontalBlock"] .stButton > button {
        font-size: 0.8rem !important;
        padding: 6px 4px !important;
        line-height: 1.3 !important;
        border-radius: 10px !important;
        min-height: 0 !important;
    }
    [data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background: #238636 !important;
        border-color: #238636 !important;
        color: #ffffff !important;
    }
    [data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] p {
        color: #ffffff !important;
    }
    [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }
    [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] p {
        color: #c9d1d9 !important;
    }
    [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"]:hover {
        background: #1c2433 !important;
        border-color: #1f6feb !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNZIONI GRAFICI
# ============================================================

CHART_LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c9d1d9', family='DM Sans'))

def chart_1x2(probs, home, away):
    vals = [probs.get(k,0)*100 for k in ['p_home','p_draw','p_away']]
    fig = go.Figure(go.Bar(x=vals, y=[f'{home}','Pareggio',f'{away}'], orientation='h',
        marker_color=['#238636','#d29922','#da3633'],
        text=[f'{v:.1f}%' for v in vals], textposition='inside', textfont=dict(size=14, color='white')))
    fig.update_layout(**CHART_LAYOUT, height=200, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0,100], showticklabels=False, gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d'), showlegend=False)
    return fig

def chart_ou(probs):
    lines = [1.5, 2.5, 3.5, 4.5]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='OVER', x=[f'{l}' for l in lines],
        y=[probs.get(f"over_{l}",0)*100 for l in lines], marker_color='#238636',
        text=[f'{probs.get(f"over_{l}",0)*100:.1f}%' for l in lines], textposition='outside'))
    fig.add_trace(go.Bar(name='UNDER', x=[f'{l}' for l in lines],
        y=[probs.get(f"under_{l}",0)*100 for l in lines], marker_color='#da3633',
        text=[f'{probs.get(f"under_{l}",0)*100:.1f}%' for l in lines], textposition='outside'))
    fig.update_layout(**CHART_LAYOUT, height=320, barmode='group',
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(range=[0,100], gridcolor='#21262d'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=10, b=30))
    return fig

def chart_cards(probs):
    lines = [2.5, 3.5, 4.5, 5.5, 6.5]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='OVER', x=[f'{l}' for l in lines],
        y=[probs.get(f"cards_over_{l}",0)*100 for l in lines], marker_color='#d29922',
        text=[f'{probs.get(f"cards_over_{l}",0)*100:.1f}%' for l in lines], textposition='outside'))
    fig.add_trace(go.Bar(name='UNDER', x=[f'{l}' for l in lines],
        y=[probs.get(f"cards_under_{l}",0)*100 for l in lines], marker_color='#8957e5',
        text=[f'{probs.get(f"cards_under_{l}",0)*100:.1f}%' for l in lines], textposition='outside'))
    fig.update_layout(**CHART_LAYOUT, height=320, barmode='group',
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(range=[0,100], gridcolor='#21262d'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=10, b=30))
    return fig

def chart_heatmap(matrix, home, away):
    M = matrix[:7,:7]*100
    fig = go.Figure(data=go.Heatmap(z=M, x=[str(i) for i in range(7)], y=[str(i) for i in range(7)],
        colorscale=[[0,'#0d1117'],[0.5,'#1f6feb'],[1,'#58a6ff']],
        text=[[f'{M[i,j]:.1f}%' if M[i,j]>0.5 else '' for j in range(7)] for i in range(7)],
        texttemplate='%{text}', textfont=dict(size=10, color='white'),
        hovertemplate=f'{home} %{{y}} - %{{x}} {away}<br>Prob: %{{z:.2f}}%<extra></extra>'))
    fig.update_layout(**CHART_LAYOUT, height=380, xaxis_title=f"Gol {away}", yaxis_title=f"Gol {home}",
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(autorange='reversed', gridcolor='#21262d'),
        margin=dict(l=10, r=10, t=10, b=40))
    return fig


# ============================================================
# FUNZIONI DISPLAY
# ============================================================

def show_exact_scores(scores):
    cols = st.columns(5)
    for i, (hg, ag, prob) in enumerate(scores[:10]):
        with cols[i%5]:
            c = "#238636" if hg > ag else "#da3633" if ag > hg else "#484f58"
            st.markdown(f'<div style="background:{c}; color:#fff; padding:8px; border-radius:8px; text-align:center; margin:4px 0;"><div style="font-size:1.3rem; font-weight:700;">{hg}-{ag}</div><div style="font-size:0.8rem; opacity:0.85;">{prob:.1f}%</div></div>', unsafe_allow_html=True)

def show_team_stats(stats, name, is_home):
    pos = "Casa" if is_home else "Trasferta"
    att = stats.get("attack_home" if is_home else "attack_away", 1.0)
    df = stats.get("defense_home" if is_home else "defense_away", 1.0)
    form = stats.get("form_factor", 1.0)
    rank = stats.get("rank", "N/A")
    fs = stats.get("form_string", "")
    ac = "#238636" if att > 1.1 else "#da3633" if att < 0.9 else "#d29922"
    dc = "#238636" if df < 0.9 else "#da3633" if df > 1.1 else "#d29922"
    border = "#1f6feb" if is_home else "#da3633"
    st.markdown(f'''<div class="team-stats-card" style="border-left:3px solid {border};">
        <h4 style="margin:0 0 10px 0; font-size:1rem;">{name}</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div><small style="font-size:0.7rem; color:#8b949e !important;">Att. ({pos})</small><div style="font-size:1.1rem; color:{ac} !important; font-weight:700;">{att:.2f}</div></div>
            <div><small style="font-size:0.7rem; color:#8b949e !important;">Dif. ({pos})</small><div style="font-size:1.1rem; color:{dc} !important; font-weight:700;">{df:.2f}</div></div>
            <div><small style="font-size:0.7rem; color:#8b949e !important;">Forma</small><div style="font-size:1rem;">{form:.2f} <small style="color:#8b949e !important;">({fs[-5:] if fs else "—"})</small></div></div>
            <div><small style="font-size:0.7rem; color:#8b949e !important;">Classifica</small><div style="font-size:1.1rem;">{rank}°</div></div>
        </div></div>''', unsafe_allow_html=True)



def calc_top_preds(probs, home, away, odds_data=None):
    """
    Seleziona i 3 pronostici piu forti del modello. (v4)
    
    - Sempre 3 pronostici (1 per mercato: 1X2, O/U, Cards)
    - Quote SOLO se reali dai bookmaker
    - Niente quote inventate, niente EV circolare
    """
    candidates = []
    
    # 1X2
    ph = probs.get('p_home', 0) * 100
    pd_ = probs.get('p_draw', 0) * 100
    pa = probs.get('p_away', 0) * 100
    candidates.append({'name': f'{home}', 'short': '1', 'prob': ph, 'icon': '🏠', 'mt': '1X2'})
    candidates.append({'name': 'Pareggio', 'short': 'X', 'prob': pd_, 'icon': '🤝', 'mt': '1X2'})
    candidates.append({'name': f'{away}', 'short': '2', 'prob': pa, 'icon': '✈️', 'mt': '1X2'})
    
    # Over/Under 2.5
    o25 = probs.get('over_2.5', 0) * 100
    u25 = probs.get('under_2.5', 0) * 100
    candidates.append({'name': 'Over 2.5', 'short': 'O2.5', 'prob': o25, 'icon': '⬆️', 'mt': 'OU'})
    candidates.append({'name': 'Under 2.5', 'short': 'U2.5', 'prob': u25, 'icon': '⬇️', 'mt': 'OU'})
    
    # Cartellini O/U 3.5
    co = probs.get('cards_over_3.5', 0) * 100
    cu = probs.get('cards_under_3.5', 0) * 100
    if co > 0:
        candidates.append({'name': 'Cart. O3.5', 'short': 'CO3.5', 'prob': co, 'icon': '🟨', 'mt': 'Cards'})
        candidates.append({'name': 'Cart. U3.5', 'short': 'CU3.5', 'prob': cu, 'icon': '🟨', 'mt': 'Cards'})
    
    # Quote reali se disponibili
    has_odds = odds_data and len(odds_data) > 0
    for p in candidates:
        short = p['short'].replace(" ", "")
        if has_odds and short in odds_data:
            p['odds'] = float(odds_data[short])
            p['has_odds'] = True
            p['ev_pct'] = (p['prob'] / 100.0 * p['odds'] - 1.0) * 100
        else:
            p['odds'] = None
            p['has_odds'] = False
            p['ev_pct'] = None
    
    # Stelle basate sulla probabilita
    for p in candidates:
        prob = p['prob']
        if prob >= 75: p['stars'] = 5
        elif prob >= 65: p['stars'] = 4
        elif prob >= 55: p['stars'] = 3
        elif prob >= 50: p['stars'] = 2
        else: p['stars'] = 1
    
    # Seleziona 1 per mercato (il piu forte)
    top = []
    best_1x2 = max([c for c in candidates if c['mt'] == '1X2'], key=lambda x: x['prob'])
    top.append(best_1x2)
    
    best_ou = max([c for c in candidates if c['mt'] == 'OU'], key=lambda x: x['prob'])
    top.append(best_ou)
    
    cards = [c for c in candidates if c['mt'] == 'Cards']
    if cards:
        top.append(max(cards, key=lambda x: x['prob']))
    else:
        remaining = [c for c in candidates if c not in top]
        if remaining:
            top.append(max(remaining, key=lambda x: abs(x['prob'] - 50)))
    
    top.sort(key=lambda x: -abs(x['prob'] - 50))
    return top[:3]


def show_top_preds(preds):
    """Mostra i 3 pronostici. Quote solo se reali."""
    st.markdown("#### 🏆 Pronostici Consigliati")
    if not preds:
        return
    
    n = min(len(preds), 3)
    cols = [st.columns([1, 2, 1])[1]] if n == 1 else st.columns(n)
    medals = ['🥇', '🥈', '🥉']
    mc = ['#d29922', '#8b949e', '#da7b30']
    
    for i in range(n):
        p = preds[i]
        with cols[i]:
            stars_str = '⭐' * p['stars'] + '☆' * (5 - p['stars'])
            
            html = '<div style="background:#161b22; border:1px solid ' + mc[i] + '; border-radius:12px; padding:16px; text-align:center;">'
            html += '<div style="font-size:1.5rem;">' + medals[i] + '</div>'
            html += '<div style="font-size:0.9rem; font-weight:600; color:#e6edf3; margin:6px 0;">' + p["icon"] + ' ' + p["name"] + '</div>'
            html += '<div style="font-size:1.6rem; font-weight:800; color:#e6edf3;">' + f'{p["prob"]:.1f}' + '%</div>'
            
            if p.get('has_odds') and p.get('odds'):
                html += '<div style="font-size:0.85rem; color:#8b949e; margin:4px 0;">🎰 Quota: <strong style="color:#e6edf3;">' + f'{p["odds"]:.2f}' + '</strong></div>'
                ev = p.get('ev_pct')
                if ev is not None and ev > 0:
                    html += '<div style="background:#238636; color:#fff; display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; margin:4px 0;">EV +' + f'{ev:.1f}' + '%</div>'
                elif ev is not None and ev > -3:
                    html += '<div style="background:#d29922; color:#fff; display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; margin:4px 0;">EV ' + f'{ev:.1f}' + '%</div>'
                else:
                    html += '<div style="background:#da3633; color:#fff; display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; margin:4px 0;">EV —</div>'
            else:
                prob = p['prob']
                if prob >= 70:
                    html += '<div style="background:#238636; color:#fff; display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; margin:4px 0;">ALTA FIDUCIA</div>'
                elif prob >= 55:
                    html += '<div style="background:#d29922; color:#fff; display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; margin:4px 0;">BUONA FIDUCIA</div>'
                else:
                    html += '<div style="background:#484f58; color:#fff; display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; margin:4px 0;">POSSIBILE</div>'
            
            html += '<div style="font-size:0.75rem; color:#d29922;">' + stars_str + '</div>'
            html += '</div>'
            
            st.markdown(html, unsafe_allow_html=True)

# ============================================================
# ANALISI COMPLETA PARTITA
# ============================================================

def show_analysis(fix, lid, lname):
    api = API_FOOTBALL_KEY; season = get_current_season()
    hn = fix["home_name"]; an = fix["away_name"]; hid = fix["home_id"]; aid = fix["away_id"]
    ref = fix.get("referee",""); hd = dn(hn); ad = dn(an)
    match_date = fix.get("date_raw", st.session_state.get("sel_date", datetime.now().strftime("%Y-%m-%d")))
    
    with st.spinner("📊 Statistiche..."): hs, aws, li = get_match_stats(api, hid, aid, lid, season)
    with st.spinner("🔍 H2H & Tiri..."):
        h2h = get_head_to_head(api, hid, aid, last_n=10)
        hshots = get_team_shots_avg(api, hid, lid, season, last_n=5)
        ashots = get_team_shots_avg(api, aid, lid, season, last_n=5)
    with st.spinner("🧮 Calcolo..."):
        pr = calculate_match_probabilities(hs, aws, lid, h2h_data=h2h, home_shots=hshots, away_shots=ashots,
            referee_name=ref if ref else None, league_name=lname)
    
    # Recupera quote reali
    odds = {}
    if ODDS_API_AVAILABLE:
        try: odds = fetch_match_odds(hn, an, lid)
        except: pass
    
    # === MARKET ANCHORING: mescola modello + mercato ===
    anchored = False
    if odds:
        pr = apply_market_anchoring(pr, odds)
        anchored = True
    
    q = assess_prediction_quality(hs, aws)
    
    # Affidabilità + indicatore anchoring
    qcol = "#238636" if q['score']>=70 else "#d29922" if q['score']>=50 else "#da3633"
    anch_badge = '<span style="background:#1f6feb; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.7rem; margin-left:8px;">📡 Market Anchored</span>' if anchored else '<span style="background:#484f58; color:#8b949e; padding:2px 8px; border-radius:10px; font-size:0.7rem; margin-left:8px;">Solo Modello</span>'
    st.markdown(f'<div style="background:#161b22; border-left:3px solid {qcol}; padding:10px 14px; border-radius:8px; margin-bottom:12px;"><span style="color:#e6edf3; font-weight:600;">Affidabilità: {q["level"]}</span> <span style="color:#8b949e;">({q["score"]}%)</span>{anch_badge}</div>', unsafe_allow_html=True)
    
    # xG
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("⚽ xG Casa", f"{pr['mu_home']:.2f}")
    with c2: st.metric("⚽ xG Trasf.", f"{pr['mu_away']:.2f}")
    with c3: st.metric("📈 Gol Attesi", f"{pr['total_expected_goals']:.2f}")
    st.markdown("---")
    
    # Pronostici consigliati
    top_preds = calc_top_preds(pr, hd, ad, odds_data=odds)
    show_top_preds(top_preds)
    
    # === SALVA IN SESSION STATE ===
    if top_preds:
        clean_league = lname.split(" ", 1)[-1] if lname and " " in lname else lname
        for pred in top_preds:
            # Evita duplicati
            key = f"{match_date}_{hd}_{ad}_{pred.get('name','')}"
            already = any(r.get('_key') == key for r in st.session_state["tracked_preds"])
            if not already:
                st.session_state["tracked_preds"].append({
                    "_key": key,
                    "Data": match_date,
                    "Lega": clean_league,
                    "Casa": hd,
                    "Trasferta": ad,
                    "Mercato": pred.get('mt', ''),
                    "Selezione": pred.get('name', ''),
                    "Probabilità": round(pred.get('prob', 0), 1),
                    "Quota": round(pred['odds'], 2) if pred.get('has_odds') and pred.get('odds') else "",
                    "EV%": round(pred['ev_pct'], 1) if pred.get('ev_pct') is not None else "",
                    "Stelle": pred.get('stars', 0),
                    "Anchored": "Sì" if anchored else "No",
                    "Risultato": "",
                })
    
    st.markdown("---")
    
    # 1X2
    st.markdown("#### 📊 1X2")
    st.plotly_chart(chart_1x2(pr, hd, ad), use_container_width=True, config={'displayModeBar': False})
    st.markdown("---")
    
    # BTTS + O/U 2.5
    st.markdown("#### ⚽ BTTS & Over/Under 2.5")
    b1, b2 = st.columns(2)
    by = pr.get('p_btts_yes',0)*100; bn = 100-by; o25 = pr.get('over_2.5',0)*100; u25 = pr.get('under_2.5',0)*100
    with b1:
        st.markdown(f'<div style="background:#161b22; border:1px solid #21262d; padding:16px; border-radius:10px; text-align:center;"><div style="color:#8b949e; font-size:0.85rem; margin-bottom:8px;">BTTS</div><div style="display:flex; justify-content:center; gap:24px;"><div><div style="font-size:1.6rem; font-weight:700; color:#238636;">{by:.1f}%</div><div style="color:#8b949e; font-size:0.8rem;">GG</div></div><div><div style="font-size:1.6rem; font-weight:700; color:#da3633;">{bn:.1f}%</div><div style="color:#8b949e; font-size:0.8rem;">NG</div></div></div></div>', unsafe_allow_html=True)
    with b2:
        st.markdown(f'<div style="background:#161b22; border:1px solid #21262d; padding:16px; border-radius:10px; text-align:center;"><div style="color:#8b949e; font-size:0.85rem; margin-bottom:8px;">Over/Under 2.5</div><div style="display:flex; justify-content:center; gap:24px;"><div><div style="font-size:1.6rem; font-weight:700; color:#238636;">{o25:.1f}%</div><div style="color:#8b949e; font-size:0.8rem;">Over</div></div><div><div style="font-size:1.6rem; font-weight:700; color:#da3633;">{u25:.1f}%</div><div style="color:#8b949e; font-size:0.8rem;">Under</div></div></div></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # O/U tutte linee
    st.markdown("#### 📈 Over/Under — Tutte le Linee")
    st.plotly_chart(chart_ou(pr), use_container_width=True, config={'displayModeBar': False})
    tbl = '<table><tr><th>Linea</th><th>OVER</th><th>UNDER</th><th>Quota O</th><th>Quota U</th></tr>'
    for l in [1.5,2.5,3.5,4.5]:
        op = pr[f'over_{l}']*100; up = pr[f'under_{l}']*100
        qo = f"{100/op:.2f}" if op > 0 else "—"
        qu = f"{100/up:.2f}" if up > 0 else "—"
        tbl += f'<tr><td>{l}</td><td>{op:.1f}%</td><td>{up:.1f}%</td><td>{qo}</td><td>{qu}</td></tr>'
    st.markdown(tbl+'</table>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Heatmap
    st.markdown("#### 🎯 Risultati Esatti")
    st.plotly_chart(chart_heatmap(pr['matrix'], hd, ad), use_container_width=True, config={'displayModeBar': False})
    show_exact_scores(pr['exact_scores'])
    st.markdown("---")
    
    # Cartellini
    st.markdown("#### 🟨 Cartellini")
    if pr.get("referee_found"):
        sv = pr.get("referee_severity",1.0); rn = pr.get("referee_name","")
        ra = pr.get("referee_avg_cards",0); rm = pr.get("referee_matches",0)
        rc = "#da3633" if sv>1.1 else "#238636" if sv<0.9 else "#d29922"
        rl = "SEVERO" if sv>1.1 else "PERMISSIVO" if sv<0.9 else "MEDIA"
        st.markdown(f'<div style="background:#161b22; border-left:3px solid {rc}; padding:12px 14px; border-radius:8px; margin-bottom:12px;"><span style="color:#e6edf3; font-weight:600;">👨‍⚖️ {rn}</span> <span style="background:{rc}; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.7rem; font-weight:600; margin-left:8px;">{rl}</span><br><span style="color:#8b949e; font-size:0.85rem;">{ra:.1f} cart/partita · {sv:.2f}x · {rm} partite</span></div>', unsafe_allow_html=True)
    
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1: st.metric("🏠 Casa", f"{pr.get('home_cards_avg',0):.2f}")
    with cc2: st.metric("✈️ Trasf.", f"{pr.get('away_cards_avg',0):.2f}")
    bc = pr.get('expected_cards_base', pr.get('expected_cards',0))
    with cc3: st.metric("📊 Base", f"{bc:.2f}")
    with cc4: st.metric("📊 Attesi", f"{pr.get('expected_cards',0):.2f}", delta=f"{pr.get('expected_cards',0)-bc:+.2f}" if pr.get("referee_found") else None)
    st.plotly_chart(chart_cards(pr), use_container_width=True, config={'displayModeBar': False})
    
    ctbl = '<table><tr><th>Linea</th><th>OVER</th><th>UNDER</th><th>Quota O</th><th>Quota U</th></tr>'
    for ln in [2.5,3.5,4.5,5.5,6.5]:
        cop = pr.get(f"cards_over_{ln}",0)*100; cup = pr.get(f"cards_under_{ln}",0)*100
        qo = f"{100/cop:.2f}" if cop > 0 else "—"
        qu = f"{100/cup:.2f}" if cup > 0 else "—"
        ctbl += f'<tr><td>{ln}</td><td>{cop:.1f}%</td><td>{cup:.1f}%</td><td>{qo}</td><td>{qu}</td></tr>'
    st.markdown(ctbl+'</table>', unsafe_allow_html=True)
    st.markdown("---")
    
    # H2H
    st.markdown("#### ⚔️ Scontri Diretti")
    if h2h.get("matches",0) > 0:
        h1_,h2_,h3_,h4_ = st.columns(4)
        with h1_: st.metric("Partite", h2h["matches"])
        with h2_: st.metric(f"{hd}", h2h["team1_wins"])
        with h3_: st.metric("Pareggi", h2h["draws"])
        with h4_: st.metric(f"{ad}", h2h["team2_wins"])
        st.markdown(f'<span style="color:#8b949e;">Media gol: {h2h["avg_goals"]}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#8b949e;">Nessuno scontro diretto recente</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Tiri
    st.markdown("#### 🎯 Media Tiri (Ultime 5)")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f'<div style="background:#161b22; border:1px solid #21262d; border-left:3px solid #1f6feb; padding:14px; border-radius:10px; text-align:center;"><div style="color:#e6edf3; font-weight:600; margin-bottom:6px;">{hd}</div><div style="font-size:1.8rem; font-weight:800; color:#e6edf3;">{hshots.get("shots_avg","—")}</div><div style="font-size:0.8rem; color:#8b949e;">tiri/partita</div><div style="font-size:1.3rem; font-weight:700; color:#238636; margin-top:4px;">{hshots.get("shots_on_target_avg","—")}</div><div style="font-size:0.8rem; color:#8b949e;">in porta</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div style="background:#161b22; border:1px solid #21262d; border-left:3px solid #da3633; padding:14px; border-radius:10px; text-align:center;"><div style="color:#e6edf3; font-weight:600; margin-bottom:6px;">{ad}</div><div style="font-size:1.8rem; font-weight:800; color:#e6edf3;">{ashots.get("shots_avg","—")}</div><div style="font-size:0.8rem; color:#8b949e;">tiri/partita</div><div style="font-size:1.3rem; font-weight:700; color:#da3633; margin-top:4px;">{ashots.get("shots_on_target_avg","—")}</div><div style="font-size:0.8rem; color:#8b949e;">in porta</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Stats
    st.markdown("#### 📋 Statistiche")
    t1, t2 = st.columns(2)
    with t1: show_team_stats(hs, hd, True)
    with t2: show_team_stats(aws, ad, False)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Opzioni")
    st.markdown("---")
    if st.button("🔄 Refresh Dati", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.markdown("---")
    adv_date = st.date_input("📅 Vai a data specifica", value=date.today(),
        min_value=date.today()-timedelta(days=7), max_value=date.today()+timedelta(days=14))
    if st.button("Vai", use_container_width=True, key="goto_date"):
        st.session_state["sel_date"] = adv_date.strftime("%Y-%m-%d"); st.rerun()
    
    # === TRACKER PRONOSTICI ===
    st.markdown("---")
    st.markdown("### 📊 Tracker")
    n_tracked = len(st.session_state.get("tracked_preds", []))
    st.markdown(f'<span style="color:#8b949e;">{n_tracked} pronostici salvati in questa sessione</span>', unsafe_allow_html=True)
    
    if n_tracked > 0:
        import pandas as pd
        # Prepara DataFrame per download
        rows = [{k: v for k, v in r.items() if k != "_key"} for r in st.session_state["tracked_preds"]]
        df_export = pd.DataFrame(rows)
        
        # Genera Excel in memoria
        buffer = io.BytesIO()
        try:
            df_export.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            st.download_button(
                label="📥 Scarica Excel",
                data=buffer,
                file_name=f"pronostici_{date.today().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception:
            # Fallback CSV se openpyxl non disponibile
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica CSV",
                data=csv_data,
                file_name=f"pronostici_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.button("🗑️ Svuota tracker", use_container_width=True):
            st.session_state["tracked_preds"] = []; st.rerun()


# ============================================================
# INIT DATE
# ============================================================
if "sel_date" not in st.session_state:
    st.session_state["sel_date"] = date.today().strftime("%Y-%m-%d")

sel = datetime.strptime(st.session_state["sel_date"], "%Y-%m-%d").date()


# ============================================================
# MAIN
# ============================================================

# Logo
cl1, cl2, cl3 = st.columns([1,1,1])
with cl2:
    try:
        with open("logo.png","rb") as f: ld = base64.b64encode(f.read()).decode()
        st.markdown(f'<div style="display:flex; justify-content:center; margin-bottom:0.5rem;"><img src="data:image/png;base64,{ld}" style="width:200px;"></div>', unsafe_allow_html=True)
    except:
        st.markdown('<h1 style="font-family:Outfit; text-align:center; font-size:2.5rem; letter-spacing:3px;">BETENGINE</h1>', unsafe_allow_html=True)

st.markdown('<p style="text-align:center; color:#8b949e; margin-bottom:1rem; font-size:0.95rem;">Analisi probabilistica partite di calcio</p>', unsafe_allow_html=True)

# === CALENDARIO A PULSANTI ===
gi_short = ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"]
gi_full = ["Lunedì","Martedì","Mercoledì","Giovedì","Venerdì","Sabato","Domenica"]

# Genera 7 giorni: da ieri a +5 giorni
cal_days = [date.today() + timedelta(days=i) for i in range(-1, 6)]

cal_cols = st.columns(len(cal_days))
for i, d in enumerate(cal_days):
    with cal_cols[i]:
        is_today = d == date.today()
        is_selected = d == sel
        
        if d == date.today(): label = "Oggi"
        elif d == date.today() + timedelta(days=1): label = "Domani"
        elif d == date.today() - timedelta(days=1): label = "Ieri"
        else: label = gi_short[d.weekday()]
        
        day_num = d.strftime("%d/%m")
        btn_label = f"{label}\n{day_num}"
        
        if st.button(btn_label, key=f"cal_{d}", use_container_width=True,
                     type="primary" if is_selected else "secondary"):
            st.session_state["sel_date"] = d.strftime("%Y-%m-%d")
            st.rerun()

st.markdown("")  # spacer

# Label data selezionata
if sel == date.today(): dl = "Oggi"
elif sel == date.today()+timedelta(days=1): dl = "Domani"
elif sel == date.today()-timedelta(days=1): dl = "Ieri"
else: dl = gi_full[sel.weekday()]
dd = sel.strftime("%d/%m/%Y")
ds = sel.strftime("%Y-%m-%d")

st.markdown(f'<div style="text-align:center; margin-bottom:1.5rem;"><span class="be-date-pill active">📅 {dl}, {dd}</span></div>', unsafe_allow_html=True)

# API check
if not API_FOOTBALL_KEY or API_FOOTBALL_KEY == "INSERISCI_QUI_LA_TUA_API_KEY":
    st.warning("⚠️ Configura la API key in config.py"); st.stop()

# Fetch
with st.spinner("⚽ Caricamento partite..."):
    fixtures = fetch_todays_fixtures(API_FOOTBALL_KEY, ds)

if not fixtures:
    st.markdown(f'<div class="be-empty"><div class="icon">⚽</div><div class="msg">Nessuna partita per {dl.lower()}</div><div class="sub">Serie A · Premier League · LaLiga · Bundesliga · Ligue 1 · Primeira Liga · Eredivisie</div></div>', unsafe_allow_html=True)
else:
    total = sum(len(f) for f in fixtures.values())
    st.markdown(f'<div class="be-total">{total} partite in programma</div>', unsafe_allow_html=True)
    
    for lid, fxs in fixtures.items():
        ln = LEAGUES.get(lid, {}).get("name", f"Lega {lid}")
        st.markdown(f'<div class="be-league"><h3>{ln}</h3><span class="count">{len(fxs)}</span></div>', unsafe_allow_html=True)
        
        for fx in fxs:
            hd = dn(fx["home_name"]); ad = dn(fx["away_name"])
            s = fx["status"]; t = fx["time"]
            if s == "NS": badge = f"🕐 {t}"
            elif s in ("1H","2H","HT","ET","P","LIVE"): badge = f"🔴 {fx.get('score_home',0)}–{fx.get('score_away',0)}"
            elif s == "FT": badge = f"✅ {fx.get('score_home',0)}–{fx.get('score_away',0)}"
            else: badge = f"🕐 {t}"
            
            with st.expander(f"{badge}   {hd}  vs  {ad}", expanded=False):
                fk = f"a_{fx['fixture_id']}"
                if fk not in st.session_state: st.session_state[fk] = False
                if not st.session_state[fk]:
                    ref_txt = fx.get('referee','') or '—'
                    st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center; padding:4px 0;"><span style="color:#e6edf3; font-weight:500;">{hd} vs {ad}</span><span style="color:#8b949e; font-size:0.85rem;">⏰ {t} · 👨‍⚖️ {ref_txt}</span></div>', unsafe_allow_html=True)
                    if st.button("🔮 ANALIZZA", key=f"b_{fx['fixture_id']}", use_container_width=True):
                        st.session_state[fk] = True; st.rerun()
                else:
                    show_analysis(fx, lid, ln)

st.markdown('<div class="be-footer">BetEngine v3.1 · Dixon-Coles · Probability Engine v3</div>', unsafe_allow_html=True)
