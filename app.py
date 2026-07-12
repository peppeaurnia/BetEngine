"""
⚽ BETENGINE v4.0 — Analisi Probabilistica Partite
====================================================
REFACTOR UI (Luglio 2026):
- Tema gestito da .streamlit/config.toml → CSS custom ridotto dell'80%
  (niente più selettori su data-testid interni di Streamlit)
- st.fragment: analizzare una partita NON ricarica più tutta la pagina
- Analisi organizzata in tab: Pronostici / Mercati / Risultati / Cartellini / Stats
- Salvataggio nel tracker ESPLICITO (bottone), non più automatico alla visualizzazione
- Tabelle native st.dataframe con barre di progresso al posto dell'HTML grezzo
- Colori semantici coerenti: blu = casa, arancio = trasferta, grigio = pareggio.
  Verde/rosso riservati ai giudizi di valore (EV, forma, severità arbitro)
- Etichetta "xG" corretta in "Gol attesi (modello)"
- Risultato della singola analisi cachato in session_state (niente ri-fetch)

Richiede: streamlit >= 1.37 (per st.fragment). Con versioni più vecchie
l'app funziona comunque, ma con rerun completi.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import base64
import os

# ============================================================
# CONFIG PAGINA
# ============================================================
st.set_page_config(
    page_title="BetEngine",
    page_icon="⚽",
    layout="centered",              # max-width nativa ~730px: addio CSS del container
    initial_sidebar_state="collapsed",
)

from probability_engine import (calculate_match_probabilities,
                                assess_prediction_quality,
                                apply_market_anchoring)
from data_fetcher import (LEAGUES, get_match_stats, get_head_to_head,
                          get_team_shots_avg, get_current_season,
                          fetch_todays_fixtures)
from config import API_FOOTBALL_KEY

try:
    from odds_api import fetch_match_odds
    ODDS_API_AVAILABLE = True
except ImportError:
    ODDS_API_AVAILABLE = False

import storage  # tracker persistente su SQLite (betengine.db)
import calibration  # calibrazione empirica data-driven (punto 4)

try:
    import dc_model  # motore Dixon-Coles MLE (punto 5) — richiede scipy
    DC_AVAILABLE = True
except ImportError:
    DC_AVAILABLE = False

# st.fragment: fallback trasparente per Streamlit < 1.37
fragment = getattr(st, "fragment", None) or (lambda f: f)

# ============================================================
# PALETTE SEMANTICA
# ============================================================
C_HOME = "#2f81f7"   # blu       → squadra di casa
C_AWAY = "#f78166"   # arancio   → squadra ospite
C_DRAW = "#8b949e"   # grigio    → pareggio / neutro
C_GOOD = "#3fb950"   # verde     → giudizio positivo (EV+, forma buona)
C_BAD  = "#f85149"   # rosso     → giudizio negativo
C_WARN = "#d29922"   # ambra     → attenzione / medio
C_TXT  = "#e6edf3"
C_MUT  = "#8b949e"
C_BG2  = "#161b22"
C_BRD  = "#30363d"

# ============================================================
# STATE
# ============================================================
if "sel_date" not in st.session_state:
    st.session_state["sel_date"] = date.today().strftime("%Y-%m-%d")

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


# ============================================================
# CSS MINIMO — solo classi custom, nessun hack su data-testid
# ============================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Outfit:wght@600;700;800&display=swap');

    html, body {{ font-family: 'DM Sans', sans-serif; }}
    h1, h2, h3, h4 {{ font-family: 'Outfit', sans-serif !important; }}

    .be-sub {{ text-align:center; color:{C_MUT}; margin-bottom:1rem; font-size:.95rem; }}
    .be-total {{ text-align:center; margin-bottom:16px; color:{C_MUT}; font-size:.9rem; }}
    .be-league {{
        display:flex; align-items:baseline; gap:8px;
        padding:12px 0 6px 0; margin-top:18px;
        border-bottom:1px solid {C_BRD};
    }}
    .be-league h3 {{ margin:0; font-size:1.05rem; }}
    .be-league .count {{ color:{C_MUT}; font-size:.8rem; }}
    .be-empty {{ text-align:center; padding:60px 20px; }}
    .be-empty .icon {{ font-size:2.5rem; margin-bottom:12px; }}
    .be-empty .msg {{ font-size:1.1rem; color:{C_MUT}; }}
    .be-empty .sub {{ font-size:.85rem; color:#484f58; margin-top:8px; }}
    .be-footer {{
        text-align:center; color:#484f58; font-size:.75rem;
        margin-top:40px; padding:15px 0; border-top:1px solid {C_BRD};
    }}

    .be-card {{
        background:{C_BG2}; border:1px solid {C_BRD};
        border-radius:12px; padding:16px; text-align:center;
    }}
    .be-badge {{
        display:inline-block; padding:3px 12px; border-radius:20px;
        font-size:.72rem; font-weight:600; letter-spacing:.03em;
        color:#fff;
    }}
    .be-chip {{
        color:#fff; padding:8px 4px; border-radius:8px;
        text-align:center; margin:4px 0;
    }}
    .be-chip .score {{ font-size:1.25rem; font-weight:700; }}
    .be-chip .pct {{ font-size:.78rem; opacity:.85; }}
    .be-note {{
        background:{C_BG2}; border-left:3px solid {C_BRD};
        padding:10px 14px; border-radius:8px; margin-bottom:12px;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# GRAFICI
# ============================================================
CHART_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C_TXT, family="DM Sans"))
PLOTLY_CFG = {"displayModeBar": False, "staticPlot": False}


def chart_1x2(probs, home, away):
    """Una sola barra impilata al 100%: la ripartizione si legge a colpo d'occhio."""
    segs = [
        (f"1 · {home}", probs.get("p_home", 0) * 100, C_HOME),
        ("X · Pareggio", probs.get("p_draw", 0) * 100, C_DRAW),
        (f"2 · {away}", probs.get("p_away", 0) * 100, C_AWAY),
    ]
    fig = go.Figure()
    for name, val, color in segs:
        fig.add_trace(go.Bar(
            y=[""], x=[val], name=name, orientation="h",
            marker=dict(color=color),
            text=f"{val:.1f}%", textposition="inside", insidetextanchor="middle",
            textfont=dict(size=14, color="white"),
            hovertemplate=f"{name}: {val:.1f}%<extra></extra>",
        ))
    fig.update_layout(**CHART_LAYOUT, barmode="stack", height=120,
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(range=[0, 100], visible=False),
                      yaxis=dict(visible=False),
                      legend=dict(orientation="h", yanchor="top", y=-0.05,
                                  x=0.5, xanchor="center"))
    return fig


def chart_over_lines(probs, lines, key_fmt, color):
    """Solo la probabilità Over per linea (Under = complemento, è ridondante).
    Linea tratteggiata al 50% come riferimento."""
    vals = [probs.get(key_fmt.format(l), 0) * 100 for l in lines]
    fig = go.Figure(go.Bar(
        x=[f"Over {l}" for l in lines], y=vals, marker_color=color,
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="#484f58",
                  annotation_text="50%", annotation_font_color=C_MUT)
    fig.update_layout(**CHART_LAYOUT, height=280, showlegend=False,
                      xaxis=dict(gridcolor="#21262d"),
                      yaxis=dict(range=[0, 105], gridcolor="#21262d",
                                 ticksuffix="%"),
                      margin=dict(l=10, r=10, t=20, b=10))
    return fig


def chart_heatmap(matrix, home, away):
    M = matrix[:7, :7] * 100
    fig = go.Figure(go.Heatmap(
        z=M, x=[str(i) for i in range(7)], y=[str(i) for i in range(7)],
        colorscale=[[0, "#0d1117"], [0.5, "#1f6feb"], [1, "#58a6ff"]],
        text=[[f"{M[i, j]:.1f}%" if M[i, j] > 0.5 else "" for j in range(7)]
              for i in range(7)],
        texttemplate="%{text}", textfont=dict(size=10, color="white"),
        hovertemplate=f"{home} %{{y}} - %{{x}} {away}<br>Prob: %{{z:.2f}}%<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(**CHART_LAYOUT, height=380,
                      xaxis_title=f"Gol {away}", yaxis_title=f"Gol {home}",
                      yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=10, b=40))
    return fig


# ============================================================
# TABELLE NATIVE (st.dataframe + ProgressColumn)
# ============================================================

def ou_dataframe(probs, lines, over_fmt, under_fmt):
    rows = []
    for l in lines:
        op = probs.get(over_fmt.format(l), 0) * 100
        up = probs.get(under_fmt.format(l), 0) * 100
        rows.append({
            "Linea": f"{l}",
            "Over": op,
            "Under": up,
            "Quota O": round(100 / op, 2) if op > 0 else None,
            "Quota U": round(100 / up, 2) if up > 0 else None,
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={
            "Over": st.column_config.ProgressColumn(
                "Over", format="%.1f%%", min_value=0, max_value=100),
            "Under": st.column_config.ProgressColumn(
                "Under", format="%.1f%%", min_value=0, max_value=100),
            "Quota O": st.column_config.NumberColumn("Quota O", format="%.2f"),
            "Quota U": st.column_config.NumberColumn("Quota U", format="%.2f"),
        },
    )


# ============================================================
# PRONOSTICI CONSIGLIATI (logica del modello INVARIATA — v7)
# ============================================================

def _fair_market_prob(short, odds_data):
    """
    Probabilità implicita FAIR della selezione (margine bookmaker rimosso
    normalizzando il gruppo di esiti). Serve come benchmark: in Performance
    si confronta il Brier del modello con quello del mercato.
    Ritorna None se il gruppo di quote non è completo.
    """
    if not odds_data or short not in odds_data:
        return None
    groups = {"1": ("1", "X", "2"), "X": ("1", "X", "2"), "2": ("1", "X", "2"),
              "O2.5": ("O2.5", "U2.5"), "U2.5": ("O2.5", "U2.5"),
              "GG": ("GG", "NG"), "NG": ("GG", "NG")}
    if short.endswith("cards"):
        partner = ("U" if short[0] == "O" else "O") + short[1:]
        group = (short, partner)
    else:
        group = groups.get(short)
    if not group:
        return None
    try:
        inv = {}
        for g in group:
            o = float(odds_data.get(g, 0))
            if o <= 1.0:
                return None
            inv[g] = 1.0 / o
        total = sum(inv.values())
        return round(inv[short] / total * 100, 1) if total > 0 else None
    except (TypeError, ValueError):
        return None


def calc_top_preds(probs, home, away, odds_data=None):
    """
    Seleziona i pronostici consigliati basandosi sulla probabilità del modello. (v8)

    LOGICA:
    - Consiglia un pronostico SOLO se la sua probabilità supera una soglia di mercato
    - v8: la probabilità viene prima corretta con la CALIBRAZIONE EMPIRICA
      (calibration.py), stimata sui pronostici realmente chiusi nel database.
      Finché il campione è insufficiente (<150 chiusi) la correzione è nulla.
    - v8: RIMOSSA la dead zone 65-70% della v7. Era stata stimata su 41 casi:
      statisticamente indifendibile (nessun meccanismo plausibile per cui il
      modello sia calibrato a 60% e a 72% ma rotto a 67%). La calibrazione
      isotonica su un campione serio è il sostituto corretto: se quella fascia
      è davvero sovrastimata, la curva la abbasserà in modo continuo.
    - Massimo 1 pronostico per famiglia (1X2, Goals, BTTS, Cards)
    - Mostra 0-3 pronostici, mai forzati
    - Le quote (se disponibili) sono info aggiuntiva nelle card, non un filtro
    """
    TH_1X2 = 55.0
    TH_OU = 65.0
    TH_BTTS = 65.0
    TH_CARDS_35 = 78.0
    TH_CARDS_45 = 70.0

    has_odds = odds_data and len(odds_data) > 0
    candidates = []

    def add_candidate(short, key, name, mt, family, threshold, anchorable=True):
        prob_raw = probs.get(key, 0) * 100
        prob_pure = probs.get(f"{key}_pure", probs.get(key, 0)) * 100 if anchorable else prob_raw
        # Calibrazione empirica: corregge la probabilità con la frequenza
        # storica reale (identità: nessuna correzione finché non ci sono dati)
        prob = calibration.apply(prob_raw, market=mt)
        if prob < threshold:
            return
        c = {"name": name, "short": short, "mt": mt, "family": family,
             "prob": prob, "prob_raw": prob_raw, "prob_pure": prob_pure,
             "threshold": threshold, "odds": None, "ev_pct": None,
             "prob_market": _fair_market_prob(short, odds_data) if has_odds else None}
        if has_odds and short in odds_data:
            c["odds"] = float(odds_data[short])
            c["ev_pct"] = (prob_pure / 100.0 * c["odds"] - 1.0) * 100
        candidates.append(c)

    # 1X2
    add_candidate("1", "p_home", home, "1X2", "1X2", TH_1X2)
    add_candidate("X", "p_draw", "Pareggio", "1X2", "1X2", TH_1X2)
    add_candidate("2", "p_away", away, "1X2", "1X2", TH_1X2)
    # Over/Under 2.5
    add_candidate("O2.5", "over_2.5", "Over 2.5", "OU", "Goals", TH_OU)
    add_candidate("U2.5", "under_2.5", "Under 2.5", "OU", "Goals", TH_OU)
    # BTTS
    add_candidate("GG", "p_btts_yes", "Gol (GG)", "BTTS", "BTTS", TH_BTTS)
    add_candidate("NG", "p_btts_no", "NoGol (NG)", "BTTS", "BTTS", TH_BTTS)
    # Cartellini
    for line, th in [(3.5, TH_CARDS_35), (4.5, TH_CARDS_45)]:
        for side, pfx in [("over", "Cart. O"), ("under", "Cart. U")]:
            add_candidate(f"{side[0].upper()}{line}cards", f"cards_{side}_{line}",
                          f"{pfx}{line}", "Cards", "Cards", th, anchorable=False)

    # Uno per famiglia (il migliore per probabilità)
    by_family = {}
    for c in candidates:
        fam = c["family"]
        if fam not in by_family or c["prob"] > by_family[fam]["prob"]:
            by_family[fam] = c

    top = sorted(by_family.values(), key=lambda x: -x["prob"])

    for p in top:
        prob = p["prob"]
        if prob >= 80:   p["stars"] = 5
        elif prob >= 72: p["stars"] = 4
        elif prob >= 65: p["stars"] = 3
        elif prob >= 58: p["stars"] = 2
        else:            p["stars"] = 1

    return top[:3]


def show_top_preds(preds):
    st.markdown("##### Pronostici consigliati")

    if not preds:
        st.markdown(
            f'<div class="be-card"><div style="color:{C_MUT}; font-size:.9rem;">'
            f'Nessun pronostico con confidenza sufficiente</div>'
            f'<div style="color:#484f58; font-size:.8rem; margin-top:4px;">'
            f'Nessun mercato supera le soglie minime del modello.</div></div>',
            unsafe_allow_html=True)
        return

    n = min(len(preds), 3)
    cols = [st.columns([1, 2, 1])[1]] if n == 1 else st.columns(n)

    for i in range(n):
        p = preds[i]
        with cols[i]:
            prob = p["prob"]
            if prob >= 75:
                badge_bg, badge_label = C_GOOD, "ALTA FIDUCIA"
            elif prob >= 65:
                badge_bg, badge_label = "#2ea043", "BUONA FIDUCIA"
            else:
                badge_bg, badge_label = C_WARN, "FIDUCIA MEDIA"

            conf_dots = "●" * p["stars"] + "○" * (5 - p["stars"])

            html = '<div class="be-card">'
            html += (f'<div style="font-size:.78rem; color:{C_MUT}; '
                     f'text-transform:uppercase; letter-spacing:.05em;">{p["mt"]}</div>')
            html += (f'<div style="font-size:.95rem; font-weight:600; color:{C_TXT}; '
                     f'margin:6px 0;">{p["name"]}</div>')
            html += (f'<div style="font-size:1.6rem; font-weight:800; '
                     f'color:{C_TXT};">{prob:.1f}%</div>')
            html += (f'<div class="be-badge" style="background:{badge_bg}; '
                     f'margin:6px 0;">{badge_label}</div>')

            if p.get("odds") is not None:
                ev = p.get("ev_pct")
                ev_color = C_GOOD if ev and ev > 5 else C_WARN if ev and ev > -3 else C_MUT
                ev_str = f"EV {ev:+.1f}%" if ev is not None else ""
                html += (f'<div style="font-size:.78rem; color:{C_MUT}; margin-top:6px; '
                         f'padding-top:6px; border-top:1px solid #21262d;">'
                         f'Quota <strong style="color:{C_TXT};">{p["odds"]:.2f}</strong>')
                if ev_str:
                    html += (f'&nbsp;·&nbsp;<span style="color:{ev_color}; '
                             f'font-weight:600;">{ev_str}</span>')
                html += "</div>"

            html += (f'<div style="font-size:.7rem; color:{C_MUT}; letter-spacing:.2em; '
                     f'margin-top:6px;">{conf_dots}</div>')
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)


# ============================================================
# COMPONENTI DISPLAY
# ============================================================

def show_exact_scores(scores, home, away):
    cols = st.columns(5)
    for i, (hg, ag, prob) in enumerate(scores[:10]):
        with cols[i % 5]:
            c = C_HOME if hg > ag else C_AWAY if ag > hg else "#484f58"
            st.markdown(
                f'<div class="be-chip" style="background:{c};">'
                f'<div class="score">{hg}-{ag}</div>'
                f'<div class="pct">{prob:.1f}%</div></div>',
                unsafe_allow_html=True)
    st.caption(f"🟦 vittoria {home} · 🟧 vittoria {away} · grigio pareggio")


def show_team_stats(stats, name, is_home):
    pos = "Casa" if is_home else "Trasferta"
    att = stats.get("attack_home" if is_home else "attack_away", 1.0)
    df_ = stats.get("defense_home" if is_home else "defense_away", 1.0)
    form = stats.get("form_factor", 1.0)
    rank = stats.get("rank", "N/A")
    fs = stats.get("form_string", "")
    # verde/rosso qui sono corretti: è un giudizio di performance
    ac = C_GOOD if att > 1.1 else C_BAD if att < 0.9 else C_WARN
    dc = C_GOOD if df_ < 0.9 else C_BAD if df_ > 1.1 else C_WARN
    border = C_HOME if is_home else C_AWAY
    st.markdown(f'''<div class="be-card" style="text-align:left; border-left:3px solid {border};">
        <h4 style="margin:0 0 10px 0; font-size:1rem;">{name}</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div><small style="font-size:.7rem; color:{C_MUT};">Attacco ({pos})</small>
                 <div style="font-size:1.1rem; color:{ac}; font-weight:700;">{att:.2f}</div></div>
            <div><small style="font-size:.7rem; color:{C_MUT};">Difesa ({pos})</small>
                 <div style="font-size:1.1rem; color:{dc}; font-weight:700;">{df_:.2f}</div></div>
            <div><small style="font-size:.7rem; color:{C_MUT};">Forma</small>
                 <div style="font-size:1rem;">{form:.2f} <small style="color:{C_MUT};">({fs[-5:] if fs else "—"})</small></div></div>
            <div><small style="font-size:.7rem; color:{C_MUT};">Classifica</small>
                 <div style="font-size:1.1rem;">{rank}°</div></div>
        </div></div>''', unsafe_allow_html=True)


def build_tracker_rows(top_preds, fix, lid, lname, hd, ad, anchored, engine="v4"):
    """Costruisce le righe da salvare su SQLite (solo su richiesta esplicita).
    fixture_id e selection_code servono all'updater per chiudere gli esiti;
    prob_market ed engine servono per benchmark vs mercato e confronto A/B."""
    match_date = fix.get("date_raw", st.session_state.get("sel_date", ""))
    clean_league = lname.split(" ", 1)[-1] if lname and " " in lname else lname
    rows = []
    for pred in top_preds:
        rows.append({
            "match_date": match_date,
            "fixture_id": fix["fixture_id"],
            "league_id": lid,
            "league": clean_league,
            "home": hd,
            "away": ad,
            "market": pred.get("mt", ""),
            "selection": pred.get("name", ""),
            "selection_code": pred.get("short", ""),
            "prob": round(pred.get("prob", 0), 1),
            "prob_pure": round(pred.get("prob_pure", 0), 1),
            "prob_market": pred.get("prob_market"),
            "odds": round(pred["odds"], 2) if pred.get("odds") else None,
            "ev_pct": round(pred["ev_pct"], 1) if pred.get("ev_pct") is not None else None,
            "stars": pred.get("stars", 0),
            "anchored": anchored,
            "engine": engine,
        })
    return rows


# ============================================================
# ANALISI PARTITA (con cache in session_state)
# ============================================================

def run_analysis(fix, lid, lname):
    """Recupera dati, calcola probabilità e ancoraggio. Eseguito UNA volta per fixture."""
    api = API_FOOTBALL_KEY
    season = get_current_season()
    hid, aid = fix["home_id"], fix["away_id"]
    ref = fix.get("referee", "")

    hs, aws, li = get_match_stats(api, hid, aid, lid, season)
    h2h = get_head_to_head(api, hid, aid, last_n=10)
    hshots = get_team_shots_avg(api, hid, lid, season, last_n=5)
    ashots = get_team_shots_avg(api, aid, lid, season, last_n=5)

    pr = calculate_match_probabilities(
        hs, aws, lid, h2h_data=h2h, home_shots=hshots, away_shots=ashots,
        referee_name=ref if ref else None, league_name=lname)

    # --- Motore Dixon-Coles (punto 5): se i parametri della lega sono
    # fittati e freschi, i mercati GOL vengono sovrascritti con il fit MLE.
    # I CARTELLINI restano al modello v4 (il DC modella solo i gol).
    # L'engine usato viene tracciato per il confronto A/B in Performance.
    engine_used = "v4"
    if DC_AVAILABLE:
        try:
            dc = dc_model.dc_match_probabilities(lid, hid, aid)
        except Exception:
            dc = None
        if dc:
            pr.update(dc)
            engine_used = "dc"

    odds = {}
    if ODDS_API_AVAILABLE:
        try:
            odds = fetch_match_odds(fix["home_name"], fix["away_name"], lid)
        except Exception:
            odds = {}

    anchored = False
    if odds:
        pr = apply_market_anchoring(pr, odds)
        anchored = True

    q = assess_prediction_quality(hs, aws)

    return {"pr": pr, "odds": odds, "anchored": anchored, "q": q,
            "engine": engine_used,
            "h2h": h2h, "hshots": hshots, "ashots": ashots,
            "hs": hs, "aws": aws}


def show_analysis(fix, lid, lname):
    hd, ad = dn(fix["home_name"]), dn(fix["away_name"])
    fid = fix["fixture_id"]

    # Cache dell'analisi: rianalizzare la stessa partita è istantaneo
    cache_key = f"an_{fid}"
    if cache_key not in st.session_state:
        with st.spinner("Analisi in corso…"):
            st.session_state[cache_key] = run_analysis(fix, lid, lname)
    data = st.session_state[cache_key]

    pr, odds, anchored, q = data["pr"], data["odds"], data["anchored"], data["q"]
    engine_used = data.get("engine", "v4")
    h2h, hshots, ashots = data["h2h"], data["hshots"], data["ashots"]
    hs, aws = data["hs"], data["aws"]

    # --- Barra affidabilità (soglie allineate ai livelli di assess: 85/65) ---
    qcol = C_GOOD if q["score"] >= 85 else C_WARN if q["score"] >= 65 else C_BAD
    anch = (f'<span class="be-badge" style="background:#1f6feb; margin-left:8px;">Market anchored</span>'
            if anchored else
            f'<span class="be-badge" style="background:#484f58; margin-left:8px;">Solo modello</span>')
    eng_badge = (f'<span class="be-badge" style="background:#8957e5; margin-left:8px;">'
                 f'Engine DC</span>' if engine_used == "dc" else "")
    st.markdown(
        f'<div class="be-note" style="border-left-color:{qcol};">'
        f'<span style="color:{C_TXT}; font-weight:600;">Affidabilità: {q["level"]}</span> '
        f'<span style="color:{C_MUT};">({q["score"]}%)</span>{anch}{eng_badge}</div>',
        unsafe_allow_html=True)

    top_preds = calc_top_preds(pr, hd, ad, odds_data=odds)

    tab_pred, tab_mkt, tab_exact, tab_cards, tab_stats = st.tabs(
        ["Pronostici", "Mercati", "Risultati esatti", "Cartellini", "Statistiche"])

    # ------------------------------------------------ TAB PRONOSTICI
    with tab_pred:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Gol attesi casa", f"{pr['mu_home']:.2f}")
        with c2: st.metric("Gol attesi trasferta", f"{pr['mu_away']:.2f}")
        with c3: st.metric("Totale attesi", f"{pr['total_expected_goals']:.2f}")
        st.caption("Stime del modello (Poisson bivariata + Dixon-Coles), non xG da dati di tiro.")

        show_top_preds(top_preds)
        if calibration.is_active():
            st.caption("Probabilità corrette con la calibrazione empirica "
                       "(pagina Performance per i dettagli).")

        # Salvataggio ESPLICITO su SQLite: guardare un'analisi non sporca il DB
        if top_preds:
            saved_codes = storage.fixture_saved_selections(fid)
            pred_codes = {p.get("short", "") for p in top_preds}
            if pred_codes and pred_codes.issubset(saved_codes):
                st.caption("✓ Pronostici già salvati nel tracker.")
            elif st.button("Salva nel tracker", key=f"save_{fid}",
                           use_container_width=True):
                rows = build_tracker_rows(top_preds, fix, lid, lname,
                                          hd, ad, anchored, engine=engine_used)
                n_saved = storage.save_predictions(rows)
                st.success(f"{n_saved} pronostici salvati. Statistiche ed "
                           f"export nella pagina Performance.")

    # ------------------------------------------------ TAB MERCATI
    with tab_mkt:
        st.markdown("##### 1X2")
        st.plotly_chart(chart_1x2(pr, hd, ad), use_container_width=True, config=PLOTLY_CFG)

        st.markdown("##### BTTS e Over/Under 2.5")
        by = pr.get("p_btts_yes", 0) * 100
        o25 = pr.get("over_2.5", 0) * 100
        b1, b2 = st.columns(2)
        with b1:
            st.markdown(
                f'<div class="be-card"><div style="color:{C_MUT}; font-size:.85rem; '
                f'margin-bottom:8px;">Both Teams To Score</div>'
                f'<div style="display:flex; justify-content:center; gap:28px;">'
                f'<div><div style="font-size:1.5rem; font-weight:700; color:{C_TXT};">{by:.1f}%</div>'
                f'<div style="color:{C_MUT}; font-size:.8rem;">GG</div></div>'
                f'<div><div style="font-size:1.5rem; font-weight:700; color:{C_MUT};">{100-by:.1f}%</div>'
                f'<div style="color:{C_MUT}; font-size:.8rem;">NG</div></div>'
                f'</div></div>', unsafe_allow_html=True)
        with b2:
            st.markdown(
                f'<div class="be-card"><div style="color:{C_MUT}; font-size:.85rem; '
                f'margin-bottom:8px;">Over/Under 2.5</div>'
                f'<div style="display:flex; justify-content:center; gap:28px;">'
                f'<div><div style="font-size:1.5rem; font-weight:700; color:{C_TXT};">{o25:.1f}%</div>'
                f'<div style="color:{C_MUT}; font-size:.8rem;">Over</div></div>'
                f'<div><div style="font-size:1.5rem; font-weight:700; color:{C_MUT};">{100-o25:.1f}%</div>'
                f'<div style="color:{C_MUT}; font-size:.8rem;">Under</div></div>'
                f'</div></div>', unsafe_allow_html=True)

        st.markdown("##### Over/Under — tutte le linee")
        st.plotly_chart(chart_over_lines(pr, [1.5, 2.5, 3.5, 4.5], "over_{}", C_HOME),
                        use_container_width=True, config=PLOTLY_CFG)
        ou_dataframe(pr, [1.5, 2.5, 3.5, 4.5], "over_{}", "under_{}")

    # ------------------------------------------------ TAB RISULTATI ESATTI
    with tab_exact:
        st.plotly_chart(chart_heatmap(pr["matrix"], hd, ad),
                        use_container_width=True, config=PLOTLY_CFG)
        show_exact_scores(pr["exact_scores"], hd, ad)

    # ------------------------------------------------ TAB CARTELLINI
    with tab_cards:
        if pr.get("referee_found"):
            sv = pr.get("referee_severity", 1.0)
            rn = pr.get("referee_name", "")
            ra = pr.get("referee_avg_cards", 0)
            rm = pr.get("referee_matches", 0)
            rc = C_BAD if sv > 1.1 else C_GOOD if sv < 0.9 else C_WARN
            rl = "SEVERO" if sv > 1.1 else "PERMISSIVO" if sv < 0.9 else "NELLA MEDIA"
            st.markdown(
                f'<div class="be-note" style="border-left-color:{rc};">'
                f'<span style="color:{C_TXT}; font-weight:600;">Arbitro: {rn}</span>'
                f'<span class="be-badge" style="background:{rc}; margin-left:8px;">{rl}</span><br>'
                f'<span style="color:{C_MUT}; font-size:.85rem;">'
                f'{ra:.1f} cartellini/partita · fattore {sv:.2f}x · {rm} partite analizzate</span></div>',
                unsafe_allow_html=True)

        cc1, cc2, cc3, cc4 = st.columns(4)
        bc = pr.get("expected_cards_base", pr.get("expected_cards", 0))
        with cc1: st.metric("Media casa", f"{pr.get('home_cards_avg', 0):.2f}")
        with cc2: st.metric("Media trasferta", f"{pr.get('away_cards_avg', 0):.2f}")
        with cc3: st.metric("Base partita", f"{bc:.2f}")
        with cc4: st.metric("Attesi (con arbitro)", f"{pr.get('expected_cards', 0):.2f}",
                            delta=f"{pr.get('expected_cards', 0) - bc:+.2f}"
                            if pr.get("referee_found") else None)

        st.plotly_chart(chart_over_lines(pr, [2.5, 3.5, 4.5, 5.5, 6.5],
                                         "cards_over_{}", C_WARN),
                        use_container_width=True, config=PLOTLY_CFG)
        ou_dataframe(pr, [2.5, 3.5, 4.5, 5.5, 6.5], "cards_over_{}", "cards_under_{}")

    # ------------------------------------------------ TAB STATISTICHE
    with tab_stats:
        st.markdown("##### Scontri diretti")
        if h2h.get("matches", 0) > 0:
            h1_, h2_, h3_, h4_ = st.columns(4)
            with h1_: st.metric("Partite", h2h["matches"])
            with h2_: st.metric(hd, h2h["team1_wins"])
            with h3_: st.metric("Pareggi", h2h["draws"])
            with h4_: st.metric(ad, h2h["team2_wins"])
            st.caption(f"Media gol negli scontri diretti: {h2h['avg_goals']}")
        else:
            st.caption("Nessuno scontro diretto recente.")

        st.markdown("##### Media tiri (ultime 5)")
        s1, s2 = st.columns(2)
        for col, name, shots, color in [(s1, hd, hshots, C_HOME), (s2, ad, ashots, C_AWAY)]:
            with col:
                st.markdown(
                    f'<div class="be-card" style="border-left:3px solid {color};">'
                    f'<div style="color:{C_TXT}; font-weight:600; margin-bottom:6px;">{name}</div>'
                    f'<div style="font-size:1.7rem; font-weight:800; color:{C_TXT};">'
                    f'{shots.get("shots_avg", "—")}</div>'
                    f'<div style="font-size:.8rem; color:{C_MUT};">tiri/partita</div>'
                    f'<div style="font-size:1.2rem; font-weight:700; color:{color}; margin-top:4px;">'
                    f'{shots.get("shots_on_target_avg", "—")}</div>'
                    f'<div style="font-size:.8rem; color:{C_MUT};">in porta</div></div>',
                    unsafe_allow_html=True)

        st.markdown("##### Forze squadra")
        t1, t2 = st.columns(2)
        with t1: show_team_stats(hs, hd, True)
        with t2: show_team_stats(aws, ad, False)


# ============================================================
# CARD PARTITA — fragment: interagire qui NON ricarica la pagina
# ============================================================

@fragment
def match_card(fx, lid, lname):
    hd, ad = dn(fx["home_name"]), dn(fx["away_name"])
    s, t = fx["status"], fx["time"]
    if s == "NS":
        badge = f"🕐 {t}"
    elif s in ("1H", "2H", "HT", "ET", "P", "LIVE"):
        badge = f"🔴 {fx.get('score_home', 0)}–{fx.get('score_away', 0)}"
    elif s == "FT":
        badge = f"FT {fx.get('score_home', 0)}–{fx.get('score_away', 0)}"
    else:
        badge = f"🕐 {t}"

    with st.expander(f"{badge}   {hd}  vs  {ad}", expanded=False):
        fk = f"a_{fx['fixture_id']}"
        if not st.session_state.get(fk):
            ref_txt = fx.get("referee", "") or "—"
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; '
                f'align-items:center; padding:4px 0;">'
                f'<span style="color:{C_TXT}; font-weight:500;">{hd} vs {ad}</span>'
                f'<span style="color:{C_MUT}; font-size:.85rem;">'
                f'{t} · Arbitro: {ref_txt}</span></div>',
                unsafe_allow_html=True)
            if st.button("Analizza partita", key=f"b_{fx['fixture_id']}",
                         type="primary", use_container_width=True):
                st.session_state[fk] = True
        if st.session_state.get(fk):
            show_analysis(fx, lid, lname)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Opzioni")
    if st.button("Aggiorna dati", use_container_width=True):
        st.cache_data.clear()
        # svuota anche le analisi cachate
        for k in [k for k in st.session_state if k.startswith("an_")]:
            del st.session_state[k]
        st.rerun()

    st.markdown("---")
    adv_date = st.date_input("Vai a data specifica", value=date.today(),
                             min_value=date.today() - timedelta(days=7),
                             max_value=date.today() + timedelta(days=14))
    if st.button("Vai", use_container_width=True, key="goto_date"):
        st.session_state["sel_date"] = adv_date.strftime("%Y-%m-%d")
        st.rerun()

    # === TRACKER (persistente su SQLite) ===
    st.markdown("---")
    st.markdown("### Tracker")
    cnt = storage.counts()
    st.caption(f"{cnt['total']} pronostici salvati · "
               f"{cnt['pending']} in attesa · {cnt['settled']} chiusi")
    st.page_link("pages/1_Performance.py",
                 label="Performance e aggiornamento risultati", icon="📈")


# ============================================================
# HEADER
# ============================================================
_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
cl1, cl2, cl3 = st.columns([1, 1, 1])
with cl2:
    try:
        with open(_logo_path, "rb") as f:
            ld = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<div style="display:flex; justify-content:center; margin-bottom:.5rem;">'
            f'<img src="data:image/png;base64,{ld}" style="width:200px;"></div>',
            unsafe_allow_html=True)
    except Exception:
        st.markdown('<h1 style="text-align:center; font-size:2.3rem; '
                    'letter-spacing:3px;">BETENGINE</h1>', unsafe_allow_html=True)

st.markdown('<p class="be-sub">Analisi probabilistica partite di calcio</p>',
            unsafe_allow_html=True)

# ============================================================
# CALENDARIO
# ============================================================
sel = datetime.strptime(st.session_state["sel_date"], "%Y-%m-%d").date()
gi_short = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
gi_full = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

cal_days = [date.today() + timedelta(days=i) for i in range(-1, 6)]
cal_cols = st.columns(len(cal_days))
for i, d in enumerate(cal_days):
    with cal_cols[i]:
        if d == date.today():
            label = "Oggi"
        elif d == date.today() + timedelta(days=1):
            label = "Domani"
        elif d == date.today() - timedelta(days=1):
            label = "Ieri"
        else:
            label = gi_short[d.weekday()]
        if st.button(f"{label}\n{d.strftime('%d/%m')}", key=f"cal_{d}",
                     use_container_width=True,
                     type="primary" if d == sel else "secondary"):
            st.session_state["sel_date"] = d.strftime("%Y-%m-%d")
            st.rerun()

if sel == date.today():
    dl = "Oggi"
elif sel == date.today() + timedelta(days=1):
    dl = "Domani"
elif sel == date.today() - timedelta(days=1):
    dl = "Ieri"
else:
    dl = gi_full[sel.weekday()]

st.markdown(f'<p class="be-sub" style="margin-top:.5rem;">{dl}, '
            f'{sel.strftime("%d/%m/%Y")}</p>', unsafe_allow_html=True)

# ============================================================
# MAIN — LISTA PARTITE
# ============================================================
if not API_FOOTBALL_KEY or API_FOOTBALL_KEY == "INSERISCI_QUI_LA_TUA_API_KEY":
    st.warning("Configura la API key (st.secrets o config.py)")
    st.stop()

ds = sel.strftime("%Y-%m-%d")
with st.spinner("Caricamento partite…"):
    fixtures = fetch_todays_fixtures(API_FOOTBALL_KEY, ds)

if not fixtures:
    st.markdown(
        f'<div class="be-empty"><div class="icon">⚽</div>'
        f'<div class="msg">Nessuna partita per {dl.lower()}</div>'
        f'<div class="sub">Serie A · Premier League · LaLiga · Bundesliga · '
        f'Ligue 1 · Primeira Liga · Eredivisie</div></div>',
        unsafe_allow_html=True)
else:
    total = sum(len(f) for f in fixtures.values())
    st.markdown(f'<div class="be-total">{total} partite in programma</div>',
                unsafe_allow_html=True)

    for lid, fxs in fixtures.items():
        ln = LEAGUES.get(lid, {}).get("name", f"Lega {lid}")
        st.markdown(f'<div class="be-league"><h3>{ln}</h3>'
                    f'<span class="count">{len(fxs)} partite</span></div>',
                    unsafe_allow_html=True)
        for fx in fxs:
            match_card(fx, lid, ln)

st.markdown('<div class="be-footer">BetEngine v4.0 · Dixon-Coles · '
            'Probability Engine v4</div>', unsafe_allow_html=True)
