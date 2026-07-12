"""
📈 PERFORMANCE — Pagina di analisi dei pronostici salvati
==========================================================
Va nella cartella: <progetto>/pages/1_Performance.py
Streamlit la aggiunge automaticamente alla navigazione in sidebar.

Contenuti:
- Aggiornamento risultati con un click (via results_updater)
- Metriche globali: win rate, yield (stake flat 1u), Brier score
- Curva di calibrazione: probabilità dette dal modello vs frequenza reale
- Breakdown per mercato e per lega
- Equity curve e storico completo

Il Brier score misura la qualità delle probabilità (0 = perfetto,
0.25 = come lanciare una moneta sul 50/50): più informativo del solo
win rate, perché penalizza la sovra/sottoconfidenza.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import storage
from results_updater import update_results
from config import API_FOOTBALL_KEY

# ============================================================
# CONFIG + PALETTE (allineata ad app.py)
# ============================================================
st.set_page_config(page_title="Performance — BetEngine", page_icon="📈",
                   layout="centered")

C_HOME = "#2f81f7"
C_GOOD = "#3fb950"
C_BAD = "#f85149"
C_WARN = "#d29922"
C_TXT = "#e6edf3"
C_MUT = "#8b949e"

CHART_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C_TXT, family="DM Sans"))
PLOTLY_CFG = {"displayModeBar": False}

st.title("Performance")
st.caption("Tracciamento e calibrazione dei pronostici del modello")

# ============================================================
# AGGIORNAMENTO RISULTATI
# ============================================================
cnt = storage.counts()
m1, m2, m3 = st.columns(3)
m1.metric("Pronostici totali", cnt["total"])
m2.metric("In attesa di esito", cnt["pending"])
m3.metric("Chiusi", cnt["settled"])

if cnt["pending"] > 0:
    st.caption(f"Aggiornare i risultati costa ~1-2 chiamate API per partita "
               f"(tetto 40 per esecuzione, piano free: 100/giorno).")
    if st.button("Aggiorna risultati dalle partite giocate", type="primary",
                 use_container_width=True):
        with st.spinner("Recupero risultati…"):
            summary = update_results(API_FOOTBALL_KEY, max_api_calls=40)
        msg = (f"Partite controllate: {summary['checked']} · "
               f"pronostici chiusi: {summary['settled']} · "
               f"annullati: {summary['voided']} · "
               f"ancora in attesa: {summary['still_pending']} · "
               f"chiamate API usate: {summary['api_calls']}")
        if summary["capped"]:
            st.warning(msg + " — raggiunto il tetto chiamate, rilancia per "
                             "completare le rimanenti.")
        else:
            st.success(msg)
        st.rerun()

df = storage.all_predictions_df()

if df.empty:
    st.markdown("---")
    st.info("Nessun pronostico salvato. Analizza una partita nella pagina "
            "principale e usa **Salva nel tracker**: da quel momento tutto "
            "quello che salvi viene tracciato qui.")
    st.stop()

# ============================================================
# EXPORT
# ============================================================
with st.expander("Esporta dati"):
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica CSV completo", data=csv_data,
                       file_name="betengine_tracker.csv", mime="text/csv",
                       use_container_width=True)

st.markdown("---")

# ============================================================
# FILTRI
# ============================================================
settled = df[df["status"].isin(["won", "lost"])].copy()

if settled.empty:
    st.info("Nessun pronostico ancora chiuso: le statistiche compariranno "
            "dopo il primo aggiornamento risultati su partite giocate.")
    st.stop()

f1, f2, f3 = st.columns(3)
with f1:
    markets = sorted(settled["market"].dropna().unique().tolist())
    sel_markets = st.multiselect("Mercato", markets, default=markets)
with f2:
    leagues = sorted(settled["league"].dropna().unique().tolist())
    sel_leagues = st.multiselect("Lega", leagues, default=leagues)
with f3:
    period = st.selectbox("Periodo", ["Tutto", "Ultimi 30 giorni",
                                      "Ultimi 90 giorni"])

flt = settled[settled["market"].isin(sel_markets) &
              settled["league"].isin(sel_leagues)].copy()
if period != "Tutto":
    days = 30 if "30" in period else 90
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    flt = flt[flt["match_date"] >= cutoff]

if flt.empty:
    st.warning("Nessun pronostico chiuso con i filtri selezionati.")
    st.stop()

# ============================================================
# METRICHE GLOBALI
# ============================================================
flt["won"] = (flt["status"] == "won").astype(int)
n = len(flt)
wr = flt["won"].mean() * 100
avg_prob = flt["prob"].mean()
gap = wr - avg_prob
brier = float(np.mean((flt["prob"] / 100.0 - flt["won"]) ** 2))

# Yield: stake flat 1 unità, solo righe con quota
with_odds = flt[flt["odds"].notna() & (flt["odds"] > 1)].copy()
if not with_odds.empty:
    with_odds["profit"] = np.where(with_odds["won"] == 1,
                                   with_odds["odds"] - 1.0, -1.0)
    yield_pct = with_odds["profit"].sum() / len(with_odds) * 100
else:
    yield_pct = None

g1, g2, g3, g4 = st.columns(4)
g1.metric("Pronostici chiusi", n)
g2.metric("Win rate", f"{wr:.1f}%",
          delta=f"{gap:+.1f} pt vs probabilità media ({avg_prob:.1f}%)")
g3.metric("Yield (stake flat)",
          f"{yield_pct:+.1f}%" if yield_pct is not None else "—",
          help="Profitto per unità puntata, solo pronostici con quota. "
               f"Campione: {len(with_odds)} pronostici quotati.")
g4.metric("Brier score", f"{brier:.4f}",
          help="0 = probabilità perfette. Sotto 0.20 è buono per il calcio; "
               "0.25 equivale a dire sempre 50%.")

if n < 50:
    st.caption(f"⚠️ Campione piccolo ({n} pronostici): win rate e yield sono "
               f"ancora molto rumorosi. Sotto i ~200 pronostici, oscillazioni "
               f"di ±10 punti sono normali e non dicono nulla sul modello.")

st.markdown("---")

# ============================================================
# CURVA DI CALIBRAZIONE
# ============================================================
st.markdown("#### Calibrazione")
st.caption("Ogni punto: pronostici raggruppati per probabilità dichiarata. "
           "Se il modello è calibrato, i punti stanno sulla diagonale. "
           "Sopra = il modello si sottostima, sotto = si sovrastima.")

bins = np.arange(50, 101, 5)
flt["bin"] = pd.cut(flt["prob"], bins=bins, right=False)
cal = flt.groupby("bin", observed=True).agg(
    n=("won", "size"), pred=("prob", "mean"), real=("won", "mean")
).reset_index()
cal = cal[cal["n"] >= 5]  # niente punti su gruppi minuscoli
cal["real"] *= 100

fig = go.Figure()
fig.add_trace(go.Scatter(x=[50, 100], y=[50, 100], mode="lines",
                         line=dict(color="#484f58", dash="dot"),
                         name="Calibrazione perfetta", hoverinfo="skip"))
if not cal.empty:
    fig.add_trace(go.Scatter(
        x=cal["pred"], y=cal["real"], mode="markers+lines",
        marker=dict(size=np.clip(cal["n"] / 2, 8, 30), color=C_HOME,
                    line=dict(color="white", width=1)),
        line=dict(color=C_HOME, width=1),
        name="Modello",
        customdata=cal["n"],
        hovertemplate="Detto: %{x:.1f}%<br>Reale: %{y:.1f}%<br>"
                      "Campione: %{customdata}<extra></extra>"))
fig.update_layout(**CHART_LAYOUT, height=380,
                  xaxis=dict(title="Probabilità dichiarata", range=[48, 102],
                             gridcolor="#21262d", ticksuffix="%"),
                  yaxis=dict(title="Frequenza reale", range=[0, 105],
                             gridcolor="#21262d", ticksuffix="%"),
                  legend=dict(orientation="h", y=1.08),
                  margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

if cal.empty:
    st.caption("Servono almeno 5 pronostici per fascia di probabilità per "
               "disegnare la curva.")

st.markdown("---")

# ============================================================
# BREAKDOWN PER MERCATO E PER LEGA
# ============================================================
st.markdown("#### Per mercato")


def breakdown(dframe: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, grp in dframe.groupby(by):
        wo = grp[grp["odds"].notna() & (grp["odds"] > 1)]
        if not wo.empty:
            profit = np.where(wo["status"] == "won", wo["odds"] - 1.0, -1.0)
            yld = profit.sum() / len(wo) * 100
        else:
            yld = None
        rows.append({
            by.capitalize(): key,
            "N": len(grp),
            "Win rate": grp["won"].mean() * 100,
            "Prob media": grp["prob"].mean(),
            "Gap": grp["won"].mean() * 100 - grp["prob"].mean(),
            "Yield": yld,
        })
    return pd.DataFrame(rows).sort_values("N", ascending=False)


bd_cfg = {
    "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
    "Prob media": st.column_config.NumberColumn(format="%.1f%%"),
    "Gap": st.column_config.NumberColumn(
        format="%+.1f", help="Win rate − probabilità media: vicino a 0 = calibrato"),
    "Yield": st.column_config.NumberColumn(format="%+.1f%%"),
}
st.dataframe(breakdown(flt, "market"), hide_index=True,
             use_container_width=True, column_config=bd_cfg)

st.markdown("#### Per lega")
st.dataframe(breakdown(flt, "league"), hide_index=True,
             use_container_width=True, column_config=bd_cfg)

st.markdown("---")

# ============================================================
# EQUITY CURVE
# ============================================================
if not with_odds.empty and len(with_odds) >= 5:
    st.markdown("#### Andamento (stake flat 1u, solo pronostici quotati)")
    eq = with_odds.sort_values(["match_date", "id"]).copy()
    eq["cum"] = eq["profit"].cumsum()
    fig2 = go.Figure(go.Scatter(
        x=list(range(1, len(eq) + 1)), y=eq["cum"], mode="lines",
        line=dict(color=C_GOOD if eq["cum"].iloc[-1] >= 0 else C_BAD, width=2),
        fill="tozeroy",
        hovertemplate="Pronostico #%{x}<br>Saldo: %{y:+.2f}u<extra></extra>"))
    fig2.add_hline(y=0, line_color="#484f58")
    fig2.update_layout(**CHART_LAYOUT, height=280,
                       xaxis=dict(title="Pronostici (in ordine di data)",
                                  gridcolor="#21262d"),
                       yaxis=dict(title="Unità", gridcolor="#21262d"),
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CFG)
    st.markdown("---")

# ============================================================
# STORICO
# ============================================================
st.markdown("#### Storico")
hist = df.copy()
status_label = {"won": "✅ Vinto", "lost": "❌ Perso",
                "void": "⚪ Annullato", "pending": "⏳ In attesa"}
hist["Esito"] = hist["status"].map(status_label).fillna(hist["status"])
hist["Partita"] = hist["home"] + " – " + hist["away"]
hist["Risultato"] = np.where(
    hist["score_home"].notna(),
    hist["score_home"].fillna(0).astype("Int64").astype(str) + "-" +
    hist["score_away"].fillna(0).astype("Int64").astype(str), "")

show_cols = ["match_date", "league", "Partita", "market", "selection",
             "prob", "odds", "Esito", "Risultato"]
st.dataframe(
    hist[show_cols].rename(columns={
        "match_date": "Data", "league": "Lega", "market": "Mercato",
        "selection": "Selezione", "prob": "Prob", "odds": "Quota"}),
    hide_index=True, use_container_width=True,
    column_config={
        "Prob": st.column_config.NumberColumn(format="%.1f%%"),
        "Quota": st.column_config.NumberColumn(format="%.2f"),
    })

# ============================================================
# ZONA PERICOLOSA
# ============================================================
with st.expander("Zona pericolosa"):
    st.warning("Elimina TUTTI i pronostici salvati. Irreversibile.")
    confirm = st.checkbox("Confermo di voler svuotare il database")
    if st.button("Svuota database", disabled=not confirm):
        deleted = storage.delete_all()
        st.success(f"{deleted} pronostici eliminati.")
        st.rerun()
