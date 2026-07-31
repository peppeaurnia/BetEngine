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
import calibration
from results_updater import update_results, update_closing_odds
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
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mercati salvati", cnt["total"],
          help=f"Di cui {cnt['shortlisted']} effettivamente consigliati. "
               f"Salvare anche i non consigliati serve a misurare la "
               f"calibrazione su tutto il range di probabilità.")
m2.metric("In attesa di esito", cnt["pending"])
m3.metric("Chiusi", cnt["settled"])
m4.metric("Con quota di chiusura", cnt["with_clv"],
          help="Necessaria per il CLV, la metrica di edge più rapida da "
               "accumulare.")

# --- CLV: da lanciare vicino al calcio d'inizio ---
with st.expander("Quote di chiusura (CLV)"):
    st.caption(
        "Il **Closing Line Value** confronta la quota che hai preso con "
        "quella finale del mercato. È il segnale di edge più rapido: lo "
        "yield ha bisogno di centinaia di esiti per staccarsi dal rumore, "
        "il CLV dà indicazioni con 40-50 scommesse, perche misura il "
        "confronto diretto con il mercato invece di passare attraverso "
        "l'esito casuale della singola partita.\n\n"
        "Va lanciato **poco prima del calcio d'inizio** delle partite di "
        "oggi: è nell'ultima ora che il mercato incorpora formazioni e "
        "ultime notizie. Usa la quota di The Odds API (quota separata da "
        "API-Football).")
    if st.button("Registra le quote di chiusura di oggi",
                 use_container_width=True):
        with st.spinner("Recupero quote…"):
            res = update_closing_odds()
        if res.get("error"):
            st.error(res["error"])
        elif res["updated"] == 0:
            st.info("Nessuna quota di chiusura registrata "
                    f"({res.get('pending_fixtures', 0)} partite in attesa). "
                    "Normale se non ci sono partite di oggi già salvate.")
        else:
            st.success(f"{res['updated']} quote di chiusura registrate su "
                       f"{res['fixtures']} partite.")
        st.rerun()

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
# CALIBRAZIONE EMPIRICA (punto 4 della roadmap)
# ============================================================
with st.expander("Calibrazione empirica"):
    cs = calibration.status()
    if cs["active"]:
        st.success(f"Calibrazione ATTIVA — stimata il {cs['fitted_at']} su "
                   f"{cs['n_settled']} pronostici chiusi. Le probabilità "
                   f"mostrate nell'app sono corrette con la frequenza reale.")
        if cs["per_market"]:
            st.caption("Curve dedicate per: " + ", ".join(cs["per_market"]) +
                       " (gli altri mercati usano la curva globale).")
    else:
        st.info(f"Calibrazione non attiva: servono almeno "
                f"{cs['min_required']} pronostici chiusi "
                f"(attuali: {cnt['settled']}). Fino ad allora l'app mostra "
                f"le probabilità del modello senza correzioni.")

    if cs.get("market_weight") is not None:
        st.markdown(f"**Peso mercato ottimale stimato: `{cs['market_weight']}`** "
                    f"(su {cs['market_weight_n']} pronostici con quota) — "
                    f"attuale in `probability_engine.py`: `MARKET_WEIGHT = 0.35`. "
                    f"Se il valore stimato è stabile per qualche settimana, "
                    f"aggiorna la costante a mano.")

    if st.button("Ricalcola calibrazione", use_container_width=True):
        with st.spinner("Fit in corso…"):
            res = calibration.fit_and_save()
        if res["active"]:
            st.success(f"Calibrazione aggiornata su {res['n_settled']} "
                       f"pronostici chiusi.")
        else:
            st.warning(f"Campione ancora insufficiente "
                       f"({res['n_settled']}/{calibration.MIN_SAMPLE_GLOBAL}): "
                       f"nessuna correzione attivata.")
        st.rerun()

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

# v5: il database contiene TUTTI i mercati valutati, non solo i consigliati.
# Sono due domande diverse e vanno tenute separate:
#  - "il modello è calibrato?"  → serve tutto il campione, tutto il range
#  - "i miei consigli rendono?" → solo i consigliati (shortlisted)
# Mescolarli darebbe uno yield calcolato su scommesse che non avresti mai
# fatto, e una calibrazione stimata solo sulla coda alta.
scope = st.radio(
    "Campione",
    ["Solo consigliati", "Tutti i mercati valutati"],
    horizontal=True,
    help="Win rate e yield hanno senso solo sui consigliati (sono le "
         "scommesse che avresti davvero piazzato). Calibrazione e Brier "
         "sono più informativi su tutto il campione.")

if "shortlisted" in settled.columns and scope == "Solo consigliati":
    settled = settled[settled["shortlisted"].fillna(1) == 1].copy()
    if settled.empty:
        st.warning("Nessun pronostico consigliato ancora chiuso.")
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

# --- CLV: il segnale di edge più rapido ---
# Nota: il CLV si calcola su TUTTI i pronostici con quota di chiusura, anche
# quelli non ancora chiusi. Non dipende dall'esito, quindi non serve
# aspettare che le partite finiscano.
clv_src = df.copy()
if "shortlisted" in clv_src.columns and scope == "Solo consigliati":
    clv_src = clv_src[clv_src["shortlisted"].fillna(1) == 1]
clv_rows = clv_src[clv_src["clv_pct"].notna()] if "clv_pct" in clv_src.columns \
    else clv_src.iloc[0:0]

if len(clv_rows) >= 10:
    clv_mean = float(clv_rows["clv_pct"].mean())
    clv_beat = float((clv_rows["clv_pct"] > 0).mean() * 100)
    c1, c2 = st.columns(2)
    c1.metric("CLV medio", f"{clv_mean:+.2f}%",
              help="Quanto sei stato migliore (o peggiore) della linea di "
                   "chiusura. Positivo e stabile = edge reale.")
    c2.metric("Battuta la chiusura", f"{clv_beat:.0f}%",
              help=f"Su {len(clv_rows)} scommesse con quota di chiusura "
                   f"registrata. Sopra il 50% in modo stabile è il segnale "
                   f"più affidabile che hai davvero un vantaggio.")
    if clv_mean > 1.0:
        st.caption("✅ CLV positivo: stai prendendo quote migliori di quelle "
                   "finali. È il predittore di profittabilità più solido che "
                   "esista, e arriva molto prima dello yield.")
    elif clv_mean < -1.0:
        st.caption("⚠️ CLV negativo: stai prendendo sistematicamente quote "
                   "peggiori della chiusura. Uno yield positivo con CLV "
                   "negativo è quasi sempre fortuna, non edge — non "
                   "aumentare le puntate su quella base.")
elif len(clv_rows) > 0:
    st.caption(f"CLV: solo {len(clv_rows)} scommesse con quota di chiusura "
               f"registrata, servono almeno 10 per un primo numero.")

# --- Benchmark contro il mercato: il test che conta davvero ---
if "prob_market" in flt.columns:
    bm = flt[flt["prob_market"].notna()].copy()
    if len(bm) >= 20:
        y = bm["won"].to_numpy()
        mcol = "prob_raw" if "prob_raw" in bm.columns else "prob"
        bm[mcol] = bm[mcol].fillna(bm["prob"])
        b_model = float(np.mean((bm[mcol].to_numpy() / 100.0 - y) ** 2))
        b_market = float(np.mean((bm["prob_market"].to_numpy() / 100.0 - y) ** 2))
        diff = b_model - b_market
        verdict = ("il modello batte il mercato" if diff < -0.002 else
                   "il mercato batte il modello" if diff > 0.002 else
                   "modello e mercato equivalenti")
        st.caption(f"**Benchmark vs mercato** (su {len(bm)} pronostici con "
                   f"quote complete): Brier modello {b_model:.4f} vs Brier "
                   f"mercato {b_market:.4f} → {verdict}. Battere stabilmente "
                   f"il Brier del mercato è la vera prova di un edge.")

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

# v5: si grafica prob_raw (ancorata ma NON calibrata), non prob.
# Graficare prob significherebbe guardare il risultato DOPO la correzione:
# la curva apparirebbe sulla diagonale per costruzione, nascondendo
# esattamente l'errore che si vuole misurare.
cal_col = "prob_raw" if "prob_raw" in flt.columns else "prob"
flt[cal_col] = flt[cal_col].fillna(flt["prob"])

# Le fasce partono da 0: il database ora contiene anche i mercati non
# consigliati, quindi l'intero range è osservabile.
lo = int(max(0, np.floor(flt[cal_col].min() / 5) * 5))
bins = np.arange(lo, 101, 5)
flt["bin"] = pd.cut(flt[cal_col], bins=bins, right=False)
cal = flt.groupby("bin", observed=True).agg(
    n=("won", "size"), pred=(cal_col, "mean"), real=("won", "mean")
).reset_index()
cal = cal[cal["n"] >= 5]  # niente punti su gruppi minuscoli
cal["real"] *= 100

if cal_col == "prob_raw":
    st.caption("La curva usa le probabilità **non calibrate**: graficare "
               "quelle già corrette le metterebbe sulla diagonale per "
               "costruzione, nascondendo l'errore da misurare.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=[lo, 100], y=[lo, 100], mode="lines",
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

# ============================================================
# CONFRONTO A/B: ENGINE v4 vs DIXON-COLES
# ============================================================
if "engine" in flt.columns and flt["engine"].nunique() > 1:
    st.markdown("#### Confronto engine (A/B)")
    st.caption("v4 = strength da medie stagionali · dc = fit Dixon-Coles con "
               "decadimento temporale e ξ scelto per validazione fuori "
               "campione. Il **Brier** è la colonna che decide: win rate e "
               "yield dipendono da quali scommesse ciascun motore ha "
               "proposto, il Brier misura la qualità delle probabilità a "
               "parità di partite.")
    st.caption("Da v5 i μ del Dixon-Coles ricevono lo stesso shrinkage "
               "James-Stein della v4. Prima non era così: il DC produceva "
               "probabilità più estreme, superava le soglie più spesso e "
               "generava più pronostici, quindi il confronto misurava "
               "insieme accuratezza e aggressività. Ora la differenza "
               "residua è attribuibile al modello.")

    ab_col = "prob_raw" if "prob_raw" in flt.columns else "prob"
    flt[ab_col] = flt[ab_col].fillna(flt["prob"])

    ab_rows = []
    for eng, grp in flt.groupby("engine"):
        wo_e = grp[grp["odds"].notna() & (grp["odds"] > 1)]
        yld_e = None
        if not wo_e.empty:
            pr_e = np.where(wo_e["status"] == "won", wo_e["odds"] - 1.0, -1.0)
            yld_e = pr_e.sum() / len(wo_e) * 100
        # Brier del mercato sulle STESSE righe: dice se quel motore batte il
        # mercato, non solo se batte l'altro motore. Due motori possono
        # essere entrambi peggiori del mercato.
        bm_e = grp[grp["prob_market"].notna()] if "prob_market" in grp.columns \
            else grp.iloc[0:0]
        brier_mkt = (float(np.mean((bm_e["prob_market"] / 100.0 - bm_e["won"]) ** 2))
                     if len(bm_e) >= 20 else None)
        clv_e = (float(grp["clv_pct"].mean())
                 if "clv_pct" in grp.columns and grp["clv_pct"].notna().any()
                 else None)
        ab_rows.append({
            "Engine": eng,
            "N": len(grp),
            "Win rate": grp["won"].mean() * 100,
            "Brier": float(np.mean((grp[ab_col] / 100.0 - grp["won"]) ** 2)),
            "Brier mercato": brier_mkt,
            "CLV": clv_e,
            "Yield": yld_e,
        })
    st.dataframe(pd.DataFrame(ab_rows), hide_index=True,
                 use_container_width=True,
                 column_config={
                     "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
                     "Brier": st.column_config.NumberColumn(format="%.4f"),
                     "Brier mercato": st.column_config.NumberColumn(
                         format="%.4f",
                         help="Brier delle quote sharp sulle stesse partite. "
                              "Se la colonna Brier non scende sotto questa, "
                              "il motore non ha edge."),
                     "CLV": st.column_config.NumberColumn(format="%+.2f%%"),
                     "Yield": st.column_config.NumberColumn(format="%+.1f%%"),
                 })
    min_n = min(r["N"] for r in ab_rows)
    if min_n < 100:
        st.caption(f"⚠️ Il gruppo più piccolo ha {min_n} pronostici: "
                   f"sotto i ~100 per engine il confronto è indicativo.")

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
