"""
🧮 ESTIMATE PROBABILITIES v4.0 — Backtest batch UNIFICATO
==========================================================
REFACTOR (Luglio 2026): questo modulo NON contiene più matematica propria.
Tutta la pipeline probabilistica (dampening, shrinkage James-Stein, λ₃,
Poisson bivariata, estrazione mercati) viene importata da probability_engine,
che è l'UNICA fonte di verità. Ciò che misuri qui è ciò che gira nell'app.

Rimosso rispetto alla v3:
- Engine duplicato (poisson_pmf, bivpois_matrix, prob_from_matrix, clamp locali)
- apply_calibration() post-hoc → non esiste più nella v4 dell'engine
- rank_factor nel soft adjustment → rimosso nella v4 (double-counting)
- Pesi anchoring divergenti (0.30/0.35) → si usa MARKET_WEIGHT dell'engine
- EV calcolato sulle probabilità ANCORATE → ora è calcolato sulle probabilità
  PURE del modello (colonna p_model), come fa l'app. L'EV su probabilità
  ancorate è circolare: più il modello converge al mercato, meno trova value.

Differenze note e volute rispetto alla produzione:
- Il backtest batch NON applica aggiustamenti H2H, tiri e arbitro
  (quei dati non sono nel dataset storico): misura il core dell'engine.

Input:  fixtures (date, home, away), stats (per squadra), odds (per riga quota)
Output: DataFrame + CSV data/probabilities_custom.csv
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from rapidfuzz import fuzz, process

# ============================================================
# UNICA FONTE DI VERITÀ: probability_engine (v4)
# ============================================================
from probability_engine import (
    calculate_expected_goals,     # dampening + shrinkage + μ (v4)
    get_lambda3,                  # λ₃ per lega + aggiustamento vivacità
    bivariate_poisson_matrix,     # matrice Poisson bivariata Dixon-Coles
    calculate_1x2,
    calculate_over_under,
    calculate_btts,
    MARKET_WEIGHT,                # peso anchoring (unico, come in produzione)
    _shin_probs,                  # rimozione margine con metodo di Shin (v5)
)


# ============================================================
# UTILITY (solo plumbing dati, nessuna matematica di modello)
# ============================================================

def normalize_team_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def find_best_team_match(team_name, candidates, threshold=80):
    """Fuzzy match nome squadra → lista candidati."""
    if not isinstance(team_name, str) or not candidates:
        return None, 0
    match = process.extractOne(team_name, candidates, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0], match[1]
    return None, 0


# ============================================================
# FORZE SQUADRA → formato atteso da calculate_expected_goals
# ============================================================

def compute_team_strengths(stats: pd.DataFrame) -> dict:
    """
    Mappa il DataFrame stats nei dict che calculate_expected_goals si aspetta.

    v4: NESSUN clamp/shrink locale — dampening, shrinkage verso 1.0 e
    James-Stein sui μ vengono applicati DENTRO l'engine, una volta sola.
    rank_factor viene ignorato (rimosso in v4: già catturato nelle strength).
    """
    if stats is None or stats.empty:
        print("⚠️ compute_team_strengths: stats vuoto.")
        return {}

    stats = stats.copy()
    if "team" not in stats.columns:
        team_col = next((c for c in stats.columns if "team" in c.lower()), None)
        stats["team"] = stats[team_col] if team_col else "unknown"
    stats["team_clean"] = stats["team"].apply(normalize_team_name)

    needed = [
        "home_attack_strength", "home_defense_strength",
        "away_attack_strength", "away_defense_strength",
        "form_factor", "momentum",
        "league_avg_gf_home", "league_avg_gf_away",
        "league_id",
    ]
    for c in needed:
        if c not in stats.columns:
            stats[c] = np.nan if ("league_avg" in c or c == "league_id") else 1.0

    strengths = {}
    for _, r in stats.iterrows():
        strengths[r["team_clean"]] = {
            # Chiavi identiche a quelle prodotte da data_fetcher.get_match_stats
            "attack_home": float(r.get("home_attack_strength", 1.0)),
            "defense_home": float(r.get("home_defense_strength", 1.0)),
            "attack_away": float(r.get("away_attack_strength", 1.0)),
            "defense_away": float(r.get("away_defense_strength", 1.0)),
            "form_factor": float(r.get("form_factor", 1.0)),
            "momentum": float(r.get("momentum", 1.0)),
            "league_avg_gf_home": float(r.get("league_avg_gf_home", np.nan)),
            "league_avg_gf_away": float(r.get("league_avg_gf_away", np.nan)),
            "league_id": r.get("league_id", None),
        }

    print(f"✅ compute_team_strengths: {len(strengths)} squadre mappate")
    return strengths


def _match_probs(sh: dict, sa: dict, league_id) -> dict:
    """
    Calcola le probabilità di UN match usando esattamente la pipeline v4
    dell'engine di produzione (senza H2H/tiri/arbitro: dati non disponibili
    nel backtest batch).
    """
    mu_home, mu_away = calculate_expected_goals(sh, sa, league_id)
    lam3 = get_lambda3(mu_home, mu_away, league_id)
    M = bivariate_poisson_matrix(mu_home, mu_away, lam3)

    p1x2 = calculate_1x2(M)
    btts = calculate_btts(M)
    ou25 = calculate_over_under(M, 2.5)

    return {
        "mu_home": mu_home, "mu_away": mu_away, "lam3": lam3, "M": M,
        "p_home": p1x2["p_home"], "p_draw": p1x2["p_draw"], "p_away": p1x2["p_away"],
        "p_over25": ou25["p_over"], "p_under25": ou25["p_under"],
        "p_btts_yes": btts["p_btts_yes"], "p_btts_no": btts["p_btts_no"],
    }


# ============================================================
# MAIN
# ============================================================

def estimate_probabilities(fixtures: pd.DataFrame, stats: pd.DataFrame,
                           odds: pd.DataFrame) -> pd.DataFrame:
    fixtures, stats, odds = fixtures.copy(), stats.copy(), odds.copy()

    for df in [fixtures, odds]:
        df["home_clean"] = df["home"].astype(str).str.lower().str.replace(" ", "", regex=False)
        df["away_clean"] = df["away"].astype(str).str.lower().str.replace(" ", "", regex=False)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.date

    fixture_home_list = fixtures["home_clean"].unique().tolist()
    fixture_away_list = fixtures["away_clean"].unique().tolist()

    # --- Merge fuzzy odds → fixtures ---
    merged_rows = []
    for _, o in odds.iterrows():
        best_home, sim_home = find_best_team_match(o["home_clean"], fixture_home_list)
        best_away, sim_away = find_best_team_match(o["away_clean"], fixture_away_list)
        if not best_home or not best_away:
            continue
        fx_candidates = fixtures[
            (fixtures["home_clean"] == best_home) &
            (fixtures["away_clean"] == best_away) &
            (fixtures["date"].between(o["date"] - timedelta(days=1),
                                      o["date"] + timedelta(days=1)))
        ]
        if fx_candidates.empty:
            continue
        fx = fx_candidates.iloc[0].to_dict()
        merged_rows.append({**o.to_dict(), **fx,
                            "sim_home": sim_home, "sim_away": sim_away})

    merged = pd.DataFrame(merged_rows)
    print(f"🔗 Merge fuzzy: {len(merged)} righe su {len(odds)} quote.")
    if merged.empty:
        print("⚠️ merged vuoto.")
        return pd.DataFrame()

    strengths = compute_team_strengths(stats)

    # ========================================================
    # PRIMO PASSAGGIO: probabilità per match (engine v4)
    # ========================================================
    match_cache = {}
    unique_matches = merged[["date", "home", "away",
                             "home_clean", "away_clean"]].drop_duplicates()

    for _, m in unique_matches.iterrows():
        sh = strengths.get(normalize_team_name(m["home_clean"]), {})
        sa = strengths.get(normalize_team_name(m["away_clean"]), {})

        league_id = sh.get("league_id") or sa.get("league_id")
        if league_id is not None:
            try:
                league_id = int(league_id)
            except (ValueError, TypeError):
                league_id = None

        match_cache[(m["date"], m["home"], m["away"])] = {
            **_match_probs(sh, sa, league_id),
            "league_id": league_id,
        }

    # ========================================================
    # SECONDO PASSAGGIO: mapping mercati
    # (v4: NESSUNA calibrazione post-hoc — rimossa come nell'engine)
    # ========================================================
    rows = []
    for _, row in merged.iterrows():
        mc = match_cache.get((row["date"], row["home"], row["away"]))
        if mc is None:
            continue

        M = mc["M"]
        market = str(row.get("market", "")).lower()
        selection = str(row.get("selection", "")).lower()
        line_val = row.get("line", None)
        try:
            line = float(line_val) if line_val is not None and str(line_val) != "None" else 2.5
        except Exception:
            line = 2.5

        p_model = np.nan
        mkt_type = "unknown"

        # 1X2
        if market in ("h2h", "1x2"):
            mkt_type = "1X2"
            if selection in ("1", "home") or selection == row["home"].lower():
                p_model = mc["p_home"]
            elif selection in ("x", "draw", "pareggio"):
                p_model = mc["p_draw"]
            elif selection in ("2", "away") or selection == row["away"].lower():
                p_model = mc["p_away"]
        # BTTS
        elif ("btts" in market) or ("both" in market) or \
             ("goal" in market and "over" not in market and "under" not in market):
            mkt_type = "BTTS"
            if selection.startswith("y") or selection in ("si", "sì", "yes") or selection == "gol":
                p_model = mc["p_btts_yes"]
            elif selection.startswith("n") or "nogol" in selection or selection == "no":
                p_model = mc["p_btts_no"]
        # Over/Under (qualsiasi linea)
        elif any(x in market for x in ["over", "under", "totals", "goals"]) \
                or "over" in selection or "under" in selection:
            mkt_type = "OU"
            ou = calculate_over_under(M, line)
            if "over" in market or "over" in selection:
                p_model = ou["p_over"]
            elif "under" in market or "under" in selection:
                p_model = ou["p_under"]

        rows.append({
            "date": row["date"], "league": row.get("league"),
            "home": row["home"], "away": row["away"],
            "bookmaker": row.get("bookmaker"), "market": row.get("market"),
            "selection": row.get("selection"), "odds": row.get("odds"),
            "line": row.get("line"),
            "mu_home": mc["mu_home"], "mu_away": mc["mu_away"],
            "p_home": mc["p_home"], "p_draw": mc["p_draw"], "p_away": mc["p_away"],
            "p_over25": mc["p_over25"], "p_under25": mc["p_under25"],
            "p_btts_yes": mc["p_btts_yes"], "p_btts_no": mc["p_btts_no"],
            "p_model": p_model,          # probabilità PURA del modello
            "_mkt_type": mkt_type,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ Nessuna riga prodotta.")
        return df

    # ========================================================
    # MARKET ANCHORING — stesso MARKET_WEIGHT della produzione,
    # per TUTTI i mercati (l'engine v4 usa un peso unico)
    # ========================================================
    df["odds_num"] = pd.to_numeric(df["odds"], errors="coerce")
    df["p_market_raw"] = np.where(df["odds_num"] > 0, 1.0 / df["odds_num"], np.nan)
    df["p_market_fair"] = np.nan

    # v5: il margine viene rimosso con il METODO DI SHIN, non per
    # normalizzazione proporzionale. La normalizzazione proporzionale assume
    # che il bookmaker spalmi il margine uniformemente, mentre lo concentra
    # sugli outsider: il risultato è che i favoriti risultano sottostimati e
    # gli outsider sovrastimati, in modo sistematico. Poiché p_market_fair è
    # sia l'ancora del blend sia il benchmark del modello, quel bias
    # inquinava entrambi.
    def _fair_group(grp, min_len):
        """Probabilità fair di un gruppo completo di esiti, o NaN."""
        if len(grp) < min_len:
            return None
        inv = grp["p_market_raw"].to_numpy(dtype=float)
        if not np.all(np.isfinite(inv)) or np.any(inv <= 0):
            return None
        p = _shin_probs(inv.tolist())
        return p if p else None

    # 1X2: solo se la terna è COMPLETA (3 esiti)
    mask_1x2 = df["_mkt_type"] == "1X2"
    for _, grp in df[mask_1x2].groupby(["date", "home", "away", "bookmaker"]):
        p = _fair_group(grp, 3)
        if p and len(grp) == 3:
            df.loc[grp.index, "p_market_fair"] = p

    # OU: coppia over+under della STESSA LINEA (non per stringa market,
    # che può differire tra "totals over"/"totals under")
    mask_ou = df["_mkt_type"] == "OU"
    df["_line_key"] = pd.to_numeric(df["line"], errors="coerce").fillna(2.5)
    for _, grp in df[mask_ou].groupby(["date", "home", "away", "bookmaker", "_line_key"]):
        p = _fair_group(grp, 2)
        if p and len(grp) == 2:
            df.loc[grp.index, "p_market_fair"] = p

    # BTTS: coppia yes+no per bookmaker
    mask_btts = df["_mkt_type"] == "BTTS"
    for _, grp in df[mask_btts].groupby(["date", "home", "away", "bookmaker"]):
        p = _fair_group(grp, 2)
        if p and len(grp) == 2:
            df.loc[grp.index, "p_market_fair"] = p

    # Blend: p_hat = (1 - w)·modello + w·mercato   (w = MARKET_WEIGHT engine)
    w = MARKET_WEIGHT
    df["p_hat"] = df["p_model"]
    can_blend = df["p_model"].notna() & df["p_market_fair"].notna()
    idx = df[can_blend].index
    if len(idx) > 0:
        df.loc[idx, "p_hat"] = (1 - w) * df.loc[idx, "p_model"] + w * df.loc[idx, "p_market_fair"]

    # Come in produzione: la terna 1X2 ancorata viene rinormalizzata a 1
    for _, grp in df[mask_1x2 & can_blend].groupby(["date", "home", "away", "bookmaker"]):
        s = grp["p_hat"].sum()
        if s > 0 and len(grp) == 3:
            df.loc[grp.index, "p_hat"] = grp["p_hat"] / s

    # ========================================================
    # EV — sulle probabilità PURE (coerente con l'app, non circolare)
    # ========================================================
    df["EV"] = np.where(
        (df["odds_num"] > 0) & df["p_model"].notna(),
        df["p_model"] * df["odds_num"] - 1.0,
        np.nan,
    )
    # EV ancorato tenuto come colonna diagnostica separata
    df["EV_anchored"] = np.where(
        (df["odds_num"] > 0) & df["p_hat"].notna(),
        df["p_hat"] * df["odds_num"] - 1.0,
        np.nan,
    )

    # Ordina colonne
    cols_first = [
        "date", "league", "home", "away", "bookmaker", "market", "selection",
        "odds", "line", "p_hat", "EV", "EV_anchored", "p_model", "p_market_fair",
        "mu_home", "mu_away", "p_home", "p_draw", "p_away",
        "p_over25", "p_under25", "p_btts_yes", "p_btts_no", "p_market_raw",
    ]
    df = df.drop(columns=["odds_num", "_mkt_type", "_line_key"], errors="ignore")
    cols = [c for c in cols_first if c in df.columns] + \
           [c for c in df.columns if c not in cols_first]
    df = df[cols]

    # Salva
    try:
        df.to_csv("data/probabilities_custom.csv", index=False)
        print(f"✅ Salvate {len(df)} righe")
    except Exception as e:
        print(f"⚠️ Errore salvataggio: {e}")

    # Diagnostica
    ev_mean = df["EV"].mean()
    ev_pos = (df["EV"] > 0).mean() * 100
    anchor = df["p_market_fair"].notna().mean() * 100
    gap = (df["p_hat"] - df["p_market_raw"]).mean() * 100

    print(f"📊 EV medio (puro): {ev_mean:.3f} | EV+: {ev_pos:.0f}% | "
          f"μ_h: {df['mu_home'].mean():.2f} | μ_a: {df['mu_away'].mean():.2f}")
    print(f"   Anchoring: {anchor:.0f}% (w={w}) | Gap p_hat vs mercato: {gap:+.1f}%")

    if ev_mean > 0.15:
        print(f"   ⚠️ EV medio alto ({ev_mean:.3f}): il modello si discosta "
              f"molto dal mercato, verificare le strength in input")
    if ev_pos > 50:
        print(f"   ⚠️ {ev_pos:.0f}% di righe EV+: troppe value bet, "
              f"probabile sovrastima sistematica")

    return df
