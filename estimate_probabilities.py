"""
🧮 ESTIMATE PROBABILITIES v3.0 — Fix completo 7 problemi
=========================================================
Problemi risolti:
  1. Strength senza clamping → aggiunto clamp [0.45, 2.0] + shrinkage
  2. clamp_lambda [0.1, 6.0] troppo largo → [0.4, 4.5]
  3. Fallback medie lega troppo bassi → 1.55/1.25
  4. Nessun market anchoring OU/BTTS → 80% modello + 20% mercato
  5. Nessuna calibrazione → apply_calibration() attiva
  6. HOME/AWAY invertiti in prob_from_matrix → corretto tril/triu
  7. Nessun μ totale minimo → floor 1.8 gol
"""

import numpy as np
import pandas as pd
from math import exp, factorial
from datetime import timedelta
from rapidfuzz import fuzz, process

# ============================================================
# PARAMETRI (allineati con probability_engine.py)
# ============================================================
MAX_GOALS = 10
SOFT_ADJ_WEIGHT = 0.25

# FIX #2: range più stretto
LAMBDA_MIN = 0.4     # era 0.1
LAMBDA_MAX = 4.5     # era 6.0

# FIX #1: clamping forze
STRENGTH_MIN = 0.45
STRENGTH_MAX = 2.0
SHRINKAGE = 0.15     # tira i valori estremi verso 1.0

# FIX #3: fallback realistici
DEFAULT_GF_HOME = 1.55   # era 1.30
DEFAULT_GF_AWAY = 1.25   # era 1.10

# FIX #7: gol totali minimi per partita
MIN_TOTAL_MU = 1.8

# FIX #4: pesi anchoring mercato
W_MARKET_1X2 = 0.30       # 70% modello + 30% mercato
W_MARKET_OU_BTTS = 0.35   # 65% modello + 35% mercato (NUOVO)

# λ₃ per lega
LEAGUE_LAMBDA3 = {
    39: 0.08, 135: 0.06, 140: 0.05, 78: 0.10,
    61: 0.08, 94: 0.08, 88: 0.08, 2: 0.07,
}

# ============================================================
# FIX #5: Calibrazione probabilità
# ============================================================
def apply_calibration(prob: float, market: str) -> float:
    """
    Corregge la sovrastima sistematica del modello.
    Basata su backtesting: prob 80%+ vincono solo 50-60%.
    """
    p = prob * 100

    # Shrinkage per probabilità alte
    if p <= 70:
        base = p
    elif p <= 80:
        base = 70 + (p - 70) * 0.80   # 80% → 78%
    elif p <= 90:
        base = 78 + (p - 80) * 0.55   # 90% → 83.5%
    else:
        base = 83.5 + (p - 90) * 0.40  # 100% → 87.5%

    # Correzione per mercato
    if market == '1X2':
        adj = -4 if p > 80 else (-2 if p > 65 else 0)
    elif market == 'BTTS':
        adj = -4
    elif market == 'OU':
        adj = -3
    else:
        adj = 0

    return max(0.02, min(0.98, (base + adj) / 100))


# ============================================================
# FUNZIONI BASE
# ============================================================
def normalize_team_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def clamp_lambda(lam) -> float:
    """FIX #2: range [0.4, 4.5] invece di [0.1, 6.0]."""
    if pd.isna(lam):
        return 1.3
    return float(np.clip(lam, LAMBDA_MIN, LAMBDA_MAX))


def clamp_strength(val) -> float:
    """FIX #1: limita strength in [0.45, 2.0]."""
    if pd.isna(val):
        return 1.0
    return float(np.clip(val, STRENGTH_MIN, STRENGTH_MAX))


def shrink_strength(val) -> float:
    """
    FIX #1: Shrinkage bayesiano — tira estremi verso 1.0.
    Pisa 0.056 → 0.198 → clamp 0.45
    Bayern 2.28 → 2.09 → clamp 2.0
    Squadra media 1.0 → 1.0 (nessun cambio)
    """
    if pd.isna(val):
        return 1.0
    shrunk = val + SHRINKAGE * (1.0 - val)
    return clamp_strength(shrunk)


# ============================================================
# λ₃ correlazione
# ============================================================
def estimate_lambda3(mu_h: float, mu_a: float, league_id=None) -> float:
    if league_id is not None and league_id in LEAGUE_LAMBDA3:
        base = LEAGUE_LAMBDA3[league_id]
    else:
        base = 0.08
    if not pd.isna(mu_h) and not pd.isna(mu_a) and mu_h > 0 and mu_a > 0:
        total = mu_h + mu_a
        adj = 0.015 * min((total - 2.0) / 1.5, 0.5)
        return float(np.clip(base + adj, 0.04, 0.14))
    return base


# ============================================================
# Poisson bivariata
# ============================================================
def bivpois_matrix(mu_h: float, mu_a: float, lam3: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    lam1 = max(mu_h - lam3, 0.01)
    lam2 = max(mu_a - lam3, 0.01)
    M = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            s = 0.0
            for k in range(min(i, j) + 1):
                s += poisson_pmf(i - k, lam1) * poisson_pmf(j - k, lam2) * poisson_pmf(k, lam3)
            M[i, j] = s
    tot = M.sum()
    if tot > 0:
        M /= tot
    return M


def prob_from_matrix(M: np.ndarray, kind: str) -> float:
    """
    FIX #6: home e away erano INVERTITI!
    M[i,j] = P(home=i gol, away=j gol)
    Home win: i > j → triangolo INFERIORE (tril)
    Away win: j > i → triangolo SUPERIORE (triu)
    """
    if M is None or M.size == 0:
        return np.nan
    # FIX #6: CORRETTO (era scambiato!)
    if kind == "home":
        return float(np.tril(M, k=-1).sum())   # era triu → SBAGLIATO
    if kind == "away":
        return float(np.triu(M, k=1).sum())     # era tril → SBAGLIATO
    if kind == "draw":
        return float(np.trace(M))
    if kind == "btts_yes":
        M2 = M.copy()
        M2[0, :] = 0
        M2[:, 0] = 0
        return float(M2.sum())
    if kind == "btts_no":
        return 1 - prob_from_matrix(M, "btts_yes")
    if kind == "over25":
        return prob_over_line(M, 2.5)
    if kind == "under25":
        return 1 - prob_over_line(M, 2.5)
    return np.nan


def prob_over_line(M: np.ndarray, line: float) -> float:
    if M is None or M.size == 0 or pd.isna(line):
        return np.nan
    n = M.shape[0]
    thr = int(np.floor(float(line))) + 1
    return float(sum(M[i, j] for i in range(n) for j in range(n) if i + j >= thr))


# ============================================================
# Fuzzy match
# ============================================================
def find_best_team_match(team_name, candidates, threshold=80):
    if not isinstance(team_name, str) or not candidates:
        return None, 0
    match = process.extractOne(team_name, candidates, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0], match[1]
    return None, 0


# ============================================================
# Soft adjustment
# ============================================================
def soft_adj(form_f, rank_f, mom_f, weight=SOFT_ADJ_WEIGHT):
    avg = np.nanmean([form_f, rank_f, mom_f])
    if pd.isna(avg):
        return 1.0
    return float(1.0 + weight * (avg - 1.0))


# ============================================================
# Forze squadra CON shrinkage (FIX #1)
# ============================================================
def compute_team_strengths(stats: pd.DataFrame):
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
        "form_factor", "rank_factor", "momentum",
        "league_avg_gf_home", "league_avg_gf_away",
        "league_avg_ga_home", "league_avg_ga_away",
        "league_id"
    ]
    for c in needed:
        if c not in stats.columns:
            stats[c] = np.nan if "league_avg" in c or c == "league_id" else 1.0

    strengths = {}
    fixes = 0
    for _, r in stats.iterrows():
        key = r["team_clean"]

        # FIX #1: shrink + clamp ogni valore di forza
        raw = {
            "att_h": float(r.get("home_attack_strength", 1.0)),
            "def_h": float(r.get("home_defense_strength", 1.0)),
            "att_a": float(r.get("away_attack_strength", 1.0)),
            "def_a": float(r.get("away_defense_strength", 1.0)),
        }
        fixed = {k: shrink_strength(v) for k, v in raw.items()}
        if any(abs(fixed[k] - raw[k]) > 0.05 for k in raw):
            fixes += 1

        strengths[key] = {
            "attack_home": fixed["att_h"],
            "defense_home": fixed["def_h"],
            "attack_away": fixed["att_a"],
            "defense_away": fixed["def_a"],
            "form_factor": float(r.get("form_factor", 1.0)),
            "rank_factor": float(r.get("rank_factor", 1.0)),
            "momentum": float(r.get("momentum", 1.0)),
            "league_avg_gf_home": float(r.get("league_avg_gf_home", np.nan)),
            "league_avg_gf_away": float(r.get("league_avg_gf_away", np.nan)),
            "league_avg_ga_home": float(r.get("league_avg_ga_home", np.nan)),
            "league_avg_ga_away": float(r.get("league_avg_ga_away", np.nan)),
            "league_id": r.get("league_id", None),
        }

    print(f"✅ compute_team_strengths: {len(strengths)} squadre ({fixes} corrette con shrinkage)")
    return strengths


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

    # Merge fuzzy odds → fixtures
    merged_rows = []
    for _, o in odds.iterrows():
        best_home, sim_home = find_best_team_match(o["home_clean"], fixture_home_list)
        best_away, sim_away = find_best_team_match(o["away_clean"], fixture_away_list)
        if not best_home or not best_away:
            continue
        fx_candidates = fixtures[
            (fixtures["home_clean"] == best_home) &
            (fixtures["away_clean"] == best_away) &
            (fixtures["date"].between(o["date"] - timedelta(days=1), o["date"] + timedelta(days=1)))
        ]
        if fx_candidates.empty:
            continue
        fx = fx_candidates.iloc[0].to_dict()
        merged_rows.append({**o.to_dict(), **fx, "sim_home": sim_home, "sim_away": sim_away})

    merged = pd.DataFrame(merged_rows)
    print(f"🔗 Merge fuzzy: {len(merged)} righe su {len(odds)} quote.")
    if merged.empty:
        print("⚠️ merged vuoto.")
        return pd.DataFrame()

    strengths = compute_team_strengths(stats)

    # ========================================================
    # PRIMO PASSAGGIO: calcolo μ con tutti i fix
    # ========================================================
    match_cache = {}
    unique_matches = merged[["date", "home", "away", "home_clean", "away_clean"]].drop_duplicates()

    for _, m in unique_matches.iterrows():
        home_key = normalize_team_name(m["home_clean"])
        away_key = normalize_team_name(m["away_clean"])
        sh = strengths.get(home_key, {})
        sa = strengths.get(away_key, {})

        att_h = sh.get("attack_home", 1.0)
        def_h = sh.get("defense_home", 1.0)
        att_a = sa.get("attack_away", 1.0)
        def_a = sa.get("defense_away", 1.0)

        adj_h = soft_adj(sh.get("form_factor", 1.0), sh.get("rank_factor", 1.0), sh.get("momentum", 1.0))
        adj_a = soft_adj(sa.get("form_factor", 1.0), sa.get("rank_factor", 1.0), sa.get("momentum", 1.0))

        # FIX #3: fallback realistici
        base_gf_home = sh.get("league_avg_gf_home", np.nan)
        base_gf_away = sa.get("league_avg_gf_away", np.nan)
        if pd.isna(base_gf_home) or base_gf_home <= 0:
            base_gf_home = DEFAULT_GF_HOME
        if pd.isna(base_gf_away) or base_gf_away <= 0:
            base_gf_away = DEFAULT_GF_AWAY

        # FIX #2: clamp più stretto
        mu_home = clamp_lambda(base_gf_home * att_h * def_a * adj_h)
        mu_away = clamp_lambda(base_gf_away * att_a * def_h * adj_a)

        # FIX #7: μ totale minimo
        total_mu = mu_home + mu_away
        if total_mu < MIN_TOTAL_MU:
            scale = MIN_TOTAL_MU / total_mu
            mu_home = clamp_lambda(mu_home * scale)
            mu_away = clamp_lambda(mu_away * scale)

        league_id = sh.get("league_id") or sa.get("league_id")
        if league_id is not None:
            try:
                league_id = int(league_id)
            except (ValueError, TypeError):
                league_id = None
        lam3 = estimate_lambda3(mu_home, mu_away, league_id=league_id)

        # FIX #6: prob_from_matrix ora è corretto
        M = bivpois_matrix(mu_home, mu_away, lam3)
        p_home = prob_from_matrix(M, "home")
        p_draw = prob_from_matrix(M, "draw")
        p_away = prob_from_matrix(M, "away")
        p_over25 = prob_over_line(M, 2.5)
        p_under25 = 1 - p_over25 if not pd.isna(p_over25) else np.nan
        p_btts_yes = prob_from_matrix(M, "btts_yes")
        p_btts_no = 1 - p_btts_yes if not pd.isna(p_btts_yes) else np.nan

        match_cache[(m["date"], m["home"], m["away"])] = dict(
            mu_home=mu_home, mu_away=mu_away, league_id=league_id, lam3=lam3, M=M,
            p_home=p_home, p_draw=p_draw, p_away=p_away,
            p_over25=p_over25, p_under25=p_under25,
            p_btts_yes=p_btts_yes, p_btts_no=p_btts_no
        )

    # ========================================================
    # SECONDO PASSAGGIO: mapping mercati + calibrazione
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
        elif ("btts" in market) or ("both" in market) or ("goal" in market and "over" not in market and "under" not in market):
            mkt_type = "BTTS"
            if selection.startswith("y") or selection in ("si", "sì", "yes") or selection == "gol":
                p_model = mc["p_btts_yes"]
            elif selection.startswith("n") or "nogol" in selection or selection == "no":
                p_model = mc["p_btts_no"]
        # Over/Under
        elif any(x in market for x in ["over", "under", "totals", "goals"]) or "over" in selection or "under" in selection:
            mkt_type = "OU"
            p_over = prob_over_line(M, line)
            p_under = 1 - p_over if not pd.isna(p_over) else np.nan
            if "over" in market or "over" in selection:
                p_model = p_over
            elif "under" in market or "under" in selection:
                p_model = p_under

        # FIX #5: applica calibrazione
        if not pd.isna(p_model) and mkt_type != "unknown":
            p_model = apply_calibration(p_model, mkt_type)

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
            "p_model": p_model,
            "_mkt_type": mkt_type,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ Nessuna riga prodotta.")
        return df

    df["ou_gap"] = (df["p_over25"] + df["p_under25"]) - 1.0
    df["btts_gap"] = (df["p_btts_yes"] + df["p_btts_no"]) - 1.0

    # ========================================================
    # FIX #4: Market anchoring per TUTTI i mercati
    # ========================================================
    df["odds_num"] = pd.to_numeric(df["odds"], errors="coerce")
    df["p_market_raw"] = np.where(df["odds_num"] > 0, 1.0 / df["odds_num"], np.nan)
    df["p_market_fair"] = np.nan

    # 1X2: normalizza per bookmaker
    mask_1x2 = df["_mkt_type"] == "1X2"
    for (d, h, a, bk), grp in df[mask_1x2].groupby(["date", "home", "away", "bookmaker"]):
        s = grp["p_market_raw"].sum()
        if s > 0:
            df.loc[grp.index, "p_market_fair"] = grp["p_market_raw"] / s

    # OU/BTTS: normalizza per coppia over+under / yes+no
    for mtype in ["OU", "BTTS"]:
        mask = df["_mkt_type"] == mtype
        for (d, h, a, bk, mkt), grp in df[mask].groupby(["date", "home", "away", "bookmaker", "market"]):
            s = grp["p_market_raw"].sum()
            if s > 0:
                df.loc[grp.index, "p_market_fair"] = grp["p_market_raw"] / s

    # Blend p_hat = modello + mercato
    df["p_hat"] = df["p_model"]

    # 1X2
    idx = df[mask_1x2 & df["p_model"].notna() & df["p_market_fair"].notna()].index
    if len(idx) > 0:
        df.loc[idx, "p_hat"] = (1 - W_MARKET_1X2) * df.loc[idx, "p_model"] + W_MARKET_1X2 * df.loc[idx, "p_market_fair"]

    # FIX #4: OU/BTTS ora hanno anche anchoring
    for mtype in ["OU", "BTTS"]:
        mask = df["_mkt_type"] == mtype
        idx = df[mask & df["p_model"].notna() & df["p_market_fair"].notna()].index
        if len(idx) > 0:
            df.loc[idx, "p_hat"] = (1 - W_MARKET_OU_BTTS) * df.loc[idx, "p_model"] + W_MARKET_OU_BTTS * df.loc[idx, "p_market_fair"]

    # EV
    df["EV"] = np.where(
        (df["odds_num"] > 0) & df["p_hat"].notna(),
        df["p_hat"] * df["odds_num"] - 1.0,
        np.nan
    )

    # Ordina colonne
    cols_first = [
        "date", "league", "home", "away", "bookmaker", "market", "selection",
        "odds", "line", "p_hat", "EV", "p_model", "p_market_fair",
        "mu_home", "mu_away", "p_home", "p_draw", "p_away",
        "p_over25", "p_under25", "p_btts_yes", "p_btts_no",
        "ou_gap", "btts_gap", "p_market_raw"
    ]
    drop_cols = ["odds_num", "_mkt_type"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    cols = [c for c in cols_first if c in df.columns] + [c for c in df.columns if c not in cols_first]
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
    mu_h = df["mu_home"].mean()
    mu_a = df["mu_away"].mean()
    gap = (df["p_hat"] - df["p_market_raw"]).mean() * 100

    print(f"📊 EV medio: {ev_mean:.3f} | EV+: {ev_pos:.0f}% | μ_h: {mu_h:.2f} | μ_a: {mu_a:.2f}")
    print(f"   Anchoring: {anchor:.0f}% | Gap vs mercato: {gap:+.1f}%")

    if ev_mean > 0.15:
        print(f"   ⚠️ EV medio ancora alto ({ev_mean:.3f}), considerare più shrinkage")
    if ev_pos > 50:
        print(f"   ⚠️ {ev_pos:.0f}% EV+, troppe value bets")

    return df
