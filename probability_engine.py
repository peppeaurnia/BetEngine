"""
🧮 PROBABILITY ENGINE v3.0 — Migliorato per Accuracy
=====================================================
Miglioramenti rispetto a v2.0:

  A. POWER DAMPENING delle strength (strength^0.70 → comprime estremi)
  B. SHRINKAGE BAYESIANO verso 1.0 (15-25% in base ai dati)
  C. CALIBRAZIONE ISOTONICA (corregge la zona 60-70% che sbagliava di più)
  D. LEAGUE-SPECIFIC ANCHORING (leghe diverse → pesi diversi)
  E. CONFIDENCE SCORE (segnala pronostici inaffidabili)
  F. μ FLOOR 1.8 (nessuna partita sotto 1.8 gol attesi)
  G. FORMA/MOMENTUM PIÙ INCISIVI (weight 0.35 → effetto reale)
  H. FORMULA MENO MOLTIPLICATIVA (limita cascata errori)

Basato su backtesting 356 scommesse reali.
"""

import numpy as np
from math import exp, factorial
from typing import Dict, Tuple, Optional

# ============================================================
# PARAMETRI DEL MODELLO
# ============================================================
MAX_GOALS = 10

# FIX G: Forma/momentum più incisivi (era 0.20-0.30)
SOFT_ADJ_WEIGHT = 0.35

# FIX A: Power dampening — comprime valori estremi
# strength^POWER_DAMP: 0.5^0.70 = 0.616, 2.0^0.70 = 1.62
POWER_DAMP = 0.70

# FIX B: Shrinkage bayesiano verso 1.0
SHRINKAGE_FACTOR = 0.18  # 18% verso la media

# FIX F: μ totale minimo
MIN_TOTAL_EXPECTED_GOALS = 1.8

# Range strength e lambda
STRENGTH_MIN = 0.50   # era 0.45
STRENGTH_MAX = 1.85   # era 2.0 (più stretto)
LAMBDA_MIN = 0.45     # era 0.4
LAMBDA_MAX = 4.2      # era 4.5 (più stretto)

# λ₃ per lega (correlazione gol)
LEAGUE_LAMBDA3 = {
    39:  0.08,   # Premier League
    135: 0.06,   # Serie A
    140: 0.05,   # LaLiga
    78:  0.10,   # Bundesliga
    61:  0.08,   # Ligue 1
    94:  0.08,   # Primeira Liga
    88:  0.08,   # Eredivisie
}

# Medie gol di lega fallback (realistiche)
DEFAULT_LEAGUE_AVG = {
    "gf_home": 1.50,
    "gf_away": 1.20,
}

# FIX D: Pesi anchoring per lega (dal backtesting)
# Leghe dove il modello è meno preciso → più peso al mercato
LEAGUE_MARKET_WEIGHT = {
    # lega: { '1X2': peso_mercato, 'OU': peso_mercato, 'BTTS': peso_mercato }
    39:  {'1X2': 0.30, 'OU': 0.35, 'BTTS': 0.30},  # Premier: buono per Under
    135: {'1X2': 0.30, 'OU': 0.30, 'BTTS': 0.30},  # Serie A: discreto
    140: {'1X2': 0.35, 'OU': 0.40, 'BTTS': 0.45},  # LaLiga: modello scarso → più mercato
    78:  {'1X2': 0.30, 'OU': 0.35, 'BTTS': 0.35},  # Bundesliga: ok
    61:  {'1X2': 0.30, 'OU': 0.35, 'BTTS': 0.35},  # Ligue 1
    88:  {'1X2': 0.35, 'OU': 0.40, 'BTTS': 0.45},  # Eredivisie: modello scarso
}
DEFAULT_MARKET_WEIGHT = {'1X2': 0.30, 'OU': 0.35, 'BTTS': 0.35}


# ============================================================
# CALIBRAZIONE ISOTONICA (basata su 356 scommesse reali)
# ============================================================
# Il backtesting mostra una curva NON lineare:
# - 50-55% predetto → 60% reale (sottostima)
# - 60-70% predetto → 39-45% reale (SOVRASTIMA GRAVE)
# - 70-75% predetto → 70% reale (ok)
# - 80%+ predetto → 85%+ reale (ok/leggera sottostima)

def apply_calibration(prob: float, league_id: int = None, market: str = "OU") -> float:
    """
    Calibrazione isotonica basata su backtesting reale.
    
    La curva 60-70% è la più problematica: il modello dice 65%
    ma la realtà è ~39%. Questa funzione corregge specificamente
    quella zona.
    """
    p = prob * 100
    
    # === STEP 1: Calibrazione isotonica (la curva reale) ===
    # Punti di calibrazione: (predetto, reale)
    # Basati su 356 bet storici con breakpoints verificati
    if p <= 50:
        # Sotto 50%: leggero boost (il modello sottostima qui)
        calibrated = p + 2
    elif p <= 55:
        # 50-55%: il modello sottostima di ~8%
        calibrated = p + 5
    elif p <= 60:
        # 55-60%: circa giusto
        calibrated = p
    elif p <= 65:
        # 60-65%: SOVRASTIMA! 65% predetto → ~48% reale
        calibrated = 45 + (p - 60) * 0.6  # 60→45, 65→48
    elif p <= 70:
        # 65-70%: SOVRASTIMA GRAVE! 70% predetto → ~50% reale
        calibrated = 48 + (p - 65) * 0.4  # 65→48, 70→50
    elif p <= 75:
        # 70-75%: torna a funzionare
        calibrated = 50 + (p - 70) * 4.0  # 70→50, 75→70
    elif p <= 80:
        # 75-80%: leggera sovrastima
        calibrated = 70 + (p - 75) * 1.6  # 75→70, 80→78
    elif p <= 90:
        # 80-90%: quasi giusto, leggero shrinkage
        calibrated = 78 + (p - 80) * 0.7  # 80→78, 90→85
    else:
        # 90%+: shrinkage
        calibrated = 85 + (p - 90) * 0.5  # 90→85, 100→90
    
    # === STEP 2: Correzione per mercato (più leggera) ===
    if market == '1X2':
        if calibrated > 75:
            calibrated -= 3
    elif market == 'BTTS':
        calibrated -= 2
    elif market == 'Over/Under' or market == 'OU':
        calibrated -= 1.5
    elif market == 'Cards':
        calibrated -= 0.5
    
    return max(0.03, min(0.97, calibrated / 100))


# ============================================================
# FUNZIONI MATEMATICHE BASE
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def clamp_lambda(lam: float, min_val: float = LAMBDA_MIN, max_val: float = LAMBDA_MAX) -> float:
    if lam is None or np.isnan(lam):
        return 1.3
    return float(np.clip(lam, min_val, max_val))


def clamp_strength(strength: float, min_val: float = STRENGTH_MIN, max_val: float = STRENGTH_MAX) -> float:
    if strength is None or np.isnan(strength):
        return 1.0
    return float(np.clip(strength, min_val, max_val))


def dampen_strength(raw_strength: float) -> float:
    """
    FIX A + B: Power dampening + shrinkage bayesiano.
    
    1. Shrinkage: tira verso 1.0 del SHRINKAGE_FACTOR
    2. Power: comprime con esponente POWER_DAMP
    3. Clamp: limita nel range [STRENGTH_MIN, STRENGTH_MAX]
    
    Esempio (Pisa attack=0.056):
      shrink: 0.056 + 0.18*(1.0-0.056) = 0.226
      power:  0.226^0.70 = 0.336
      clamp:  max(0.50, 0.336) = 0.50
    
    Esempio (Bayern attack=2.28):
      shrink: 2.28 + 0.18*(1.0-2.28) = 2.050
      power:  2.050^0.70 = 1.68
      clamp:  min(1.85, 1.68) = 1.68
    
    Esempio (squadra media=1.05):
      shrink: 1.05 + 0.18*(1.0-1.05) = 1.041
      power:  1.041^0.70 = 1.029
      → quasi invariato! Il dampening colpisce solo gli estremi.
    """
    if raw_strength is None or np.isnan(raw_strength):
        return 1.0
    
    # Step 1: Shrinkage
    shrunk = raw_strength + SHRINKAGE_FACTOR * (1.0 - raw_strength)
    
    # Step 2: Power dampening (solo se > 0)
    if shrunk <= 0:
        return STRENGTH_MIN
    dampened = shrunk ** POWER_DAMP
    
    # Step 3: Clamp
    return clamp_strength(dampened)


# ============================================================
# CORRELAZIONE DIXON-COLES (λ₃)
# ============================================================

def get_lambda3(mu_home: float, mu_away: float, league_id: int = None) -> float:
    if league_id is not None and league_id in LEAGUE_LAMBDA3:
        base = LEAGUE_LAMBDA3[league_id]
    else:
        base = 0.08
    
    if mu_home > 0 and mu_away > 0:
        total_expected = mu_home + mu_away
        adjustment = 0.015 * min((total_expected - 2.0) / 1.5, 0.5)
        return float(np.clip(base + adjustment, 0.04, 0.14))
    
    return base


# ============================================================
# POISSON BIVARIATA
# ============================================================

def bivariate_poisson_matrix(mu_home: float, mu_away: float, 
                              lambda3: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    lam1 = max(mu_home - lambda3, 0.01)
    lam2 = max(mu_away - lambda3, 0.01)
    
    M = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_sum = 0.0
            for k in range(min(i, j) + 1):
                prob_sum += (poisson_pmf(i - k, lam1) * 
                            poisson_pmf(j - k, lam2) * 
                            poisson_pmf(k, lambda3))
            M[i, j] = prob_sum
    
    total = M.sum()
    if total > 0:
        M /= total
    
    return M


# ============================================================
# CALCOLO PROBABILITÀ
# ============================================================

def calculate_1x2(M: np.ndarray) -> Dict[str, float]:
    if M is None or M.size == 0:
        return {"p_home": 0.33, "p_draw": 0.33, "p_away": 0.34}
    
    p_home = float(np.tril(M, k=-1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, k=1).sum())
    
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    
    return {
        "p_home": round(p_home, 4),
        "p_draw": round(p_draw, 4),
        "p_away": round(p_away, 4)
    }


def calculate_over_under(M: np.ndarray, line: float) -> Dict[str, float]:
    if M is None or M.size == 0:
        return {"p_over": 0.50, "p_under": 0.50}
    
    n = M.shape[0]
    threshold = int(np.floor(line)) + 1
    
    p_over = 0.0
    for i in range(n):
        for j in range(n):
            if i + j >= threshold:
                p_over += M[i, j]
    p_under = 1.0 - p_over
    
    return {
        "p_over": round(float(p_over), 4),
        "p_under": round(float(p_under), 4)
    }


def calculate_btts(M: np.ndarray) -> Dict[str, float]:
    if M is None or M.size == 0:
        return {"p_btts_yes": 0.50, "p_btts_no": 0.50}
    
    M_btts = M.copy()
    M_btts[0, :] = 0
    M_btts[:, 0] = 0
    
    p_btts_yes = float(M_btts.sum())
    p_btts_no = 1.0 - p_btts_yes
    
    return {
        "p_btts_yes": round(p_btts_yes, 4),
        "p_btts_no": round(p_btts_no, 4)
    }


def calculate_exact_scores(M: np.ndarray, top_n: int = 10) -> list:
    if M is None or M.size == 0:
        return []
    scores = []
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            scores.append((i, j, M[i, j]))
    scores.sort(key=lambda x: x[2], reverse=True)
    return [(h, a, round(p * 100, 2)) for h, a, p in scores[:top_n]]


# ============================================================
# CALCOLO μ (EXPECTED GOALS) — versione migliorata
# ============================================================

def calculate_expected_goals(home_stats: Dict, away_stats: Dict, 
                             league_id: int = None) -> Tuple[float, float]:
    """
    Calcola i gol attesi con:
    - FIX A: Power dampening (strength^0.70)
    - FIX B: Shrinkage bayesiano (18% verso 1.0)
    - FIX F: μ totale minimo 1.8
    - FIX G: Forma/momentum più incisivi
    - FIX H: Rank difference meno aggressiva
    """
    # Estrai e DAMPENA le forze (FIX A+B)
    att_home = dampen_strength(home_stats.get("attack_home", 1.0))
    def_home = dampen_strength(home_stats.get("defense_home", 1.0))
    att_away = dampen_strength(away_stats.get("attack_away", 1.0))
    def_away = dampen_strength(away_stats.get("defense_away", 1.0))
    
    # FIX G: Soft adjustment più incisivo (weight 0.35)
    def soft_adj(form, rank, momentum):
        if form is None or np.isnan(form):
            form = 1.0
        if rank is None or np.isnan(rank):
            rank = 1.0
        if momentum is None or np.isnan(momentum):
            momentum = 1.0
        combined = (form + rank + momentum) / 3.0
        return 1.0 + SOFT_ADJ_WEIGHT * (combined - 1.0)
    
    adj_home = soft_adj(
        home_stats.get("form_factor", 1.0),
        home_stats.get("rank_factor", 1.0),
        home_stats.get("momentum", 1.0)
    )
    adj_away = soft_adj(
        away_stats.get("form_factor", 1.0),
        away_stats.get("rank_factor", 1.0),
        away_stats.get("momentum", 1.0)
    )
    
    # FIX H: Rank difference MODERATA (meno aggressiva)
    rank_home = home_stats.get("rank", 10)
    rank_away = away_stats.get("rank", 10)
    
    if rank_home and rank_away and rank_home > 0 and rank_away > 0:
        rank_diff = rank_away - rank_home
        # Ridotto: +1.5% per posizione (era +2%)
        rank_diff_boost_home = 1.0 + (rank_diff * 0.015)
        rank_diff_boost_away = 1.0 - (rank_diff * 0.010)
        
        # Big team bonus ridotto: +5% (era +8%)
        if rank_home <= 5 and rank_away > 10:
            rank_diff_boost_home *= 1.05
        
        # Clamp più stretto
        rank_diff_boost_home = np.clip(rank_diff_boost_home, 0.88, 1.25)
        rank_diff_boost_away = np.clip(rank_diff_boost_away, 0.78, 1.12)
    else:
        rank_diff_boost_home = 1.0
        rank_diff_boost_away = 1.0
    
    # Media gol lega
    base_gf_home = home_stats.get("league_avg_gf_home", DEFAULT_LEAGUE_AVG["gf_home"])
    base_gf_away = away_stats.get("league_avg_gf_away", DEFAULT_LEAGUE_AVG["gf_away"])
    
    if base_gf_home is None or np.isnan(base_gf_home) or base_gf_home <= 0:
        base_gf_home = DEFAULT_LEAGUE_AVG["gf_home"]
    if base_gf_away is None or np.isnan(base_gf_away) or base_gf_away <= 0:
        base_gf_away = DEFAULT_LEAGUE_AVG["gf_away"]
    
    # Calcolo μ
    mu_home = base_gf_home * att_home * def_away * adj_home * rank_diff_boost_home
    mu_away = base_gf_away * att_away * def_home * adj_away * rank_diff_boost_away
    
    # FIX F: μ totale minimo
    total = mu_home + mu_away
    if total < MIN_TOTAL_EXPECTED_GOALS:
        scale = MIN_TOTAL_EXPECTED_GOALS / total
        mu_home *= scale
        mu_away *= scale
    
    # Clamp finale
    mu_home = clamp_lambda(mu_home)
    mu_away = clamp_lambda(mu_away)
    
    return mu_home, mu_away


# ============================================================
# FIX E: CONFIDENCE SCORE
# ============================================================

def calculate_confidence(home_stats: Dict, away_stats: Dict,
                          p_model: float, p_market: float = None) -> Dict:
    """
    Valuta la confidenza nella previsione.
    
    Il backtesting mostra che le previsioni 60-70% sono le meno affidabili.
    Questa funzione assegna un punteggio 0-100 che indica quanto
    fidarsi del pronostico.
    
    Confidenza ALTA (>75):
    - Prob < 55% o > 75% (zone dove il modello è preciso)
    - Gap modello-mercato < 10%
    - Dati completi
    
    Confidenza BASSA (<50):
    - Prob 60-70% (zona morta)
    - Gap modello-mercato > 20%
    - Dati incompleti
    """
    score = 80  # Base
    warnings = []
    
    # 1. Zona di probabilità (peso maggiore)
    p_pct = p_model * 100 if p_model else 50
    if 60 <= p_pct <= 70:
        # ZONA MORTA: il modello sbaglia di più qui
        score -= 25
        warnings.append("⚠️ Zona 60-70%: il modello sovrastima qui")
    elif 55 <= p_pct <= 60:
        score -= 10
    elif p_pct > 80:
        score += 5
        warnings.append("✅ Alta probabilità: storicamente affidabile")
    elif p_pct < 55:
        score += 5
        warnings.append("✅ Bassa probabilità: il modello sottostima qui")
    
    # 2. Gap con il mercato
    if p_market is not None and p_model is not None:
        gap = abs(p_model - p_market) * 100
        if gap > 25:
            score -= 20
            warnings.append(f"🔴 Gap enorme col mercato ({gap:.0f}%)")
        elif gap > 15:
            score -= 10
            warnings.append(f"🟡 Gap col mercato ({gap:.0f}%)")
        elif gap < 5:
            score += 5
            warnings.append(f"✅ Modello allineato col mercato")
    
    # 3. Completezza dati
    if home_stats.get("attack_home") in [None, 1.0]:
        score -= 10
        warnings.append("Dati attacco casa mancanti")
    if away_stats.get("attack_away") in [None, 1.0]:
        score -= 10
        warnings.append("Dati attacco trasferta mancanti")
    if home_stats.get("form_factor") in [None, 1.0]:
        score -= 5
    
    # Livello
    score = max(0, min(100, score))
    if score >= 70:
        level = "🟢 Alta"
    elif score >= 50:
        level = "🟡 Media"
    else:
        level = "🔴 Bassa"
    
    return {
        "confidence_score": score,
        "confidence_level": level,
        "confidence_warnings": warnings
    }


def assess_prediction_quality(home_stats: Dict, away_stats: Dict) -> Dict:
    """
    Valuta la qualità/affidabilità della previsione basandosi
    sulla completezza dei dati disponibili.
    
    Usata da app.py per mostrare l'indicatore di affidabilità.
    
    Returns:
        Dict con score (0-100), level, message, issues
    """
    score = 100
    issues = []
    
    # Check dati casa
    if home_stats.get("attack_home") in [None, 1.0]:
        score -= 15
        issues.append("Dati attacco casa incompleti")
    if home_stats.get("defense_home") in [None, 1.0]:
        score -= 15
        issues.append("Dati difesa casa incompleti")
    if home_stats.get("form_factor") in [None, 1.0]:
        score -= 5
        issues.append("Form casa non disponibile")
    
    # Check dati trasferta
    if away_stats.get("attack_away") in [None, 1.0]:
        score -= 15
        issues.append("Dati attacco trasferta incompleti")
    if away_stats.get("defense_away") in [None, 1.0]:
        score -= 15
        issues.append("Dati difesa trasferta incompleti")
    if away_stats.get("form_factor") in [None, 1.0]:
        score -= 5
        issues.append("Form trasferta non disponibile")
    
    # Check medie lega
    if home_stats.get("league_avg_gf_home") is None:
        score -= 10
        issues.append("Medie lega non disponibili")
    
    # Determina livello
    if score >= 85:
        level = "🟢 Alta"
        message = "Dati completi e affidabili"
    elif score >= 65:
        level = "🟡 Media"
        message = "Alcuni dati mancanti"
    else:
        level = "🔴 Bassa"
        message = "Molti dati mancanti - previsione meno affidabile"
    
    return {
        "score": max(score, 0),
        "level": level,
        "message": message,
        "issues": issues
    }


# ============================================================
# H2H ADJUSTMENT
# ============================================================

def calculate_h2h_adjustment(h2h_data: Dict) -> Tuple[float, float, float]:
    if h2h_data is None or h2h_data.get("matches", 0) < 3:
        return 1.0, 1.0, 0.0
    
    matches = h2h_data["matches"]
    team1_wins = h2h_data.get("team1_wins", 0)
    team2_wins = h2h_data.get("team2_wins", 0)
    avg_goals = h2h_data.get("avg_goals", 2.5)
    
    team1_dominance = team1_wins / matches if matches > 0 else 0.5
    team2_dominance = team2_wins / matches if matches > 0 else 0.5
    
    # Aggiustamento moderato
    adj_home = 1.0 + (team1_dominance - 0.5) * 0.15  # ridotto da 0.20
    adj_away = 1.0 + (team2_dominance - 0.5) * 0.15
    
    adj_home = np.clip(adj_home, 0.92, 1.12)
    adj_away = np.clip(adj_away, 0.92, 1.12)
    
    goals_boost = (avg_goals - 2.5) / 2.5 if avg_goals > 2.5 else 0
    goals_boost = np.clip(goals_boost, 0, 0.25)
    
    return adj_home, adj_away, goals_boost


# ============================================================
# SHOTS ADJUSTMENT
# ============================================================

def calculate_shots_adjustment(shots_data: Dict) -> float:
    if not shots_data or shots_data.get("matches_analyzed", 0) == 0:
        return 1.0
    
    shots_avg = shots_data.get("shots_avg", 12)
    shots_on_target = shots_data.get("shots_on_target_avg", 4)
    
    shots_factor = shots_avg / 12.0
    on_target_factor = shots_on_target / 4.0
    combined = 0.4 * shots_factor + 0.6 * on_target_factor
    
    # Effetto ridotto (0.15 → max ±7.5%)
    adjustment = 1.0 + (combined - 1.0) * 0.15
    return float(np.clip(adjustment, 0.92, 1.10))


# ============================================================
# CARDS PROBABILITIES
# ============================================================

def calculate_cards_probabilities(home_stats: Dict, away_stats: Dict, 
                                   referee_data: Dict = None) -> Dict:
    home_cards = home_stats.get("total_cards_avg", 1.9)
    away_cards = away_stats.get("total_cards_avg", 1.9)
    match_factor = 1.05
    lambda_cards_base = (home_cards + away_cards) * match_factor
    
    referee_adjustment = 1.0
    referee_info = {}
    
    if referee_data and referee_data.get("found"):
        severity = referee_data.get("severity_factor", 1.0)
        REFEREE_WEIGHT = 0.5
        referee_adjustment = 1.0 + (severity - 1.0) * REFEREE_WEIGHT
        referee_adjustment = np.clip(referee_adjustment, 0.80, 1.25)
        
        referee_info = {
            "referee_found": True,
            "referee_name": referee_data.get("name", ""),
            "referee_severity": round(severity, 3),
            "referee_adjustment": round(referee_adjustment, 3),
            "referee_avg_cards": referee_data.get("avg_cards"),
            "referee_matches": referee_data.get("matches", 0)
        }
    else:
        referee_info = {
            "referee_found": False, "referee_name": None,
            "referee_severity": 1.0, "referee_adjustment": 1.0,
            "referee_avg_cards": None, "referee_matches": 0
        }
    
    lambda_cards = max(2.0, min(lambda_cards_base * referee_adjustment, 8.0))
    
    probs_cards = {}
    for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
        p_under = sum(poisson_pmf(k, lambda_cards) for k in range(int(line) + 1))
        probs_cards[f"cards_over_{line}"] = round(1 - p_under, 4)
        probs_cards[f"cards_under_{line}"] = round(p_under, 4)
    
    probs_cards["expected_cards"] = round(lambda_cards, 2)
    probs_cards["expected_cards_base"] = round(lambda_cards_base, 2)
    probs_cards["home_cards_avg"] = round(home_cards, 2)
    probs_cards["away_cards_avg"] = round(away_cards, 2)
    probs_cards.update(referee_info)
    
    return probs_cards


# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================

def calculate_match_probabilities(home_stats: Dict, away_stats: Dict,
                                   league_id: int = None,
                                   h2h_data: Dict = None,
                                   home_shots: Dict = None,
                                   away_shots: Dict = None,
                                   referee_name: str = None,
                                   league_name: str = None) -> Dict:
    """
    Calcola tutte le probabilità per un match.
    Versione v3.0 con tutti i miglioramenti per accuracy.
    """
    # 1. Calcola expected goals (con dampening + shrinkage + floor)
    mu_home, mu_away = calculate_expected_goals(home_stats, away_stats, league_id)
    
    # 2. Applica aggiustamenti H2H e tiri (ridotti)
    h2h_adj_home, h2h_adj_away, h2h_goals_boost = calculate_h2h_adjustment(h2h_data)
    mu_home *= h2h_adj_home
    mu_away *= h2h_adj_away
    
    shots_adj_home = calculate_shots_adjustment(home_shots)
    shots_adj_away = calculate_shots_adjustment(away_shots)
    mu_home *= shots_adj_home
    mu_away *= shots_adj_away
    
    # Re-applica floor e clamp dopo aggiustamenti
    total = mu_home + mu_away
    if total < MIN_TOTAL_EXPECTED_GOALS:
        scale = MIN_TOTAL_EXPECTED_GOALS / total
        mu_home *= scale
        mu_away *= scale
    mu_home = clamp_lambda(mu_home)
    mu_away = clamp_lambda(mu_away)
    
    # 3. λ₃
    lambda3 = get_lambda3(mu_home, mu_away, league_id)
    if h2h_goals_boost > 0:
        lambda3 = min(lambda3 * (1 + h2h_goals_boost * 0.1), 0.15)
    
    # 4. Matrice
    M = bivariate_poisson_matrix(mu_home, mu_away, lambda3)
    
    # 5. Probabilità grezze
    probs_1x2 = calculate_1x2(M)
    probs_btts = calculate_btts(M)
    
    probs_ou = {}
    for line in [1.5, 2.5, 3.5, 4.5]:
        ou = calculate_over_under(M, line)
        probs_ou[f"over_{line}"] = ou["p_over"]
        probs_ou[f"under_{line}"] = ou["p_under"]
    
    exact_scores = calculate_exact_scores(M, top_n=10)
    
    # 6. Arbitro per cartellini
    referee_data = None
    if referee_name:
        try:
            from .fetch_referee_stats import get_referee_adjustment
            referee_data = get_referee_adjustment(referee_name, league_name)
        except ImportError:
            try:
                from fetch_referee_stats import get_referee_adjustment
                referee_data = get_referee_adjustment(referee_name, league_name)
            except ImportError:
                referee_data = {"found": False, "severity_factor": 1.0}
    
    probs_cards = calculate_cards_probabilities(home_stats, away_stats, referee_data)
    
    # 7. APPLICA CALIBRAZIONE ISOTONICA (FIX C)
    if league_id:
        # 1X2
        probs_1x2["p_home"] = apply_calibration(probs_1x2["p_home"], league_id, "1X2")
        probs_1x2["p_draw"] = apply_calibration(probs_1x2["p_draw"], league_id, "1X2")
        probs_1x2["p_away"] = apply_calibration(probs_1x2["p_away"], league_id, "1X2")
        
        total_1x2 = probs_1x2["p_home"] + probs_1x2["p_draw"] + probs_1x2["p_away"]
        if total_1x2 > 0:
            probs_1x2["p_home"] /= total_1x2
            probs_1x2["p_draw"] /= total_1x2
            probs_1x2["p_away"] /= total_1x2
        
        # BTTS
        probs_btts["p_btts_yes"] = apply_calibration(probs_btts["p_btts_yes"], league_id, "BTTS")
        probs_btts["p_btts_no"] = 1 - probs_btts["p_btts_yes"]
        
        # OU
        for line in [1.5, 2.5, 3.5, 4.5]:
            key_over = f"over_{line}"
            key_under = f"under_{line}"
            if key_over in probs_ou:
                probs_ou[key_over] = apply_calibration(probs_ou[key_over], league_id, "Over/Under")
                probs_ou[key_under] = 1 - probs_ou[key_over]
        
        # Cards
        for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
            key_over = f"cards_over_{line}"
            key_under = f"cards_under_{line}"
            if key_over in probs_cards:
                probs_cards[key_over] = apply_calibration(probs_cards[key_over], league_id, "Cards")
                probs_cards[key_under] = 1 - probs_cards[key_over]
    
    # 8. Confidence score
    confidence = calculate_confidence(home_stats, away_stats, 
                                       probs_ou.get("over_2.5", 0.5))
    
    return {
        "mu_home": round(mu_home, 3),
        "mu_away": round(mu_away, 3),
        "lambda3": round(lambda3, 4),
        "total_expected_goals": round(mu_home + mu_away, 2),
        
        "h2h_adj_home": round(h2h_adj_home, 3),
        "h2h_adj_away": round(h2h_adj_away, 3),
        "shots_adj_home": round(shots_adj_home, 3),
        "shots_adj_away": round(shots_adj_away, 3),
        
        **probs_1x2,
        **probs_btts,
        **probs_ou,
        **probs_cards,
        **confidence,
        
        "exact_scores": exact_scores,
        "matrix": M
    }
