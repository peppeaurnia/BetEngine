"""
🧮 PROBABILITY ENGINE - Motore di calcolo probabilità
=====================================================
Questo modulo contiene tutta la matematica per calcolare le probabilità:
- Distribuzione Poisson univariata e bivariata
- Correlazione Dixon-Coles (λ₃)
- Calcolo 1X2, Over/Under, BTTS
- Matrice punteggi esatti

Autore: Sistema sviluppato con Peppe
Versione: 2.0 (Gennaio 2025)
"""

import numpy as np
from math import exp, factorial
from typing import Dict, Tuple, Optional

# ============================================================
# PARAMETRI DEL MODELLO (v4 — Aprile 2026)
# ============================================================
MAX_GOALS = 10          # Massimo gol per squadra nella matrice

# Dampening delle strength (comprime valori estremi)
POWER_DAMP = 0.70       # Comprime strength estremi (0.5→0.62, 2.0→1.62)
SHRINKAGE = 0.15        # Tira strength verso 1.0 del 15%

# Soft adjustment: peso di forma + momentum (rank RIMOSSO — già nelle strength)
SOFT_ADJ_WEIGHT = 0.25  # Peso per forma/momentum

# Floor gol attesi
MIN_TOTAL_MU = 1.8      # Gol attesi minimi per partita

# v4: CALIBRAZIONE A LIVELLO MU (sostituisce la vecchia calibrazione post-hoc)
# Tira mu verso la media di lega — riduce overconfidence senza rompere coerenza
MU_SHRINK = 0.85        # 1.0 = nessun shrinkage, 0.0 = tutto appiattito a media lega

# ============================================================
# CALIBRAZIONE (v4)
# ============================================================
# La calibrazione post-hoc con breakpoints manuali (v3) è stata rimossa.
# Era fragile, aveva discontinuità nella derivata (coefficiente 0.4→4.0),
# e rompeva la coerenza tra mercati (1X2 calibrato ≠ matrice ≠ O/U calibrato).
#
# v4: La calibrazione avviene a livello di μ (expected goals) PRIMA di 
# costruire la matrice Poisson. Questo garantisce che TUTTE le probabilità
# derivate (1X2, O/U, BTTS, esatti) siano coerenti tra loro.
# Metodo: James-Stein shrinkage → tira μ verso la media di lega.

# λ₃ calibrato empiricamente per ogni lega (correlazione gol)
LEAGUE_LAMBDA3 = {
    39:  0.08,   # Premier League (partite aperte)
    135: 0.06,   # Serie A (tattica, meno gol correlati)
    140: 0.05,   # LaLiga (possesso palla)
    78:  0.10,   # Bundesliga (pressing alto, contropiedi)
    61:  0.08,   # Ligue 1
    94:  0.08,   # Primeira Liga
    88:  0.08,   # Eredivisie
}

# Medie di lega di fallback (aumentate per essere più realistiche)
DEFAULT_LEAGUE_AVG = {
    "gf_home": 1.55,  # Gol fatti in casa (media europea)
    "gf_away": 1.25,  # Gol fatti fuori casa
}


# ============================================================
# FUNZIONI MATEMATICHE BASE
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    """
    Funzione di massa di probabilità Poisson.
    P(X = k) = e^(-λ) × λ^k / k!
    
    Args:
        k: Numero di eventi (gol)
        lam: Valore atteso (λ)
    
    Returns:
        Probabilità che X = k
    """
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def clamp_lambda(lam: float, min_val: float = 0.4, max_val: float = 4.5) -> float:
    """
    Limita λ in un range ragionevole per stabilità numerica.
    
    Range [0.4, 4.5]:
    - 0.4: Una squadra che segna pochissimo
    - 4.5: Una squadra estremamente offensiva (es. big vs neopromossa)
    
    Args:
        lam: Valore lambda grezzo
        min_val: Minimo ammesso
        max_val: Massimo ammesso
    
    Returns:
        Lambda limitato nel range
    """
    if lam is None or np.isnan(lam):
        return 1.3  # Default: media ragionevole
    return float(np.clip(lam, min_val, max_val))


def clamp_strength(strength: float, min_val: float = 0.45, max_val: float = 2.0) -> float:
    """
    Limita i valori di forza squadra in un range realistico.
    
    Range [0.45, 2.0]:
    - 0.45: Squadra molto debole (neopromossa in difficoltà)
    - 2.0: Squadra dominante (es. Inter/Juve attacco vs neopromossa)
    
    Args:
        strength: Valore di forza grezzo
        min_val: Minimo ammesso
        max_val: Massimo ammesso
    
    Returns:
        Forza limitata nel range
    """
    if strength is None or np.isnan(strength):
        return 1.0  # Default: media
    return float(np.clip(strength, min_val, max_val))


# ============================================================
# CORRELAZIONE DIXON-COLES (λ₃)
# ============================================================

def get_lambda3(mu_home: float, mu_away: float, league_id: int = None) -> float:
    """
    Calcola il parametro di correlazione λ₃ per la Poisson bivariata.
    
    λ₃ cattura la correlazione tra gol casa e trasferta:
    - Valori più alti = partite più "aperte" (se una segna, anche l'altra)
    - Valori più bassi = partite più tattiche
    
    Args:
        mu_home: Gol attesi squadra casa
        mu_away: Gol attesi squadra trasferta
        league_id: ID della lega (per calibrazione specifica)
    
    Returns:
        Valore λ₃
    """
    # Se la lega è nota, usa il valore calibrato
    if league_id is not None and league_id in LEAGUE_LAMBDA3:
        base = LEAGUE_LAMBDA3[league_id]
    else:
        base = 0.08  # Default per leghe sconosciute
    
    # Leggero aggiustamento basato sulla "vivacità" della partita
    # Partite più offensive tendono ad avere più correlazione
    if mu_home > 0 and mu_away > 0:
        total_expected = mu_home + mu_away
        # Da 2.0 gol (difensivo) a 3.5+ (offensivo): max +0.03
        adjustment = 0.015 * min((total_expected - 2.0) / 1.5, 0.5)
        return float(np.clip(base + adjustment, 0.04, 0.14))
    
    return base


# ============================================================
# POISSON BIVARIATA E MATRICE PROBABILITÀ
# ============================================================

def bivariate_poisson_matrix(mu_home: float, mu_away: float, 
                              lambda3: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Costruisce la matrice di probabilità per la distribuzione Poisson bivariata.
    
    La Poisson bivariata modella la correlazione tra gol casa e trasferta
    usando un terzo parametro λ₃ (componente comune).
    
    Formula:
    P(X=i, Y=j) = Σ_k [ P_poisson(i-k, λ₁) × P_poisson(j-k, λ₂) × P_poisson(k, λ₃) ]
    
    dove:
    - λ₁ = mu_home - λ₃ (componente indipendente casa)
    - λ₂ = mu_away - λ₃ (componente indipendente trasferta)
    - λ₃ = correlazione (componente comune)
    
    Args:
        mu_home: Gol attesi squadra casa
        mu_away: Gol attesi squadra trasferta
        lambda3: Parametro di correlazione
        max_goals: Massimo gol da considerare
    
    Returns:
        Matrice (max_goals+1) x (max_goals+1) con probabilità congiunte
        M[i,j] = P(home=i, away=j)
    """
    # Componenti indipendenti (non possono essere negative)
    lam1 = max(mu_home - lambda3, 0.01)
    lam2 = max(mu_away - lambda3, 0.01)
    
    # Inizializza matrice
    M = np.zeros((max_goals + 1, max_goals + 1))
    
    # Calcola ogni cella della matrice
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_sum = 0.0
            # Somma su k (componente comune)
            for k in range(min(i, j) + 1):
                prob_sum += (poisson_pmf(i - k, lam1) * 
                            poisson_pmf(j - k, lam2) * 
                            poisson_pmf(k, lambda3))
            M[i, j] = prob_sum
    
    # Normalizza per assicurare che la somma sia 1
    total = M.sum()
    if total > 0:
        M /= total
    
    return M


# ============================================================
# CALCOLO PROBABILITÀ DAI RISULTATI
# ============================================================

def calculate_1x2(M: np.ndarray) -> Dict[str, float]:
    """
    Calcola le probabilità 1X2 dalla matrice.
    
    La matrice M[i,j] rappresenta P(home=i, away=j)
    - Vittoria casa: home > away, quindi i > j (triangolo INFERIORE)
    - Pareggio: home = away, quindi i = j (diagonale)
    - Vittoria trasferta: away > home, quindi j > i (triangolo SUPERIORE)
    
    Args:
        M: Matrice Poisson bivariata
    
    Returns:
        Dict con p_home, p_draw, p_away
    """
    if M is None or M.size == 0:
        return {"p_home": 0.33, "p_draw": 0.33, "p_away": 0.34}
    
    # Vittoria casa: i > j → triangolo INFERIORE (sotto la diagonale)
    p_home = float(np.tril(M, k=-1).sum())
    
    # Pareggio: diagonale principale (i = j)
    p_draw = float(np.trace(M))
    
    # Vittoria trasferta: j > i → triangolo SUPERIORE (sopra la diagonale)
    p_away = float(np.triu(M, k=1).sum())
    
    # Normalizza per sicurezza
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
    """
    Calcola le probabilità Over/Under per una data linea.
    
    Args:
        M: Matrice Poisson bivariata
        line: Linea di gol (es. 2.5, 3.5)
    
    Returns:
        Dict con p_over e p_under
    """
    if M is None or M.size == 0:
        return {"p_over": 0.50, "p_under": 0.50}
    
    n = M.shape[0]
    threshold = int(np.floor(line)) + 1  # Per Over 2.5, serve >= 3 gol
    
    # Somma tutte le celle dove i+j >= threshold
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
    """
    Calcola le probabilità BTTS (Both Teams To Score / Gol-NoGol).
    
    BTTS Yes: entrambe le squadre segnano almeno 1 gol
    BTTS No: almeno una squadra non segna
    
    Args:
        M: Matrice Poisson bivariata
    
    Returns:
        Dict con p_btts_yes e p_btts_no
    """
    if M is None or M.size == 0:
        return {"p_btts_yes": 0.50, "p_btts_no": 0.50}
    
    # BTTS Yes: esclude riga 0 e colonna 0 (quando qualcuno non segna)
    M_btts = M.copy()
    M_btts[0, :] = 0  # Casa non segna
    M_btts[:, 0] = 0  # Trasferta non segna
    
    p_btts_yes = float(M_btts.sum())
    p_btts_no = 1.0 - p_btts_yes
    
    return {
        "p_btts_yes": round(p_btts_yes, 4),
        "p_btts_no": round(p_btts_no, 4)
    }


def calculate_exact_scores(M: np.ndarray, top_n: int = 10) -> list:
    """
    Restituisce i punteggi esatti più probabili.
    
    Args:
        M: Matrice Poisson bivariata
        top_n: Numero di risultati da restituire
    
    Returns:
        Lista di tuple (home_goals, away_goals, probability)
    """
    if M is None or M.size == 0:
        return []
    
    scores = []
    n = M.shape[0]
    
    for i in range(n):
        for j in range(n):
            scores.append((i, j, M[i, j]))
    
    # Ordina per probabilità decrescente
    scores.sort(key=lambda x: x[2], reverse=True)
    
    # Formatta e restituisce i top N
    return [(h, a, round(p * 100, 2)) for h, a, p in scores[:top_n]]


# ============================================================
# CALCOLO μ (EXPECTED GOALS) PER SQUADRA
# ============================================================

def calculate_expected_goals(home_stats: Dict, away_stats: Dict, 
                             league_id: int = None) -> Tuple[float, float]:
    """
    Calcola i gol attesi per entrambe le squadre. (v4)
    
    Formula semplificata (4 fattori, non più 7):
    μ_home = base_gf_home × attack_home × defense_away × soft_adj_home
    μ_away = base_gf_away × attack_away × defense_home × soft_adj_away
    
    Poi applica James-Stein shrinkage verso la media di lega per ridurre
    overconfidence senza rompere la coerenza tra mercati.
    
    v4 CAMBIAMENTI rispetto a v3:
    - RIMOSSO rank_diff_boost (era double-counting con rank_factor nelle strength)
    - RIMOSSO big team bonus (arbitrario, non giustificato dai dati)
    - RIMOSSO correzione difesa-classifica (sovrascriveva dati reali)
    - RIMOSSO rank_factor dal soft_adj (già catturato nelle strength)
    - AGGIUNTO mu shrinkage (James-Stein) come calibrazione pulita
    """
    # Estrai forze (con clamping + dampening + shrinkage bayesiano)
    def _dampen(s):
        """Applica power dampening + shrinkage verso 1.0."""
        s = clamp_strength(s)
        s = s + SHRINKAGE * (1.0 - s)  # Tira verso 1.0
        if s > 0:
            s = s ** POWER_DAMP  # Comprime estremi
        return s
    
    att_home = _dampen(home_stats.get("attack_home", 1.0))
    def_home = _dampen(home_stats.get("defense_home", 1.0))
    att_away = _dampen(away_stats.get("attack_away", 1.0))
    def_away = _dampen(away_stats.get("defense_away", 1.0))
    
    # Soft adjustment: solo forma + momentum (rank rimosso — già nelle strength)
    def soft_adj(form, momentum):
        """Combina forma e momentum recente."""
        if form is None or np.isnan(form):
            form = 1.0
        if momentum is None or np.isnan(momentum):
            momentum = 1.0
        combined = (form + momentum) / 2.0
        return 1.0 + SOFT_ADJ_WEIGHT * (combined - 1.0)
    
    adj_home = soft_adj(
        home_stats.get("form_factor", 1.0),
        home_stats.get("momentum", 1.0)
    )
    adj_away = soft_adj(
        away_stats.get("form_factor", 1.0),
        away_stats.get("momentum", 1.0)
    )
    
    # Media gol della lega (base per il calcolo)
    base_gf_home = home_stats.get("league_avg_gf_home", DEFAULT_LEAGUE_AVG["gf_home"])
    base_gf_away = away_stats.get("league_avg_gf_away", DEFAULT_LEAGUE_AVG["gf_away"])
    
    if base_gf_home is None or np.isnan(base_gf_home) or base_gf_home <= 0:
        base_gf_home = DEFAULT_LEAGUE_AVG["gf_home"]
    if base_gf_away is None or np.isnan(base_gf_away) or base_gf_away <= 0:
        base_gf_away = DEFAULT_LEAGUE_AVG["gf_away"]
    
    # Calcolo μ grezzo (4 fattori: base × attacco × difesa × soft_adj)
    mu_home_raw = base_gf_home * att_home * def_away * adj_home
    mu_away_raw = base_gf_away * att_away * def_home * adj_away
    
    # === v4: JAMES-STEIN SHRINKAGE ===
    # Tira μ verso la media di lega. Riduce overconfidence in modo liscio.
    # mu_calibrated = league_avg + MU_SHRINK * (mu_raw - league_avg)
    mu_home = base_gf_home + MU_SHRINK * (mu_home_raw - base_gf_home)
    mu_away = base_gf_away + MU_SHRINK * (mu_away_raw - base_gf_away)
    
    # Clamp per stabilità
    mu_home = clamp_lambda(mu_home)
    mu_away = clamp_lambda(mu_away)
    
    # Floor minimo gol totali
    total_mu = mu_home + mu_away
    if total_mu < MIN_TOTAL_MU:
        scale = MIN_TOTAL_MU / total_mu
        mu_home *= scale
        mu_away *= scale
    
    return mu_home, mu_away


# ============================================================
# FUNZIONE PRINCIPALE: CALCOLA TUTTE LE PROBABILITÀ
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
    
    Questa è la funzione principale che orchestra tutti i calcoli:
    1. Calcola μ per entrambe le squadre
    2. Applica aggiustamenti H2H e tiri
    3. Calcola λ₃ per la correlazione
    4. Costruisce la matrice Poisson bivariata
    5. Estrae tutte le probabilità
    6. Calcola probabilità cartellini con aggiustamento arbitro
    
    Args:
        home_stats: Dict con statistiche squadra casa
        away_stats: Dict con statistiche squadra trasferta
        league_id: ID della lega
        h2h_data: Dict con dati scontri diretti
        home_shots: Dict con media tiri squadra casa
        away_shots: Dict con media tiri squadra trasferta
        referee_name: Nome dell'arbitro (opzionale)
        league_name: Nome della lega per aggiustamento arbitro specifico
    
    Returns:
        Dict completo con tutte le probabilità e metriche
    """
    # 1. Calcola expected goals base
    mu_home, mu_away = calculate_expected_goals(home_stats, away_stats, league_id)
    
    # 2. Applica aggiustamento H2H
    h2h_adj_home, h2h_adj_away, h2h_goals_boost = calculate_h2h_adjustment(h2h_data)
    mu_home *= h2h_adj_home
    mu_away *= h2h_adj_away
    
    # 3. Applica aggiustamento tiri
    shots_adj_home = calculate_shots_adjustment(home_shots)
    shots_adj_away = calculate_shots_adjustment(away_shots)
    mu_home *= shots_adj_home
    mu_away *= shots_adj_away
    
    # Ri-applica clamping dopo aggiustamenti
    mu_home = clamp_lambda(mu_home)
    mu_away = clamp_lambda(mu_away)
    
    # 4. Calcola λ₃ (aumentato se H2H ha tanti gol)
    lambda3 = get_lambda3(mu_home, mu_away, league_id)
    if h2h_goals_boost > 0:
        lambda3 = min(lambda3 * (1 + h2h_goals_boost * 0.1), 0.15)
    
    # 5. Costruisci matrice
    M = bivariate_poisson_matrix(mu_home, mu_away, lambda3)
    
    # 6. Calcola tutte le probabilità
    probs_1x2 = calculate_1x2(M)
    probs_btts = calculate_btts(M)
    
    # Over/Under per varie linee
    probs_ou = {}
    for line in [1.5, 2.5, 3.5, 4.5]:
        ou = calculate_over_under(M, line)
        probs_ou[f"over_{line}"] = ou["p_over"]
        probs_ou[f"under_{line}"] = ou["p_under"]
    
    # Punteggi esatti più probabili
    exact_scores = calculate_exact_scores(M, top_n=10)
    
    # 7. Carica dati arbitro (se disponibile)
    referee_data = None
    if referee_name:
        try:
            # Prova import relativo (se usato come modulo)
            from .fetch_referee_stats import get_referee_adjustment
            referee_data = get_referee_adjustment(referee_name, league_name)
        except ImportError:
            try:
                # Prova import assoluto (se usato come script)
                from fetch_referee_stats import get_referee_adjustment
                referee_data = get_referee_adjustment(referee_name, league_name)
            except ImportError:
                # Modulo non disponibile
                referee_data = {"found": False, "severity_factor": 1.0}
    
    # 8. Calcola probabilità cartellini con aggiustamento arbitro
    probs_cards = calculate_cards_probabilities(home_stats, away_stats, referee_data)
    
    # v4: NESSUNA calibrazione post-hoc.
    # La calibrazione avviene a livello μ (James-Stein shrinkage in calculate_expected_goals).
    # Questo garantisce che 1X2, O/U, BTTS e risultati esatti siano tutti coerenti
    # con la stessa matrice Poisson bivariata.
    
    return {
        # Metriche base
        "mu_home": round(mu_home, 3),
        "mu_away": round(mu_away, 3),
        "lambda3": round(lambda3, 4),
        "total_expected_goals": round(mu_home + mu_away, 2),
        
        # Aggiustamenti applicati (per debug/trasparenza)
        "h2h_adj_home": round(h2h_adj_home, 3),
        "h2h_adj_away": round(h2h_adj_away, 3),
        "shots_adj_home": round(shots_adj_home, 3),
        "shots_adj_away": round(shots_adj_away, 3),
        
        # 1X2
        **probs_1x2,
        
        # BTTS
        **probs_btts,
        
        # Over/Under
        **probs_ou,
        
        # Cartellini (include anche info arbitro: referee_name, referee_found, etc.)
        **probs_cards,
        
        # Punteggi esatti
        "exact_scores": exact_scores,
        
        # Matrice (per visualizzazione avanzata)
        "matrix": M
    }


def calculate_h2h_adjustment(h2h_data: Dict) -> Tuple[float, float, float]:
    """
    Calcola gli aggiustamenti basati sugli scontri diretti. (v4)
    
    v4: Il peso dell'H2H scala con il numero di partite disponibili.
    Con 3 partite l'effetto è minimo (~30%), con 10+ è pieno.
    Questo evita di dare troppa importanza a campioni troppo piccoli.
    """
    if not h2h_data or h2h_data.get("matches", 0) < 3:
        return 1.0, 1.0, 0.0
    
    matches = h2h_data["matches"]
    team1_wins = h2h_data.get("team1_wins", 0)
    team2_wins = h2h_data.get("team2_wins", 0)
    avg_goals = h2h_data.get("avg_goals", 2.5)
    
    # v4: Peso proporzionale al campione (3 partite → 0.3, 10 → 1.0)
    sample_weight = min(matches / 10.0, 1.0)
    
    # Dominanza nei precedenti
    team1_dominance = team1_wins / matches if matches > 0 else 0.5
    team2_dominance = team2_wins / matches if matches > 0 else 0.5
    
    # Aggiustamento scalato per campione
    adj_home = 1.0 + (team1_dominance - 0.5) * 0.20 * sample_weight
    adj_away = 1.0 + (team2_dominance - 0.5) * 0.20 * sample_weight
    
    # Clamp
    adj_home = np.clip(adj_home, 0.92, 1.10)
    adj_away = np.clip(adj_away, 0.92, 1.10)
    
    # Goals boost (anche questo scalato per campione)
    goals_boost = ((avg_goals - 2.5) / 2.5 * sample_weight) if avg_goals > 2.5 else 0
    goals_boost = np.clip(goals_boost, 0, 0.20)
    
    return adj_home, adj_away, goals_boost


def calculate_shots_adjustment(shots_data: Dict) -> float:
    """
    Calcola l'aggiustamento basato sui tiri. (v4)
    
    v4: Ridotto impatto massimo da ±15% a ±10%.
    I tiri sono indicativi ma rumorosi su 5 partite.
    """
    if not shots_data or shots_data.get("matches_analyzed", 0) == 0:
        return 1.0
    
    shots_avg = shots_data.get("shots_avg", 12)
    shots_on_target = shots_data.get("shots_on_target_avg", 4)
    
    REFERENCE_SHOTS = 12.0
    REFERENCE_ON_TARGET = 4.0
    
    shots_factor = shots_avg / REFERENCE_SHOTS if REFERENCE_SHOTS > 0 else 1.0
    on_target_factor = shots_on_target / REFERENCE_ON_TARGET if REFERENCE_ON_TARGET > 0 else 1.0
    
    # Tiri in porta contano di più
    combined = 0.4 * shots_factor + 0.6 * on_target_factor
    
    # v4: ridotto da 0.20 a 0.15 (max ±10% circa)
    adjustment = 1.0 + (combined - 1.0) * 0.15
    
    return float(np.clip(adjustment, 0.90, 1.10))


def calculate_cards_probabilities(home_stats: Dict, away_stats: Dict, 
                                   referee_data: Dict = None) -> Dict:
    """
    Calcola le probabilità Over/Under per i cartellini.
    Usa distribuzione Poisson semplice sul totale cartellini attesi,
    aggiustato in base alla severità dell'arbitro.
    
    Args:
        home_stats: Statistiche squadra casa
        away_stats: Statistiche squadra trasferta
        referee_data: Dict con dati arbitro (severity_factor, avg_cards, etc.)
    
    Returns:
        Dict con probabilità per ogni linea cartellini
    """
    # Media cartellini per squadra (default se non disponibile)
    home_cards = home_stats.get("total_cards_avg", 1.9)
    away_cards = away_stats.get("total_cards_avg", 1.9)
    
    # Fattore partita: le partite tendono ad avere più cartellini
    # rispetto alla somma delle medie individuali (effetto competizione)
    match_factor = 1.05
    
    # Lambda totale atteso per la partita (base)
    lambda_cards_base = (home_cards + away_cards) * match_factor
    
    # ============================================================
    # AGGIUSTAMENTO ARBITRO
    # ============================================================
    referee_adjustment = 1.0
    referee_info = {}
    
    if referee_data and referee_data.get("found"):
        severity = referee_data.get("severity_factor", 1.0)
        
        # L'aggiustamento è proporzionale alla deviazione dalla media
        # severity = 1.0 -> nessun aggiustamento
        # severity = 1.3 -> arbitro 30% più severo -> +15% cartellini attesi
        # severity = 0.7 -> arbitro 30% più permissivo -> -15% cartellini attesi
        # 
        # Usiamo un peso moderato (0.5) per non sovrappesare l'arbitro
        # rispetto alle caratteristiche delle squadre
        REFEREE_WEIGHT = 0.5
        referee_adjustment = 1.0 + (severity - 1.0) * REFEREE_WEIGHT
        
        # Clamp per evitare estremi
        referee_adjustment = np.clip(referee_adjustment, 0.80, 1.25)
        
        referee_info = {
            "referee_found": True,
            "referee_name": referee_data.get("name", ""),  # Nome completo dal database
            "referee_severity": round(severity, 3),
            "referee_adjustment": round(referee_adjustment, 3),
            "referee_avg_cards": referee_data.get("avg_cards"),
            "referee_matches": referee_data.get("matches", 0)
        }
    else:
        referee_info = {
            "referee_found": False,
            "referee_name": None,
            "referee_severity": 1.0,
            "referee_adjustment": 1.0,
            "referee_avg_cards": None,
            "referee_matches": 0
        }
    
    # Lambda finale con aggiustamento arbitro
    lambda_cards = lambda_cards_base * referee_adjustment
    
    # Clamp per stabilità
    lambda_cards = max(2.0, min(lambda_cards, 8.0))
    
    # Calcola probabilità cumulative con Poisson
    probs_cards = {}
    
    for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
        # P(X > line) = 1 - P(X <= floor(line))
        p_under = 0
        for k in range(int(line) + 1):
            p_under += poisson_pmf(k, lambda_cards)
        
        p_over = 1 - p_under
        
        probs_cards[f"cards_over_{line}"] = round(p_over, 4)
        probs_cards[f"cards_under_{line}"] = round(p_under, 4)
    
    # Aggiungi lambda e info per riferimento
    probs_cards["expected_cards"] = round(lambda_cards, 2)
    probs_cards["expected_cards_base"] = round(lambda_cards_base, 2)
    probs_cards["home_cards_avg"] = round(home_cards, 2)
    probs_cards["away_cards_avg"] = round(away_cards, 2)
    
    # Aggiungi info arbitro
    probs_cards.update(referee_info)
    
    return probs_cards


# ============================================================
# UTILITY: VALUTAZIONE QUALITÀ PREVISIONE
# ============================================================

def assess_prediction_quality(home_stats: Dict, away_stats: Dict) -> Dict:
    """
    Valuta la qualità/affidabilità della previsione basandosi
    sulla completezza dei dati disponibili.
    
    Args:
        home_stats: Statistiche squadra casa
        away_stats: Statistiche squadra trasferta
    
    Returns:
        Dict con score (0-100) e messaggio
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
