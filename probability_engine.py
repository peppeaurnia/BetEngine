# ============================================================
# PROBABILITY ENGINE - BetEngine
# ============================================================
# Calibrazione intelligente delle probabilità basata su 356 scommesse storiche

# ============================================================
# CALIBRAZIONE BILANCIATA
# ============================================================
# Problema: il modello sovrastima le probabilità alte (80%+ vincono solo 50-60%)
# Soluzione: shrinkage moderato + correzione leggera per mercato

def apply_calibration(prob: float, league_id: int, market: str) -> float:
    """
    Calibrazione bilanciata delle probabilità.
    
    Applica:
    1. Shrinkage moderato solo per probabilità > 80%
    2. Correzione leggera per mercato
    
    Args:
        prob: Probabilità calcolata dal modello (0-1)
        league_id: ID della lega (non usato)
        market: Tipo di mercato ('1X2', 'BTTS', 'Over/Under', 'Cards')
    
    Returns:
        Probabilità calibrata (0-1)
    """
    # Converti in percentuale
    prob_pct = prob * 100
    
    # === STEP 1: SHRINKAGE MODERATO (solo per prob > 80%) ===
    if prob_pct <= 80:
        # Sotto 80%: nessuna modifica shrinkage
        base = prob_pct
    elif prob_pct <= 90:
        # 80-90%: riduzione leggera (90% -> 86%)
        base = 80 + (prob_pct - 80) * 0.6
    else:
        # 90%+: riduzione moderata (95% -> 89%, 100% -> 92%)
        base = 86 + (prob_pct - 90) * 0.6
    
    # === STEP 2: CORREZIONE LEGGERA PER MERCATO ===
    if market == '1X2':
        # 1X2: correzione solo per probabilità alte
        if prob_pct > 85:
            calibrated = base - 6
        elif prob_pct > 75:
            calibrated = base - 3
        else:
            calibrated = base
    elif market == 'BTTS':
        # BTTS: correzione leggera (-5%)
        calibrated = base - 5
    elif market == 'Over/Under':
        # Over/Under: correzione minima (-3%)
        calibrated = base - 3
    elif market == 'Cards':
        # Cards: quasi nulla (-1%)
        calibrated = base - 1
    else:
        calibrated = base
    
    # Converti in decimale (minimo 1%, massimo 99%)
    return max(0.01, min(0.99, calibrated / 100))


# ============================================================
# λ₃ calibrato per lega (correlazione gol)
# ============================================================
LEAGUE_LAMBDA3 = {
    39:  0.08,   # Premier League (partite aperte)
    135: 0.06,   # Serie A (tattica, meno gol correlati)
    140: 0.05,   # LaLiga (possesso palla)
    78:  0.10,   # Bundesliga (pressing alto, contropiedi)
    61:  0.08,   # Ligue 1
    94:  0.08,   # Primeira Liga
    88:  0.08,   # Eredivisie
}

# Medie di lega di fallback
DEFAULT_LEAGUE_AVG = {
    "gf_home": 1.55,
    "gf_away": 1.25,
    "ga_home": 1.25,
    "ga_away": 1.55,
}
