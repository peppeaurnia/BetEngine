"""
📊 TRACKER — Salvataggio automatico pronostici su Excel
========================================================
Salva ogni pronostico consigliato con data, lega, partita, mercato,
selezione, probabilità, quota. Permette di calcolare accuracy per
lega e tipo di pronostico.

File: predictions_tracker.xlsx
"""

import os
import pandas as pd
from datetime import datetime, date


TRACKER_FILE = "predictions_tracker.xlsx"

# Colonne del tracker
COLUMNS = [
    "data",           # Data partita (YYYY-MM-DD)
    "lega",           # Nome lega (es. "Serie A")
    "lega_id",        # ID lega (es. 135)
    "casa",           # Squadra casa
    "trasferta",      # Squadra trasferta
    "mercato",        # Tipo: 1X2, OU, Cards
    "selezione",      # Es: "1", "Over 2.5", "Cart. O3.5"
    "probabilita",    # Probabilità modello (%)
    "quota",          # Quota bookmaker (se disponibile)
    "ev_pct",         # Expected Value % (se quota disponibile)
    "stelle",         # Rating 1-5
    "anchored",       # Se market anchoring applicato (True/False)
    "risultato",      # VUOTO — da compilare: "W" o "L"
    "note",           # Note opzionali
    "timestamp",      # Quando è stato salvato
]


def load_tracker() -> pd.DataFrame:
    """Carica il file tracker esistente o ne crea uno nuovo."""
    if os.path.exists(TRACKER_FILE):
        try:
            df = pd.read_excel(TRACKER_FILE)
            # Assicura che tutte le colonne esistano
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def save_tracker(df: pd.DataFrame):
    """Salva il tracker su Excel."""
    try:
        df.to_excel(TRACKER_FILE, index=False, engine='openpyxl')
        return True
    except Exception as e:
        print(f"Errore salvataggio tracker: {e}")
        # Fallback: salva come CSV
        try:
            df.to_csv(TRACKER_FILE.replace('.xlsx', '.csv'), index=False)
            return True
        except:
            return False


def save_predictions(predictions: list, match_info: dict) -> int:
    """
    Salva i pronostici consigliati nel tracker.
    
    Args:
        predictions: Lista di dict con pronostici (da calc_top_preds)
        match_info: Dict con {date, league_name, league_id, home, away, anchored}
    
    Returns:
        Numero di pronostici salvati
    """
    df = load_tracker()
    
    match_date = match_info.get("date", date.today().strftime("%Y-%m-%d"))
    league_name = match_info.get("league_name", "")
    league_id = match_info.get("league_id", 0)
    home = match_info.get("home", "")
    away = match_info.get("away", "")
    anchored = match_info.get("anchored", False)
    
    saved = 0
    for pred in predictions:
        # Controlla se già salvato (stessa data + stessa partita + stessa selezione)
        exists = df[
            (df['data'] == match_date) & 
            (df['casa'] == home) & 
            (df['trasferta'] == away) & 
            (df['selezione'] == pred.get('name', ''))
        ]
        if len(exists) > 0:
            continue  # Già salvato, skip
        
        row = {
            "data": match_date,
            "lega": league_name,
            "lega_id": league_id,
            "casa": home,
            "trasferta": away,
            "mercato": pred.get('mt', ''),
            "selezione": pred.get('name', ''),
            "probabilita": round(pred.get('prob', 0), 1),
            "quota": pred.get('odds') if pred.get('has_odds') else None,
            "ev_pct": round(pred.get('ev_pct', 0), 1) if pred.get('ev_pct') is not None else None,
            "stelle": pred.get('stars', 0),
            "anchored": anchored,
            "risultato": "",
            "note": "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        saved += 1
    
    if saved > 0:
        save_tracker(df)
    
    return saved


def get_stats() -> dict:
    """
    Calcola le statistiche di accuracy dal tracker.
    
    Returns:
        Dict con stats per lega e per mercato
    """
    df = load_tracker()
    
    if len(df) == 0:
        return {"total": 0, "by_league": {}, "by_market": {}}
    
    # Solo pronostici con risultato compilato
    completed = df[df['risultato'].isin(['W', 'L', 'w', 'l'])].copy()
    completed['risultato'] = completed['risultato'].str.upper()
    
    stats = {
        "total": len(df),
        "completed": len(completed),
        "pending": len(df) - len(completed),
    }
    
    if len(completed) > 0:
        wins = len(completed[completed['risultato'] == 'W'])
        stats['win_rate'] = round(wins / len(completed) * 100, 1)
        stats['wins'] = wins
        stats['losses'] = len(completed) - wins
        
        # Per lega
        by_league = {}
        for league in completed['lega'].unique():
            lg = completed[completed['lega'] == league]
            w = len(lg[lg['risultato'] == 'W'])
            by_league[league] = {
                'total': len(lg),
                'wins': w,
                'losses': len(lg) - w,
                'win_rate': round(w / len(lg) * 100, 1) if len(lg) > 0 else 0
            }
        stats['by_league'] = by_league
        
        # Per mercato
        by_market = {}
        for market in completed['mercato'].unique():
            mk = completed[completed['mercato'] == market]
            w = len(mk[mk['risultato'] == 'W'])
            by_market[market] = {
                'total': len(mk),
                'wins': w,
                'losses': len(mk) - w,
                'win_rate': round(w / len(mk) * 100, 1) if len(mk) > 0 else 0
            }
        stats['by_market'] = by_market
    else:
        stats['win_rate'] = None
        stats['by_league'] = {}
        stats['by_market'] = {}
    
    return stats
