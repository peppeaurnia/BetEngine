"""
🎰 THE ODDS API - Quote Reali Bookmaker
========================================
Recupera quote reali da bookmaker per calcolare ROI accurato.
"""

import requests
from typing import Dict, Optional, List
import streamlit as st

# Mapping leghe API-Football -> The Odds API
LEAGUE_MAPPING = {
    39: "soccer_epl",           # Premier League
    135: "soccer_italy_serie_a", # Serie A
    140: "soccer_spain_la_liga", # La Liga
    78: "soccer_germany_bundesliga",  # Bundesliga
    61: "soccer_france_ligue_one",    # Ligue 1
    2: "soccer_uefa_champs_league",   # Champions League
}


def get_odds_api_key() -> Optional[str]:
    """Recupera la chiave API da secrets o config."""
    # Prima prova Streamlit secrets
    try:
        return st.secrets.get("THE_ODDS_API_KEY")
    except:
        pass
    
    # Poi prova config.py
    try:
        from config import THE_ODDS_API_KEY
        if THE_ODDS_API_KEY and "INSERISCI" not in THE_ODDS_API_KEY:
            return THE_ODDS_API_KEY
    except:
        pass
    
    return None


def fetch_match_odds(
    home_team: str, 
    away_team: str, 
    league_id: int
) -> Dict[str, float]:
    """
    Recupera le migliori quote per una partita specifica.
    
    Args:
        home_team: Nome squadra casa
        away_team: Nome squadra trasferta
        league_id: ID lega (API-Football)
    
    Returns:
        Dict con le migliori quote per ogni mercato:
        {
            '1': 1.85,      # Vittoria casa
            'X': 3.40,      # Pareggio
            '2': 4.20,      # Vittoria trasferta
            'O2.5': 1.95,   # Over 2.5
            'U2.5': 1.88,   # Under 2.5
            'GG': 1.75,     # BTTS Yes
            'NG': 2.05      # BTTS No
        }
    """
    api_key = get_odds_api_key()
    
    if not api_key:
        return {}
    
    # Converti league_id in sport_key
    sport_key = LEAGUE_MAPPING.get(league_id)
    if not sport_key:
        return {}
    
    odds_result = {}
    
    try:
        # Recupera quote 1X2
        h2h_odds = _fetch_market_odds(api_key, sport_key, home_team, away_team, "h2h")
        if h2h_odds:
            odds_result.update(h2h_odds)
        
        # Recupera quote Over/Under
        totals_odds = _fetch_market_odds(api_key, sport_key, home_team, away_team, "totals")
        if totals_odds:
            odds_result.update(totals_odds)
        
        # Recupera quote BTTS
        btts_odds = _fetch_market_odds(api_key, sport_key, home_team, away_team, "btts")
        if btts_odds:
            odds_result.update(btts_odds)
            
    except Exception as e:
        # Silenzioso in caso di errore
        pass
    
    return odds_result


def _fetch_market_odds(
    api_key: str,
    sport_key: str,
    home_team: str,
    away_team: str,
    market: str
) -> Dict[str, float]:
    """
    Recupera quote per un mercato specifico.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",  # Bookmaker europei e UK
        "markets": market if market != "btts" else "btts",
        "oddsFormat": "decimal"
    }
    
    # Correggi nome mercato per API
    if market == "btts":
        params["markets"] = "btts"
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            return {}
        
        data = response.json()
        
        # Cerca la partita
        for event in data:
            if _match_teams(event, home_team, away_team):
                return _extract_best_odds(event, market)
        
        return {}
        
    except Exception as e:
        return {}


def _match_teams(event: dict, home_team: str, away_team: str) -> bool:
    """
    Verifica se l'evento corrisponde alla partita cercata.
    Usa matching fuzzy per gestire differenze nei nomi.
    """
    api_home = event.get("home_team", "").lower()
    api_away = event.get("away_team", "").lower()
    
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    
    # Partial matching
    home_match = (
        home_lower in api_home or 
        api_home in home_lower or
        _fuzzy_match(home_lower, api_home)
    )
    
    away_match = (
        away_lower in api_away or 
        api_away in away_lower or
        _fuzzy_match(away_lower, api_away)
    )
    
    return home_match and away_match


def _fuzzy_match(name1: str, name2: str) -> bool:
    """
    Matching fuzzy per nomi squadre.
    Gestisce casi come "Inter" vs "Inter Milan" vs "Internazionale".
    """
    # Parole chiave comuni
    keywords1 = set(name1.replace("-", " ").split())
    keywords2 = set(name2.replace("-", " ").split())
    
    # Se almeno una parola significativa corrisponde
    common = keywords1 & keywords2
    
    # Rimuovi parole comuni non significative
    noise = {"fc", "ac", "as", "sc", "ss", "us", "afc", "cf", "cd", "real", "sporting"}
    significant = common - noise
    
    return len(significant) > 0


def _extract_best_odds(event: dict, market: str) -> Dict[str, float]:
    """
    Estrae le migliori quote da tutti i bookmaker per un evento.
    """
    best_odds = {}
    
    for bookmaker in event.get("bookmakers", []):
        for mkt in bookmaker.get("markets", []):
            mkt_key = mkt.get("key", "")
            
            for outcome in mkt.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price", 0)
                point = outcome.get("point")  # Per Over/Under
                
                # 1X2
                if mkt_key == "h2h":
                    if name == event.get("home_team"):
                        key = "1"
                    elif name == event.get("away_team"):
                        key = "2"
                    elif name.lower() == "draw":
                        key = "X"
                    else:
                        continue
                
                # Over/Under 2.5
                elif mkt_key == "totals" and point == 2.5:
                    if name.lower() == "over":
                        key = "O2.5"
                    elif name.lower() == "under":
                        key = "U2.5"
                    else:
                        continue
                
                # BTTS
                elif mkt_key == "btts":
                    if name.lower() == "yes":
                        key = "GG"
                    elif name.lower() == "no":
                        key = "NG"
                    else:
                        continue
                
                else:
                    continue
                
                # Salva la quota migliore (più alta)
                if key not in best_odds or price > best_odds[key]:
                    best_odds[key] = price
    
    return best_odds


def get_best_odds_for_selection(
    home_team: str,
    away_team: str,
    league_id: int,
    selection: str
) -> Optional[float]:
    """
    Recupera la migliore quota per una selezione specifica.
    
    Args:
        home_team: Nome squadra casa
        away_team: Nome squadra trasferta
        league_id: ID lega
        selection: Selezione (1, X, 2, O2.5, U2.5, GG, NG, CO3.5, CU3.5)
    
    Returns:
        Migliore quota o None se non trovata
    """
    # Per cartellini non abbiamo quote reali
    if selection.startswith("CO") or selection.startswith("CU"):
        return None
    
    # Normalizza selezione
    selection_map = {
        "1": "1",
        "X": "X", 
        "2": "2",
        "O2.5": "O2.5",
        "U2.5": "U2.5",
        "GG": "GG",
        "NG": "NG"
    }
    
    normalized = selection_map.get(selection)
    if not normalized:
        return None
    
    # Recupera tutte le quote
    all_odds = fetch_match_odds(home_team, away_team, league_id)
    
    return all_odds.get(normalized)
