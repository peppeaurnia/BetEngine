"""
📡 DATA FETCHER - Recupero dati da API-Football
===============================================
Questo modulo gestisce tutte le chiamate a API-Football:
- Lista squadre per lega
- Statistiche squadra
- Classifica
- Forma e momentum

Include sistema di caching per ridurre chiamate API.

Autore: Sistema sviluppato con Peppe
Versione: 2.0 (Gennaio 2025)
"""

import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import streamlit as st

# ============================================================
# CONFIGURAZIONE API
# ============================================================
BASE_URL = "https://v3.football.api-sports.io"

# Leghe supportate (ID API-Football -> Nome con bandiera)
LEAGUES = {
    39: {"name": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "country": "England"},
    135: {"name": "🇮🇹 Serie A", "country": "Italy"},
    140: {"name": "🇪🇸 LaLiga", "country": "Spain"},
    78: {"name": "🇩🇪 Bundesliga", "country": "Germany"},
    61: {"name": "🇫🇷 Ligue 1", "country": "France"},
    94: {"name": "🇵🇹 Primeira Liga", "country": "Portugal"},
    88: {"name": "🇳🇱 Eredivisie", "country": "Netherlands"},
}


# ============================================================
# FUNZIONI HELPER
# ============================================================

def _get_headers(api_key: str) -> Dict:
    """Genera headers per le richieste API."""
    return {"x-apisports-key": api_key}


def _safe_float(x) -> float:
    """Converte in float in modo sicuro."""
    try:
        val = float(x)
        return val if not np.isnan(val) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _form_factor(form_str: str) -> float:
    """
    Trasforma la stringa form (es. 'WDWLW') in un moltiplicatore.
    W=3, D=1, L=0 -> normalizzato in range [0.8, 1.2]
    """
    if not form_str or not isinstance(form_str, str):
        return 1.0
    
    mapping = {"W": 3, "D": 1, "L": 0}
    vals = [mapping.get(ch.upper(), 0) for ch in form_str.strip()]
    
    if not vals:
        return 1.0
    
    # Normalizza: media punti / max possibile (3) -> range 0-1
    avg = sum(vals) / (3 * len(vals))
    # Scala a range 0.8-1.2
    return 0.8 + 0.4 * avg


def _rank_factor(rank: int, total: int) -> float:
    """
    Moltiplicatore basato sulla posizione in classifica.
    1° posto -> 1.25, ultimo -> 0.75
    """
    if not rank or not total or rank <= 0:
        return 1.0
    
    # Normalizza posizione (1° = 1.0, ultimo = 0.0)
    x = 1 - (rank - 1) / max(total - 1, 1)
    # Scala a range 0.75-1.25 (aumentato per maggiore impatto)
    return 0.75 + 0.5 * x


# ============================================================
# FUNZIONI API
# ============================================================

def get_current_season() -> int:
    """
    Determina la stagione calcistica corrente.
    Le stagioni europee vanno da agosto a maggio.
    Es: Agosto 2024 - Maggio 2025 = stagione 2024
    """
    now = datetime.now()
    # Se siamo tra gennaio e luglio, la stagione è iniziata l'anno precedente
    if now.month <= 7:
        return now.year - 1
    else:
        return now.year


@st.cache_data(ttl=3600, show_spinner=False)  # Cache 1 ora
def fetch_teams_for_league(api_key: str, league_id: int, season: int = None) -> List[Dict]:
    """
    Recupera lista squadre per una lega.
    
    Args:
        api_key: Chiave API-Football
        league_id: ID della lega
        season: Stagione (default: stagione calcistica corrente)
    
    Returns:
        Lista di dict con id e name delle squadre
    """
    if season is None:
        season = get_current_season()
    
    url = f"{BASE_URL}/teams?league={league_id}&season={season}"
    
    try:
        response = requests.get(url, headers=_get_headers(api_key), timeout=15)
        response.raise_for_status()
        data = response.json()
        
        teams = []
        for item in data.get("response", []):
            team = item.get("team", {})
            teams.append({
                "id": team.get("id"),
                "name": team.get("name"),
                "logo": team.get("logo")
            })
        
        return sorted(teams, key=lambda x: x.get("name", ""))
    
    except requests.exceptions.RequestException as e:
        st.error(f"Errore API: {str(e)}")
        return []


@st.cache_data(ttl=1800, show_spinner=False)  # Cache 30 minuti
def fetch_team_statistics(api_key: str, team_id: int, league_id: int, 
                         season: int = None) -> Dict:
    """
    Recupera statistiche complete per una squadra.
    Combina stagione corrente (90%) e precedente (10%).
    
    Args:
        api_key: Chiave API-Football
        team_id: ID squadra
        league_id: ID lega
        season: Stagione (default: stagione calcistica corrente)
    
    Returns:
        Dict con tutte le statistiche
    """
    if season is None:
        season = get_current_season()
    
    def _fetch_year(year: int) -> Dict:
        """Recupera stats per un singolo anno."""
        url = f"{BASE_URL}/teams/statistics?team={team_id}&league={league_id}&season={year}"
        try:
            r = requests.get(url, headers=_get_headers(api_key), timeout=15)
            r.raise_for_status()
            resp = r.json().get("response", {})
            if isinstance(resp, list) and resp:
                resp = resp[0]
            return resp if resp else {}
        except Exception:
            return {}
    
    # Recupera dati stagione corrente e precedente
    curr = _fetch_year(season)
    prev = _fetch_year(season - 1)
    
    def _get_goals(stat_type: str, side: str) -> float:
        """Estrae media gol con fallback a totali."""
        # Prima prova media diretta
        c = _safe_float(
            curr.get("goals", {}).get(stat_type, {}).get("average", {}).get(side)
        )
        p = _safe_float(
            prev.get("goals", {}).get(stat_type, {}).get("average", {}).get(side)
        )
        
        # Se non c'è media, calcola da totali
        if np.isnan(c):
            total = curr.get("goals", {}).get(stat_type, {}).get("total", {}).get(side)
            played = curr.get("fixtures", {}).get("played", {}).get(side)
            if total and played:
                total_val = total.get("total") if isinstance(total, dict) else total
                c = _safe_float(total_val) / max(_safe_float(played), 1)
        
        if np.isnan(p):
            total = prev.get("goals", {}).get(stat_type, {}).get("total", {}).get(side)
            played = prev.get("fixtures", {}).get("played", {}).get(side)
            if total and played:
                total_val = total.get("total") if isinstance(total, dict) else total
                p = _safe_float(total_val) / max(_safe_float(played), 1)
        
        # Combina: 90% corrente, 10% precedente
        if np.isnan(c) and np.isnan(p):
            return np.nan
        if np.isnan(c):
            return p
        if np.isnan(p):
            return c
        return 0.9 * c + 0.1 * p
    
    # Estrai medie gol
    goals_for_home = _get_goals("for", "home")
    goals_against_home = _get_goals("against", "home")
    goals_for_away = _get_goals("for", "away")
    goals_against_away = _get_goals("against", "away")
    
    # Estrai cartellini
    def _get_cards() -> Dict:
        """Estrae statistiche cartellini."""
        # Calcola cartellini medi per partita dalla stagione corrente
        cards_curr = curr.get("cards", {})
        cards_prev = prev.get("cards", {})
        
        # Somma tutti i cartellini gialli
        yellow_total_curr = 0
        yellow_total_prev = 0
        red_total_curr = 0
        red_total_prev = 0
        
        for period in ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "91-105", "106-120"]:
            yt_c = cards_curr.get("yellow", {}).get(period, {}).get("total")
            yt_p = cards_prev.get("yellow", {}).get(period, {}).get("total")
            rt_c = cards_curr.get("red", {}).get(period, {}).get("total")
            rt_p = cards_prev.get("red", {}).get(period, {}).get("total")
            
            if yt_c: yellow_total_curr += int(yt_c)
            if yt_p: yellow_total_prev += int(yt_p)
            if rt_c: red_total_curr += int(rt_c)
            if rt_p: red_total_prev += int(rt_p)
        
        # Partite giocate
        played_curr = curr.get("fixtures", {}).get("played", {}).get("total", 1) or 1
        played_prev = prev.get("fixtures", {}).get("played", {}).get("total", 1) or 1
        
        # Media per partita
        yellow_avg_curr = yellow_total_curr / played_curr if played_curr else 0
        yellow_avg_prev = yellow_total_prev / played_prev if played_prev else 0
        red_avg_curr = red_total_curr / played_curr if played_curr else 0
        red_avg_prev = red_total_prev / played_prev if played_prev else 0
        
        # Combina 90% corrente, 10% precedente
        if yellow_avg_curr == 0 and yellow_avg_prev == 0:
            yellow_avg = 1.8  # Default medio Serie A
        elif yellow_avg_curr == 0:
            yellow_avg = yellow_avg_prev
        elif yellow_avg_prev == 0:
            yellow_avg = yellow_avg_curr
        else:
            yellow_avg = 0.9 * yellow_avg_curr + 0.1 * yellow_avg_prev
        
        if red_avg_curr == 0 and red_avg_prev == 0:
            red_avg = 0.08  # Default medio
        elif red_avg_curr == 0:
            red_avg = red_avg_prev
        elif red_avg_prev == 0:
            red_avg = red_avg_curr
        else:
            red_avg = 0.9 * red_avg_curr + 0.1 * red_avg_prev
        
        return {
            "yellow_cards_avg": yellow_avg,
            "red_cards_avg": red_avg,
            "total_cards_avg": yellow_avg + red_avg
        }
    
    cards_stats = _get_cards()
    
    # Form factor
    form_str = curr.get("form", "")
    form_mult = _form_factor(form_str)
    
    return {
        "goals_for_home": goals_for_home,
        "goals_against_home": goals_against_home,
        "goals_for_away": goals_for_away,
        "goals_against_away": goals_against_away,
        "form_factor": form_mult,
        "form_string": form_str,
        "fixtures_played_home": curr.get("fixtures", {}).get("played", {}).get("home", 0),
        "fixtures_played_away": curr.get("fixtures", {}).get("played", {}).get("away", 0),
        "yellow_cards_avg": cards_stats["yellow_cards_avg"],
        "red_cards_avg": cards_stats["red_cards_avg"],
        "total_cards_avg": cards_stats["total_cards_avg"],
    }


@st.cache_data(ttl=3600, show_spinner=False)  # Cache 1 ora
def fetch_standings(api_key: str, league_id: int, season: int = None) -> List[Dict]:
    """
    Recupera la classifica della lega.
    
    Returns:
        Lista di dict con rank, team_name, team_id, points, etc.
    """
    if season is None:
        season = get_current_season()
    
    url = f"{BASE_URL}/standings?league={league_id}&season={season}"
    
    try:
        response = requests.get(url, headers=_get_headers(api_key), timeout=15)
        response.raise_for_status()
        data = response.json()
        
        resp = data.get("response", [])
        if not resp:
            return []
        
        standings_data = resp[0].get("league", {}).get("standings", [])
        if not standings_data:
            return []
        
        # Prendi il primo gruppo (alcuni campionati hanno gironi)
        standings = standings_data[0] if standings_data else []
        
        result = []
        for item in standings:
            team = item.get("team", {})
            result.append({
                "rank": item.get("rank"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "points": item.get("points"),
                "played": item.get("all", {}).get("played"),
                "won": item.get("all", {}).get("win"),
                "drawn": item.get("all", {}).get("draw"),
                "lost": item.get("all", {}).get("lose"),
                "goals_for": item.get("all", {}).get("goals", {}).get("for"),
                "goals_against": item.get("all", {}).get("goals", {}).get("against"),
                "goal_diff": item.get("goalsDiff"),
                "form": item.get("form")
            })
        
        return result
    
    except requests.exceptions.RequestException as e:
        st.error(f"Errore API standings: {str(e)}")
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_last_fixtures(api_key: str, team_id: int, league_id: int, 
                        season: int = None, last_n: int = 5) -> List[Dict]:
    """
    Recupera le ultime N partite di una squadra.
    Utile per calcolare momentum.
    """
    if season is None:
        season = get_current_season()
    
    url = f"{BASE_URL}/fixtures?team={team_id}&league={league_id}&season={season}&last={last_n}"
    
    try:
        response = requests.get(url, headers=_get_headers(api_key), timeout=15)
        response.raise_for_status()
        data = response.json()
        
        fixtures = []
        for item in data.get("response", []):
            fixture = item.get("fixture", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            
            fixtures.append({
                "date": fixture.get("date"),
                "home_team": teams.get("home", {}).get("name"),
                "away_team": teams.get("away", {}).get("name"),
                "home_id": teams.get("home", {}).get("id"),
                "away_id": teams.get("away", {}).get("id"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "home_winner": teams.get("home", {}).get("winner"),
                "away_winner": teams.get("away", {}).get("winner"),
            })
        
        return fixtures
    
    except Exception:
        return []


def calculate_momentum(fixtures: List[Dict], team_id: int) -> float:
    """
    Calcola momentum basato sui punti delle ultime partite.
    Media punti -> moltiplicatore [0.85, 1.15]
    """
    if not fixtures:
        return 1.0
    
    points = 0
    for f in fixtures:
        is_home = f.get("home_id") == team_id
        if is_home:
            if f.get("home_winner") is True:
                points += 3
            elif f.get("home_winner") is None and f.get("away_winner") is None:
                points += 1
        else:
            if f.get("away_winner") is True:
                points += 3
            elif f.get("home_winner") is None and f.get("away_winner") is None:
                points += 1
    
    # Media punti per partita (max 3)
    avg_points = points / len(fixtures)
    
    # Scala: 1.5 punti/partita = neutro (1.0)
    # 3.0 punti = +15%, 0 punti = -15%
    momentum = 1.0 + 0.1 * ((avg_points / 1.5) - 1.0)
    return max(min(momentum, 1.15), 0.85)


# ============================================================
# FUNZIONE PRINCIPALE: OTTIENI STATS COMPLETE PER IL MATCH
# ============================================================

def get_match_stats(api_key: str, home_team_id: int, away_team_id: int,
                    league_id: int, season: int = None) -> Tuple[Dict, Dict, Dict]:
    """
    Ottiene tutte le statistiche necessarie per calcolare le probabilità.
    
    Returns:
        Tuple (home_stats, away_stats, league_info)
    """
    if season is None:
        season = get_current_season()
    
    # 1. Recupera statistiche base squadre
    home_stats_raw = fetch_team_statistics(api_key, home_team_id, league_id, season)
    away_stats_raw = fetch_team_statistics(api_key, away_team_id, league_id, season)
    
    # 2. Recupera classifica per rank factor
    standings = fetch_standings(api_key, league_id, season)
    total_teams = len(standings)
    
    home_rank = None
    away_rank = None
    for s in standings:
        if s["team_id"] == home_team_id:
            home_rank = s["rank"]
        if s["team_id"] == away_team_id:
            away_rank = s["rank"]
    
    # 3. Calcola rank factor
    home_rank_factor = _rank_factor(home_rank, total_teams)
    away_rank_factor = _rank_factor(away_rank, total_teams)
    
    # 4. Recupera ultime partite per momentum
    home_fixtures = fetch_last_fixtures(api_key, home_team_id, league_id, season)
    away_fixtures = fetch_last_fixtures(api_key, away_team_id, league_id, season)
    
    home_momentum = calculate_momentum(home_fixtures, home_team_id)
    away_momentum = calculate_momentum(away_fixtures, away_team_id)
    
    # 5. Calcola medie di lega
    league_stats = calculate_league_averages(standings)
    
    # 6. Calcola forze relative
    def _calc_strength(team_stat: float, league_avg: float) -> float:
        if np.isnan(team_stat) or np.isnan(league_avg) or league_avg <= 0:
            return 1.0
        return team_stat / league_avg
    
    # Assembla statistiche finali
    home_stats = {
        "attack_home": _calc_strength(home_stats_raw.get("goals_for_home", np.nan), 
                                       league_stats.get("avg_gf_home", 1.4)),
        "defense_home": _calc_strength(home_stats_raw.get("goals_against_home", np.nan),
                                        league_stats.get("avg_ga_home", 1.1)),
        "attack_away": _calc_strength(home_stats_raw.get("goals_for_away", np.nan),
                                       league_stats.get("avg_gf_away", 1.1)),
        "defense_away": _calc_strength(home_stats_raw.get("goals_against_away", np.nan),
                                        league_stats.get("avg_ga_away", 1.4)),
        "form_factor": home_stats_raw.get("form_factor", 1.0),
        "form_string": home_stats_raw.get("form_string", ""),
        "rank_factor": home_rank_factor,
        "momentum": home_momentum,
        "rank": home_rank,
        "league_avg_gf_home": league_stats.get("avg_gf_home", 1.4),
        "league_avg_gf_away": league_stats.get("avg_gf_away", 1.1),
        "matches_played": home_stats_raw.get("fixtures_played_home", 0) + home_stats_raw.get("fixtures_played_away", 0),
    }
    
    away_stats = {
        "attack_home": _calc_strength(away_stats_raw.get("goals_for_home", np.nan),
                                       league_stats.get("avg_gf_home", 1.4)),
        "defense_home": _calc_strength(away_stats_raw.get("goals_against_home", np.nan),
                                        league_stats.get("avg_ga_home", 1.1)),
        "attack_away": _calc_strength(away_stats_raw.get("goals_for_away", np.nan),
                                       league_stats.get("avg_gf_away", 1.1)),
        "defense_away": _calc_strength(away_stats_raw.get("goals_against_away", np.nan),
                                        league_stats.get("avg_ga_away", 1.4)),
        "form_factor": away_stats_raw.get("form_factor", 1.0),
        "form_string": away_stats_raw.get("form_string", ""),
        "rank_factor": away_rank_factor,
        "momentum": away_momentum,
        "rank": away_rank,
        "league_avg_gf_home": league_stats.get("avg_gf_home", 1.4),
        "league_avg_gf_away": league_stats.get("avg_gf_away", 1.1),
        "matches_played": away_stats_raw.get("fixtures_played_home", 0) + away_stats_raw.get("fixtures_played_away", 0),
    }
    
    league_info = {
        "id": league_id,
        "name": LEAGUES.get(league_id, {}).get("name", f"League {league_id}"),
        "total_teams": total_teams,
        **league_stats
    }
    
    return home_stats, away_stats, league_info


def calculate_league_averages(standings: List[Dict]) -> Dict:
    """
    Calcola medie di lega dalla classifica.
    
    NOTA: Questo è un calcolo approssimato.
    Per dati più precisi servirebbe una chiamata API aggiuntiva.
    """
    if not standings:
        return {
            "avg_gf_home": 1.45,
            "avg_ga_home": 1.15,
            "avg_gf_away": 1.15,
            "avg_ga_away": 1.45,
        }
    
    total_gf = sum(s.get("goals_for", 0) or 0 for s in standings)
    total_ga = sum(s.get("goals_against", 0) or 0 for s in standings)
    total_played = sum(s.get("played", 0) or 0 for s in standings)
    
    # Stima: circa 55% dei gol vengono segnati in casa
    HOME_RATIO = 0.55
    
    if total_played > 0:
        avg_gf_total = total_gf / total_played
        avg_ga_total = total_ga / total_played
        
        return {
            "avg_gf_home": avg_gf_total * HOME_RATIO * 2,  # x2 perché played è per squadra
            "avg_ga_home": avg_ga_total * (1 - HOME_RATIO) * 2,
            "avg_gf_away": avg_gf_total * (1 - HOME_RATIO) * 2,
            "avg_ga_away": avg_ga_total * HOME_RATIO * 2,
        }
    
    return {
        "avg_gf_home": 1.45,
        "avg_ga_home": 1.15,
        "avg_gf_away": 1.15,
        "avg_ga_away": 1.45,
    }


# ============================================================
# FUNZIONE DI TEST API
# ============================================================

def test_api_connection(api_key: str) -> Tuple[bool, str]:
    """
    Testa se la chiave API è valida.
    
    Returns:
        Tuple (success: bool, message: str)
    """
    if not api_key or len(api_key) < 10:
        return False, "Chiave API non fornita o troppo corta"
    
    url = f"{BASE_URL}/status"
    try:
        response = requests.get(url, headers=_get_headers(api_key), timeout=10)
        response.raise_for_status()
        data = response.json()
        
        account = data.get("response", {}).get("account", {})
        requests_today = data.get("response", {}).get("requests", {})
        
        return True, f"✅ API connessa! Piano: {account.get('firstname', 'N/A')}, Richieste oggi: {requests_today.get('current', 0)}/{requests_today.get('limit_day', 'N/A')}"
    
    except requests.exceptions.RequestException as e:
        return False, f"❌ Errore connessione: {str(e)}"
