"""
📊 BACKTESTING SYSTEM - BetEngine
==================================
Sistema per salvare, tracciare e analizzare le previsioni.

Funzionalità:
- Salvataggio automatico previsioni
- Recupero risultati reali via API
- Calcolo accuracy e ROI
- Storico e statistiche
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import requests
import streamlit as st
import traceback

# Import configurazione database
from database import get_connection


# ============================================================
# KELLY CRITERION - CALCOLO STAKE DINAMICO
# ============================================================

def kelly_stake(prob: float, odds: float, bankroll: float = 200, 
                min_stake: float = 10, max_stake: float = 30) -> float:
    """
    Calcola lo stake ottimale con Kelly Criterion.
    
    Args:
        prob: Probabilità stimata dal modello (0-1)
        odds: Quota decimale del bookmaker
        bankroll: Bankroll di riferimento (default €200)
        min_stake: Stake minimo (default €10)
        max_stake: Stake massimo (default €30)
    
    Returns:
        Stake consigliato in euro
    
    Formula Kelly: f = (p * b - q) / b
    dove b = odds - 1, q = 1 - p
    
    Usa Quarter Kelly (25%) per bilanciare rischio/rendimento.
    """
    if odds <= 1 or prob <= 0:
        return 0
    
    # Formula Kelly
    kelly_pct = (prob * odds - 1) / (odds - 1)
    
    # Se Kelly negativo = no value
    if kelly_pct <= 0:
        return 0
    
    # Quarter Kelly (25%)
    kelly_pct = kelly_pct * 0.25
    
    # Cap al 15% del bankroll = €30 max
    kelly_pct = min(kelly_pct, 0.15)
    
    # Calcola stake
    stake = bankroll * kelly_pct
    
    # Arrotonda a €5
    stake = round(stake / 5) * 5
    
    # Applica limiti min/max
    if stake < min_stake:
        stake = min_stake
    if stake > max_stake:
        stake = max_stake
    
    return stake


# ============================================================
# INIZIALIZZAZIONE TABELLE
# ============================================================

def init_predictions_table():
    """Crea la tabella predictions se non esiste."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            match_id INTEGER,
            match_date DATE,
            league_id INTEGER,
            league_name VARCHAR(100),
            home_team VARCHAR(100),
            away_team VARCHAR(100),
            
            -- Previsione
            market VARCHAR(50),
            selection VARCHAR(50),
            predicted_prob DECIMAL(5,4),
            best_odds DECIMAL(5,2),
            expected_value DECIMAL(5,4),
            confidence_stars INTEGER,
            
            -- Risultato (aggiornato dopo la partita)
            home_goals INTEGER,
            away_goals INTEGER,
            actual_result VARCHAR(50),
            is_won INTEGER,
            profit DECIMAL(10,2),
            
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            
            -- Chiave univoca basata su squadre, data e mercato (non match_id!)
            UNIQUE(user_id, home_team, away_team, match_date, market, selection)
        )
    """)
    
    # Rimuovi vecchio constraint se esiste e aggiungi nuovo
    try:
        cursor.execute("""
            ALTER TABLE predictions DROP CONSTRAINT IF EXISTS predictions_user_id_match_id_market_selection_key
        """)
    except:
        pass
    
    try:
        cursor.execute("""
            ALTER TABLE predictions ADD CONSTRAINT predictions_match_unique 
            UNIQUE(user_id, home_team, away_team, match_date, market, selection)
        """)
    except:
        pass  # Già esiste
    
    # Crea indici per performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_user 
        ON predictions(user_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_date 
        ON predictions(match_date)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_market 
        ON predictions(market)
    """)
    
    # Aggiungi colonna archived se non esiste (per mantenere statistiche dopo eliminazione)
    try:
        # Verifica se la colonna esiste
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'predictions' AND column_name = 'archived'
        """)
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE predictions ADD COLUMN archived BOOLEAN DEFAULT FALSE")
            conn.commit()
    except Exception as e:
        pass
    
    conn.commit()
    cursor.close()
    conn.close()


def ensure_archived_column():
    """Assicura che la colonna archived esista nel database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'predictions' AND column_name = 'archived'
        """)
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE predictions ADD COLUMN archived BOOLEAN DEFAULT FALSE")
            conn.commit()
    except:
        pass
    finally:
        cursor.close()
        conn.close()


# ============================================================
# SALVATAGGIO PREVISIONI
# ============================================================

def save_prediction(
    user_id: int,
    match_id: int,
    match_date: str,
    league_id: int,
    league_name: str,
    home_team: str,
    away_team: str,
    market: str,
    selection: str,
    predicted_prob: float,
    best_odds: float = None,
    expected_value: float = None,
    confidence_stars: int = None
) -> bool:
    """
    Salva una previsione nel database.
    
    Args:
        user_id: ID utente che fa la previsione
        match_id: ID partita (da API-Football)
        match_date: Data partita (YYYY-MM-DD)
        league_id: ID lega
        league_name: Nome lega
        home_team: Nome squadra casa
        away_team: Nome squadra trasferta
        market: Tipo mercato (1X2, Over/Under, BTTS, Cards)
        selection: Selezione specifica (1, X, 2, Over 2.5, Under 2.5, GG, NG, etc.)
        predicted_prob: Probabilità prevista (0-1)
        best_odds: Migliore quota trovata
        expected_value: EV calcolato
        confidence_stars: Stelle confidence (1-5)
    
    Returns:
        True se salvato, False se errore o già esiste
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO predictions 
            (user_id, match_id, match_date, league_id, league_name, 
             home_team, away_team, market, selection, predicted_prob,
             best_odds, expected_value, confidence_stars)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, home_team, away_team, match_date, market, selection) 
            DO UPDATE SET
                predicted_prob = EXCLUDED.predicted_prob,
                best_odds = EXCLUDED.best_odds,
                expected_value = EXCLUDED.expected_value,
                confidence_stars = EXCLUDED.confidence_stars,
                match_id = EXCLUDED.match_id,
                updated_at = CURRENT_TIMESTAMP
        """, (
            user_id, match_id, match_date, league_id, league_name,
            home_team, away_team, market, selection, predicted_prob,
            best_odds, expected_value, confidence_stars
        ))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        # Log dettagliato per debug
        st.error(f"❌ DB Error [{market}:{selection}]: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def save_match_predictions(
    user_id: int,
    match_id: int,
    match_date: str,
    league_id: int,
    league_name: str,
    home_team: str,
    away_team: str,
    probabilities: Dict,
    top_predictions: List[Dict] = None
) -> int:
    """
    Salva tutte le previsioni di una partita (1X2, O/U, BTTS, Cards).
    
    Returns:
        Numero di previsioni salvate
    """
    saved = 0
    
    # === 1X2 ===
    markets_1x2 = [
        ('1X2', '1', probabilities.get('p_home', 0)),
        ('1X2', 'X', probabilities.get('p_draw', 0)),
        ('1X2', '2', probabilities.get('p_away', 0)),
    ]
    
    for market, selection, prob in markets_1x2:
        if prob > 0:
            if save_prediction(
                user_id, match_id, match_date, league_id, league_name,
                home_team, away_team, market, selection, prob
            ):
                saved += 1
    
    # === BTTS ===
    markets_btts = [
        ('BTTS', 'GG', probabilities.get('p_btts_yes', 0)),
        ('BTTS', 'NG', probabilities.get('p_btts_no', 0)),
    ]
    
    for market, selection, prob in markets_btts:
        if prob > 0:
            if save_prediction(
                user_id, match_id, match_date, league_id, league_name,
                home_team, away_team, market, selection, prob
            ):
                saved += 1
    
    # === Over/Under ===
    for line in [1.5, 2.5, 3.5, 4.5]:
        over_prob = probabilities.get(f'over_{line}', 0)
        under_prob = probabilities.get(f'under_{line}', 0)
        
        if over_prob > 0:
            if save_prediction(
                user_id, match_id, match_date, league_id, league_name,
                home_team, away_team, 'Over/Under', f'Over {line}', over_prob
            ):
                saved += 1
        
        if under_prob > 0:
            if save_prediction(
                user_id, match_id, match_date, league_id, league_name,
                home_team, away_team, 'Over/Under', f'Under {line}', under_prob
            ):
                saved += 1
    
    # === Cards Over/Under ===
    for line in [2.5, 3.5, 4.5, 5.5]:
        cards_over = probabilities.get(f'cards_over_{line}', 0)
        cards_under = probabilities.get(f'cards_under_{line}', 0)
        
        if cards_over > 0:
            if save_prediction(
                user_id, match_id, match_date, league_id, league_name,
                home_team, away_team, 'Cards', f'Cards Over {line}', cards_over
            ):
                saved += 1
        
        if cards_under > 0:
            if save_prediction(
                user_id, match_id, match_date, league_id, league_name,
                home_team, away_team, 'Cards', f'Cards Under {line}', cards_under
            ):
                saved += 1
    
    return saved


# ============================================================
# RECUPERO RISULTATI REALI
# ============================================================

def get_match_result_from_api(api_key: str, match_id: int, 
                              expected_home: str = None, expected_away: str = None) -> Optional[Dict]:
    """
    Recupera il risultato reale di una partita dall'API.
    
    Args:
        api_key: API key
        match_id: ID partita
        expected_home: Nome atteso squadra casa (per verifica)
        expected_away: Nome atteso squadra trasferta (per verifica)
    
    Returns:
        Dict con home_goals, away_goals, status o None se errore
    """
    url = f"https://v3.football.api-sports.io/fixtures?id={match_id}"
    headers = {"x-apisports-key": api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        # Debug: salva info errore
        if data.get("errors"):
            return {"error": str(data["errors"]), "not_finished": False}
        
        if not data.get("response") or len(data["response"]) == 0:
            return {"error": f"Nessun risultato per match_id {match_id}", "not_finished": False}
        
        if data.get("response"):
            fixture = data["response"][0]
            
            # VERIFICA: la partita corrisponde ai nomi attesi?
            if expected_home and expected_away:
                api_home = fixture["teams"]["home"]["name"].lower()
                api_away = fixture["teams"]["away"]["name"].lower()
                
                home_ok = expected_home.lower() in api_home or api_home in expected_home.lower()
                away_ok = expected_away.lower() in api_away or api_away in expected_away.lower()
                
                if not (home_ok and away_ok):
                    # Partita sbagliata! Restituisci errore per forzare ricerca per nome
                    return {
                        "error": f"ID:{match_id} è {api_home} vs {api_away}, non {expected_home} vs {expected_away}",
                        "wrong_match": True,
                        "not_finished": False
                    }
            
            status = fixture["fixture"]["status"]["short"]
            
            # Solo partite finite (FT, AET, PEN)
            if status in ["FT", "AET", "PEN"]:
                goals = fixture["goals"]
                return {
                    "home_goals": goals["home"],
                    "away_goals": goals["away"],
                    "status": status,
                    "total_goals": goals["home"] + goals["away"]
                }
            else:
                return {"status": status, "not_finished": True}
        
        return None
    except Exception as e:
        return {"error": str(e), "not_finished": False}


def search_match_result_by_teams(api_key: str, home_team: str, away_team: str, 
                                  match_date: str, league_id: int = None) -> Optional[Dict]:
    """
    Cerca il risultato di una partita tramite nomi squadre e data.
    Include anche i cartellini totali.
    """
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    
    params = {
        "date": match_date,
        "season": 2025
    }
    if league_id:
        params["league"] = league_id
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        
        if data.get("response"):
            for fixture in data["response"]:
                api_home = fixture["teams"]["home"]["name"].lower()
                api_away = fixture["teams"]["away"]["name"].lower()
                
                # Confronta nomi (case-insensitive e partial match)
                home_match = home_team.lower() in api_home or api_home in home_team.lower()
                away_match = away_team.lower() in api_away or api_away in away_team.lower()
                
                if home_match and away_match:
                    status = fixture["fixture"]["status"]["short"]
                    fixture_id = fixture["fixture"]["id"]
                    
                    if status in ["FT", "AET", "PEN"]:
                        goals = fixture["goals"]
                        
                        # Recupera cartellini dalla partita
                        total_cards = get_match_cards(api_key, fixture_id)
                        
                        return {
                            "home_goals": goals["home"],
                            "away_goals": goals["away"],
                            "status": status,
                            "total_goals": goals["home"] + goals["away"],
                            "total_cards": total_cards,
                            "fixture_id": fixture_id
                        }
                    else:
                        return {"status": status, "not_finished": True, 
                                "fixture_id": fixture_id}
        
        return None
    except Exception as e:
        return {"error": str(e), "not_finished": False}


def get_match_cards(api_key: str, fixture_id: int) -> Optional[int]:
    """
    Recupera il totale cartellini (gialli + rossi) di una partita.
    Usa l'endpoint /fixtures/events che contiene tutti gli eventi.
    """
    url = f"https://v3.football.api-sports.io/fixtures/events?fixture={fixture_id}"
    headers = {"x-apisports-key": api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        total_cards = 0
        
        if data.get("response"):
            for event in data["response"]:
                event_type = event.get("type", "").lower()
                
                # Conta cartellini gialli e rossi
                if event_type == "card":
                    detail = event.get("detail", "").lower()
                    if "yellow" in detail or "red" in detail:
                        total_cards += 1
        
        return total_cards if total_cards > 0 else None
        
    except Exception as e:
        return None


def determine_prediction_outcome(
    market: str,
    selection: str,
    home_goals: int,
    away_goals: int,
    total_cards: int = None
) -> Tuple[bool, str]:
    """
    Determina se una previsione è stata vinta o persa.
    
    Returns:
        (is_won: bool, actual_result: str)
    """
    try:
        total_goals = home_goals + away_goals
        
        # === 1X2 ===
        if market == '1X2':
            if home_goals > away_goals:
                actual = '1'
            elif home_goals < away_goals:
                actual = '2'
            else:
                actual = 'X'
            
            return (selection == actual, actual)
        
        # === BTTS ===
        elif market == 'BTTS':
            both_scored = home_goals > 0 and away_goals > 0
            actual = 'GG' if both_scored else 'NG'
            return (selection == actual, actual)
        
        # === Over/Under ===
        elif market == 'Over/Under' or 'Over' in str(selection) or 'Under' in str(selection) or str(selection).startswith('O') or str(selection).startswith('U'):
            selection_str = str(selection).strip()
            
            # Gestisci formato abbreviato: "O2.5" o "U2.5"
            if selection_str.startswith('O') and selection_str[1:2].isdigit():
                is_over = True
                try:
                    line = float(selection_str[1:])
                except ValueError:
                    return (None, None)
            elif selection_str.startswith('U') and selection_str[1:2].isdigit():
                is_over = False
                try:
                    line = float(selection_str[1:])
                except ValueError:
                    return (None, None)
            else:
                # Formato standard: "Over 2.5" o "Under 2.5"
                parts = selection_str.split()
                if len(parts) < 2:
                    return (None, None)
                
                try:
                    line = float(parts[1])
                except (ValueError, IndexError):
                    return (None, None)
                    
                is_over = parts[0].lower() == 'over'
            
            if is_over:
                won = total_goals > line
                actual = f"Over {line}" if total_goals > line else f"Under {line}"
            else:
                won = total_goals < line
                actual = f"Under {line}" if total_goals < line else f"Over {line}"
            
            return (won, actual)
        
        # === Cards ===
        elif market == 'Cards' and total_cards is not None:
            selection_str = str(selection)
            
            # Formato abbreviato: "CO3.5" o "CU3.5"
            if selection_str.startswith('CO') and selection_str[2:3].isdigit():
                is_over = True
                try:
                    line = float(selection_str[2:])  # "CO3.5" -> 3.5
                except ValueError:
                    return (None, None)
            elif selection_str.startswith('CU') and selection_str[2:3].isdigit():
                is_over = False
                try:
                    line = float(selection_str[2:])  # "CU3.5" -> 3.5
                except ValueError:
                    return (None, None)
            else:
                # Formato esteso: "Cards Over 3.5"
                parts = selection_str.split()
                if len(parts) < 3:
                    return (None, None)
                
                try:
                    line = float(parts[2])
                except (ValueError, IndexError):
                    return (None, None)
                    
                is_over = parts[1].lower() == 'over'
            
            if is_over:
                won = total_cards > line
                actual = f"Cards Over {line}" if total_cards > line else f"Cards Under {line}"
            else:
                won = total_cards < line
                actual = f"Cards Under {line}" if total_cards < line else f"Cards Over {line}"
            
            return (won, actual)
        
        return (None, None)
    
    except Exception as e:
        # In caso di qualsiasi errore, ritorna None
        return (None, None)


def update_predictions_with_results(api_key: str, user_id: int = None) -> Dict:
    """
    Aggiorna tutte le previsioni pendenti con i risultati reali.
    SEMPLIFICATO: cerca SEMPRE per nome squadra e data (più affidabile).
    
    Returns:
        Dict con statistiche aggiornamento
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Trova previsioni senza risultato per partite passate (escludi archiviate)
    query = """
        SELECT DISTINCT home_team, away_team, match_date, league_id
        FROM predictions 
        WHERE is_won IS NULL 
        AND match_date <= CURRENT_DATE
        AND (archived IS NULL OR archived = FALSE)
    """
    if user_id:
        query += f" AND user_id = {user_id}"
    
    cursor.execute(query)
    pending_matches = cursor.fetchall()
    
    stats = {
        "checked": 0,
        "updated": 0,
        "not_finished": 0,
        "errors": 0,
        "debug_matches": []
    }
    
    for match in pending_matches:
        home_team = match["home_team"]
        away_team = match["away_team"]
        match_date = str(match["match_date"])
        league_id = match.get("league_id")
        
        stats["checked"] += 1
        
        # Cerca SEMPRE per nome squadra e data (più affidabile!)
        result = search_match_result_by_teams(
            api_key, home_team, away_team, match_date, league_id
        )
        
        if not result:
            stats["errors"] += 1
            stats["debug_matches"].append(f"❌ {home_team} vs {away_team} ({match_date}) - Non trovato")
            continue
        
        if result.get("error"):
            stats["errors"] += 1
            stats["debug_matches"].append(f"❌ {home_team} vs {away_team} - {result['error']}")
            continue
        
        if result.get("not_finished"):
            stats["not_finished"] += 1
            stats["debug_matches"].append(f"⏳ {home_team} vs {away_team} - {result.get('status', 'N/A')}")
            continue
        
        home_goals = result["home_goals"]
        away_goals = result["away_goals"]
        total_cards = result.get("total_cards")  # Può essere None
        
        # Aggiorna tutte le previsioni per questa partita (cerca per nome squadra e data)
        cursor.execute("""
            SELECT id, market, selection FROM predictions
            WHERE home_team = %s AND away_team = %s AND match_date = %s AND is_won IS NULL
            AND (archived IS NULL OR archived = FALSE)
        """, (home_team, away_team, match_date))
        
        predictions = cursor.fetchall()
        
        for pred in predictions:
            is_won, actual_result = determine_prediction_outcome(
                pred["market"],
                pred["selection"],
                home_goals,
                away_goals,
                total_cards  # Passa i cartellini!
            )
            
            if is_won is not None:
                cursor.execute("""
                    UPDATE predictions SET
                        home_goals = %s,
                        away_goals = %s,
                        actual_result = %s,
                        is_won = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (home_goals, away_goals, actual_result, int(is_won), pred["id"]))
                
                stats["updated"] += 1
        
        if predictions:
            cards_info = f" | 🟨 {total_cards} carte" if total_cards else ""
            stats["debug_matches"].append(f"✅ {home_team} vs {away_team} - {home_goals}-{away_goals}{cards_info}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return stats


# ============================================================
# STATISTICHE E ANALISI
# ============================================================

def get_user_statistics(user_id: int, days: int = 30) -> Dict:
    """
    Calcola statistiche complete per un utente.
    
    Args:
        user_id: ID utente
        days: Giorni da considerare (default 30)
    
    Returns:
        Dict con tutte le statistiche
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    date_limit = datetime.now() - timedelta(days=days)
    
    # Statistiche generali
    cursor.execute("""
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(CASE WHEN is_won IS NOT NULL THEN 1 END) as settled,
            COUNT(CASE WHEN is_won = 1 THEN 1 END) as won,
            COUNT(CASE WHEN is_won = 0 THEN 1 END) as lost,
            COUNT(CASE WHEN is_won IS NULL AND match_date < CURRENT_DATE THEN 1 END) as pending_update,
            COUNT(CASE WHEN is_won IS NULL AND match_date >= CURRENT_DATE THEN 1 END) as pending_match
        FROM predictions
        WHERE user_id = %s AND created_at >= %s
    """, (user_id, date_limit))
    
    general = cursor.fetchone()
    
    # Statistiche per mercato
    cursor.execute("""
        SELECT 
            market,
            COUNT(*) as total,
            COUNT(CASE WHEN is_won = 1 THEN 1 END) as won,
            COUNT(CASE WHEN is_won = 0 THEN 1 END) as lost,
            ROUND(
                CASE WHEN COUNT(CASE WHEN is_won IS NOT NULL THEN 1 END) > 0
                THEN COUNT(CASE WHEN is_won = 1 THEN 1 END)::decimal / 
                     COUNT(CASE WHEN is_won IS NOT NULL THEN 1 END) * 100
                ELSE 0 END, 1
            ) as accuracy
        FROM predictions
        WHERE user_id = %s AND created_at >= %s
        GROUP BY market
        ORDER BY accuracy DESC
    """, (user_id, date_limit))
    
    by_market = cursor.fetchall()
    
    # Statistiche per lega
    cursor.execute("""
        SELECT 
            league_name,
            COUNT(*) as total,
            COUNT(CASE WHEN is_won = 1 THEN 1 END) as won,
            ROUND(
                CASE WHEN COUNT(CASE WHEN is_won IS NOT NULL THEN 1 END) > 0
                THEN COUNT(CASE WHEN is_won = 1 THEN 1 END)::decimal / 
                     COUNT(CASE WHEN is_won IS NOT NULL THEN 1 END) * 100
                ELSE 0 END, 1
            ) as accuracy
        FROM predictions
        WHERE user_id = %s AND created_at >= %s
        GROUP BY league_name
        ORDER BY total DESC
        LIMIT 10
    """, (user_id, date_limit))
    
    by_league = cursor.fetchall()
    
    # Trend ultimi 7 giorni
    cursor.execute("""
        SELECT 
            match_date,
            COUNT(CASE WHEN is_won = 1 THEN 1 END) as won,
            COUNT(CASE WHEN is_won = 0 THEN 1 END) as lost
        FROM predictions
        WHERE user_id = %s 
        AND match_date >= CURRENT_DATE - INTERVAL '7 days'
        AND is_won IS NOT NULL
        GROUP BY match_date
        ORDER BY match_date
    """, (user_id,))
    
    daily_trend = cursor.fetchall()
    
    # Calcola ROI con quote REALI e KELLY CRITERION
    cursor2 = conn.cursor(cursor_factory=RealDictCursor)
    cursor2.execute("""
        SELECT is_won, best_odds, predicted_prob
        FROM predictions
        WHERE user_id = %s AND created_at >= %s AND is_won IS NOT NULL
    """, (user_id, date_limit))
    
    settled_bets = cursor2.fetchall()
    cursor2.close()
    
    cursor.close()
    conn.close()
    
    # Calcola accuracy
    total_settled = general["settled"] or 0
    total_won = general["won"] or 0
    accuracy = (total_won / total_settled * 100) if total_settled > 0 else 0
    
    # ROI con KELLY CRITERION (stake dinamico €10-30)
    total_staked = 0
    total_return = 0
    
    for bet in settled_bets:
        # Calcola stake con Kelly
        prob = float(bet["predicted_prob"]) if bet["predicted_prob"] else 0.55
        odds = float(bet["best_odds"]) if bet["best_odds"] else 1.85
        
        stake = kelly_stake(prob, odds)
        
        # Se no value (stake = 0), usa minimo €10 per il calcolo storico
        if stake == 0:
            stake = 10
        
        total_staked += stake
        if bet["is_won"] == 1:
            total_return += stake * odds
    
    roi = ((total_return - total_staked) / total_staked * 100) if total_staked > 0 else 0
    profit = total_return - total_staked
    
    return {
        "general": dict(general),
        "accuracy": round(accuracy, 1),
        "roi": round(roi, 1),
        "profit": round(profit, 2),
        "by_market": [dict(m) for m in by_market],
        "by_league": [dict(l) for l in by_league],
        "daily_trend": [dict(d) for d in daily_trend]
    }


def get_monthly_roi(user_id: int) -> List[Dict]:
    """
    Calcola il ROI per ogni mese con Kelly Criterion.
    
    Returns:
        Lista di dict con: month, year, total_bets, won, lost, roi, profit
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Prendi tutte le scommesse concluse (inclusa predicted_prob per Kelly)
    cursor.execute("""
        SELECT 
            EXTRACT(YEAR FROM match_date) as year,
            EXTRACT(MONTH FROM match_date) as month,
            is_won,
            best_odds,
            predicted_prob
        FROM predictions
        WHERE user_id = %s AND is_won IS NOT NULL
        ORDER BY match_date DESC
    """, (user_id,))
    
    raw_data = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Aggrega per mese con Kelly Criterion
    monthly_data = {}
    
    for row in raw_data:
        key = f"{int(row['year'])}-{int(row['month']):02d}"
        
        if key not in monthly_data:
            monthly_data[key] = {
                'year': int(row['year']),
                'month': int(row['month']),
                'won': 0,
                'lost': 0,
                'staked': 0,
                'returns': 0
            }
        
        # Calcola stake con Kelly
        prob = float(row['predicted_prob']) if row['predicted_prob'] else 0.55
        odds = float(row['best_odds']) if row['best_odds'] else 1.85
        stake = kelly_stake(prob, odds)
        
        # Se no value, usa minimo €10
        if stake == 0:
            stake = 10
        
        monthly_data[key]['staked'] += stake
        
        if row['is_won'] == 1:
            monthly_data[key]['won'] += 1
            monthly_data[key]['returns'] += stake * odds
        else:
            monthly_data[key]['lost'] += 1
    
    # Calcola ROI per ogni mese
    result = []
    for key in sorted(monthly_data.keys(), reverse=True):
        data = monthly_data[key]
        staked = data['staked']
        returns = data['returns']
        
        roi = ((returns - staked) / staked * 100) if staked > 0 else 0
        profit = returns - staked
        total_bets = data['won'] + data['lost']
        accuracy = (data['won'] / total_bets * 100) if total_bets > 0 else 0
        
        result.append({
            'year': data['year'],
            'month': data['month'],
            'month_name': get_month_name(data['month']),
            'total_bets': total_bets,
            'won': data['won'],
            'lost': data['lost'],
            'accuracy': round(accuracy, 1),
            'roi': round(roi, 1),
            'profit': round(profit, 2)
        })
    
    return result


def get_month_name(month: int) -> str:
    """Restituisce il nome del mese in italiano."""
    months = {
        1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile",
        5: "Maggio", 6: "Giugno", 7: "Luglio", 8: "Agosto",
        9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre"
    }
    return months.get(month, str(month))


def get_predictions_history(
    user_id: int,
    limit: int = 50,
    market: str = None,
    only_settled: bool = False
) -> List[Dict]:
    """
    Recupera lo storico previsioni di un utente.
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
        SELECT 
            id, match_date, league_name, home_team, away_team,
            market, selection, predicted_prob, best_odds, expected_value,
            home_goals, away_goals, actual_result, is_won,
            created_at
        FROM predictions
        WHERE user_id = %s AND (archived IS NULL OR archived = FALSE)
    """
    params = [user_id]
    
    if market:
        query += " AND market = %s"
        params.append(market)
    
    if only_settled:
        query += " AND is_won IS NOT NULL"
    
    query += " ORDER BY match_date DESC, created_at DESC LIMIT %s"
    params.append(limit)
    
    cursor.execute(query, params)
    predictions = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return [dict(p) for p in predictions]


def get_best_predictions(user_id: int, min_prob: float = 0.55, limit: int = 20) -> List[Dict]:
    """
    Recupera le migliori previsioni (alta probabilità) ancora da giocare.
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT 
            match_date, league_name, home_team, away_team,
            market, selection, predicted_prob, best_odds, expected_value
        FROM predictions
        WHERE user_id = %s
        AND match_date >= CURRENT_DATE
        AND predicted_prob >= %s
        AND (archived IS NULL OR archived = FALSE)
        ORDER BY predicted_prob DESC
        LIMIT %s
    """, (user_id, min_prob, limit))
    
    predictions = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return [dict(p) for p in predictions]


def export_predictions_to_csv(user_id: int) -> str:
    """
    Esporta tutte le previsioni in formato CSV per analisi.
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT 
            match_date,
            league_id,
            league_name,
            home_team,
            away_team,
            market,
            selection,
            predicted_prob,
            best_odds,
            home_goals,
            away_goals,
            is_won,
            created_at
        FROM predictions
        WHERE user_id = %s
        ORDER BY match_date DESC
    """, (user_id,))
    
    predictions = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not predictions:
        return None
    
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'data_partita', 'league_id', 'lega', 'casa', 'trasferta',
        'mercato', 'selezione', 'prob_modello', 'quota',
        'gol_casa', 'gol_trasferta', 'vinta', 'creata_il'
    ])
    
    # Righe
    for p in predictions:
        writer.writerow([
            p['match_date'],
            p['league_id'],
            p['league_name'],
            p['home_team'],
            p['away_team'],
            p['market'],
            p['selection'],
            round(p['predicted_prob'] * 100, 1) if p['predicted_prob'] else '',
            p['best_odds'],
            p['home_goals'],
            p['away_goals'],
            'SI' if p['is_won'] == 1 else ('NO' if p['is_won'] == 0 else ''),
            p['created_at']
        ])
    
    return output.getvalue()


# ============================================================
# UI STREAMLIT
# ============================================================

def delete_prediction(prediction_id: int) -> bool:
    """
    Elimina o archivia una singola previsione.
    - Se già calcolata (is_won NOT NULL): archivia (mantiene statistiche)
    - Se non calcolata: elimina completamente
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Archivia le previsioni già calcolate, elimina le altre
        cursor.execute("""
            UPDATE predictions SET archived = TRUE 
            WHERE id = %s AND is_won IS NOT NULL
        """, (prediction_id,))
        
        if cursor.rowcount == 0:
            # Non era calcolata, elimina completamente
            cursor.execute("DELETE FROM predictions WHERE id = %s AND is_won IS NULL", (prediction_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Errore eliminazione: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def delete_match_predictions(user_id: int, match_id: int) -> int:
    """
    Elimina o archivia tutte le previsioni di una partita.
    - Previsioni calcolate: archiviate (mantengono statistiche)
    - Previsioni non calcolate: eliminate
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Archivia previsioni già calcolate
        cursor.execute("""
            UPDATE predictions SET archived = TRUE 
            WHERE user_id = %s AND match_id = %s AND is_won IS NOT NULL
        """, (user_id, match_id))
        archived = cursor.rowcount
        
        # Elimina previsioni non calcolate
        cursor.execute("""
            DELETE FROM predictions 
            WHERE user_id = %s AND match_id = %s AND is_won IS NULL
        """, (user_id, match_id))
        deleted = cursor.rowcount
        
        conn.commit()
        return archived + deleted
    except Exception as e:
        conn.rollback()
        print(f"Errore eliminazione: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()


def delete_all_predictions(user_id: int) -> int:
    """
    Elimina o archivia TUTTE le previsioni di un utente.
    - Previsioni calcolate: archiviate (mantengono statistiche)
    - Previsioni non calcolate: eliminate
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Archivia previsioni già calcolate
        cursor.execute("""
            UPDATE predictions SET archived = TRUE 
            WHERE user_id = %s AND is_won IS NOT NULL
        """, (user_id,))
        archived = cursor.rowcount
        
        # Elimina previsioni non calcolate
        cursor.execute("""
            DELETE FROM predictions 
            WHERE user_id = %s AND is_won IS NULL
        """, (user_id,))
        deleted = cursor.rowcount
        
        conn.commit()
        return archived + deleted
    except Exception as e:
        conn.rollback()
        print(f"Errore eliminazione: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()


def display_backtesting_dashboard(user_id: int, api_key: str):
    """Mostra la dashboard completa di backtesting."""
    
    # Assicura che la colonna archived esista (migrazione database)
    ensure_archived_column()
    
    try:
        st.header("📊 Backtesting & Statistiche")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Statistiche", 
            "📋 Storico", 
            "🔄 Aggiorna",
            "🗑️ Gestione"
        ])
        
        with tab1:
            try:
                display_statistics_tab(user_id)
            except Exception as e:
                st.error(f"Errore Statistiche: {e}")
        
        with tab2:
            try:
                display_history_tab(user_id)
            except Exception as e:
                st.error(f"Errore Storico: {e}")
        
        with tab3:
            try:
                display_update_tab(user_id, api_key)
            except Exception as e:
                st.error(f"Errore Aggiorna: {e}")
        
        with tab4:
            try:
                display_manage_tab(user_id)
            except Exception as e:
                st.error(f"Errore Gestione: {e}")
    
    except Exception as e:
        st.error(f"Errore backtesting: {e}")
        st.code(traceback.format_exc())


def display_statistics_tab(user_id: int):
    """Tab statistiche."""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        days = st.selectbox("Periodo", [3, 7, 14, 30, 90, 365], index=3, key="stats_days")
    
    with col2:
        if st.button("🔄 Aggiorna", use_container_width=True, key="refresh_stats"):
            st.rerun()
    
    # Carica statistiche
    stats = get_user_statistics(user_id, days)
    
    if stats["general"]["total_predictions"] == 0:
        st.info("📭 Nessuna previsione salvata. Calcola qualche partita per iniziare!")
        return
    
    # KPI principali
    st.markdown("### 📊 Performance Generale")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("🎯 Accuracy", f"{stats['accuracy']}%")
    
    with kpi2:
        st.metric("📈 ROI", f"{stats['roi']:+.1f}%", delta=f"€{stats['profit']:+.2f}")
    
    with kpi3:
        st.metric("✅ Vinte", stats['general']['won'])
    
    with kpi4:
        st.metric("❌ Perse", stats['general']['lost'])
    
    st.markdown("---")
    
    # Dettaglio per mercato
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Per Mercato")
        
        if stats['by_market']:
            for market in stats['by_market']:
                accuracy = market['accuracy'] or 0
                emoji = "🟢" if accuracy >= 55 else "🟡" if accuracy >= 50 else "🔴"
                
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.1); padding:10px; border-radius:8px; margin-bottom:8px;">
                    <strong style="color:#ffffff;">{emoji} {market['market']}</strong><br>
                    <span style="color:#a8d4f0;">Totale: {market['total']} | ✅ {market['won']} | ❌ {market['lost']}</span><br>
                    <span style="color:#4fc3f7; font-size:1.2em;"><strong>{accuracy}%</strong></span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Nessun dato")
    
    with col2:
        st.markdown("### 🏆 Per Lega")
        
        if stats['by_league']:
            for league in stats['by_league'][:10]:  # Mostra fino a 10 leghe
                accuracy = league['accuracy'] or 0
                emoji = "🟢" if accuracy >= 55 else "🟡" if accuracy >= 50 else "🔴"
                
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.1); padding:10px; border-radius:8px; margin-bottom:8px;">
                    <strong style="color:#ffffff;">{emoji} {league['league_name']}</strong><br>
                    <span style="color:#a8d4f0;">Totale: {league['total']}</span><br>
                    <span style="color:#4fc3f7; font-size:1.2em;"><strong>{accuracy}%</strong></span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Nessun dato")
    
    st.markdown("---")
    
    # ROI MENSILE
    st.markdown("### 📅 ROI Mensile")
    
    monthly_stats = get_monthly_roi(user_id)
    
    if monthly_stats:
        # Crea tabella
        for month_data in monthly_stats[:12]:  # Ultimi 12 mesi
            roi = month_data['roi']
            profit = month_data['profit']
            
            # Colore in base al ROI
            if roi > 5:
                roi_color = "#2ecc71"  # Verde
                emoji = "🟢"
            elif roi > 0:
                roi_color = "#f1c40f"  # Giallo
                emoji = "🟡"
            elif roi > -5:
                roi_color = "#e67e22"  # Arancione
                emoji = "🟠"
            else:
                roi_color = "#e74c3c"  # Rosso
                emoji = "🔴"
            
            profit_sign = "+" if profit >= 0 else ""
            
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.1); padding:12px; border-radius:8px; margin-bottom:8px;
                        border-left:4px solid {roi_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong style="color:#ffffff; font-size:1.1em;">
                            {emoji} {month_data['month_name']} {month_data['year']}
                        </strong><br>
                        <span style="color:#a8d4f0;">
                            🎯 {month_data['total_bets']} scommesse | 
                            ✅ {month_data['won']} | ❌ {month_data['lost']} |
                            Accuracy: {month_data['accuracy']}%
                        </span>
                    </div>
                    <div style="text-align:right;">
                        <span style="color:{roi_color}; font-size:1.4em; font-weight:bold;">
                            {roi:+.1f}%
                        </span><br>
                        <span style="color:#a8d4f0;">
                            {profit_sign}€{profit:.2f}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📭 Nessun dato mensile disponibile")
    
    st.markdown("---")
    
    # Stato
    st.markdown("### 📋 Riepilogo")
    status1, status2, status3 = st.columns(3)
    
    with status1:
        st.metric("📊 Totali", stats['general']['total_predictions'])
    
    with status2:
        st.metric("⏳ Da aggiornare", stats['general']['pending_update'])
    
    with status3:
        st.metric("🔮 Future", stats['general']['pending_match'])


def display_history_tab(user_id: int):
    """Tab storico previsioni."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_filter = st.selectbox(
            "Mercato",
            ["Tutti", "1X2", "BTTS", "Over/Under", "Cards"],
            key="history_market"
        )
    
    with col2:
        only_settled = st.checkbox("Solo concluse", value=False, key="history_settled")
    
    with col3:
        limit = st.selectbox("Mostra", [20, 50, 100, 500], index=0, key="history_limit")
    
    # Carica storico
    market = market_filter if market_filter != "Tutti" else None
    history = get_predictions_history(user_id, limit, market, only_settled)
    
    if not history:
        st.info("📭 Nessuna previsione trovata")
        return
    
    # EXPORT CSV
    st.markdown("---")
    col_exp1, col_exp2 = st.columns([3, 1])
    
    with col_exp1:
        st.markdown(f"### 📋 Ultime {len(history)} Previsioni")
    
    with col_exp2:
        # Prepara CSV
        import io
        csv_data = export_predictions_to_csv(user_id)
        if csv_data:
            st.download_button(
                label="📥 Esporta CSV",
                data=csv_data,
                file_name="betengine_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Mostra previsioni
    for pred in history:
        # Colore stato
        if pred['is_won'] is None:
            status = "⏳"
            border_color = "#f39c12"
            bg_color = "rgba(243, 156, 18, 0.2)"
        elif pred['is_won'] == 1:
            status = "✅"
            border_color = "#27ae60"
            bg_color = "rgba(39, 174, 96, 0.2)"
        else:
            status = "❌"
            border_color = "#e74c3c"
            bg_color = "rgba(231, 76, 60, 0.2)"
        
        # Formatta data
        match_date = pred['match_date']
        if isinstance(match_date, str):
            date_str = match_date
        else:
            date_str = match_date.strftime("%d/%m/%Y") if match_date else "N/A"
        
        prob_pct = pred['predicted_prob'] * 100 if pred['predicted_prob'] else 0
        
        result_str = ""
        if pred['home_goals'] is not None:
            result_str = f"<strong style='color:#4fc3f7;'>({pred['home_goals']}-{pred['away_goals']})</strong>"
        
        actual_str = ""
        if pred['actual_result']:
            actual_str = f"<span style='color:#a8d4f0;'>→ {pred['actual_result']}</span>"
        
        st.markdown(f"""
        <div style="background:{bg_color}; border-left:4px solid {border_color}; 
                    padding:12px; border-radius:8px; margin-bottom:10px;">
            <div style="color:#ffffff; font-weight:bold; font-size:1.1em;">
                {status} {pred['home_team']} vs {pred['away_team']} {result_str}
            </div>
            <div style="color:#a8d4f0; font-size:0.9em; margin-top:4px;">
                📅 {date_str} | 🏆 {pred['league_name']}
            </div>
            <div style="color:#ffffff; margin-top:8px;">
                <strong>{pred['market']}</strong>: 
                <span style="color:#4fc3f7; font-weight:bold;">{pred['selection']}</span> 
                @ {prob_pct:.0f}% {actual_str}
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_update_tab(user_id: int, api_key: str):
    """Tab aggiornamento risultati."""
    
    st.markdown("### 🔄 Aggiorna Risultati")
    st.markdown("""
    <p style="color:#a8d4f0;">
    Clicca per recuperare i risultati delle partite concluse e calcolare le statistiche.
    </p>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Aggiorna Risultati Partite", type="primary", use_container_width=True):
        with st.spinner("Recupero risultati in corso..."):
            stats = update_predictions_with_results(api_key, user_id)
        
        st.success(f"""
        ✅ Aggiornamento completato!
        
        - Partite controllate: **{stats['checked']}**
        - Previsioni aggiornate: **{stats['updated']}**
        - Non ancora concluse: {stats['not_finished']}
        - Errori API: {stats['errors']}
        """)
        
        # Mostra debug
        if stats.get('debug_matches'):
            with st.expander("🔍 Dettaglio partite"):
                for msg in stats['debug_matches']:
                    st.write(msg)
        
        if stats['updated'] > 0:
            st.balloons()
    
    st.markdown("---")
    
    # === RESET PREVISIONI ERRATE ===
    st.markdown("### 🔧 Correggi Previsioni Errate")
    st.markdown("""
    <p style="color:#e74c3c;">
    Se i risultati mostrati sono sbagliati, seleziona la partita per resettarla e riaggiornarla.
    </p>
    """, unsafe_allow_html=True)
    
    # Carica partite già aggiornate (per poterle resettare)
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT DISTINCT match_id, match_date, home_team, away_team, league_name, home_goals, away_goals
        FROM predictions
        WHERE user_id = %s AND is_won IS NOT NULL AND (archived IS NULL OR archived = FALSE)
        ORDER BY match_date DESC
        LIMIT 20
    """, (user_id,))
    
    settled_matches = cursor.fetchall()
    
    if settled_matches:
        reset_options = {
            f"{m['home_team']} vs {m['away_team']} ({m['home_goals']}-{m['away_goals']}) - {m['match_date']}": m 
            for m in settled_matches
        }
        
        selected_reset = st.selectbox(
            "Seleziona partita da correggere:",
            options=list(reset_options.keys()),
            key="reset_match"
        )
        
        if st.button("🔄 Reset e Riaggiorna", type="secondary"):
            match_to_reset = reset_options[selected_reset]
            
            # Reset previsioni di questa partita (cerca per nomi squadre e data)
            cursor2 = conn.cursor()
            cursor2.execute("""
                UPDATE predictions SET
                    home_goals = NULL,
                    away_goals = NULL,
                    actual_result = NULL,
                    is_won = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE home_team = %s AND away_team = %s AND match_date = %s AND user_id = %s
            """, (match_to_reset['home_team'], match_to_reset['away_team'], 
                  match_to_reset['match_date'], user_id))
            conn.commit()
            cursor2.close()
            
            st.success(f"✅ Reset completato per {match_to_reset['home_team']} vs {match_to_reset['away_team']}")
            st.info("Clicca 'Aggiorna Risultati Partite' per recuperare il risultato corretto.")
            st.rerun()
    else:
        st.info("Nessuna partita con risultato da correggere.")
    
    cursor.close()
    conn.close()
    
    st.markdown("---")
    
    # === AGGIORNAMENTO MANUALE ===
    st.markdown("### ✏️ Aggiornamento Manuale")
    st.markdown("""
    <p style="color:#f39c12;">
    Per partite rinviate o con errori API, inserisci il risultato manualmente.
    </p>
    """, unsafe_allow_html=True)
    
    # Carica partite con errori
    conn2 = get_connection()
    cursor2 = conn2.cursor(cursor_factory=RealDictCursor)
    
    cursor2.execute("""
        SELECT DISTINCT match_id, match_date, home_team, away_team, league_name
        FROM predictions
        WHERE user_id = %s AND is_won IS NULL AND match_date <= CURRENT_DATE
        AND (archived IS NULL OR archived = FALSE)
        ORDER BY match_date DESC
        LIMIT 20
    """, (user_id,))
    
    pending_for_manual = cursor2.fetchall()
    cursor2.close()
    conn2.close()
    
    if pending_for_manual and len(pending_for_manual) > 0:
        # Dropdown per selezionare partita
        match_options = {f"{m['home_team']} vs {m['away_team']} ({m['match_date']})": m for m in pending_for_manual}
        
        options_list = list(match_options.keys())
        if not options_list:
            st.info("Nessuna partita disponibile per aggiornamento manuale.")
        else:
            selected_match_key = st.selectbox(
                "Seleziona partita da aggiornare:",
                options=options_list,
                index=0
            )
            
            if selected_match_key and selected_match_key in match_options:
                selected_match = match_options[selected_match_key]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    home_goals = st.number_input(f"Gol {selected_match['home_team']}", min_value=0, max_value=15, value=0)
                with col2:
                    away_goals = st.number_input(f"Gol {selected_match['away_team']}", min_value=0, max_value=15, value=0)
                with col3:
                    total_cards = st.number_input("🟨 Cartellini totali", min_value=0, max_value=20, value=0, 
                                                   help="Somma cartellini gialli + rossi di entrambe le squadre")
                
                if st.button("💾 Salva Risultato Manuale", type="secondary"):
                    # Aggiorna tutte le previsioni per questa partita (cerca per nomi squadre e data)
                    conn = get_connection()
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    
                    cursor.execute("""
                        SELECT id, market, selection FROM predictions
                        WHERE home_team = %s AND away_team = %s AND match_date = %s 
                        AND user_id = %s AND is_won IS NULL
                        AND (archived IS NULL OR archived = FALSE)
                    """, (selected_match['home_team'], selected_match['away_team'],
                          selected_match['match_date'], user_id))
                    
                    predictions = cursor.fetchall()
                    updated = 0
                    
                    # Passa None se cartellini = 0 (non inserito)
                    cards_value = total_cards if total_cards > 0 else None
                    
                    for pred in predictions:
                        is_won, actual_result = determine_prediction_outcome(
                            pred["market"],
                            pred["selection"],
                            home_goals,
                            away_goals,
                            cards_value  # Passa i cartellini!
                        )
                        
                        if is_won is not None:
                            cursor.execute("""
                                UPDATE predictions SET
                                    home_goals = %s,
                                    away_goals = %s,
                                    actual_result = %s,
                                    is_won = %s,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE id = %s
                            """, (home_goals, away_goals, actual_result, int(is_won), pred["id"]))
                            updated += 1
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    st.success(f"✅ Aggiornate {updated} previsioni con risultato {home_goals}-{away_goals}!")
                    st.rerun()
    else:
        st.info("Nessuna partita da aggiornare manualmente.")
    
    st.markdown("---")
    
    # DEBUG: Mostra previsioni pendenti
    st.markdown("### ⏳ Partite da Aggiornare")
    
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # DEBUG: Mostra data corrente del database
    cursor.execute("SELECT CURRENT_DATE as today, CURRENT_TIMESTAMP as now")
    db_time = cursor.fetchone()
    st.caption(f"🕐 Data DB: {db_time['today']} | Ora: {db_time['now']}")
    
    cursor.execute("""
        SELECT DISTINCT match_id, match_date, home_team, away_team, league_name
        FROM predictions
        WHERE user_id = %s AND is_won IS NULL AND (archived IS NULL OR archived = FALSE)
        ORDER BY match_date DESC
        LIMIT 15
    """, (user_id,))
    
    pending = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if pending:
        st.markdown(f"**Trovate {len(pending)} partite senza risultato:**")
        for match in pending:
            date_str = match['match_date'].strftime("%d/%m/%Y") if match['match_date'] else "N/A"
            is_past = match['match_date'] < db_time['today'] if match['match_date'] else False
            status_icon = "✅" if is_past else "⏳"
            st.markdown(f"""
            <div style="background:rgba(243, 156, 18, 0.2); border-left:4px solid #f39c12; 
                        padding:10px; border-radius:8px; margin-bottom:8px;">
                <span style="color:#ffffff;">{status_icon} <strong>{match['home_team']} vs {match['away_team']}</strong></span><br>
                <span style="color:#a8d4f0;">📅 {date_str} | 🏆 {match['league_name']} | ID: {match['match_id']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Tutte le previsioni sono aggiornate!")


def display_manage_tab(user_id: int):
    """Tab gestione - elimina previsioni."""
    
    # Assicura che la colonna archived esista
    ensure_archived_column()
    
    st.markdown("### 🗑️ Gestione Previsioni")
    
    # Carica partite
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT DISTINCT match_id, match_date, home_team, away_team, league_name,
               COUNT(*) as num_predictions
        FROM predictions
        WHERE user_id = %s AND (archived IS NULL OR archived = FALSE)
        GROUP BY match_id, match_date, home_team, away_team, league_name
        ORDER BY match_date DESC
        LIMIT 50
    """, (user_id,))
    
    matches = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not matches:
        st.info("📭 Nessuna partita da gestire")
        return
    
    st.markdown(f"""
    <p style="color:#a8d4f0;">
    Hai <strong style="color:#4fc3f7;">{len(matches)}</strong> partite salvate. 
    Elimina quelle che non vuoi più vedere.
    <br><small style="color:#f39c12;">📊 Le partite già calcolate saranno archiviate e le statistiche mantenute!</small>
    </p>
    """, unsafe_allow_html=True)
    
    # Lista partite con checkbox
    st.markdown("---")
    
    # Container per le selezioni
    matches_to_delete = []
    
    for i, match in enumerate(matches):
        date_str = match['match_date'].strftime("%d/%m/%Y") if match['match_date'] else "N/A"
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.1); padding:10px; border-radius:8px;">
                <span style="color:#ffffff;"><strong>{match['home_team']} vs {match['away_team']}</strong></span><br>
                <span style="color:#a8d4f0; font-size:0.9em;">
                    📅 {date_str} | 🏆 {match['league_name']} | 
                    📊 {match['num_predictions']} pronostici
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🗑️", key=f"del_{match['match_id']}_{i}", help="Elimina questa partita"):
                deleted = delete_match_predictions(user_id, match['match_id'])
                if deleted > 0:
                    st.success(f"✅ Eliminati {deleted} pronostici!")
                    st.rerun()
                else:
                    st.error("❌ Errore eliminazione")
        
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
    
    # Pulsante elimina tutto
    st.markdown("---")
    st.markdown("### ⚠️ Zona Pericolosa")
    
    st.markdown("""
    <p style="color:#f39c12;">
    📊 <strong>Nota:</strong> Le previsioni già calcolate verranno archiviate, non eliminate.
    Le statistiche (accuracy, ROI, vincite/perdite) saranno mantenute!
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        confirm = st.checkbox("✅ Confermo di voler eliminare/archiviare TUTTE le previsioni", key="confirm_delete_all")
    
    with col2:
        if st.button("🗑️ ELIMINA TUTTO", type="primary", disabled=not confirm):
            deleted = delete_all_predictions(user_id)
            if deleted > 0:
                st.success(f"✅ Eliminate {deleted} previsioni!")
                st.rerun()
            else:
                st.info("Nessuna previsione da eliminare")


# ============================================================
# INIT
# ============================================================

def init_backtesting():
    """Inizializza il sistema di backtesting."""
    try:
        init_predictions_table()
        ensure_archived_column()  # Assicura che la colonna archived esista
        return True
    except Exception as e:
        print(f"Errore init backtesting: {e}")
        return False
