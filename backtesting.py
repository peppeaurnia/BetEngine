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

# Import configurazione database
from database import get_connection


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
            
            -- Indici per query veloci
            UNIQUE(user_id, match_id, market, selection)
        )
    """)
    
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
    
    conn.commit()
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
            ON CONFLICT (user_id, match_id, market, selection) 
            DO UPDATE SET
                predicted_prob = EXCLUDED.predicted_prob,
                best_odds = EXCLUDED.best_odds,
                expected_value = EXCLUDED.expected_value,
                confidence_stars = EXCLUDED.confidence_stars,
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
        print(f"Errore salvataggio previsione: {e}")
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

def get_match_result_from_api(api_key: str, match_id: int) -> Optional[Dict]:
    """
    Recupera il risultato reale di una partita dall'API.
    
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
    elif market == 'Over/Under':
        # Estrai la linea dalla selezione (es. "Over 2.5" -> 2.5)
        parts = selection.split()
        line = float(parts[1])
        is_over = parts[0] == 'Over'
        
        if is_over:
            won = total_goals > line
            actual = f"Over {line}" if total_goals > line else f"Under {line}"
        else:
            won = total_goals < line
            actual = f"Under {line}" if total_goals < line else f"Over {line}"
        
        return (won, actual)
    
    # === Cards ===
    elif market == 'Cards' and total_cards is not None:
        parts = selection.split()
        line = float(parts[2])  # "Cards Over 3.5" -> 3.5
        is_over = parts[1] == 'Over'
        
        if is_over:
            won = total_cards > line
            actual = f"Cards Over {line}" if total_cards > line else f"Cards Under {line}"
        else:
            won = total_cards < line
            actual = f"Cards Under {line}" if total_cards < line else f"Cards Over {line}"
        
        return (won, actual)
    
    return (None, None)


def update_predictions_with_results(api_key: str, user_id: int = None) -> Dict:
    """
    Aggiorna tutte le previsioni pendenti con i risultati reali.
    
    Returns:
        Dict con statistiche aggiornamento
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Trova previsioni senza risultato per partite passate
    # Usa <= CURRENT_DATE per includere anche partite di oggi che potrebbero essere finite
    query = """
        SELECT DISTINCT match_id, match_date, home_team, away_team
        FROM predictions 
        WHERE is_won IS NULL 
        AND match_date <= CURRENT_DATE
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
        "debug_matches": []  # Per debug
    }
    
    for match in pending_matches:
        match_id = match["match_id"]
        stats["checked"] += 1
        
        # Recupera risultato
        result = get_match_result_from_api(api_key, match_id)
        
        if not result:
            stats["errors"] += 1
            stats["debug_matches"].append(f"❌ {match['home_team']} vs {match['away_team']} (ID:{match_id}) - Nessuna risposta API")
            continue
        
        # Gestisci errore specifico
        if result.get("error"):
            stats["errors"] += 1
            stats["debug_matches"].append(f"❌ {match['home_team']} vs {match['away_team']} (ID:{match_id}) - {result['error']}")
            continue
        
        if result.get("not_finished"):
            stats["not_finished"] += 1
            stats["debug_matches"].append(f"⏳ {match['home_team']} vs {match['away_team']} (ID:{match_id}) - Status: {result.get('status', 'N/A')}")
            continue
        
        home_goals = result["home_goals"]
        away_goals = result["away_goals"]
        
        # Aggiorna tutte le previsioni per questa partita
        cursor.execute("""
            SELECT id, market, selection FROM predictions
            WHERE match_id = %s AND is_won IS NULL
        """, (match_id,))
        
        predictions = cursor.fetchall()
        
        for pred in predictions:
            is_won, actual_result = determine_prediction_outcome(
                pred["market"],
                pred["selection"],
                home_goals,
                away_goals
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
            stats["debug_matches"].append(f"✅ {match['home_team']} vs {match['away_team']} - {home_goals}-{away_goals}")
    
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
    
    cursor.close()
    conn.close()
    
    # Calcola accuracy e ROI simulato
    total_settled = general["settled"] or 0
    total_won = general["won"] or 0
    accuracy = (total_won / total_settled * 100) if total_settled > 0 else 0
    
    # ROI simulato (assumendo stake €10 e quota media 1.85)
    avg_odds = 1.85
    stake = 10
    total_staked = total_settled * stake
    total_return = total_won * stake * avg_odds
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
        WHERE user_id = %s
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
        ORDER BY predicted_prob DESC
        LIMIT %s
    """, (user_id, min_prob, limit))
    
    predictions = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return [dict(p) for p in predictions]


# ============================================================
# UI STREAMLIT
# ============================================================

def delete_prediction(prediction_id: int) -> bool:
    """Elimina una singola previsione."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM predictions WHERE id = %s", (prediction_id,))
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
    """Elimina tutte le previsioni di una partita."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "DELETE FROM predictions WHERE user_id = %s AND match_id = %s",
            (user_id, match_id)
        )
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    except Exception as e:
        conn.rollback()
        print(f"Errore eliminazione: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()


def delete_all_predictions(user_id: int) -> int:
    """Elimina TUTTE le previsioni di un utente."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM predictions WHERE user_id = %s", (user_id,))
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    except Exception as e:
        conn.rollback()
        print(f"Errore eliminazione: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()


def display_backtesting_dashboard(user_id: int, api_key: str):
    """Mostra la dashboard completa di backtesting."""
    
    st.header("📊 Backtesting & Statistiche")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Statistiche", 
        "📋 Storico", 
        "🔄 Aggiorna",
        "🗑️ Gestione"
    ])
    
    with tab1:
        display_statistics_tab(user_id)
    
    with tab2:
        display_history_tab(user_id)
    
    with tab3:
        display_update_tab(user_id, api_key)
    
    with tab4:
        display_manage_tab(user_id)


def display_statistics_tab(user_id: int):
    """Tab statistiche."""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        days = st.selectbox("Periodo", [7, 14, 30, 90, 365], index=2, key="stats_days")
    
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
            for league in stats['by_league'][:5]:
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
        limit = st.selectbox("Mostra", [20, 50, 100], index=0, key="history_limit")
    
    # Carica storico
    market = market_filter if market_filter != "Tutti" else None
    history = get_predictions_history(user_id, limit, market, only_settled)
    
    if not history:
        st.info("📭 Nessuna previsione trovata")
        return
    
    st.markdown(f"### 📋 Ultime {len(history)} Previsioni")
    
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
    
    # === AGGIORNAMENTO MANUALE ===
    st.markdown("### ✏️ Aggiornamento Manuale")
    st.markdown("""
    <p style="color:#f39c12;">
    Per partite rinviate o con errori API, inserisci il risultato manualmente.
    </p>
    """, unsafe_allow_html=True)
    
    # Carica partite con errori
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT DISTINCT match_id, match_date, home_team, away_team, league_name
        FROM predictions
        WHERE user_id = %s AND is_won IS NULL AND match_date <= CURRENT_DATE
        ORDER BY match_date DESC
        LIMIT 20
    """, (user_id,))
    
    pending_for_manual = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if pending_for_manual:
        # Dropdown per selezionare partita
        match_options = {f"{m['home_team']} vs {m['away_team']} ({m['match_date']})": m for m in pending_for_manual}
        
        selected_match_key = st.selectbox(
            "Seleziona partita da aggiornare:",
            options=list(match_options.keys())
        )
        
        if selected_match_key:
            selected_match = match_options[selected_match_key]
            
            col1, col2 = st.columns(2)
            with col1:
                home_goals = st.number_input(f"Gol {selected_match['home_team']}", min_value=0, max_value=15, value=0)
            with col2:
                away_goals = st.number_input(f"Gol {selected_match['away_team']}", min_value=0, max_value=15, value=0)
            
            if st.button("💾 Salva Risultato Manuale", type="secondary"):
                # Aggiorna tutte le previsioni per questa partita
                conn = get_connection()
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT id, market, selection FROM predictions
                    WHERE match_id = %s AND user_id = %s AND is_won IS NULL
                """, (selected_match['match_id'], user_id))
                
                predictions = cursor.fetchall()
                updated = 0
                
                for pred in predictions:
                    is_won, actual_result = determine_prediction_outcome(
                        pred["market"],
                        pred["selection"],
                        home_goals,
                        away_goals
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
        WHERE user_id = %s AND is_won IS NULL
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
    
    st.markdown("### 🗑️ Gestione Previsioni")
    
    # Carica partite
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT DISTINCT match_id, match_date, home_team, away_team, league_name,
               COUNT(*) as num_predictions
        FROM predictions
        WHERE user_id = %s
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
    Seleziona quelle da eliminare.
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
    <p style="color:#e74c3c;">
    Attenzione: questa azione è irreversibile!
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        confirm = st.checkbox("✅ Confermo di voler eliminare TUTTE le previsioni", key="confirm_delete_all")
    
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
        return True
    except Exception as e:
        print(f"Errore init backtesting: {e}")
        return False
