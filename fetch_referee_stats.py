"""
fetch_referee_stats.py - Scarica e analizza statistiche arbitri

Questo script:
1. Scarica tutte le partite passate delle leghe configurate
2. Estrae cartellini gialli/rossi per ogni partita
3. Calcola statistiche per ogni arbitro
4. Salva in referee_data.json

COSTO: ~5-10 chiamate API totali (una per lega/stagione)
"""

import requests
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

# Costanti API
BASE_URL = "https://v3.football.api-sports.io"

# Carica API key da config o variabile ambiente
try:
    from config import API_FOOTBALL_KEY
except ImportError:
    API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")

# ============================================================
# CONFIGURAZIONE
# ============================================================

# Leghe da analizzare con i relativi ID
LEAGUES_TO_ANALYZE = {
    135: "Serie A",
    39: "Premier League", 
    78: "Bundesliga",
    61: "Ligue 1",
    140: "La Liga",
    2: "Champions League"
}

# Stagioni da analizzare (1 stagione = meno chiamate API)
# Ogni lega + stagione costa ~20 chiamate API
SEASONS_TO_ANALYZE = [2024]  # Solo stagione corrente

# File di output
OUTPUT_FILE = "referee_data.json"


# ============================================================
# FUNZIONI API
# ============================================================

def get_headers() -> Dict:
    """Header per autenticazione API-Football"""
    return {"x-apisports-key": API_FOOTBALL_KEY}


def fetch_finished_fixtures(league_id: int, season: int) -> Tuple[List[Dict], int]:
    """
    Scarica tutte le partite finite di una lega/stagione.
    Prima ottiene la lista ID, poi scarica i dettagli con eventi in batch.
    
    Args:
        league_id: ID della lega
        season: Anno della stagione
    
    Returns:
        Tuple (Lista di fixtures con eventi, numero chiamate API)
    """
    api_calls = 0
    
    # STEP 1: Ottieni lista delle partite finite (solo ID)
    url = f"{BASE_URL}/fixtures"
    params = {
        "league": league_id,
        "season": season,
        "status": "FT"  # Solo partite finite
    }
    
    try:
        response = requests.get(url, headers=get_headers(), params=params, timeout=30)
        api_calls += 1
        response.raise_for_status()
        data = response.json()
        
        if data.get("errors"):
            print(f"  ⚠️ Errore API: {data['errors']}")
            return [], api_calls
        
        fixtures_basic = data.get("response", [])
        print(f"  ✅ Trovate {len(fixtures_basic)} partite finite")
        
        if not fixtures_basic:
            return [], api_calls
        
        # STEP 2: Scarica dettagli con eventi in batch di 20
        all_fixtures_with_events = []
        fixture_ids = [fx["fixture"]["id"] for fx in fixtures_basic]
        
        # Dividi in batch di 20
        batch_size = 20
        batches = [fixture_ids[i:i+batch_size] for i in range(0, len(fixture_ids), batch_size)]
        
        print(f"  📦 Scarico eventi in {len(batches)} batch...")
        
        for i, batch in enumerate(batches):
            ids_str = "-".join(str(id) for id in batch)
            batch_url = f"{BASE_URL}/fixtures"
            batch_params = {"ids": ids_str}
            
            try:
                batch_response = requests.get(batch_url, headers=get_headers(), params=batch_params, timeout=30)
                api_calls += 1
                batch_response.raise_for_status()
                batch_data = batch_response.json()
                
                if batch_data.get("response"):
                    all_fixtures_with_events.extend(batch_data["response"])
                
                # Progress ogni 5 batch
                if (i + 1) % 5 == 0:
                    print(f"    Batch {i+1}/{len(batches)} completato...")
                    
            except Exception as e:
                print(f"    ⚠️ Errore batch {i+1}: {e}")
                continue
        
        print(f"  ✅ Scaricate {len(all_fixtures_with_events)} partite con eventi ({api_calls} chiamate)")
        return all_fixtures_with_events, api_calls
        
    except Exception as e:
        print(f"  ❌ Errore: {e}")
        return [], api_calls


def fetch_fixture_events(fixture_id: int) -> List[Dict]:
    """
    Scarica gli eventi (cartellini, gol, etc.) di una partita specifica.
    
    NOTA: Questo è necessario solo se la chiamata fixtures non include gli eventi.
    Verifichiamo prima se gli eventi sono già inclusi.
    """
    url = f"{BASE_URL}/fixtures/events"
    params = {"fixture": fixture_id}
    
    try:
        response = requests.get(url, headers=get_headers(), params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("response", [])
    except Exception as e:
        print(f"  ❌ Errore eventi fixture {fixture_id}: {e}")
        return []


# ============================================================
# ANALISI DATI
# ============================================================

def extract_referee_stats(fixtures: List[Dict], league_name: str) -> Dict:
    """
    Estrae statistiche arbitri dalle partite.
    
    Args:
        fixtures: Lista di fixtures da API-Football
        league_name: Nome della lega
    
    Returns:
        Dict con statistiche per ogni arbitro
    """
    referee_stats = defaultdict(lambda: {
        "matches": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "total_cards": 0,
        "leagues": set(),
        "matches_list": []  # Per debug
    })
    
    for fx in fixtures:
        fixture = fx.get("fixture", {})
        referee_name = fixture.get("referee")
        
        # Salta se arbitro non assegnato
        if not referee_name:
            continue
        
        # Pulisci nome arbitro (rimuovi nazionalità)
        # Es: "Daniele Orsato, Italy" -> "Daniele Orsato"
        referee_clean = referee_name.split(",")[0].strip()
        
        # Conta cartellini dagli eventi
        events = fx.get("events", [])
        yellow = 0
        red = 0
        
        for event in events:
            event_type = event.get("type", "").lower()
            detail = event.get("detail", "").lower()
            
            if event_type == "card":
                if "yellow" in detail and "red" not in detail:
                    yellow += 1
                elif "red" in detail:
                    red += 1
        
        # Aggiorna statistiche
        stats = referee_stats[referee_clean]
        stats["matches"] += 1
        stats["yellow_cards"] += yellow
        stats["red_cards"] += red
        stats["total_cards"] += yellow + red
        stats["leagues"].add(league_name)
        
        # Salva dettaglio partita (per debug)
        home = fx.get("teams", {}).get("home", {}).get("name", "?")
        away = fx.get("teams", {}).get("away", {}).get("name", "?")
        stats["matches_list"].append({
            "match": f"{home} vs {away}",
            "date": fixture.get("date", "")[:10],
            "yellow": yellow,
            "red": red
        })
    
    return referee_stats


def calculate_league_average(referee_stats: Dict) -> Tuple[float, float]:
    """
    Calcola la media cartellini per partita della lega.
    
    Returns:
        Tuple (media_cartellini_totali, media_gialli)
    """
    total_matches = 0
    total_cards = 0
    total_yellow = 0
    
    for stats in referee_stats.values():
        total_matches += stats["matches"]
        total_cards += stats["total_cards"]
        total_yellow += stats["yellow_cards"]
    
    if total_matches == 0:
        return 4.0, 3.5  # Default ragionevoli
    
    return total_cards / total_matches, total_yellow / total_matches


def build_referee_database() -> Dict:
    """
    Costruisce il database completo degli arbitri.
    
    Returns:
        Dict con tutti i dati arbitri e medie per lega
    """
    # Stima costo API
    estimated_matches_per_league = 380  # Circa
    batches_per_league = (estimated_matches_per_league // 20) + 1
    total_estimated_calls = len(LEAGUES_TO_ANALYZE) * len(SEASONS_TO_ANALYZE) * (batches_per_league + 1)
    
    print("🏃 Avvio costruzione database arbitri...")
    print(f"📊 Leghe da analizzare: {list(LEAGUES_TO_ANALYZE.values())}")
    print(f"📅 Stagioni: {SEASONS_TO_ANALYZE}")
    print(f"💰 Chiamate API stimate: ~{total_estimated_calls}")
    print("-" * 50)
    
    all_referee_stats = defaultdict(lambda: {
        "matches": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "total_cards": 0,
        "avg_cards_per_match": 0,
        "avg_yellow_per_match": 0,
        "leagues": [],
        "severity_by_league": {}
    })
    
    league_averages = {}
    api_calls = 0
    
    for league_id, league_name in LEAGUES_TO_ANALYZE.items():
        print(f"\n⚽ {league_name} (ID: {league_id})")
        
        league_total_matches = 0
        league_total_cards = 0
        league_total_yellow = 0
        
        for season in SEASONS_TO_ANALYZE:
            print(f"  📅 Stagione {season}...")
            
            fixtures, calls_used = fetch_finished_fixtures(league_id, season)
            api_calls += calls_used
            
            if not fixtures:
                continue
            
            # Estrai statistiche per questa lega/stagione
            season_stats = extract_referee_stats(fixtures, league_name)
            
            # Aggiorna statistiche globali arbitri
            for ref_name, stats in season_stats.items():
                all_stats = all_referee_stats[ref_name]
                all_stats["matches"] += stats["matches"]
                all_stats["yellow_cards"] += stats["yellow_cards"]
                all_stats["red_cards"] += stats["red_cards"]
                all_stats["total_cards"] += stats["total_cards"]
                
                if league_name not in all_stats["leagues"]:
                    all_stats["leagues"].append(league_name)
                
                # Accumula per media lega
                league_total_matches += stats["matches"]
                league_total_cards += stats["total_cards"]
                league_total_yellow += stats["yellow_cards"]
        
        # Calcola media per questa lega
        if league_total_matches > 0:
            league_averages[league_name] = {
                "avg_cards": round(league_total_cards / league_total_matches, 2),
                "avg_yellow": round(league_total_yellow / league_total_matches, 2),
                "total_matches_analyzed": league_total_matches
            }
            print(f"  📈 Media lega: {league_averages[league_name]['avg_cards']} cartellini/partita")
    
    # Calcola medie per ogni arbitro
    print("\n" + "-" * 50)
    print("📊 Calcolo statistiche finali arbitri...")
    
    final_referees = {}
    for ref_name, stats in all_referee_stats.items():
        if stats["matches"] >= 5:  # Solo arbitri con almeno 5 partite
            avg_cards = stats["total_cards"] / stats["matches"]
            avg_yellow = stats["yellow_cards"] / stats["matches"]
            
            final_referees[ref_name] = {
                "matches": stats["matches"],
                "yellow_cards": stats["yellow_cards"],
                "red_cards": stats["red_cards"],
                "total_cards": stats["total_cards"],
                "avg_cards_per_match": round(avg_cards, 2),
                "avg_yellow_per_match": round(avg_yellow, 2),
                "leagues": stats["leagues"]
            }
    
    # Calcola severity factor per ogni arbitro rispetto alla media globale
    if league_averages:
        valid_avgs = [la["avg_cards"] for la in league_averages.values() if la["avg_cards"] > 0]
        global_avg = sum(valid_avgs) / len(valid_avgs) if valid_avgs else 4.0
    else:
        global_avg = 4.0  # Default se nessun dato
    
    # Evita divisione per zero
    if global_avg == 0:
        global_avg = 4.0
    
    for ref_name, stats in final_referees.items():
        stats["severity_factor"] = round(stats["avg_cards_per_match"] / global_avg, 3)
    
    # Costruisci output finale
    database = {
        "last_updated": datetime.now().isoformat(),
        "api_calls_used": api_calls,
        "seasons_analyzed": SEASONS_TO_ANALYZE,
        "league_averages": league_averages,
        "global_average_cards": round(global_avg, 2),
        "referees": final_referees,
        "total_referees": len(final_referees)
    }
    
    print(f"\n✅ Database costruito!")
    print(f"   - Arbitri con almeno 5 partite: {len(final_referees)}")
    print(f"   - Chiamate API usate: {api_calls}")
    print(f"   - Media globale cartellini: {global_avg:.2f}")
    
    return database


def save_database(database: Dict, filepath: str = None):
    """Salva il database in JSON"""
    if filepath is None:
        filepath = OUTPUT_FILE
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Database salvato in: {filepath}")


def load_referee_database(filepath: str = None) -> Dict:
    """
    Carica il database arbitri da file JSON.
    
    Returns:
        Dict con dati arbitri o None se non esiste
    """
    if filepath is None:
        # Cerca in vari percorsi possibili
        possible_paths = [
            OUTPUT_FILE,
            os.path.join(os.path.dirname(__file__), OUTPUT_FILE),
            os.path.join(os.path.dirname(__file__), "..", OUTPUT_FILE)
        ]
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
    
    if filepath is None or not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Errore caricamento database arbitri: {e}")
        return None


def get_referee_adjustment(referee_name: str, league_name: str = None) -> Dict:
    """
    Ottiene i fattori di aggiustamento per un arbitro specifico.
    
    Args:
        referee_name: Nome arbitro (può includere nazionalità)
        league_name: Nome della lega (opzionale, per media specifica)
    
    Returns:
        Dict con severity_factor e altre statistiche
    """
    database = load_referee_database()
    
    if not database:
        return {
            "found": False,
            "severity_factor": 1.0,
            "avg_cards": None,
            "matches": 0,
            "message": "Database arbitri non trovato"
        }
    
    # Pulisci nome arbitro
    ref_clean = referee_name.split(",")[0].strip() if referee_name else ""
    
    referees = database.get("referees", {})
    
    if ref_clean in referees:
        ref_data = referees[ref_clean]
        
        # Usa media della lega specifica se disponibile
        league_avg = database.get("global_average_cards", 4.0)
        if league_name and league_name in database.get("league_averages", {}):
            league_avg = database["league_averages"][league_name]["avg_cards"]
        
        # Ricalcola severity rispetto alla lega specifica
        severity = ref_data["avg_cards_per_match"] / league_avg if league_avg > 0 else 1.0
        
        return {
            "found": True,
            "severity_factor": round(severity, 3),
            "avg_cards": ref_data["avg_cards_per_match"],
            "avg_yellow": ref_data["avg_yellow_per_match"],
            "matches": ref_data["matches"],
            "leagues": ref_data["leagues"],
            "message": f"Arbitro trovato: {ref_data['matches']} partite analizzate"
        }
    else:
        return {
            "found": False,
            "severity_factor": 1.0,
            "avg_cards": None,
            "matches": 0,
            "message": f"Arbitro '{ref_clean}' non trovato nel database"
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏟️  COSTRUZIONE DATABASE STATISTICHE ARBITRI")
    print("=" * 60)
    
    # Costruisci database
    database = build_referee_database()
    
    # Salva
    save_database(database)
    
    # Mostra top 10 arbitri più severi
    print("\n" + "=" * 60)
    print("🔴 TOP 10 ARBITRI PIÙ SEVERI:")
    print("=" * 60)
    
    referees = database.get("referees", {})
    sorted_refs = sorted(referees.items(), key=lambda x: x[1]["severity_factor"], reverse=True)
    
    for i, (name, stats) in enumerate(sorted_refs[:10], 1):
        print(f"{i:2}. {name:<25} | {stats['avg_cards_per_match']:.1f} cart/partita | severity: {stats['severity_factor']:.2f} | {stats['matches']} partite")
    
    print("\n" + "=" * 60)
    print("🟢 TOP 10 ARBITRI PIÙ PERMISSIVI:")
    print("=" * 60)
    
    for i, (name, stats) in enumerate(sorted_refs[-10:][::-1], 1):
        print(f"{i:2}. {name:<25} | {stats['avg_cards_per_match']:.1f} cart/partita | severity: {stats['severity_factor']:.2f} | {stats['matches']} partite")
