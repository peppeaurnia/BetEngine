"""
🏥 INJURIES IMPACT - Sistema di calcolo impatto infortuni
==========================================================
Calcola l'impatto degli infortuni/squalifiche sui gol attesi (xG).

Funzionalità:
- Recupero infortuni dall'API con posizione giocatore
- Calcolo impatto xG per posizione
- Database giocatori chiave con impatto personalizzato
- Aggiustamento mu_home/mu_away

Autore: Sistema sviluppato con Peppe
Versione: 1.0 (Gennaio 2025)
"""

import requests
from typing import Dict, List, Tuple, Optional
import streamlit as st

# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_URL = "https://v3.football.api-sports.io"

# Impatto xG per posizione (stima conservativa)
# Basato su contributo medio alla produzione offensiva
POSITION_IMPACT = {
    # Attaccanti - alto impatto
    "Attacker": 0.35,
    "Forward": 0.35,
    "Striker": 0.40,
    "Centre-Forward": 0.40,
    "Second Striker": 0.30,
    "Left Winger": 0.25,
    "Right Winger": 0.25,
    
    # Centrocampisti offensivi - medio-alto impatto
    "Attacking Midfield": 0.25,
    "Attacking Midfielder": 0.25,
    "Central Midfield": 0.15,
    "Central Midfielder": 0.15,
    "Midfielder": 0.15,
    
    # Centrocampisti difensivi - basso impatto offensivo
    "Defensive Midfield": 0.08,
    "Defensive Midfielder": 0.08,
    "Left Midfield": 0.12,
    "Right Midfield": 0.12,
    
    # Difensori - impatto minimo sull'attacco
    "Defender": 0.05,
    "Centre-Back": 0.04,
    "Left-Back": 0.08,
    "Right-Back": 0.08,
    
    # Portiere - nessun impatto sull'attacco (ma sulla difesa sì)
    "Goalkeeper": 0.00,
}

# Impatto difensivo (gol subiti in più senza questo giocatore)
POSITION_DEFENSE_IMPACT = {
    "Goalkeeper": 0.15,      # Portiere titolare out = più gol subiti
    "Centre-Back": 0.10,
    "Defender": 0.08,
    "Left-Back": 0.06,
    "Right-Back": 0.06,
    "Defensive Midfield": 0.05,
    "Defensive Midfielder": 0.05,
}

# Database giocatori chiave con impatto PERSONALIZZATO
# Formato: "Nome Giocatore": {"attack": xG_impact, "defense": xGA_impact}
# Questi override il valore standard della posizione
KEY_PLAYERS = {
    # Serie A
    "Lautaro Martínez": {"attack": 0.55, "defense": 0},
    "Marcus Thuram": {"attack": 0.45, "defense": 0},
    "Rafael Leão": {"attack": 0.45, "defense": 0},
    "Dusan Vlahovic": {"attack": 0.50, "defense": 0},
    "Victor Osimhen": {"attack": 0.55, "defense": 0},
    "Khvicha Kvaratskhelia": {"attack": 0.40, "defense": 0},
    "Paulo Dybala": {"attack": 0.35, "defense": 0},
    "Romelu Lukaku": {"attack": 0.45, "defense": 0},
    "Theo Hernández": {"attack": 0.15, "defense": 0.08},
    "Federico Dimarco": {"attack": 0.12, "defense": 0.06},
    "Nicolò Barella": {"attack": 0.18, "defense": 0.05},
    "Hakan Çalhanoğlu": {"attack": 0.15, "defense": 0.05},
    "Mike Maignan": {"attack": 0, "defense": 0.12},
    
    # Premier League
    "Erling Haaland": {"attack": 0.70, "defense": 0},
    "Mohamed Salah": {"attack": 0.55, "defense": 0},
    "Kevin De Bruyne": {"attack": 0.35, "defense": 0},
    "Bukayo Saka": {"attack": 0.40, "defense": 0},
    "Son Heung-min": {"attack": 0.45, "defense": 0},
    "Bruno Fernandes": {"attack": 0.30, "defense": 0},
    "Cole Palmer": {"attack": 0.40, "defense": 0},
    "Alexander Isak": {"attack": 0.45, "defense": 0},
    "Virgil van Dijk": {"attack": 0.05, "defense": 0.15},
    "Alisson Becker": {"attack": 0, "defense": 0.15},
    
    # LaLiga
    "Robert Lewandowski": {"attack": 0.55, "defense": 0},
    "Vinícius Júnior": {"attack": 0.50, "defense": 0},
    "Jude Bellingham": {"attack": 0.40, "defense": 0},
    "Lamine Yamal": {"attack": 0.35, "defense": 0},
    "Antoine Griezmann": {"attack": 0.40, "defense": 0},
    "Raphinha": {"attack": 0.35, "defense": 0},
    "Kylian Mbappé": {"attack": 0.60, "defense": 0},
    "Pedri": {"attack": 0.15, "defense": 0.05},
    
    # Bundesliga
    "Harry Kane": {"attack": 0.65, "defense": 0},
    "Jamal Musiala": {"attack": 0.35, "defense": 0},
    "Florian Wirtz": {"attack": 0.40, "defense": 0},
    "Xavi Simons": {"attack": 0.30, "defense": 0},
    "Serhou Guirassy": {"attack": 0.45, "defense": 0},
    
    # Ligue 1
    "Bradley Barcola": {"attack": 0.35, "defense": 0},
    "Ousmane Dembélé": {"attack": 0.35, "defense": 0},
    "Jonathan David": {"attack": 0.45, "defense": 0},
    "Mason Greenwood": {"attack": 0.40, "defense": 0},
}


# ============================================================
# FUNZIONI DI RECUPERO DATI
# ============================================================

def _get_headers(api_key: str) -> Dict:
    """Genera headers per le richieste API."""
    return {"x-apisports-key": api_key}


@st.cache_data(ttl=1800, show_spinner=False)
def get_team_injuries(api_key: str, team_id: int, season: int = 2025) -> List[Dict]:
    """
    Recupera i giocatori infortunati/squalificati di una squadra.
    
    Args:
        api_key: Chiave API
        team_id: ID della squadra
        season: Stagione corrente
    
    Returns:
        Lista di giocatori assenti con dettagli
    """
    url = f"{BASE_URL}/injuries"
    params = {
        "team": team_id,
        "season": season
    }
    
    try:
        response = requests.get(url, headers=_get_headers(api_key), params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        injuries = []
        seen_players = set()  # Evita duplicati
        
        for item in data.get("response", []):
            player = item.get("player", {})
            player_name = player.get("name", "")
            
            # Evita duplicati
            if player_name in seen_players:
                continue
            seen_players.add(player_name)
            
            # Estrai info
            injury_info = {
                "name": player_name,
                "id": player.get("id"),
                "photo": player.get("photo"),
                "type": player.get("type", ""),  # "Missing Fixture", "Questionable"
                "reason": player.get("reason", ""),  # "Injury", "Suspended", etc.
            }
            
            injuries.append(injury_info)
        
        return injuries
    
    except Exception as e:
        print(f"Errore recupero infortuni team {team_id}: {e}")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_player_position(api_key: str, player_id: int, team_id: int, season: int = 2025) -> str:
    """
    Recupera la posizione di un giocatore.
    
    Args:
        api_key: Chiave API
        player_id: ID del giocatore
        team_id: ID della squadra
        season: Stagione
    
    Returns:
        Posizione del giocatore (es. "Attacker", "Midfielder", etc.)
    """
    url = f"{BASE_URL}/players"
    params = {
        "id": player_id,
        "season": season
    }
    
    try:
        response = requests.get(url, headers=_get_headers(api_key), params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("response"):
            player_data = data["response"][0]
            # Cerca nelle statistiche della squadra corrente
            for stat in player_data.get("statistics", []):
                if stat.get("team", {}).get("id") == team_id:
                    position = stat.get("games", {}).get("position", "")
                    if position:
                        return position
            
            # Fallback: prendi la prima posizione disponibile
            if player_data.get("statistics"):
                return player_data["statistics"][0].get("games", {}).get("position", "Unknown")
        
        return "Unknown"
    
    except Exception as e:
        print(f"Errore recupero posizione player {player_id}: {e}")
        return "Unknown"


def get_injuries_with_positions(api_key: str, team_id: int, season: int = 2025) -> List[Dict]:
    """
    Recupera infortuni completi con posizione di ogni giocatore.
    
    Nota: Questa funzione fa più chiamate API. Usare con cautela.
    Per risparmiare chiamate, usa KEY_PLAYERS se il giocatore è nel database.
    
    Args:
        api_key: Chiave API
        team_id: ID squadra
        season: Stagione
    
    Returns:
        Lista di infortuni con posizione
    """
    injuries = get_team_injuries(api_key, team_id, season)
    
    for injury in injuries:
        player_name = injury.get("name", "")
        
        # Prima controlla se è un giocatore chiave noto
        if player_name in KEY_PLAYERS:
            injury["position"] = "Key Player"
            injury["is_key_player"] = True
        elif injury.get("id"):
            # Altrimenti chiama l'API (usa cache)
            position = get_player_position(api_key, injury["id"], team_id, season)
            injury["position"] = position
            injury["is_key_player"] = False
        else:
            injury["position"] = "Unknown"
            injury["is_key_player"] = False
    
    return injuries


# ============================================================
# FUNZIONI DI CALCOLO IMPATTO
# ============================================================

def calculate_player_impact(player_name: str, position: str = "Unknown") -> Tuple[float, float]:
    """
    Calcola l'impatto xG di un giocatore assente.
    
    Args:
        player_name: Nome del giocatore
        position: Posizione (se nota)
    
    Returns:
        Tuple (attack_impact, defense_impact)
        - attack_impact: Riduzione xG squadra (gol che segnerà in meno)
        - defense_impact: Aumento xG avversario (gol che subirà in più)
    """
    # Prima controlla se è un giocatore chiave noto
    if player_name in KEY_PLAYERS:
        key_data = KEY_PLAYERS[player_name]
        return (key_data.get("attack", 0), key_data.get("defense", 0))
    
    # Altrimenti usa la posizione
    attack_impact = POSITION_IMPACT.get(position, 0.10)  # Default conservativo
    defense_impact = POSITION_DEFENSE_IMPACT.get(position, 0)
    
    return (attack_impact, defense_impact)


def calculate_team_injury_impact(injuries: List[Dict], max_impact: float = 0.6) -> Tuple[float, float]:
    """
    Calcola l'impatto totale degli infortuni su una squadra.
    
    L'impatto è LIMITATO per evitare valori irrealistici:
    - Max riduzione xG: 0.6 (anche se perdi 5 attaccanti)
    - Gli infortuni "Questionable" contano solo 50%
    
    Args:
        injuries: Lista di infortuni con name, position, type
        max_impact: Impatto massimo consentito
    
    Returns:
        Tuple (total_attack_impact, total_defense_impact)
    """
    total_attack = 0.0
    total_defense = 0.0
    
    for injury in injuries:
        player_name = injury.get("name", "")
        position = injury.get("position", "Unknown")
        injury_type = injury.get("type", "")
        
        # Calcola impatto giocatore
        attack_imp, defense_imp = calculate_player_impact(player_name, position)
        
        # Se "Questionable", riduce l'impatto del 50% (potrebbe giocare)
        if "Questionable" in injury_type:
            attack_imp *= 0.5
            defense_imp *= 0.5
        
        total_attack += attack_imp
        total_defense += defense_imp
    
    # Limita l'impatto massimo
    total_attack = min(total_attack, max_impact)
    total_defense = min(total_defense, max_impact * 0.5)  # Difesa max 0.3
    
    return (total_attack, total_defense)


def get_injury_adjustment(
    api_key: str,
    home_team_id: int,
    away_team_id: int,
    season: int = 2025,
    use_api_positions: bool = False
) -> Dict:
    """
    Funzione principale: calcola l'aggiustamento xG per infortuni.
    
    Args:
        api_key: Chiave API
        home_team_id: ID squadra casa
        away_team_id: ID squadra trasferta
        season: Stagione
        use_api_positions: Se True, recupera posizioni dall'API (più chiamate)
                          Se False, usa solo KEY_PLAYERS database
    
    Returns:
        Dict con:
        - home_injuries: Lista infortuni casa
        - away_injuries: Lista infortuni trasferta
        - home_attack_reduction: Riduzione xG casa
        - home_defense_increase: Aumento xG subiti casa
        - away_attack_reduction: Riduzione xG trasferta
        - away_defense_increase: Aumento xG subiti trasferta
        - mu_home_adj: Fattore moltiplicativo per mu_home
        - mu_away_adj: Fattore moltiplicativo per mu_away
    """
    # Recupera infortuni
    if use_api_positions:
        home_injuries = get_injuries_with_positions(api_key, home_team_id, season)
        away_injuries = get_injuries_with_positions(api_key, away_team_id, season)
    else:
        home_injuries = get_team_injuries(api_key, home_team_id, season)
        away_injuries = get_team_injuries(api_key, away_team_id, season)
        
        # Aggiungi posizione solo per key players
        for inj in home_injuries:
            if inj.get("name") in KEY_PLAYERS:
                inj["position"] = "Key Player"
                inj["is_key_player"] = True
            else:
                inj["position"] = "Unknown"
                inj["is_key_player"] = False
        
        for inj in away_injuries:
            if inj.get("name") in KEY_PLAYERS:
                inj["position"] = "Key Player"
                inj["is_key_player"] = True
            else:
                inj["position"] = "Unknown"
                inj["is_key_player"] = False
    
    # Calcola impatto
    home_attack_red, home_def_inc = calculate_team_injury_impact(home_injuries)
    away_attack_red, away_def_inc = calculate_team_injury_impact(away_injuries)
    
    # Calcola aggiustamento mu
    # mu_home = gol che segna la casa
    # - Si riduce per i suoi infortuni (attacco)
    # - Si aumenta per infortuni avversari (difesa avversaria peggiore)
    
    # Esempio: mu_home = 1.5
    # Casa perde Leao (-0.45 attack) -> mu_home = 1.5 - 0.45 = 1.05
    # Trasferta perde Van Dijk (+0.15 defense) -> mu_home = 1.05 + 0.15 = 1.20
    
    # Invece di sottrarre direttamente, uso un fattore moltiplicativo
    # per mantenere la proporzione
    
    # Stimo mu base = 1.5 per calcolare il fattore
    BASE_MU = 1.5
    
    # Aggiustamento casa
    # Riduzione per suoi infortuni offensivi
    home_adj = 1.0 - (home_attack_red / BASE_MU) * 0.8  # Scala a 80% per essere conservativo
    # Aumento per infortuni difensivi avversari
    home_adj += (away_def_inc / BASE_MU) * 0.5
    home_adj = max(0.5, min(home_adj, 1.3))  # Limita tra 0.5 e 1.3
    
    # Aggiustamento trasferta (simmetrico)
    away_adj = 1.0 - (away_attack_red / BASE_MU) * 0.8
    away_adj += (home_def_inc / BASE_MU) * 0.5
    away_adj = max(0.5, min(away_adj, 1.3))
    
    return {
        "home_injuries": home_injuries,
        "away_injuries": away_injuries,
        "home_attack_reduction": round(home_attack_red, 3),
        "home_defense_increase": round(home_def_inc, 3),
        "away_attack_reduction": round(away_attack_red, 3),
        "away_defense_increase": round(away_def_inc, 3),
        "mu_home_adj": round(home_adj, 3),
        "mu_away_adj": round(away_adj, 3),
        "home_key_players_out": [i["name"] for i in home_injuries if i.get("is_key_player")],
        "away_key_players_out": [i["name"] for i in away_injuries if i.get("is_key_player")],
    }


# ============================================================
# FUNZIONE DI DISPLAY
# ============================================================

def display_injuries_impact(injury_data: Dict):
    """
    Mostra le informazioni sugli infortuni nell'app.
    
    Args:
        injury_data: Output di get_injury_adjustment()
    """
    st.markdown("### 🏥 Infortuni & Squalifiche")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏠 Casa")
        home_injuries = injury_data.get("home_injuries", [])
        
        if not home_injuries:
            st.success("✅ Rosa al completo")
        else:
            for inj in home_injuries[:5]:  # Max 5 mostrati
                name = inj.get("name", "N/A")
                reason = inj.get("reason", "")
                is_key = inj.get("is_key_player", False)
                
                icon = "⭐" if is_key else "🤕"
                color = "#e74c3c" if is_key else "#f39c12"
                
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.1); padding:8px; border-radius:6px; 
                            margin-bottom:5px; border-left:3px solid {color};">
                    <span style="color:#ffffff;">{icon} <strong>{name}</strong></span>
                    <span style="color:#a8d4f0; font-size:0.9em;"> - {reason}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Impatto
            attack_red = injury_data.get("home_attack_reduction", 0)
            if attack_red > 0:
                st.markdown(f"""
                <div style="color:#e74c3c; font-weight:bold;">
                    📉 Impatto: -{attack_red:.2f} xG
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ✈️ Trasferta")
        away_injuries = injury_data.get("away_injuries", [])
        
        if not away_injuries:
            st.success("✅ Rosa al completo")
        else:
            for inj in away_injuries[:5]:
                name = inj.get("name", "N/A")
                reason = inj.get("reason", "")
                is_key = inj.get("is_key_player", False)
                
                icon = "⭐" if is_key else "🤕"
                color = "#e74c3c" if is_key else "#f39c12"
                
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.1); padding:8px; border-radius:6px; 
                            margin-bottom:5px; border-left:3px solid {color};">
                    <span style="color:#ffffff;">{icon} <strong>{name}</strong></span>
                    <span style="color:#a8d4f0; font-size:0.9em;"> - {reason}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Impatto
            attack_red = injury_data.get("away_attack_reduction", 0)
            if attack_red > 0:
                st.markdown(f"""
                <div style="color:#e74c3c; font-weight:bold;">
                    📉 Impatto: -{attack_red:.2f} xG
                </div>
                """, unsafe_allow_html=True)
    
    # Riepilogo impatto
    st.markdown("---")
    
    mu_home_adj = injury_data.get("mu_home_adj", 1.0)
    mu_away_adj = injury_data.get("mu_away_adj", 1.0)
    
    if mu_home_adj != 1.0 or mu_away_adj != 1.0:
        st.markdown(f"""
        <div style="background:rgba(231, 76, 60, 0.2); padding:12px; border-radius:8px; border:1px solid #e74c3c;">
            <strong style="color:#ffffff;">⚠️ Aggiustamento per Infortuni:</strong><br>
            <span style="color:#a8d4f0;">
                🏠 xG Casa: <strong style="color:{'#e74c3c' if mu_home_adj < 1 else '#27ae60'};">×{mu_home_adj:.2f}</strong> | 
                ✈️ xG Trasferta: <strong style="color:{'#e74c3c' if mu_away_adj < 1 else '#27ae60'};">×{mu_away_adj:.2f}</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)
