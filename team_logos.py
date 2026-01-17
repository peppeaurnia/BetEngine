#!/usr/bin/env python3
"""
team_logos.py - Gestione loghi squadre

Mappa i nomi delle squadre (dall'API) ai file dei loghi locali.
Converte le immagini in base64 per visualizzarle in Streamlit.
"""

import os
import base64
from pathlib import Path
from typing import Optional, Dict

# Percorso base della cartella Loghi - cerca in più posizioni
def _find_logos_path():
    """Trova la cartella Loghi cercando in varie posizioni."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "Loghi"),  # Stessa cartella dello script
        os.path.join(os.getcwd(), "Loghi"),  # Directory corrente
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Loghi"),  # Path assoluto
        "Loghi",  # Relativo
        os.path.join("..", "Loghi"),  # Cartella parent
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    # Fallback: usa il primo path anche se non esiste
    return possible_paths[0]

LOGOS_BASE_PATH = _find_logos_path()

# ============================================================
# MAPPING NOMI API -> FILE LOGHI
# ============================================================
# Chiave: nome squadra come arriva dall'API (case-insensitive)
# Valore: (cartella, nome_file)

TEAM_LOGO_MAPPING = {
    # ============ SERIE A ============
    "atalanta": ("Serie A", "Atalanta_logo.png"),
    "bologna": ("Serie A", "Bologna_logo.png"),
    "cagliari": ("Serie A", "Cagliari_logo.png"),
    "como": ("Serie A", "Como_logo.png"),
    "como 1907": ("Serie A", "Como_logo.png"),
    "cremonese": ("Serie A", "Cremonese_logo.png"),
    "empoli": ("Serie A", "Empoli_logo.png"),
    "fiorentina": ("Serie A", "Fiorentina_logo.png"),
    "genoa": ("Serie A", "Genoa_logo.png"),
    "hellas verona": ("Serie A", "Verona_logo.png"),
    "verona": ("Serie A", "Verona_logo.png"),
    "inter": ("Serie A", "Inter_logo.png"),
    "internazionale": ("Serie A", "Inter_logo.png"),
    "juventus": ("Serie A", "Juventus_logo.png"),
    "lazio": ("Serie A", "Lazio_logo.png"),
    "lecce": ("Serie A", "Lecce_logo.png"),
    "ac milan": ("Serie A", "Milan_logo.png"),
    "milan": ("Serie A", "Milan_logo.png"),
    "monza": ("Serie A", "Monza_logo.png"),
    "napoli": ("Serie A", "Napoli_logo.png"),
    "parma": ("Serie A", "Parma_logo.png"),
    "pisa": ("Serie A", "Pisa_logo.png"),
    "as roma": ("Serie A", "Roma_logo.png"),
    "roma": ("Serie A", "Roma_logo.png"),
    "salernitana": ("Serie A", "Salernitana_logo.png"),
    "sassuolo": ("Serie A", "Sassuolo_logo.png"),
    "spezia": ("Serie A", "Spezia_logo.png"),
    "torino": ("Serie A", "Torino_logo.png"),
    "udinese": ("Serie A", "Udinese_logo.png"),
    "venezia": ("Serie A", "Venezia_logo.png"),
    
    # ============ PREMIER LEAGUE ============
    "arsenal": ("Premier league", "Arsenal_logo.png"),
    "aston villa": ("Premier league", "astonvilla_logo.png"),
    "bournemouth": ("Premier league", "bournemouth_logo.png"),
    "afc bournemouth": ("Premier league", "bournemouth_logo.png"),
    "brentford": ("Premier league", "Brentford_logo.png"),
    "brighton": ("Premier league", "Brighton_logo.png"),
    "brighton & hove albion": ("Premier league", "Brighton_logo.png"),
    "burnley": ("Premier league", "Burnley_logo.png"),
    "chelsea": ("Premier league", "Chelsea_logo.png"),
    "crystal palace": ("Premier league", "CrystalPalace_logo.png"),
    "everton": ("Premier league", "Everton_logo.png"),
    "fulham": ("Premier league", "Fulham_logo.png"),
    "ipswich": ("Premier league", "Ipswich_logo.png"),
    "ipswich town": ("Premier league", "Ipswich_logo.png"),
    "leeds": ("Premier league", "Leeds_logo.png"),
    "leeds united": ("Premier league", "Leeds_logo.png"),
    "leicester": ("Premier league", "Leicester_logo.png"),
    "leicester city": ("Premier league", "Leicester_logo.png"),
    "liverpool": ("Premier league", "Liverpool_logo.png"),
    "manchester city": ("Premier league", "ManchesterCity_logo.png"),
    "man city": ("Premier league", "ManchesterCity_logo.png"),
    "manchester united": ("Premier league", "ManchesterUnited_logo.png"),
    "man united": ("Premier league", "ManchesterUnited_logo.png"),
    "newcastle": ("Premier league", "NewcastleUnited_logo.png"),
    "newcastle united": ("Premier league", "NewcastleUnited_logo.png"),
    "nottingham forest": ("Premier league", "NottinghamForest_logo.png"),
    "southampton": ("Premier league", "Southampton_logo.png"),
    "sunderland": ("Premier league", "Sunderland_logo.png"),
    "tottenham": ("Premier league", "Tottenham_logo.png"),
    "tottenham hotspur": ("Premier league", "Tottenham_logo.png"),
    "spurs": ("Premier league", "Tottenham_logo.png"),
    "west ham": ("Premier league", "Westham_logo.png"),
    "west ham united": ("Premier league", "Westham_logo.png"),
    "wolverhampton": ("Premier league", "Wolverhampton_logo.png"),
    "wolves": ("Premier league", "Wolverhampton_logo.png"),
    "wolverhampton wanderers": ("Premier league", "Wolverhampton_logo.png"),
    
    # ============ BUNDESLIGA ============
    "augsburg": ("Bundesliga", "Augsburg_logo.png"),
    "fc augsburg": ("Bundesliga", "Augsburg_logo.png"),
    "bayer leverkusen": ("Bundesliga", "BayerLeverkusen_logo.png"),
    "leverkusen": ("Bundesliga", "BayerLeverkusen_logo.png"),
    "bayern munich": ("Bundesliga", "BayernMonaco_logo.png"),
    "bayern munchen": ("Bundesliga", "BayernMonaco_logo.png"),
    "bayern münchen": ("Bundesliga", "BayernMonaco_logo.png"),
    "fc bayern münchen": ("Bundesliga", "BayernMonaco_logo.png"),
    "borussia dortmund": ("Bundesliga", "BorussiaDortmund_logo.png"),
    "dortmund": ("Bundesliga", "BorussiaDortmund_logo.png"),
    "bvb": ("Bundesliga", "BorussiaDortmund_logo.png"),
    "borussia monchengladbach": ("Bundesliga", "BorussiaMönchengladbach_logo.png"),
    "borussia m'gladbach": ("Bundesliga", "BorussiaMönchengladbach_logo.png"),
    "monchengladbach": ("Bundesliga", "BorussiaMönchengladbach_logo.png"),
    "gladbach": ("Bundesliga", "BorussiaMönchengladbach_logo.png"),
    "eintracht frankfurt": ("Bundesliga", "EintrachtFrankfurt_logo.png"),
    "frankfurt": ("Bundesliga", "EintrachtFrankfurt_logo.png"),
    "freiburg": ("Bundesliga", "Freiburg_logo.png"),
    "sc freiburg": ("Bundesliga", "Freiburg_logo.png"),
    "hamburger sv": ("Bundesliga", "hamburger_logo.png"),
    "hamburg": ("Bundesliga", "hamburger_logo.png"),
    "heidenheim": ("Bundesliga", "Heidenheim_logo.png"),
    "1. fc heidenheim 1846": ("Bundesliga", "Heidenheim_logo.png"),
    "hoffenheim": ("Bundesliga", "Hoffenheim_logo.png"),
    "tsg hoffenheim": ("Bundesliga", "Hoffenheim_logo.png"),
    "fc koln": ("Bundesliga", "Koln_logo.png"),
    "köln": ("Bundesliga", "Koln_logo.png"),
    "cologne": ("Bundesliga", "Koln_logo.png"),
    "rb leipzig": ("Bundesliga", "Leipzig_logo.png"),
    "leipzig": ("Bundesliga", "Leipzig_logo.png"),
    "mainz": ("Bundesliga", "Mainz_logo.png"),
    "mainz 05": ("Bundesliga", "Mainz_logo.png"),
    "1. fsv mainz 05": ("Bundesliga", "Mainz_logo.png"),
    "st pauli": ("Bundesliga", "StPauli_logo.png"),
    "fc st. pauli": ("Bundesliga", "StPauli_logo.png"),
    "stuttgart": ("Bundesliga", "Stuttgart_logo.png"),
    "vfb stuttgart": ("Bundesliga", "Stuttgart_logo.png"),
    "union berlin": ("Bundesliga", "UnionBerlin_logo.png"),
    "1. fc union berlin": ("Bundesliga", "UnionBerlin_logo.png"),
    "werder bremen": ("Bundesliga", "WerderBremen_logo.png"),
    "bremen": ("Bundesliga", "WerderBremen_logo.png"),
    "wolfsburg": ("Bundesliga", "Wolfsburg_logo.png"),
    "vfl wolfsburg": ("Bundesliga", "Wolfsburg_logo.png"),
    "bochum": ("Bundesliga", "Bochum_logo.png"),
    "vfl bochum": ("Bundesliga", "Bochum_logo.png"),
    "holstein kiel": ("Bundesliga", "HolsteinKiel_logo.png"),
    "kiel": ("Bundesliga", "HolsteinKiel_logo.png"),
    
    # ============ LA LIGA ============
    "athletic bilbao": ("La Liga", "AthleticBilbao.png"),
    "athletic club": ("La Liga", "AthleticBilbao.png"),
    "atletico madrid": ("La Liga", "AtleticoMadrid.png"),
    "atletico de madrid": ("La Liga", "AtleticoMadrid.png"),
    "barcelona": ("La Liga", "Barcellona.png"),
    "fc barcelona": ("La Liga", "Barcellona.png"),
    "celta vigo": ("La Liga", "CeltaDeVigo.png"),
    "celta de vigo": ("La Liga", "CeltaDeVigo.png"),
    "alaves": ("La Liga", "DeportivoAlaves.png"),
    "deportivo alaves": ("La Liga", "DeportivoAlaves.png"),
    "elche": ("La Liga", "Elche.png"),
    "espanyol": ("La Liga", "Espanyol.png"),
    "rcd espanyol": ("La Liga", "Espanyol.png"),
    "getafe": ("La Liga", "Getafe.jpg"),
    "girona": ("La Liga", "Girona.png"),
    "las palmas": ("La Liga", "LasPalmas.png"),
    "ud las palmas": ("La Liga", "LasPalmas.png"),
    "leganes": ("La Liga", "Leganes.png"),
    "cd leganes": ("La Liga", "Leganes.png"),
    "levante": ("La Liga", "Levante.png"),
    "mallorca": ("La Liga", "Mallorca.png"),
    "rcd mallorca": ("La Liga", "Mallorca.png"),
    "osasuna": ("La Liga", "Osasuna.png"),
    "ca osasuna": ("La Liga", "Osasuna.png"),
    "rayo vallecano": ("La Liga", "RayoVallecano.png"),
    "real betis": ("La Liga", "RealBetis.png"),
    "betis": ("La Liga", "RealBetis.png"),
    "real madrid": ("La Liga", "RealMadrid.png"),
    "real oviedo": ("La Liga", "RealOviedo.png"),
    "real sociedad": ("La Liga", "RealSociedad.png"),
    "sevilla": ("La Liga", "Sevilla.png"),
    "sevilla fc": ("La Liga", "Sevilla.png"),
    "valencia": ("La Liga", "Valencia.png"),
    "valencia cf": ("La Liga", "Valencia.png"),
    "valladolid": ("La Liga", "Valladolid.png"),
    "real valladolid": ("La Liga", "Valladolid.png"),
    "villarreal": ("La Liga", "Villarreal.png"),
    "villarreal cf": ("La Liga", "Villarreal.png"),
    
    # ============ LIGUE 1 ============
    "angers": ("Ligue 1", "Angers.png"),
    "angers sco": ("Ligue 1", "Angers.png"),
    "auxerre": ("Ligue 1", "Auxerre.png"),
    "aj auxerre": ("Ligue 1", "Auxerre.png"),
    "brest": ("Ligue 1", "Brest.png"),
    "stade brestois 29": ("Ligue 1", "Brest.png"),
    "lorient": ("Ligue 1", "FCLorient.png"),
    "fc lorient": ("Ligue 1", "FCLorient.png"),
    "metz": ("Ligue 1", "FCMetz.png"),
    "fc metz": ("Ligue 1", "FCMetz.png"),
    "nantes": ("Ligue 1", "FCNantes.png"),
    "fc nantes": ("Ligue 1", "FCNantes.png"),
    "le havre": ("Ligue 1", "LeHavre.png"),
    "le havre ac": ("Ligue 1", "LeHavre.png"),
    "lille": ("Ligue 1", "Lille.png"),
    "losc lille": ("Ligue 1", "Lille.png"),
    "losc": ("Ligue 1", "Lille.png"),
    "monaco": ("Ligue 1", "Monaco.png"),
    "as monaco": ("Ligue 1", "Monaco.png"),
    "montpellier": ("Ligue 1", "Montpellier.png"),
    "montpellier hsc": ("Ligue 1", "Montpellier.png"),
    "nice": ("Ligue 1", "Nice.png"),
    "ogc nice": ("Ligue 1", "Nice.png"),
    "lyon": ("Ligue 1", "OlympiqueLyonnais.png"),
    "olympique lyon": ("Ligue 1", "OlympiqueLyonnais.png"),
    "olympique lyonnais": ("Ligue 1", "OlympiqueLyonnais.png"),
    "marseille": ("Ligue 1", "OlympiqueMarseille.png"),
    "olympique marseille": ("Ligue 1", "OlympiqueMarseille.png"),
    "olympique de marseille": ("Ligue 1", "OlympiqueMarseille.png"),
    "om": ("Ligue 1", "OlympiqueMarseille.png"),
    "paris fc": ("Ligue 1", "ParisFC.png"),
    "paris saint germain": ("Ligue 1", "ParisSaintGermain.png"),
    "paris saint-germain": ("Ligue 1", "ParisSaintGermain.png"),
    "psg": ("Ligue 1", "ParisSaintGermain.png"),
    "reims": ("Ligue 1", "Reims.png"),
    "stade de reims": ("Ligue 1", "Reims.png"),
    "rennes": ("Ligue 1", "StadeRennais.png"),
    "stade rennais": ("Ligue 1", "StadeRennais.png"),
    "strasbourg": ("Ligue 1", "Strasburgo.png"),
    "rc strasbourg": ("Ligue 1", "Strasburgo.png"),
    "rc strasbourg alsace": ("Ligue 1", "Strasburgo.png"),
    "toulouse": ("Ligue 1", "Toulouse.png"),
    "toulouse fc": ("Ligue 1", "Toulouse.png"),
    "lens": ("Ligue 1", "Lens.png"),
    "rc lens": ("Ligue 1", "Lens.png"),
    "saint etienne": ("Ligue 1", "SaintEtienne.png"),
    "saint-etienne": ("Ligue 1", "SaintEtienne.png"),
    "as saint-etienne": ("Ligue 1", "SaintEtienne.png"),
    
    # ============ PRIMEIRA LIGA (PORTOGALLO) ============
    "avs": ("Premeira liga", "AVSFutebol.png"),
    "avs futebol": ("Premeira liga", "AVSFutebol.png"),
    "braga": ("Premeira liga", "Braga.png"),
    "sc braga": ("Premeira liga", "Braga.png"),
    "sporting braga": ("Premeira liga", "Braga.png"),
    "casa pia": ("Premeira liga", "CasaPia.png"),
    "casa pia ac": ("Premeira liga", "CasaPia.png"),
    "nacional": ("Premeira liga", "CDNacional.png"),
    "cd nacional": ("Premeira liga", "CDNacional.png"),
    "tondela": ("Premeira liga", "CDTondela.png"),
    "cd tondela": ("Premeira liga", "CDTondela.png"),
    "estoril": ("Premeira liga", "EstorilPraia.png"),
    "estoril praia": ("Premeira liga", "EstorilPraia.png"),
    "estrela amadora": ("Premeira liga", "EstrelaAmadora.png"),
    "estrela da amadora": ("Premeira liga", "EstrelaAmadora.png"),
    "alverca": ("Premeira liga", "FCAlverca.png"),
    "fc alverca": ("Premeira liga", "FCAlverca.png"),
    "arouca": ("Premeira liga", "FcArouca.png"),
    "fc arouca": ("Premeira liga", "FcArouca.png"),
    "famalicao": ("Premeira liga", "FCFamalicão.png"),
    "fc famalicao": ("Premeira liga", "FCFamalicão.png"),
    "porto": ("Premeira liga", "FCPorto.png"),
    "fc porto": ("Premeira liga", "FCPorto.png"),
    "gil vicente": ("Premeira liga", "GilVicente.png"),
    "moreirense": ("Premeira liga", "Moreirense.png"),
    "moreirense fc": ("Premeira liga", "Moreirense.png"),
    "rio ave": ("Premeira liga", "RioAve.png"),
    "rio ave fc": ("Premeira liga", "RioAve.png"),
    "santa clara": ("Premeira liga", "SantaClara.png"),
    "cd santa clara": ("Premeira liga", "SantaClara.png"),
    "benfica": ("Premeira liga", "SLBenfica.png"),
    "sl benfica": ("Premeira liga", "SLBenfica.png"),
    "sporting": ("Premeira liga", "SportingLisbona.png"),
    "sporting cp": ("Premeira liga", "SportingLisbona.png"),
    "sporting lisbon": ("Premeira liga", "SportingLisbona.png"),
    "vitoria guimaraes": ("Premeira liga", "VitóriaGuimarães.png"),
    "vitoria sc": ("Premeira liga", "VitóriaGuimarães.png"),
    "guimaraes": ("Premeira liga", "VitóriaGuimarães.png"),
    "boavista": ("Premeira liga", "Boavista.png"),
    "boavista fc": ("Premeira liga", "Boavista.png"),
    "farense": ("Premeira liga", "Farense.png"),
    "sc farense": ("Premeira liga", "Farense.png"),
}

# Cache per le immagini già caricate (evita di rileggere il file ogni volta)
_logo_cache: Dict[str, str] = {}


def get_logo_path(team_name: str) -> Optional[str]:
    """
    Ottiene il percorso completo del file logo per una squadra.
    
    Args:
        team_name: Nome della squadra (case-insensitive)
    
    Returns:
        Percorso completo del file logo o None se non trovato
    """
    team_key = team_name.lower().strip()
    
    if team_key not in TEAM_LOGO_MAPPING:
        # Prova a cercare match parziale
        for key in TEAM_LOGO_MAPPING:
            if key in team_key or team_key in key:
                team_key = key
                break
        else:
            return None
    
    folder, filename = TEAM_LOGO_MAPPING[team_key]
    logo_path = os.path.join(LOGOS_BASE_PATH, folder, filename)
    
    # Controlla anche varianti del nome file (con caratteri speciali)
    if not os.path.exists(logo_path):
        # Cerca file simili nella cartella
        folder_path = os.path.join(LOGOS_BASE_PATH, folder)
        if os.path.exists(folder_path):
            base_name = filename.split('.')[0].lower()
            for f in os.listdir(folder_path):
                if base_name in f.lower() or f.lower().startswith(base_name[:5]):
                    logo_path = os.path.join(folder_path, f)
                    if os.path.exists(logo_path):
                        break
    
    return logo_path if os.path.exists(logo_path) else None


def get_logo_base64(team_name: str, size: int = 40) -> Optional[str]:
    """
    Ottiene il logo di una squadra come stringa base64 per HTML.
    
    Args:
        team_name: Nome della squadra
        size: Dimensione desiderata (non usata direttamente, solo per cache key)
    
    Returns:
        Stringa base64 dell'immagine o None se non trovata
    """
    cache_key = f"{team_name.lower()}_{size}"
    
    if cache_key in _logo_cache:
        return _logo_cache[cache_key]
    
    logo_path = get_logo_path(team_name)
    
    if not logo_path:
        return None
    
    try:
        with open(logo_path, "rb") as f:
            img_data = f.read()
        
        # Determina il tipo MIME
        ext = logo_path.lower().split('.')[-1]
        mime_type = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }.get(ext, 'image/png')
        
        base64_str = base64.b64encode(img_data).decode('utf-8')
        result = f"data:{mime_type};base64,{base64_str}"
        
        _logo_cache[cache_key] = result
        return result
        
    except Exception as e:
        print(f"⚠️ Errore caricamento logo {team_name}: {e}")
        return None


def get_team_logo_html(team_name: str, size: int = 40, position: str = "left") -> str:
    """
    Genera HTML per visualizzare il logo di una squadra.
    
    Args:
        team_name: Nome della squadra
        size: Dimensione in pixel
        position: "left" (logo a sinistra del nome) o "right" (logo a destra)
    
    Returns:
        Stringa HTML con logo e nome squadra
    """
    logo_base64 = get_logo_base64(team_name, size)
    
    if logo_base64:
        logo_html = f'<img src="{logo_base64}" style="width:{size}px; height:{size}px; object-fit:contain; vertical-align:middle;">'
    else:
        # Emoji placeholder se logo non trovato
        logo_html = f'<span style="font-size:{size-10}px;">⚽</span>'
    
    name_html = f'<span style="font-size:1.1em; font-weight:600; vertical-align:middle;">{team_name}</span>'
    
    if position == "left":
        return f'{logo_html}&nbsp;&nbsp;{name_html}'
    else:
        return f'{name_html}&nbsp;&nbsp;{logo_html}'


def get_match_header_html(home_team: str, away_team: str, logo_size: int = 50) -> str:
    """
    Genera HTML per l'header della partita con loghi.
    
    Layout: [Logo Casa] Nome Casa    vs    Nome Trasferta [Logo Trasferta]
    
    Args:
        home_team: Nome squadra di casa
        away_team: Nome squadra in trasferta
        logo_size: Dimensione loghi in pixel
    
    Returns:
        Stringa HTML completa
    """
    try:
        home_logo = get_logo_base64(home_team, logo_size)
        away_logo = get_logo_base64(away_team, logo_size)
    except Exception as e:
        print(f"⚠️ Errore caricamento loghi: {e}")
        home_logo = None
        away_logo = None
    
    # Logo casa (o placeholder)
    if home_logo and len(home_logo) < 500000:  # Limita dimensione base64
        home_logo_html = f'<img src="{home_logo}" style="width:{logo_size}px; height:{logo_size}px; object-fit:contain;">'
    else:
        home_logo_html = f'<div style="width:{logo_size}px; height:{logo_size}px; background:#3498db; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:24px;">🏠</div>'
    
    # Logo trasferta (o placeholder)
    if away_logo and len(away_logo) < 500000:
        away_logo_html = f'<img src="{away_logo}" style="width:{logo_size}px; height:{logo_size}px; object-fit:contain;">'
    else:
        away_logo_html = f'<div style="width:{logo_size}px; height:{logo_size}px; background:#e74c3c; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:24px;">✈️</div>'
    
    html = f'''
    <div style="display:flex; align-items:center; justify-content:center; padding:20px; 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
                border-radius:15px; margin:10px 0;">
        
        <div style="display:flex; align-items:center; flex:1; justify-content:flex-end;">
            {home_logo_html}
            <span style="margin-left:15px; font-size:1.4em; font-weight:700; color:#ffffff;">
                {home_team}
            </span>
        </div>
        
        <div style="margin:0 30px; padding:10px 20px; background:rgba(255,255,255,0.1); border-radius:10px;">
            <span style="font-size:1.2em; font-weight:600; color:#f39c12;">VS</span>
        </div>
        
        <div style="display:flex; align-items:center; flex:1; justify-content:flex-start;">
            <span style="margin-right:15px; font-size:1.4em; font-weight:700; color:#ffffff;">
                {away_team}
            </span>
            {away_logo_html}
        </div>
        
    </div>
    '''
    
    return html


def get_match_header_simple(home_team: str, away_team: str) -> str:
    """
    Versione semplificata dell'header senza loghi (fallback).
    """
    return f'''
    <div style="display:flex; align-items:center; justify-content:center; padding:20px; 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
                border-radius:15px; margin:10px 0;">
        
        <span style="font-size:1.5em; font-weight:700; color:#ffffff;">
            🏠 {home_team}
        </span>
        
        <span style="margin:0 25px; padding:8px 16px; background:rgba(255,255,255,0.1); 
                     border-radius:8px; font-size:1.2em; font-weight:600; color:#f39c12;">
            VS
        </span>
        
        <span style="font-size:1.5em; font-weight:700; color:#ffffff;">
            {away_team} ✈️
        </span>
        
    </div>
    '''


def check_logos_availability(teams: list) -> Dict[str, bool]:
    """
    Controlla quali squadre hanno il logo disponibile.
    
    Args:
        teams: Lista di nomi squadre
    
    Returns:
        Dict con {nome_squadra: True/False}
    """
    return {team: get_logo_path(team) is not None for team in teams}


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test con alcune squadre
    test_teams = [
        "Inter", "Milan", "Juventus", "Napoli", "Roma",
        "Liverpool", "Manchester City", "Arsenal",
        "Real Madrid", "Barcelona",
        "Bayern Munich", "Borussia Dortmund",
        "PSG", "Monaco"
    ]
    
    print("🔍 Test disponibilità loghi:\n")
    
    for team in test_teams:
        path = get_logo_path(team)
        if path:
            print(f"✅ {team}: {path}")
        else:
            print(f"❌ {team}: Non trovato")
    
    print("\n" + "="*50)
    print("📊 Riepilogo mapping:")
    print(f"   Squadre mappate: {len(TEAM_LOGO_MAPPING)}")
