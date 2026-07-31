"""
🎰 THE ODDS API - Quote Reali Bookmaker
========================================
v5 — SEPARAZIONE TRA "QUOTE PER PREZZARE" E "QUOTE PER STIMARE"

Il bug della v4: `fetch_match_odds` restituiva, per ogni esito, la quota
MASSIMA su tutti i bookmaker. Quel dizionario veniva poi usato per DUE cose
incompatibili:

  1. calcolare l'EV e mostrare la quota migliore   → corretto
  2. stimare le probabilità implicite del mercato  → SBAGLIATO

Prendere il massimo esito per esito costruisce un book sintetico che nessun
operatore ha mai quotato. Il margine risulta già eroso (a volte sotto 1.00,
cioè un arbitraggio virtuale) e soprattutto la FORMA della distribuzione è
distorta, perché i massimi arrivano da bookmaker con opinioni diverse. Le
probabilità normalizzate da lì non sono le probabilità del mercato: sono un
artefatto. E quell'artefatto finiva sia nel market anchoring sia nel
benchmark Brier che dovrebbe dire se il modello ha un edge.

v5 restituisce entrambe le cose, separate:
  - `best`  : quota massima per esito         → EV, staking, display
  - `sharp` : quote di UN SOLO book sharp     → probabilità di mercato

Un book "sharp" è un operatore che vive di volume e margini bassi e le cui
quote sono il miglior stimatore pubblico disponibile della probabilità vera
(Pinnacle su tutti, poi gli exchange). Se nessuno dei book sharp è presente
per quella partita, si ripiega sul book con il margine più basso tra quelli
disponibili — sempre UN book solo, mai un mosaico.

Nota costi: la v4 faceva 3 chiamate per partita (una per mercato). La v5 ne
fa 1 sola, chiedendo i tre mercati insieme.
"""

import requests
from typing import Dict, Optional, Tuple
import streamlit as st

# Mapping leghe API-Football -> The Odds API
LEAGUE_MAPPING = {
    39: "soccer_epl",                     # Premier League
    135: "soccer_italy_serie_a",          # Serie A
    140: "soccer_spain_la_liga",          # La Liga
    78: "soccer_germany_bundesliga",      # Bundesliga
    61: "soccer_france_ligue_one",        # Ligue 1
    94: "soccer_portugal_primeira_liga",  # Liga Portugal
    88: "soccer_netherlands_eredivisie",  # Eredivisie
    2: "soccer_uefa_champs_league",       # Champions League
}

# Bookmaker "sharp" in ordine di preferenza. Le loro quote sono il miglior
# stimatore pubblico della probabilità vera: margini bassi, limiti alti,
# nessun interesse a sbilanciare il book per marketing.
SHARP_BOOKS = [
    "pinnacle",         # riferimento assoluto del settore
    "betfair_ex_eu",    # exchange: quote = ordini reali, margine ~0
    "betfair_ex_uk",
    "smarkets",         # exchange
    "matchbook",        # exchange
]


def get_odds_api_key() -> Optional[str]:
    """Recupera la chiave API da secrets o config."""
    try:
        return st.secrets.get("THE_ODDS_API_KEY")
    except Exception:
        pass

    try:
        from config import THE_ODDS_API_KEY
        if THE_ODDS_API_KEY and "INSERISCI" not in THE_ODDS_API_KEY:
            return THE_ODDS_API_KEY
    except Exception:
        pass

    return None


# ============================================================
# FETCH PRINCIPALE
# ============================================================

def fetch_match_odds_full(home_team: str, away_team: str,
                          league_id: int) -> Dict:
    """
    Recupera le quote di una partita in forma COMPLETA.

    Returns:
        {
          "best":       {'1': 1.85, 'X': 3.40, ...},   # max per esito
          "sharp":      {'1': 1.80, 'X': 3.30, ...},   # un solo book sharp
          "sharp_book": "pinnacle",                     # quale book
          "n_books":    12,
        }
        Tutti i campi possono essere vuoti/None se le quote non sono trovate.
    """
    empty = {"best": {}, "sharp": {}, "sharp_book": None, "n_books": 0}

    api_key = get_odds_api_key()
    if not api_key:
        return empty

    sport_key = LEAGUE_MAPPING.get(league_id)
    if not sport_key:
        return empty

    # Un'unica chiamata per tutti e tre i mercati
    event = _fetch_event(api_key, sport_key, home_team, away_team,
                         "h2h,totals,btts")
    if not event:
        return empty

    per_book = _odds_by_bookmaker(event)
    if not per_book:
        return empty

    sharp_key, sharp = _pick_sharp_book(per_book)

    return {
        "best": _best_across_books(per_book),
        "sharp": sharp,
        "sharp_book": sharp_key,
        "n_books": len(per_book),
    }


def fetch_match_odds(home_team: str, away_team: str,
                     league_id: int) -> Dict[str, float]:
    """
    Compatibilità con la v4: restituisce solo le quote MIGLIORI.
    Usa `fetch_match_odds_full` se ti servono anche le quote sharp.
    """
    return fetch_match_odds_full(home_team, away_team, league_id).get("best", {})


# ============================================================
# INTERNI
# ============================================================

def _fetch_event(api_key: str, sport_key: str, home_team: str,
                 away_team: str, markets: str) -> Optional[dict]:
    """Scarica gli eventi della lega e restituisce quello che matcha. 1 chiamata."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",
        "markets": markets,
        "oddsFormat": "decimal",
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return None
        for event in response.json():
            if _match_teams(event, home_team, away_team):
                return event
    except Exception:
        return None
    return None


def _outcome_to_code(event: dict, mkt_key: str, outcome: dict) -> Optional[str]:
    """Traduce un outcome di The Odds API nel codice interno ('1','O2.5',...)."""
    name = outcome.get("name", "") or ""
    point = outcome.get("point")

    if mkt_key == "h2h":
        if name == event.get("home_team"):
            return "1"
        if name == event.get("away_team"):
            return "2"
        if name.lower() == "draw":
            return "X"
        return None

    if mkt_key == "totals":
        if point is None:
            return None
        try:
            line = float(point)
        except (TypeError, ValueError):
            return None
        line_str = f"{line:g}"        # 2.5 -> "2.5", evita "2.50"
        if name.lower() == "over":
            return f"O{line_str}"
        if name.lower() == "under":
            return f"U{line_str}"
        return None

    if mkt_key == "btts":
        if name.lower() == "yes":
            return "GG"
        if name.lower() == "no":
            return "NG"
        return None

    return None


def _odds_by_bookmaker(event: dict) -> Dict[str, Dict[str, float]]:
    """
    Riorganizza l'evento in {bookmaker_key: {codice: quota}}.
    Mantenere la struttura per bookmaker è ciò che permette di stimare le
    probabilità da UN solo book invece che da un mosaico.
    """
    per_book: Dict[str, Dict[str, float]] = {}
    for bookmaker in event.get("bookmakers", []):
        bkey = bookmaker.get("key", "")
        if not bkey:
            continue
        book_odds: Dict[str, float] = {}
        for mkt in bookmaker.get("markets", []):
            mkt_key = mkt.get("key", "")
            for outcome in mkt.get("outcomes", []):
                code = _outcome_to_code(event, mkt_key, outcome)
                if not code:
                    continue
                try:
                    price = float(outcome.get("price", 0))
                except (TypeError, ValueError):
                    continue
                if price > 1.0:
                    book_odds[code] = price
        if book_odds:
            per_book[bkey] = book_odds
    return per_book


def _best_across_books(per_book: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Quota massima per ogni esito. Serve per EV e per piazzare, NON per stimare."""
    best: Dict[str, float] = {}
    for book_odds in per_book.values():
        for code, price in book_odds.items():
            if code not in best or price > best[code]:
                best[code] = price
    return best


# Gruppi completi: un book è utilizzabile per stimare probabilità solo se
# quota TUTTI gli esiti di almeno un gruppo.
_GROUPS = [("1", "X", "2"), ("O2.5", "U2.5"), ("GG", "NG")]


def _book_margin(book_odds: Dict[str, float]) -> Optional[float]:
    """
    Margine medio del book sui gruppi che quota per intero.
    Più basso = più sharp. None se non quota nessun gruppo completo.
    """
    margins = []
    for group in _GROUPS:
        if all(g in book_odds for g in group):
            s = sum(1.0 / book_odds[g] for g in group)
            if s > 0:
                margins.append(s)
    if not margins:
        return None
    return sum(margins) / len(margins)


def _pick_sharp_book(per_book: Dict[str, Dict[str, float]]
                     ) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Sceglie UN bookmaker da cui stimare le probabilità di mercato.

    1. Il primo book sharp disponibile in ordine di preferenza (Pinnacle...)
    2. Altrimenti: il book con il margine medio più basso tra quelli che
       quotano almeno un gruppo completo.

    Mai un mosaico di book diversi: la coerenza interna del book è ciò che
    rende le sue probabilità normalizzabili.
    """
    for key in SHARP_BOOKS:
        if key in per_book and _book_margin(per_book[key]) is not None:
            return key, per_book[key]

    best_key, best_margin = None, None
    for key, book_odds in per_book.items():
        m = _book_margin(book_odds)
        if m is None:
            continue
        if best_margin is None or m < best_margin:
            best_key, best_margin = key, m

    return best_key, (per_book.get(best_key, {}) if best_key else {})


# ============================================================
# MATCHING SQUADRE
# ============================================================

def _match_teams(event: dict, home_team: str, away_team: str) -> bool:
    """Verifica se l'evento corrisponde alla partita cercata (matching fuzzy)."""
    api_home = (event.get("home_team") or "").lower()
    api_away = (event.get("away_team") or "").lower()

    home_lower = (home_team or "").lower()
    away_lower = (away_team or "").lower()
    if not api_home or not api_away or not home_lower or not away_lower:
        return False

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
    """Matching fuzzy per nomi squadre ("Inter" vs "Inter Milan")."""
    keywords1 = set(name1.replace("-", " ").split())
    keywords2 = set(name2.replace("-", " ").split())
    common = keywords1 & keywords2
    noise = {"fc", "ac", "as", "sc", "ss", "us", "afc", "cf", "cd",
             "real", "sporting"}
    return len(common - noise) > 0


# ============================================================
# QUOTE DI CHIUSURA (per il CLV)
# ============================================================

def fetch_closing_odds_map(home_team: str, away_team: str,
                           league_id: int) -> Dict[str, float]:
    """
    Tutte le quote migliori disponibili adesso, da chiamare vicino al calcio
    d'inizio per registrare la linea di chiusura.

    Il CLV (Closing Line Value) — differenza tra la quota presa e quella di
    chiusura — è il predittore più affidabile di profittabilità a lungo
    termine, ed è misurabile SUBITO invece di aspettare centinaia di esiti
    come serve per lo yield.
    """
    return fetch_match_odds_full(home_team, away_team, league_id).get("best", {})


def get_best_odds_for_selection(home_team: str, away_team: str,
                                league_id: int,
                                selection: str) -> Optional[float]:
    """Migliore quota per una selezione specifica."""
    if selection.startswith("CO") or selection.startswith("CU") \
            or selection.endswith("cards"):
        return None
    return fetch_match_odds_full(home_team, away_team, league_id) \
        .get("best", {}).get(selection)
