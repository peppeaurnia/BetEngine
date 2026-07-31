"""
🔄 RESULTS UPDATER — Aggiornamento automatico esiti pronostici
===============================================================
Recupera i risultati reali delle partite via API-Football e chiude
i pronostici pendenti nel database (won / lost / void).

Costo API per esecuzione:
- 1 chiamata per fixture pendente (risultato)
- +1 chiamata per fixture se ci sono pronostici Cartellini (statistiche)
Il parametro max_api_calls (default 40) fa da tetto per non bruciare
il piano gratuito (100 chiamate/giorno). Le fixture non processate
restano pending e verranno chiuse all'esecuzione successiva.

Nota sui cartellini: il totale è conteggiato come gialli + rossi di
entrambe le squadre (un'espulsione per doppio giallo può contare in modo
diverso presso alcuni bookmaker — qui vale il conteggio API).
"""

from datetime import date
from typing import Dict, Optional

import requests

import storage

BASE_URL = "https://v3.football.api-sports.io"

# Stati partita conclusa (si può assegnare l'esito)
FINISHED_STATUSES = {"FT", "AET", "PEN"}
# Stati che invalidano il pronostico (partita non giocata regolarmente)
VOID_STATUSES = {"CANC", "ABD", "AWD", "WO"}
# PST (rinviata) resta pending: la partita verrà recuperata


def _headers(api_key: str) -> Dict:
    return {"x-apisports-key": api_key}


def fetch_fixture_result(api_key: str, fixture_id: int) -> Optional[Dict]:
    """Stato e punteggio finale di una fixture. 1 chiamata API."""
    try:
        r = requests.get(f"{BASE_URL}/fixtures",
                         params={"id": fixture_id},
                         headers=_headers(api_key), timeout=15)
        r.raise_for_status()
        resp = r.json().get("response", [])
        if not resp:
            return None
        fx = resp[0]
        return {
            "status": fx.get("fixture", {}).get("status", {}).get("short", ""),
            "home_goals": fx.get("goals", {}).get("home"),
            "away_goals": fx.get("goals", {}).get("away"),
        }
    except Exception:
        return None


def fetch_fixture_cards(api_key: str, fixture_id: int) -> Optional[int]:
    """Totale cartellini (gialli + rossi, entrambe le squadre). 1 chiamata API."""
    try:
        r = requests.get(f"{BASE_URL}/fixtures/statistics",
                         params={"fixture": fixture_id},
                         headers=_headers(api_key), timeout=15)
        r.raise_for_status()
        resp = r.json().get("response", [])
        if not resp:
            return None
        total = 0
        found = False
        for team_block in resp:
            for stat in team_block.get("statistics", []):
                if stat.get("type") in ("Yellow Cards", "Red Cards"):
                    val = stat.get("value")
                    total += int(val) if val is not None else 0
                    found = True
        return total if found else None
    except Exception:
        return None


def determine_outcome(selection_code: str, home_goals: int, away_goals: int,
                      total_cards: Optional[int] = None) -> Optional[str]:
    """
    Dato il codice selezione e il risultato reale, ritorna 'won'/'lost',
    oppure None se non determinabile (es. cartellini non disponibili).

    Codici: 1, X, 2, GG, NG, O{linea}, U{linea}, O{linea}cards, U{linea}cards
    """
    if home_goals is None or away_goals is None:
        return None
    code = str(selection_code).strip()
    total_goals = home_goals + away_goals

    # --- 1X2 ---
    if code == "1":
        return "won" if home_goals > away_goals else "lost"
    if code == "X":
        return "won" if home_goals == away_goals else "lost"
    if code == "2":
        return "won" if away_goals > home_goals else "lost"

    # --- BTTS ---
    if code == "GG":
        return "won" if (home_goals > 0 and away_goals > 0) else "lost"
    if code == "NG":
        return "won" if (home_goals == 0 or away_goals == 0) else "lost"

    # --- Cartellini: O3.5cards / U4.5cards ---
    if code.endswith("cards") and code[0] in ("O", "U"):
        if total_cards is None:
            return None  # riprova alla prossima esecuzione
        try:
            line = float(code[1:-5])
        except ValueError:
            return None
        if code[0] == "O":
            return "won" if total_cards > line else "lost"
        return "won" if total_cards < line else "lost"

    # --- Over/Under gol: O2.5 / U2.5 ---
    if code[0] in ("O", "U"):
        try:
            line = float(code[1:])
        except ValueError:
            return None
        if code[0] == "O":
            return "won" if total_goals > line else "lost"
        return "won" if total_goals < line else "lost"

    return None


def update_closing_odds(max_fixtures: int = 25) -> Dict:
    """
    Registra la quota di CHIUSURA dei pronostici delle partite di oggi e
    calcola il CLV (Closing Line Value).

    Perché conta più di tutto il resto all'inizio: lo yield ha bisogno di
    centinaia di esiti prima di distinguersi dal rumore, il Brier di almeno
    un centinaio. Il CLV no — è una misura diretta del confronto tra la tua
    stima e quella finale del mercato, e con 40-50 scommesse dà già un
    segnale leggibile. Se batti sistematicamente la linea di chiusura hai un
    edge, quasi indipendentemente da come sono andati i singoli risultati.

    Va lanciato VICINO AL CALCIO D'INIZIO (l'ultima ora è quella che conta:
    è lì che il mercato incorpora formazioni e ultime notizie).

    Costo: chiamate a The Odds API (quota separata da API-Football),
    1 per fixture. `max_fixtures` fa da tetto.
    """
    try:
        from odds_api import fetch_closing_odds_map
    except ImportError:
        return {"updated": 0, "fixtures": 0, "error": "odds_api non disponibile"}

    today = date.today().isoformat()
    rows = storage.predictions_awaiting_closing_odds(today)
    if not rows:
        return {"updated": 0, "fixtures": 0, "error": None}

    # Raggruppa per fixture: una sola chiamata copre tutte le sue selezioni
    by_fixture: Dict[int, list] = {}
    for r in rows:
        fid = r.get("fixture_id")
        if fid:
            by_fixture.setdefault(int(fid), []).append(r)

    updated = 0
    fixtures_done = 0
    for fid, preds in list(by_fixture.items())[:max_fixtures]:
        ref = preds[0]
        try:
            closing = fetch_closing_odds_map(ref.get("home", ""),
                                             ref.get("away", ""),
                                             ref.get("league_id"))
        except Exception:
            continue
        fixtures_done += 1
        if not closing:
            continue
        for p in preds:
            code = p.get("selection_code")
            if code in closing:
                storage.set_closing_odds(p["id"], float(closing[code]))
                updated += 1

    return {"updated": updated, "fixtures": fixtures_done,
            "pending_fixtures": len(by_fixture), "error": None}


def update_results(api_key: str, max_api_calls: int = 40) -> Dict:
    """
    Chiude i pronostici pendenti recuperando gli esiti via API.

    Returns: dict riepilogo {checked, settled, voided, still_pending,
             api_calls, capped}
    """
    pending = storage.pending_predictions()
    today = date.today().isoformat()

    # Solo partite già giocate (data <= oggi), raggruppate per fixture
    by_fixture: Dict[int, list] = {}
    for p in pending:
        fid = p.get("fixture_id")
        if fid and p.get("match_date") and p["match_date"] <= today:
            by_fixture.setdefault(int(fid), []).append(p)

    api_calls = 0
    settled = voided = 0
    capped = False

    for fid, preds in by_fixture.items():
        if api_calls >= max_api_calls:
            capped = True
            break

        res = fetch_fixture_result(api_key, fid)
        api_calls += 1
        if res is None:
            continue

        status = res["status"]

        # Partita annullata/assegnata a tavolino → void
        if status in VOID_STATUSES:
            for p in preds:
                storage.settle_prediction(p["id"], "void")
                voided += 1
            continue

        # Non ancora finita (o rinviata) → resta pending
        if status not in FINISHED_STATUSES:
            continue

        hg, ag = res["home_goals"], res["away_goals"]

        # Cartellini: serve una chiamata extra, solo se necessario
        total_cards = None
        if any(str(p.get("selection_code", "")).endswith("cards") for p in preds):
            if api_calls < max_api_calls:
                total_cards = fetch_fixture_cards(api_key, fid)
                api_calls += 1
            else:
                capped = True

        for p in preds:
            outcome = determine_outcome(p["selection_code"], hg, ag, total_cards)
            if outcome:
                storage.settle_prediction(p["id"], outcome, hg, ag, total_cards)
                settled += 1
            # outcome None (es. cartellini mancanti) → resta pending

    still_pending = storage.counts()["pending"]
    return {
        "checked": len(by_fixture),
        "settled": settled,
        "voided": voided,
        "still_pending": still_pending,
        "api_calls": api_calls,
        "capped": capped,
    }
